"""
YuResearchAgent — 核心编排器 (M1: Multi-Agent Orchestrator)

10 状态状态机驱动的异步任务编排引擎：
  IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING
       → ADVERSARIAL（可选）→ EVIDENCE_REFINING → DONE
  失败时进入 REPLANNING，最终可进入 FAILED。

设计亮点:
  - 自研 asyncio + DAG executor，不依赖 LangGraph/AutoGen
  - 拓扑排序后按层并发执行，Semaphore 控制最大并发度
  - 三级降级策略：单任务超时→标记继续；>50%失败→re-plan；全局超时→强制合成
  - 状态机用字典映射实现，便于扩展新状态和转换逻辑
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import logging
import threading
import time
from typing import Any, Callable

from .schemas import (
    OrchestratorState,
    SubTask,
    AgentResult,
    AgentStatus,
    ResearchReport,
    RunConfig,
    TaskType,
)
from .agent_pool import AgentPool
from ..planner.dag import DAG
from ..planner.planner import Planner, PlanParseError
from ..planner.budget_tracker import BudgetTracker
from ..utils.tracing import trace_chain
from ..runtime.events import CancellationToken, RunEvent

# M4: Memory Store 类型提示（延迟导入避免循环依赖）
SharedMemoryStore = Any


logger = logging.getLogger("orchestrator")

__all__ = ["Orchestrator"]


def _looks_like_error(text: str) -> bool:
    """判断子任务输出是否为 LLM/工具错误文本（如限流报错），避免当作有效 claim 写入记忆。"""
    head = (text or "").strip()[:80]
    low = head.lower()
    return low.startswith("error") or "error code:" in low or (head.startswith("[") and "error]" in low)


class Orchestrator:
    """YuResearchAgent 核心编排器。

    Attributes:
        planner: 自适应规划器，负责初始规划和增量重规划。
        agent_pool: Agent 对象池，管理 worker agent 生命周期。
        budget_tracker: Token 预算追踪器。
        memory_store: 全局共享内存，存储所有子任务结果和中间上下文。
        compressor: （预留）上下文压缩器接口。
    """

    def __init__(
        self,
        planner: Planner,
        agent_pool: AgentPool,
        budget_tracker: BudgetTracker | None = None,
        compressor: Any | None = None,
        adversarial_loop: Any | None = None,
        memory_store: Any | None = None,
        summarizer_policy: Any | None = None,
        evidence_store: Any | None = None,
        evidence_verifier: Any | None = None,
        evidence_reviser: Any | None = None,
        run_id: str = "",
        event_sink: Callable[[RunEvent], None] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        self.planner = planner
        self.agent_pool = agent_pool
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.compressor = compressor
        self.adversarial_loop = adversarial_loop
        self.memory_store = memory_store
        self.summarizer_policy = summarizer_policy
        self.evidence_store = evidence_store
        self.evidence_verifier = evidence_verifier
        self.evidence_reviser = evidence_reviser
        self.run_id = run_id or f"run_{time.time_ns()}"
        self.event_sink = event_sink
        self.cancellation_token = cancellation_token
        self._event_sequence = 0
        self._event_lock = threading.Lock()

        # 运行时状态（保留 dict 作为快速缓存，M4 提供持久化 + 语义检索）
        self._memory_store: dict[str, Any] = {}
        self._results: list[AgentResult] = []
        self._all_results: list[AgentResult] = []
        self._dag: DAG | None = None
        self._task_map: dict[str, SubTask] = {}
        # Plans are replaced during gap filling and replanning. Keep the original
        # task descriptions so synthesis can still account for every requested
        # research dimension.
        self._task_history: dict[str, SubTask] = {}
        self._current_state = OrchestratorState.IDLE
        self._query: str = ""
        self._config: RunConfig = RunConfig()
        self._start_time: float = 0.0
        self._replan_count: int = 0
        self._adversarial_count: int = 0
        self._evidence_gap_rounds: int = 0
        self._evidence_audit: Any | None = None

        # 状态机处理器映射
        self._state_handlers: dict[OrchestratorState, Callable[[], asyncio.Future[OrchestratorState]]] = {
            OrchestratorState.IDLE: self._on_idle,
            OrchestratorState.PLANNING: self._do_planning,
            OrchestratorState.DISPATCHING: self._do_dispatching,
            OrchestratorState.COLLECTING: self._do_collecting,
            OrchestratorState.SYNTHESIZING: self._do_synthesizing,
            OrchestratorState.ADVERSARIAL: self._do_adversarial,
            OrchestratorState.EVIDENCE_REFINING: self._do_evidence_refining,
            OrchestratorState.REPLANNING: self._do_replanning,
            OrchestratorState.DONE: self._on_done,
            OrchestratorState.FAILED: self._on_failed,
        }

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @trace_chain(name="orchestrator.run", tags=["m1", "orchestrator"])
    async def run(self, query: str, config: RunConfig | None = None) -> ResearchReport:
        """主入口：执行完整的研究流程。

        Args:
            query: 研究问题。
            config: 运行配置，默认使用 RunConfig()。

        Returns:
            ResearchReport: 最终研究报告。
        """
        self._query = query
        self._config = config or RunConfig()
        self._start_time = time.monotonic()
        self._replan_count = 0
        self._adversarial_count = 0
        self._evidence_gap_rounds = 0
        self._evidence_audit = None
        self._memory_store.clear()
        self._results.clear()
        self._all_results.clear()
        self._dag = None
        self._task_map.clear()
        self._task_history.clear()
        self._current_state = OrchestratorState.IDLE
        with self._event_lock:
            self._event_sequence = 0
        if self.evidence_store is not None:
            self.evidence_store.reset(query)
        self._emit_event(
            "run_started",
            "Research run started.",
            {"query": query[:500]},
            state=OrchestratorState.IDLE,
        )

        # 状态机主循环
        while self._current_state not in (OrchestratorState.DONE, OrchestratorState.FAILED):
            if self._is_cancelled():
                self._apply_cancellation()
                break
            # 全局超时检查
            if self._is_global_timeout():
                if self._memory_store.get("final_report") is not None:
                    logger.warning(
                        "[Orchestrator] Global budget expired after report generation; "
                        "preserving the synthesized report and using the local audit fallback"
                    )
                    self._current_state = OrchestratorState.DONE
                elif self._all_results or self._results:
                    self._memory_store["final_report"] = self._build_timeout_report()
                    self._current_state = OrchestratorState.DONE
                else:
                    self._current_state = OrchestratorState.FAILED
                break

            handler = self._state_handlers.get(self._current_state)
            if handler is None:
                raise RuntimeError(f"Unknown state: {self._current_state}")

            next_state = await handler()
            self._current_state = next_state

            logger.info(f"[Orchestrator] State transition: {self._current_state.value}")
            self._emit_event(
                "state_transition",
                f"Entered {self._current_state.value}.",
                state=self._current_state,
            )
            if self._is_cancelled():
                self._apply_cancellation()
                break

        # 返回结果
        if self._current_state == OrchestratorState.DONE:
            # 最终报告应在 memory 中
            report = self._memory_store.get("final_report")
            if report is None:
                report = ResearchReport(query=query, content="Report generation failed unexpectedly.")
            report.num_replan = self._replan_count
            report.adversarial_rounds = self._adversarial_count
            report.evidence_gap_rounds = self._evidence_gap_rounds
            # Evidence refinement already audits the exact text users receive.
            # The fallback keeps timeout/legacy state transitions auditable.
            if (
                self._config.enable_evidence
                and not report.evidence_audit
                and not self._is_cancelled()
            ):
                await self._finalize_evidence(report)

            # M4: 将最终报告存入 SharedMemoryStore
            if (
                self.memory_store is not None
                and report.run_status not in {"cancelled", "cancelled_partial"}
            ):
                try:
                    from src.memory.long_term import MemoryEntry
                    entry = MemoryEntry(
                        entry_id=f"final_report:{int(time.time())}",
                        claim=str(report.content)[:800],
                        source="orchestrator",
                        confidence=report.confidence,
                        agent_id="orchestrator",
                        timestamp=time.time(),
                        evidence_type="primary",
                        embedding=[],
                        topic=query[:50],
                        metadata={
                            "num_searches": report.num_searches,
                            "num_replan": report.num_replan,
                            "adversarial_rounds": report.adversarial_rounds,
                        },
                    )
                    self.memory_store.put(entry)
                    logger.info(f"[M4] Final report stored to memory (confidence={report.confidence:.2f})")
                except Exception as e:
                    logger.warning(f"[M4] Failed to store final report: {e}")

            self._emit_event(
                "run_completed",
                "Research run reached a terminal state.",
                {
                    "status": report.run_status,
                    "confidence": report.confidence,
                    "num_searches": report.num_searches,
                    "num_replan": report.num_replan,
                    "evidence_gap_rounds": report.evidence_gap_rounds,
                },
                state=OrchestratorState.DONE,
            )
            return report

        # FAILED 状态
        failed_report = ResearchReport(
            query=query,
            content="Research failed due to persistent errors or global timeout.",
            num_replan=self._replan_count,
            adversarial_rounds=self._adversarial_count,
            run_status="failed",
        )
        self._emit_event(
            "run_failed",
            "Research failed before a report could be produced.",
            {"status": "failed"},
            state=OrchestratorState.FAILED,
        )
        return failed_report

    def get_evidence_snapshot(self) -> dict[str, Any]:
        """Return a read-only UI/observability snapshot of the current evidence graph."""
        if self._evidence_audit is None or self.evidence_store is None:
            return {}
        snapshot = self._evidence_audit.to_dict(self.evidence_store.evidence)
        snapshot["sources"] = self.evidence_store.to_source_dicts()
        snapshot["gap_rounds"] = self._evidence_gap_rounds
        report = self._memory_store.get("final_report")
        snapshot["artifact"] = getattr(report, "evidence_artifact", "") if report is not None else ""
        snapshot["revision"] = getattr(report, "evidence_revision", {}) if report is not None else {}
        snapshot["task_coverage"] = getattr(report, "task_coverage", {}) if report is not None else {}
        return snapshot

    def _emit_event(
        self,
        kind: str,
        message: str = "",
        payload: dict[str, Any] | None = None,
        *,
        state: OrchestratorState | None = None,
    ) -> None:
        """Emit a typed event without allowing observers to break a run."""
        if self.event_sink is None:
            return
        with self._event_lock:
            self._event_sequence += 1
            sequence = self._event_sequence
        event = RunEvent(
            run_id=self.run_id,
            sequence=sequence,
            kind=kind,
            state=(state or self._current_state).value,
            message=str(message or "")[:1000],
            payload=dict(payload or {}),
        )
        try:
            self.event_sink(event)
        except Exception as exc:
            logger.warning("[Observability] event sink failed: %s", exc)

    def _is_cancelled(self) -> bool:
        return bool(
            self.cancellation_token is not None
            and self.cancellation_token.is_cancelled
        )

    def _apply_cancellation(self) -> None:
        """Preserve completed work and move the state machine to DONE."""
        report = self._memory_store.get("final_report")
        if report is None:
            report = self._build_partial_report(
                run_status="cancelled_partial" if self._successful_results() else "cancelled",
                heading="Partial results from a cancelled run",
                explanation=(
                    "The run was cancelled before full synthesis. Completed sub-task "
                    "outputs are preserved below."
                ),
                empty_message="Research was cancelled before any sub-task completed successfully.",
            )
            self._memory_store["final_report"] = report
        elif report.run_status != "cancelled":
            report.run_status = "cancelled_partial"
        reason = (
            self.cancellation_token.reason
            if self.cancellation_token is not None
            else "Cancelled by user."
        )
        logger.info("[Orchestrator] Cancellation observed: %s", reason)
        self._emit_event(
            "cancellation_requested",
            reason,
            {
                "requested_at": (
                    self.cancellation_token.requested_at
                    if self.cancellation_token is not None
                    else None
                ),
            },
            state=self._current_state,
        )
        self._emit_event(
            "cancellation_observed",
            reason,
            {
                "status": report.run_status,
                "completed_tasks": len(self._successful_results()),
            },
            state=self._current_state,
        )
        self._current_state = OrchestratorState.DONE

    # ------------------------------------------------------------------
    # 状态机处理器
    # ------------------------------------------------------------------

    async def _on_idle(self) -> OrchestratorState:
        """从 IDLE 自动进入 PLANNING。"""
        return OrchestratorState.PLANNING

    async def _do_planning(self) -> OrchestratorState:
        """调用 Planner 生成初始 DAG。

        失败时直接转入 FAILED（初始计划失败无法恢复）。
        """
        try:
            context_deadline = time.monotonic() + min(
                20.0,
                max(0.25, self._remaining_seconds() - 1.0),
            )
            try:
                memory_ctx = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._build_memory_context,
                        request_deadline_monotonic=context_deadline,
                        cancellation_token=self.cancellation_token,
                    ),
                    timeout=max(0.25, context_deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[M3] Planning context exceeded its budget; continuing without memory context"
                )
                memory_ctx = ""
                self._emit_event(
                    "planning_context_timeout",
                    "Planning context exceeded its budget; using an empty context.",
                    state=OrchestratorState.PLANNING,
                )
            if self._is_cancelled():
                return OrchestratorState.DONE
            bounded_planner = getattr(self.planner, "generate_plan_with_timeout", None)
            if callable(bounded_planner):
                self._dag = await asyncio.to_thread(
                    bounded_planner,
                    self._query,
                    memory_ctx,
                    min(60.0, max(0.25, self._remaining_seconds() - 1.0)),
                )
            else:
                self._dag = await asyncio.to_thread(
                    self.planner.generate_plan,
                    self._query,
                    memory_ctx,
                )
            # 从 planner 获取完整的 SubTask 信息（包括 description、search_hints 等）
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                # 降级：如果解析失败，使用占位符
                self._task_map = self._rebuild_task_map_from_dag()
            self._task_history.update(self._task_map)
        except PlanParseError as e:
            logger.warning(f"[Planning] Failed: {e}")
            return OrchestratorState.FAILED
        except Exception as e:
            logger.warning(f"[Planning] Unexpected error: {e}")
            return OrchestratorState.FAILED

        n_tasks = len(self._dag)
        n_layers = len(self._dag.get_parallel_groups()) if self._dag else 0
        logger.info(f"[Planning] ✓ DAG 生成完成: {n_tasks} 个子任务, {n_layers} 个执行层")
        self._emit_event(
            "plan_created",
            f"Planned {n_tasks} sub-tasks across {n_layers} execution layers.",
            {
                "task_count": n_tasks,
                "layer_count": n_layers,
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "type": task.task_type.value,
                        "description": task.description[:300],
                    }
                    for task in self._task_map.values()
                ],
            },
            state=OrchestratorState.PLANNING,
        )
        # 打印子任务描述以便诊断
        for tid, task in self._task_map.items():
            logger.info(f"[Planning]   {tid}: {task.description}")
        return OrchestratorState.DISPATCHING

    async def _do_dispatching(self) -> OrchestratorState:
        """拓扑排序 + 并发调度 sub-agents。

        核心逻辑:
          1. 获取并行执行层 (parallel groups)
          2. 每层内用 asyncio.gather + Semaphore 并发执行
          3. 每个 sub-task 设置单独超时 (asyncio.wait_for)
          4. 收集结果到 self._results
        """
        if self._dag is None or len(self._dag) == 0:
            return OrchestratorState.COLLECTING

        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        parallel_groups = self._dag.get_parallel_groups()
        all_results: list[AgentResult] = []

        for layer_idx, group in enumerate(parallel_groups):
            if self._is_cancelled():
                break
            logger.info(f"[Dispatch] ▶ Layer {layer_idx + 1}/{len(parallel_groups)}: {group} (并行执行)")
            self._emit_event(
                "dispatch_layer_started",
                f"Executing layer {layer_idx + 1}/{len(parallel_groups)}.",
                {
                    "layer": layer_idx + 1,
                    "layer_count": len(parallel_groups),
                    "task_ids": list(group),
                },
                state=OrchestratorState.DISPATCHING,
            )

            # 构建本层的 coroutine 列表
            async def _run_one(task_id: str) -> AgentResult:
                async with semaphore:
                    if self._is_cancelled():
                        return AgentResult(
                            task_id=task_id,
                            status=AgentStatus.CANCELLED,
                            output=self.cancellation_token.reason,
                        )
                    subtask = self._task_map.get(task_id)
                    if subtask is None:
                        return AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"SubTask '{task_id}' not found in task_map",
                        )

                    # 准备上下文：先执行依赖任务的结果
                    context = self._build_task_context(subtask)

                    result = AgentResult(
                        task_id=task_id,
                        status=AgentStatus.FAILED,
                        output="Task did not start",
                    )
                    attempts = max(1, self._config.max_subagent_retries + 1)
                    for attempt in range(1, attempts + 1):
                        if self._is_cancelled():
                            return AgentResult(
                                task_id=task_id,
                                status=AgentStatus.CANCELLED,
                                output=self.cancellation_token.reason,
                            )
                        task_timeout = self._task_execution_seconds(
                            task_id,
                            subtask.timeout_seconds,
                        )
                        if task_timeout <= 0.25:
                            return AgentResult(
                                task_id=task_id,
                                status=AgentStatus.TIMEOUT,
                                output="No execution budget remained before task execution",
                            )
                        agent = await self.agent_pool.get_agent(subtask.task_type)
                        context["_request_deadline_monotonic"] = (
                            time.monotonic() + max(0.25, task_timeout - 0.5)
                        )
                        self._emit_event(
                            "task_started",
                            subtask.description[:300],
                            {
                                "task_id": task_id,
                                "task_type": subtask.task_type.value,
                                "attempt": attempt,
                                "attempts": attempts,
                            },
                            state=OrchestratorState.DISPATCHING,
                        )
                        try:
                            result = await asyncio.wait_for(
                                agent.run(subtask, context),
                                timeout=task_timeout,
                            )
                        except asyncio.TimeoutError:
                            result = AgentResult(
                                task_id=task_id,
                                status=AgentStatus.TIMEOUT,
                                output=f"Task timed out after {task_timeout:.1f}s",
                            )
                        except Exception as e:
                            result = AgentResult(
                                task_id=task_id,
                                status=AgentStatus.FAILED,
                                output=f"Exception: {type(e).__name__}: {e}",
                            )
                        finally:
                            await self.agent_pool.release_agent(agent)

                        self._emit_event(
                            "task_completed",
                            f"{task_id} finished with status {result.status.value}.",
                            {
                                "task_id": task_id,
                                "status": result.status.value,
                                "confidence": result.confidence,
                                "tool_calls": sum(
                                    1
                                    for step in result.trajectory
                                    if step.get("role") == "tool"
                                ),
                            },
                            state=OrchestratorState.DISPATCHING,
                        )

                        if (
                            result.status in {AgentStatus.SUCCESS, AgentStatus.CANCELLED}
                            or attempt >= attempts
                            or self._is_cancelled()
                        ):
                            return result
                        logger.warning(
                            f"[Dispatch] {task_id} attempt {attempt}/{attempts} failed; retrying"
                        )
                    return result

            # 并发执行本层
            coros = [_run_one(tid) for tid in group]
            layer_results = await asyncio.gather(*coros, return_exceptions=True)

            for lr in layer_results:
                if isinstance(lr, Exception):
                    # 将异常包装为 FAILED 结果
                    # 这种情况理论上不会发生（_run_one 内部已捕获），但保险起见
                    all_results.append(AgentResult(
                        task_id="unknown",
                        status=AgentStatus.FAILED,
                        output=f"Dispatch exception: {lr}",
                    ))
                else:
                    all_results.append(lr)

            self._emit_event(
                "dispatch_layer_completed",
                f"Layer {layer_idx + 1} completed.",
                {
                    "layer": layer_idx + 1,
                    "success_count": sum(
                        result.status == AgentStatus.SUCCESS for result in all_results
                    ),
                    "completed_count": len(all_results),
                },
                state=OrchestratorState.DISPATCHING,
            )

        self._results = all_results
        self._record_result_history(all_results)
        return OrchestratorState.COLLECTING

    async def _do_collecting(self) -> OrchestratorState:
        """收集结果，写入 memory，检查是否需要重规划。

        三级降级策略检查点:
          - 单任务超时/失败：已在 dispatch 层处理（标记状态，继续执行）
          - >50% 失败：触发 REPLANNING
          - 全局超时：由外层 run() 的循环检查处理
        """
        # 将结果写入运行时 memory dict
        for r in self._results:
            self._memory_store[f"result:{r.task_id}"] = r

        # M4: 将成功结果同步写入 SharedMemoryStore（持久化 + 向量索引）
        if self.memory_store is not None:
            for r in self._results:
                if r.status == AgentStatus.SUCCESS and r.output and not _looks_like_error(str(r.output)):
                    self._sync_result_to_memory_store(r)

        if (
            self._config.enable_evidence
            and self.evidence_store is not None
            and self.evidence_verifier is not None
        ):
            self.evidence_store.ingest_results(self._results)
            self._evidence_audit = await asyncio.to_thread(
                self.evidence_verifier.audit_results,
                self._all_results,
                self.evidence_store,
                use_llm=False,
            )
            remaining = self._remaining_seconds()
            hybrid_budget = min(
                20.0,
                max(
                    0.0,
                    remaining
                    - self._effective_synthesis_reserve_seconds()
                    - self._effective_final_audit_reserve_seconds()
                    - 2.0,
                ),
            )
            if (
                self.evidence_verifier.mode == "hybrid"
                and self.evidence_verifier.policy is not None
                and hybrid_budget >= 3.0
            ):
                try:
                    self._evidence_audit = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.evidence_verifier.audit_results,
                            self._all_results,
                            self.evidence_store,
                            use_llm=True,
                            timeout_seconds=max(0.25, hybrid_budget - 1.0),
                        ),
                        timeout=hybrid_budget,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[Evidence] hybrid pre-synthesis audit timed out; using heuristic audit"
                    )
                    self._evidence_audit.verification_mode = "hybrid_unavailable"
            audit = self._evidence_audit
            logger.info(
                "[Evidence] coverage=%.1f%% | claims=%d | supported=%d | NEI=%d | "
                "sources=%d | fulltext=%.1f%% | primary=%.1f%%",
                audit.coverage * 100,
                len(audit.claims),
                audit.supported_count,
                audit.nei_count,
                audit.source_count,
                audit.fulltext_source_ratio * 100,
                audit.primary_source_ratio * 100,
            )
            self._emit_event(
                "evidence_snapshot",
                f"Evidence coverage is {audit.coverage:.1%}.",
                {
                    "coverage": audit.coverage,
                    "claim_count": len(audit.claims),
                    "supported_count": audit.supported_count,
                    "refuted_count": audit.refuted_count,
                    "not_enough_evidence_count": audit.nei_count,
                    "source_count": audit.source_count,
                    "fulltext_source_ratio": audit.fulltext_source_ratio,
                    "primary_source_ratio": audit.primary_source_ratio,
                    "gap_rounds": self._evidence_gap_rounds,
                },
                state=OrchestratorState.COLLECTING,
            )

        success_count = sum(1 for r in self._results if r.status == AgentStatus.SUCCESS)
        total_count = len(self._results)
        fail_count = total_count - success_count
        status_icon = "✓" if success_count == total_count else "⚠"
        if fail_count > 0:
            logger.info(f"[Collect] {status_icon} 子任务完成: {success_count}/{total_count} 成功 ({fail_count} 失败)")
        else:
            logger.info(f"[Collect] {status_icon} 子任务完成: {success_count}/{total_count} 成功")
        self._emit_event(
            "collection_completed",
            f"Collected {success_count}/{total_count} successful sub-tasks.",
            {
                "success_count": success_count,
                "failed_count": fail_count,
                "total_count": total_count,
            },
            state=OrchestratorState.COLLECTING,
        )

        # 检查是否需要重规划
        if self._config.enable_replan and self._should_replan(self._results):
            if self._replan_count < self._config.max_replan_rounds:
                self._replan_count += 1
                return OrchestratorState.REPLANNING
            else:
                logger.warning("[Collect] Max replan rounds reached, proceeding with partial results")
                # 超过最大重规划次数，继续合成（用已有结果）

        if self._should_fill_evidence_gaps():
            self._prepare_evidence_gap_round()
            return OrchestratorState.DISPATCHING

        return OrchestratorState.SYNTHESIZING

    def _sync_result_to_memory_store(self, result: AgentResult) -> None:
        """将 AgentResult 同步到 M4 SharedMemoryStore。

        提取 output 中的关键 claim 作为记忆条目，支持后续语义检索。
        """
        try:
            # 延迟导入避免循环依赖
            from src.memory.long_term import MemoryEntry
            claim_text = str(result.output)[:500]  # 取前 500 字作为 claim
            entry = MemoryEntry(
                entry_id=result.task_id,
                claim=claim_text,
                source=f"task:{result.task_id}",
                confidence=getattr(result, "confidence", 0.5),
                agent_id=result.task_id,
                timestamp=time.time(),
                evidence_type="primary",
                embedding=[],  # SharedMemoryStore.put() 会自动生成 embedding
                topic=self._query[:50],
                metadata={
                    "status": result.status.value,
                    "token_usage": getattr(result, "token_usage", 0),
                },
            )
            self.memory_store.put(entry)
            logger.info(f"[M4] Memory stored: {result.task_id} (claim={claim_text[:60]}...)")
        except Exception as e:
            logger.warning(f"[M4] Failed to store memory for {result.task_id}: {e}")

    async def _do_synthesizing(self) -> OrchestratorState:
        """调用 SummarizerAgent 合成研究报告。"""
        if self._is_cancelled():
            return OrchestratorState.DONE
        # 创建合成任务
        synth_task = SubTask(
            task_id="synthesize_final",
            task_type=TaskType.ANALYZE,  # 使用 ANALYZE 类型，实际由 SummarizerAgent 处理
            description="Synthesize all sub-task results into a final research report.",
            timeout_seconds=300,
        )

        raw_synthesis_results = self._all_results or self._results
        remaining = self._remaining_seconds()
        final_audit_reserve = self._effective_final_audit_reserve_seconds()
        compression_window = min(
            30.0,
            max(0.25, (remaining - final_audit_reserve - 2.0) * 0.20),
        )
        compression_deadline = time.monotonic() + compression_window
        self._emit_event(
            "context_compression_started",
            "Preparing bounded synthesis context.",
            {
                "result_count": len(raw_synthesis_results),
                "timeout_seconds": compression_window,
            },
            state=OrchestratorState.SYNTHESIZING,
        )
        try:
            synthesis_results = await asyncio.wait_for(
                asyncio.to_thread(
                    self._prepare_synthesis_results,
                    raw_synthesis_results,
                    request_deadline_monotonic=compression_deadline,
                    cancellation_token=self.cancellation_token,
                ),
                timeout=compression_window,
            )
            compression_status = "complete"
        except asyncio.TimeoutError:
            logger.warning(
                "[M3] Context compression exceeded %.1fs; using raw synthesis results",
                compression_window,
            )
            synthesis_results = raw_synthesis_results
            compression_status = "timeout_fallback"
        self._emit_event(
            "context_compression_completed",
            "Synthesis context is ready.",
            {
                "status": compression_status,
                "input_characters": sum(
                    len(str(result.output or "")) for result in raw_synthesis_results
                ),
                "output_characters": sum(
                    len(str(result.output or "")) for result in synthesis_results
                ),
            },
            state=OrchestratorState.SYNTHESIZING,
        )
        if self._is_cancelled():
            return OrchestratorState.DONE
        context = {
            "query": self._query,
            "as_of_date": self._config.as_of_date,
            "results": synthesis_results,
            "coverage_requirements": self._synthesis_coverage_requirements(raw_synthesis_results),
            "evidence_audit": (
                self._evidence_audit.to_dict(self.evidence_store.evidence)
                if self._evidence_audit is not None and self.evidence_store is not None
                else {}
            ),
            "evidence_sources": (
                self.evidence_store.to_source_dicts() if self.evidence_store is not None else []
            ),
            "_cancellation_token": self.cancellation_token,
        }

        self._emit_event(
            "synthesis_started",
            "Synthesizing the final report.",
            {"result_count": len(synthesis_results)},
            state=OrchestratorState.SYNTHESIZING,
        )

        agent = await self.agent_pool.get_agent(TaskType.ANALYZE)
        pooled_agent = agent
        release_pooled = True
        # 需要 SummarizerAgent，但 agent_pool 可能返回 ResearcherAgent
        # 这里我们通过类型检查或强制创建 SummarizerAgent
        from ..agents.summarizer import SummarizerAgent
        if not isinstance(agent, SummarizerAgent):
            # 优先使用配置的 summarizer_policy（更大的 max_tokens），fallback 到 agent.policy
            policy = self.summarizer_policy or agent.policy
            tools = agent.tools
            await self.agent_pool.release_agent(pooled_agent)
            release_pooled = False
            agent = SummarizerAgent(name="summarizer", policy=policy, tools=tools)

        remaining = self._remaining_seconds()
        synthesis_timeout = min(
            synth_task.timeout_seconds,
            max(1.0, remaining - final_audit_reserve),
        )
        context["_request_deadline_monotonic"] = (
            time.monotonic() + max(0.25, synthesis_timeout - 0.5)
        )
        try:
            result = await asyncio.wait_for(
                agent.run(synth_task, context),
                timeout=synthesis_timeout,
            )
        except asyncio.TimeoutError:
            result = AgentResult(
                task_id="synthesize_final",
                status=AgentStatus.TIMEOUT,
                output="Synthesis timed out",
            )
        except Exception as e:
            result = AgentResult(
                task_id="synthesize_final",
                status=AgentStatus.FAILED,
                output=f"Synthesis error: {type(e).__name__}: {e}",
            )
        finally:
            if release_pooled:
                await self.agent_pool.release_agent(pooled_agent)

        if result.status == AgentStatus.SUCCESS and isinstance(result.output, ResearchReport):
            self._memory_store["final_report"] = result.output
        else:
            # 合成失败时仍保留已完成子任务，而不是只返回错误字符串。
            if result.status == AgentStatus.CANCELLED:
                fallback = self._build_partial_report(
                    run_status=(
                        "cancelled_partial" if self._successful_results() else "cancelled"
                    ),
                    heading="Partial results from a cancelled run",
                    explanation=(
                        "The run was cancelled during synthesis. Completed sub-task "
                        "outputs are preserved below."
                    ),
                    empty_message=(
                        "Research was cancelled before any sub-task completed successfully."
                    ),
                )
            else:
                fallback = self._build_timeout_report()
                fallback.run_status = (
                    "partial_timeout"
                    if result.status == AgentStatus.TIMEOUT
                    else "partial_failure"
                )
            self._memory_store["final_report"] = fallback

        final_report = self._memory_store.get("final_report")
        self._emit_event(
            "synthesis_completed",
            f"Synthesis finished with status {result.status.value}.",
            {
                "status": result.status.value,
                "report_status": getattr(final_report, "run_status", "partial_failure"),
                "confidence": getattr(final_report, "confidence", 0.0),
            },
            state=OrchestratorState.SYNTHESIZING,
        )

        if self._config.enable_adversarial:
            logger.info("[Synthesize] ✓ 报告合成完成，进入对抗优化")
            return OrchestratorState.ADVERSARIAL
        logger.info("[Synthesize] ✓ 报告合成完成")
        return self._post_generation_state()

    async def _do_adversarial(self) -> OrchestratorState:
        """M5: Red-Blue 对抗降噪循环。

        调用 AdversarialLoop 对报告进行 challenge-verify 迭代优化。
        仅在报告置信度低于阈值时触发，避免资源浪费。
        """
        report = self._memory_store.get("final_report")
        if report is None:
            return self._post_generation_state()
        if self._is_cancelled():
            return OrchestratorState.DONE

        # 置信度足够高时跳过对抗
        if report.confidence >= 0.8:
            logger.info("[Adversarial] ✓ 报告置信度已达标 (≥0.8)，跳过对抗优化")
            return self._post_generation_state()

        if self.adversarial_loop is None:
            logger.info("[Adversarial] AdversarialLoop 未配置，跳过")
            return self._post_generation_state()

        # 剩余全局时间预算：对抗循环是单个长状态，不限时会绕过状态机层面的全局超时检查，
        # 故用 asyncio.wait_for 显式套上剩余预算，避免限流重试时跑失控（实测曾达 1394s）。
        remaining = self._config.global_timeout_seconds - (time.monotonic() - self._start_time)
        if remaining <= 0:
            logger.warning("[Adversarial] 全局时间已耗尽，跳过对抗优化")
            return self._post_generation_state()

        try:
            logger.info(
                f"[Adversarial] ▶ 启动 Red-Blue 对抗优化 "
                f"(当前置信度={report.confidence:.2f}, 剩余预算={remaining:.0f}s)"
            )
            run_parameters = inspect.signature(self.adversarial_loop.run).parameters
            run_kwargs: dict[str, Any] = {}
            if "cancellation_token" in run_parameters:
                run_kwargs["cancellation_token"] = self.cancellation_token
            if "timeout_seconds" in run_parameters:
                run_kwargs["timeout_seconds"] = max(0.25, remaining - 0.5)
            adversarial_run = self.adversarial_loop.run(report, **run_kwargs)
            optimized_report, history = await asyncio.wait_for(
                adversarial_run,
                timeout=remaining,
            )
            self._adversarial_count += len(history)
            # 防截断保护：Blue 重写整篇报告时可能被 max_tokens 截断，
            # 若优化后正文显著变短（<60%）则判定截断，保留更完整的原报告。
            orig_len = len(report.content or "")
            new_len = len(optimized_report.content or "")
            if orig_len > 0 and new_len < orig_len * 0.6:
                logger.warning(
                    f"[Adversarial] 优化后报告显著变短 ({orig_len}→{new_len} 字)，疑似截断，保留原报告"
                )
            else:
                self._memory_store["final_report"] = optimized_report
                logger.info(
                    f"[Adversarial] ✓ 对抗优化完成: {len(history)} 轮, "
                    f"最终置信度={optimized_report.confidence:.2f}"
                )
        except asyncio.TimeoutError:
            logger.warning(
                f"[Adversarial] ✗ 超出剩余全局时间预算 ({remaining:.0f}s)，中止对抗，保留当前报告"
            )
        except Exception as e:
            logger.warning(f"[Adversarial] ✗ 对抗优化失败: {e}，使用原始报告")

        return self._post_generation_state()

    def _post_generation_state(self) -> OrchestratorState:
        if (
            self._config.enable_evidence
            and self.evidence_store is not None
            and self.evidence_verifier is not None
        ):
            return OrchestratorState.EVIDENCE_REFINING
        return OrchestratorState.DONE

    async def _do_evidence_refining(self) -> OrchestratorState:
        """Audit the final text, optionally revise it, then accept only a better draft."""
        report = self._memory_store.get("final_report")
        if report is None:
            return OrchestratorState.DONE
        if self._is_cancelled():
            return OrchestratorState.DONE

        self._emit_event(
            "final_audit_started",
            "Auditing claims in the synthesized report.",
            state=OrchestratorState.EVIDENCE_REFINING,
        )

        # Acceptance uses a deterministic audit. Hybrid verification is also
        # computed once here so cross-language evidence is not mislabeled as NEI
        # in the editor prompt; it remains the final audit when no edit is used.
        before_gate_audit = await self._audit_report_text(report, use_hybrid=False)
        if before_gate_audit is None:
            return OrchestratorState.DONE
        if self._is_cancelled():
            return OrchestratorState.DONE
        is_complete_report = report.run_status == "complete"
        if is_complete_report:
            before_audit = (
                await self._audit_report_text(report, use_hybrid=True)
                or before_gate_audit
            )
        else:
            before_audit = before_gate_audit

        report.evidence_audit = before_audit.to_dict(self.evidence_store.evidence)
        self._evidence_audit = before_audit
        self._emit_event(
            "final_audit_completed",
            f"Final report evidence coverage is {before_audit.coverage:.1%}.",
            {
                "coverage": before_audit.coverage,
                "claim_count": len(before_audit.claims),
                "supported_count": before_audit.supported_count,
                "refuted_count": before_audit.refuted_count,
                "not_enough_evidence_count": before_audit.nei_count,
                "source_count": before_audit.source_count,
                "fulltext_source_ratio": before_audit.fulltext_source_ratio,
                "primary_source_ratio": before_audit.primary_source_ratio,
                "gap_rounds": self._evidence_gap_rounds,
            },
            state=OrchestratorState.EVIDENCE_REFINING,
        )
        if self._is_cancelled():
            return OrchestratorState.DONE
        if not is_complete_report:
            reason = f"Revision is disabled for {report.run_status} reports."
            report.evidence_revision = {
                "attempted": False,
                "accepted": False,
                "reason": reason,
                "before": self._audit_metrics(before_audit),
                "after": self._audit_metrics(before_audit),
            }
            await self._commit_evidence_audit(report, before_audit)
            self._emit_event(
                "revision_skipped",
                reason,
                {
                    "coverage": before_audit.coverage,
                    "unresolved_count": before_audit.refuted_count + before_audit.nei_count,
                },
                state=OrchestratorState.EVIDENCE_REFINING,
            )
            return OrchestratorState.DONE
        should_revise = (
            self._config.enable_evidence_revision
            and self.evidence_reviser is not None
            and before_audit.claims
            and before_audit.verification_mode not in {
                "hybrid_unavailable",
                "hybrid_partial",
            }
            and before_audit.coverage < self._config.evidence_revision_trigger_coverage
            and (before_audit.refuted_count > 0 or before_audit.nei_count > 0)
        )
        if not should_revise:
            report.evidence_revision = {
                "attempted": False,
                "accepted": False,
                "reason": "Revision trigger was not met.",
                "before": self._audit_metrics(before_audit),
                "after": self._audit_metrics(before_audit),
            }
            await self._commit_evidence_audit(report, before_audit)
            logger.info(
                "[EvidenceRevision] skipped | coverage=%.1f%% | unresolved=%d",
                before_audit.coverage * 100,
                before_audit.refuted_count + before_audit.nei_count,
            )
            self._emit_event(
                "revision_skipped",
                "Evidence revision trigger was not met.",
                {
                    "coverage": before_audit.coverage,
                    "unresolved_count": before_audit.refuted_count + before_audit.nei_count,
                },
                state=OrchestratorState.EVIDENCE_REFINING,
            )
            return OrchestratorState.DONE

        remaining = self._remaining_seconds()
        timeout = min(self._config.evidence_revision_timeout_seconds, max(0.0, remaining - 2.0))
        if timeout < 1.0:
            report.evidence_revision = {
                "attempted": False,
                "accepted": False,
                "reason": "Insufficient global time remained for evidence revision.",
                "before": self._audit_metrics(before_audit),
                "after": self._audit_metrics(before_audit),
            }
            await self._commit_evidence_audit(report, before_audit)
            return OrchestratorState.DONE

        original_content = report.content
        sources = list(report.sources or self.evidence_store.to_source_dicts())
        logger.info(
            "[EvidenceRevision] starting | coverage=%.1f%% | refuted=%d | NEI=%d | budget=%.0fs",
            before_audit.coverage * 100,
            before_audit.refuted_count,
            before_audit.nei_count,
            timeout,
        )
        self._emit_event(
            "revision_started",
            "Generating one evidence-bounded revision candidate.",
            {
                "coverage": before_audit.coverage,
                "refuted_count": before_audit.refuted_count,
                "not_enough_evidence_count": before_audit.nei_count,
                "timeout_seconds": timeout,
            },
            state=OrchestratorState.EVIDENCE_REFINING,
        )
        try:
            draft = await asyncio.wait_for(
                asyncio.to_thread(
                    self.evidence_reviser.revise,
                    query=self._query,
                    content=original_content,
                    audit=report.evidence_audit,
                    sources=sources,
                    request_timeout_seconds=max(0.25, timeout - 1.0),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            draft = None
            rejection_reason = "Evidence revision timed out."
        except Exception as exc:
            draft = None
            rejection_reason = f"Evidence revision failed: {type(exc).__name__}: {exc}"
        else:
            rejection_reason = draft.reason if not draft.valid else ""

        if self._is_cancelled():
            return OrchestratorState.DONE

        if draft is None or not draft.valid:
            revision_metadata = {
                "attempted": True,
                "accepted": False,
                "reason": rejection_reason,
                "before": self._audit_metrics(before_audit),
                "after": self._audit_metrics(before_audit),
                "original_sha256": self._text_sha256(original_content),
            }
            if draft is not None and draft.content:
                revision_metadata["candidate_sha256"] = self._text_sha256(draft.content)
            report.evidence_revision = revision_metadata
            await self._commit_evidence_audit(report, before_audit)
            logger.warning("[EvidenceRevision] rejected before re-audit: %s", rejection_reason)
            self._emit_event(
                "revision_rejected",
                rejection_reason,
                {"accepted": False, "before": revision_metadata["before"]},
                state=OrchestratorState.EVIDENCE_REFINING,
            )
            return OrchestratorState.DONE

        # Keep the acceptance pass deterministic. Hybrid verdict noise must not
        # decide whether a model-authored revision replaces the original report.
        after_gate_audit = await self._audit_report_text_content(draft.content, report, use_hybrid=False)
        if self._is_cancelled():
            return OrchestratorState.DONE
        if after_gate_audit is None:
            report.evidence_revision = {
                "attempted": True,
                "accepted": False,
                "reason": "The revision could not be re-audited.",
                "before": self._audit_metrics(before_audit),
                "after": self._audit_metrics(before_audit),
                "original_sha256": self._text_sha256(original_content),
                "candidate_sha256": self._text_sha256(draft.content),
            }
            await self._commit_evidence_audit(report, before_audit)
            self._emit_event(
                "revision_rejected",
                "The revision could not be re-audited.",
                {"accepted": False},
                state=OrchestratorState.EVIDENCE_REFINING,
            )
            return OrchestratorState.DONE

        from ..evidence import evaluate_revision

        decision = evaluate_revision(
            before_gate_audit,
            after_gate_audit,
            min_coverage_gain=self._config.evidence_revision_min_coverage_gain,
            min_claim_retention=self._config.evidence_revision_min_claim_retention,
        )
        report.evidence_revision = {
            "attempted": True,
            "accepted": decision.accepted,
            "reason": decision.reason,
            "before": decision.before,
            "after": decision.after,
            "original_sha256": self._text_sha256(original_content),
            "candidate_sha256": self._text_sha256(draft.content),
        }
        accepted_audit = before_audit
        if decision.accepted:
            report.content = draft.content
            accepted_audit = after_gate_audit
            logger.info(
                "[EvidenceRevision] accepted | coverage %.1f%% -> %.1f%% | claims %d -> %d",
                before_gate_audit.coverage * 100,
                after_gate_audit.coverage * 100,
                len(before_gate_audit.claims),
                len(after_gate_audit.claims),
            )
            self._emit_event(
                "revision_accepted",
                decision.reason,
                {
                    "accepted": True,
                    "before": decision.before,
                    "after": decision.after,
                },
                state=OrchestratorState.EVIDENCE_REFINING,
            )
        else:
            logger.warning("[EvidenceRevision] rejected by quality gate: %s", decision.reason)
            self._emit_event(
                "revision_rejected",
                decision.reason,
                {
                    "accepted": False,
                    "before": decision.before,
                    "after": decision.after,
                },
                state=OrchestratorState.EVIDENCE_REFINING,
            )

        # The accepted candidate has already passed a deterministic re-audit.
        # A second hybrid pass adds latency and noise without affecting the gate.
        await self._commit_evidence_audit(report, accepted_audit)
        return OrchestratorState.DONE

    async def _do_replanning(self) -> OrchestratorState:
        """触发增量重规划。

        保留 confidence≥0.6 的成功结果，修改失败子问题。
        """
        failed_tasks = []
        for r in self._results:
            if r.status != AgentStatus.SUCCESS:
                st = self._task_map.get(r.task_id)
                if st:
                    failed_tasks.append(st)

        reason = self._build_failure_reason(self._results)
        logger.warning(f"[Replan] Round {self._replan_count}/{self._config.max_replan_rounds}. Failed tasks: {[t.task_id for t in failed_tasks]}")
        self._emit_event(
            "replan_started",
            f"Starting replan round {self._replan_count}.",
            {
                "round": self._replan_count,
                "failed_task_ids": [task.task_id for task in failed_tasks],
                "reason": reason,
            },
            state=OrchestratorState.REPLANNING,
        )

        try:
            bounded_replan = getattr(self.planner, "replan_with_timeout", None)
            if callable(bounded_replan):
                new_dag = await asyncio.to_thread(
                    bounded_replan,
                    self._query,
                    failed_tasks,
                    self._all_results or self._results,
                    reason,
                    min(60.0, max(0.25, self._remaining_seconds() - 1.0)),
                )
            else:
                new_dag = await asyncio.to_thread(
                    self.planner.replan,
                    query=self._query,
                    failed_tasks=failed_tasks,
                    existing_results=self._all_results or self._results,
                    reason=reason,
                )
            self._dag = new_dag
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                self._task_map = self._rebuild_task_map_from_dag()
            self._task_history.update(self._task_map)
            # 清空上一轮结果（保留在 memory 中，新任务可通过 context_keys 引用）
            self._results = []
        except PlanParseError as e:
            logger.warning(f"[Replan] Failed: {e}")
            # 重规划失败，如果已有部分成功结果，尝试直接合成
            if any(r.status == AgentStatus.SUCCESS for r in self._all_results or self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED
        except Exception as e:
            logger.warning(f"[Replan] Unexpected error: {e}")
            if any(r.status == AgentStatus.SUCCESS for r in self._all_results or self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED

        return OrchestratorState.DISPATCHING

    async def _on_done(self) -> OrchestratorState:
        """终态，不应再转换。"""
        return OrchestratorState.DONE

    async def _on_failed(self) -> OrchestratorState:
        """终态，不应再转换。"""
        return OrchestratorState.FAILED

    # ------------------------------------------------------------------
    # 决策逻辑
    # ------------------------------------------------------------------

    def _should_replan(self, results: list[AgentResult]) -> bool:
        """判断是否需要重规划。

        策略:
          - 失败率 > 50% 时触发
          - 或存在任何 TIMEOUT 且成功结果不足 30%
        """
        if not results:
            return False
        total = len(results)
        failed = sum(1 for r in results if r.status in (AgentStatus.FAILED, AgentStatus.TIMEOUT))
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)

        failure_rate = failed / total
        if failure_rate > 0.5:
            return True
        if success / total < 0.3 and failed > 0:
            return True
        return False

    def _should_fill_evidence_gaps(self) -> bool:
        if not (
            self._config.enable_evidence
            and self._config.enable_completeness_check
            and self._evidence_audit is not None
            and self.evidence_store is not None
        ):
            return False
        if self._evidence_gap_rounds >= self._config.max_evidence_gap_rounds:
            return False
        if getattr(self._evidence_audit, "verification_mode", "") in {
            "hybrid_unavailable",
            "hybrid_partial",
        }:
            # A failed semantic audit does not establish an evidence gap. In
            # particular, cross-language claim/evidence pairs often have zero
            # lexical support until the hybrid verifier can adjudicate them.
            return False
        if self._evidence_audit.coverage >= self._config.min_evidence_coverage:
            return False
        required_seconds = (
            self._effective_synthesis_reserve_seconds()
            + self._effective_final_audit_reserve_seconds()
            + self._config.evidence_gap_min_seconds
        )
        if not self._evidence_audit.claims or self._remaining_seconds() < required_seconds:
            return False
        return any(
            claim.status.value in {"refuted", "not_enough_evidence"}
            for claim in self._evidence_audit.claims
        )

    def _task_execution_seconds(self, task_id: str, requested_seconds: float) -> float:
        """Keep every pre-synthesis task outside the final-report reserve."""
        available = self._remaining_seconds() - (
            self._effective_synthesis_reserve_seconds()
            + self._effective_final_audit_reserve_seconds()
        )
        return max(0.0, min(float(requested_seconds), available))

    def _effective_synthesis_reserve_seconds(self) -> float:
        """Scale configured reserves down for intentionally short smoke runs."""
        return min(
            float(self._config.synthesis_reserve_seconds),
            max(1.0, float(self._config.global_timeout_seconds) * 0.30),
        )

    def _effective_final_audit_reserve_seconds(self) -> float:
        configured = self._config.final_audit_reserve_seconds if self._config.enable_evidence else 8.0
        fraction = 0.15 if self._config.enable_evidence else 0.05
        return min(
            float(configured),
            max(1.0, float(self._config.global_timeout_seconds) * fraction),
        )

    def _prepare_evidence_gap_round(self) -> None:
        from ..evidence import build_evidence_gap_tasks

        round_index = self._evidence_gap_rounds + 1
        tasks = build_evidence_gap_tasks(
            self._evidence_audit,
            round_index=round_index,
            max_tasks=self._config.max_evidence_gap_tasks,
            query=self._query,
        )
        dag = DAG()
        for task in tasks:
            dag.add_node(task.task_id)
        self._dag = dag
        self._task_map = {task.task_id: task for task in tasks}
        self._task_history.update(self._task_map)
        self._results = []
        self._evidence_gap_rounds = round_index
        logger.info(
            "[Evidence] coverage %.1f%% < %.1f%%; launching gap round %d with %d verification tasks",
            self._evidence_audit.coverage * 100,
            self._config.min_evidence_coverage * 100,
            round_index,
            len(tasks),
        )
        self._emit_event(
            "evidence_gap_started",
            f"Launching evidence gap round {round_index}.",
            {
                "round": round_index,
                "task_count": len(tasks),
                "coverage": self._evidence_audit.coverage,
                "target_coverage": self._config.min_evidence_coverage,
            },
            state=OrchestratorState.COLLECTING,
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _is_global_timeout(self) -> bool:
        """检查是否超过全局超时。"""
        elapsed = time.monotonic() - self._start_time
        return elapsed > self._config.global_timeout_seconds

    def _remaining_seconds(self) -> float:
        return max(0.0, self._config.global_timeout_seconds - (time.monotonic() - self._start_time))

    def _record_result_history(self, results: list[AgentResult]) -> None:
        positions = {result.task_id: index for index, result in enumerate(self._all_results)}
        for result in results:
            if result.task_id not in positions:
                positions[result.task_id] = len(self._all_results)
                self._all_results.append(result)
                continue
            index = positions[result.task_id]
            previous = self._all_results[index]
            if previous.status != AgentStatus.SUCCESS or result.status == AgentStatus.SUCCESS:
                self._all_results[index] = result

    def _successful_results(self) -> list[AgentResult]:
        return [
            result
            for result in (self._all_results or self._results)
            if result.status == AgentStatus.SUCCESS and result.output
        ]

    def _synthesis_coverage_requirements(
        self,
        results: list[AgentResult],
    ) -> list[dict[str, str]]:
        """Return original research tasks that the final report must address."""

        requirements: list[dict[str, str]] = []
        seen: set[str] = set()
        for result in results:
            task_id = str(result.task_id or "")
            if not task_id or task_id in seen or task_id.startswith("evidence_gap_"):
                continue
            task = self._task_history.get(task_id)
            if task is None or not task.description:
                continue
            seen.add(task_id)
            requirements.append(
                {
                    "task_id": task_id,
                    "description": str(task.description),
                    "status": result.status.value,
                }
            )
        return requirements

    def _build_partial_report(
        self,
        *,
        run_status: str,
        heading: str,
        explanation: str,
        empty_message: str,
    ) -> ResearchReport:
        successful = self._successful_results()
        if successful:
            sections = [
                f"## {heading}",
                "",
                explanation,
            ]
            for result in successful:
                sections.extend(["", f"### {result.task_id}", "", str(result.output)])
            content = "\n".join(sections)
            confidence = round(
                sum(result.confidence for result in successful) / len(successful) * 0.7,
                2,
            )
        else:
            content = empty_message
            confidence = 0.0
        sources = self.evidence_store.to_source_dicts() if self.evidence_store is not None else []
        return ResearchReport(
            query=self._query,
            content=content,
            sources=sources,
            confidence=confidence,
            num_searches=sum(
                len([step for step in result.trajectory if step.get("role") == "tool"])
                for result in successful
            ),
            run_status=run_status,
        )

    def _build_timeout_report(self) -> ResearchReport:
        return self._build_partial_report(
            run_status="partial_timeout",
            heading="Partial results",
            explanation=(
                "The global time budget expired before full synthesis. The following "
                "completed sub-task outputs are preserved."
            ),
            empty_message="Research timed out before any sub-task completed successfully.",
        )

    def _prepare_synthesis_results(
        self,
        results: list[AgentResult],
        *,
        request_deadline_monotonic: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[AgentResult]:
        """Compress long worker outputs before synthesis while preserving raw evidence."""
        if self.compressor is None or not results:
            return results
        eligible = [
            result
            for result in results
            if result.status == AgentStatus.SUCCESS and isinstance(result.output, str)
        ]
        if not eligible:
            return results
        texts = [result.output for result in eligible]
        calculate_tokens = getattr(self.compressor, "calculate_tokens", None)
        available_budget = getattr(self.compressor, "available_budget", 0)
        l1_threshold = getattr(self.compressor, "l1_threshold", 0.6)
        if callable(calculate_tokens) and available_budget:
            if calculate_tokens(texts) <= available_budget * l1_threshold:
                return results
        try:
            compress_parameters = inspect.signature(self.compressor.compress).parameters
            compress_kwargs: dict[str, Any] = {
                "texts": texts,
                "query": self._query,
                "system_prompt_tokens": 1000,
            }
            if "request_deadline_monotonic" in compress_parameters:
                compress_kwargs["request_deadline_monotonic"] = request_deadline_monotonic
            if "cancellation_token" in compress_parameters:
                compress_kwargs["cancellation_token"] = cancellation_token
            compressed = self.compressor.compress(**compress_kwargs)
        except Exception as exc:
            logger.warning("[M3] Synthesis compression failed, using raw results: %s", exc)
            return results
        if not compressed:
            return results

        if len(compressed) == len(eligible):
            replacements = dict(zip((result.task_id for result in eligible), compressed))
            prepared = []
            for result in results:
                cloned = copy.copy(result)
                if result.task_id in replacements:
                    cloned.output = replacements[result.task_id]
                prepared.append(cloned)
        else:
            prepared = [result for result in results if result.status != AgentStatus.SUCCESS]
            prepared.extend(
                AgentResult(
                    task_id=f"compressed_context_{index}",
                    status=AgentStatus.SUCCESS,
                    output=text,
                    confidence=0.7,
                )
                for index, text in enumerate(compressed, 1)
            )
        logger.info(
            "[M3] Prepared synthesis context: %d raw chars -> %d compressed chars",
            sum(len(text) for text in texts),
            sum(len(text) for text in compressed),
        )
        return prepared

    async def _finalize_evidence(self, report: ResearchReport) -> None:
        if self.evidence_store is None or self.evidence_verifier is None:
            return
        audit = await self._audit_report_text(
            report,
            use_hybrid=report.run_status == "complete",
        )
        if audit is None and self._evidence_audit is not None:
            logger.warning(
                "[Evidence] no final-text audit available; falling back to pre-synthesis audit"
            )
            audit = self._evidence_audit
        if audit is None:
            return
        await self._commit_evidence_audit(report, audit)

    async def _audit_report_text(
        self,
        report: ResearchReport,
        *,
        use_hybrid: bool,
    ) -> Any | None:
        return await self._audit_report_text_content(
            report.content,
            report,
            use_hybrid=use_hybrid,
        )

    async def _audit_report_text_content(
        self,
        content: str,
        report: ResearchReport,
        *,
        use_hybrid: bool,
    ) -> Any | None:
        if (
            not content
            or self.evidence_store is None
            or self.evidence_verifier is None
            or self._is_cancelled()
        ):
            return None
        citation_source_ids = [source.get("source_id", "") for source in report.sources]
        try:
            # Always produce a local result, even after the global network budget
            # expires. Hybrid refinement may replace it only if time remains.
            audit = await asyncio.to_thread(
                self.evidence_verifier.audit_text,
                content,
                self.evidence_store,
                "final_report",
                citation_source_ids,
                use_llm=False,
            )
            if self._is_cancelled():
                return audit
            remaining = self._remaining_seconds()
            if (
                use_hybrid
                and self.evidence_verifier.mode == "hybrid"
                and self.evidence_verifier.policy is not None
                and remaining > 5
                and not self._is_cancelled()
            ):
                hybrid_timeout = min(20.0, max(1.0, remaining - 2.0))
                try:
                    audit = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.evidence_verifier.audit_text,
                            content,
                            self.evidence_store,
                            "final_report",
                            citation_source_ids,
                            use_llm=True,
                            timeout_seconds=max(0.25, hybrid_timeout - 1.0),
                        ),
                        timeout=hybrid_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[Evidence] hybrid final audit timed out; using heuristic audit")
                    audit.verification_mode = "hybrid_unavailable"
            return audit
        except Exception as exc:
            logger.warning("[Evidence] report audit failed: %s", exc)
            return None

    async def _commit_evidence_audit(self, report: ResearchReport, audit: Any) -> None:
        from ..evidence import audit_task_coverage

        self._evidence_audit = audit
        report.evidence_audit = audit.to_dict(self.evidence_store.evidence)
        report.task_coverage = audit_task_coverage(
            report.content,
            self._synthesis_coverage_requirements(self._all_results or self._results),
            report.sources,
        )
        task_coverage = report.task_coverage
        self._emit_event(
            "task_coverage_snapshot",
            f"Planned-dimension coverage is {task_coverage.get('coverage', 0.0):.1%}.",
            {
                "coverage": task_coverage.get("coverage", 0.0),
                "required_count": task_coverage.get("required_count", 0),
                "covered_count": task_coverage.get("covered_count", 0),
                "synthesis_gap_count": task_coverage.get("synthesis_gap_count", 0),
                "research_gap_count": task_coverage.get("research_gap_count", 0),
                "source_gap_count": task_coverage.get("source_gap_count", 0),
            },
            state=OrchestratorState.EVIDENCE_REFINING,
        )
        if audit.claims:
            evidence_factor = 0.7 + 0.3 * audit.coverage
            report.confidence = round(report.confidence * evidence_factor, 2)
        try:
            report.evidence_artifact = self.evidence_store.persist(
                audit,
                query=self._query,
                metadata={
                    "final_report_sha256": self._text_sha256(report.content),
                    "evidence_revision": report.evidence_revision,
                    "task_coverage": report.task_coverage,
                },
            )
        except Exception as exc:
            logger.warning("[Evidence] failed to persist evidence artifact: %s", exc)
        logger.info(
            "[Evidence] final audit: coverage=%.1f%%, supported=%d/%d, artifact=%s",
            audit.coverage * 100,
            audit.supported_count,
            len(audit.claims),
            report.evidence_artifact or "disabled",
        )

    @staticmethod
    def _audit_metrics(audit: Any) -> dict[str, Any]:
        from ..evidence import audit_summary

        return audit_summary(audit)

    @staticmethod
    def _text_sha256(content: str) -> str:
        return hashlib.sha256((content or "").encode("utf-8")).hexdigest()

    def _build_memory_context(
        self,
        *,
        request_deadline_monotonic: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> str:
        """构建给 planner 的上下文摘要。

        优先使用 M4 SharedMemoryStore 的语义检索（如果已接入），
        否则回退到运行时 dict 遍历。
        """
        # M4: 语义检索相关记忆
        if self.memory_store is not None:
            try:
                ctx = self.memory_store.get_context_for_query(
                    self._query, max_tokens=2000
                )
                if ctx:
                    logger.info(f"[M4] Retrieved {len(ctx)} chars of semantic memory context")
                    return ctx
            except Exception as e:
                logger.warning(f"[M4] Semantic memory query failed: {e}, falling back to dict")

        # 回退：运行时 dict 遍历
        parts = []
        for key, value in self._memory_store.items():
            if key.startswith("result:"):
                continue
            parts.append(f"{key}: {str(value)[:200]}")

        # M3: 如果上下文过长，启用压缩
        if self.compressor is not None and parts:
            total_chars = sum(len(p) for p in parts)
            if total_chars > 6000:  # 约 2000 tokens 的启发式阈值
                try:
                    compress_parameters = inspect.signature(self.compressor.compress).parameters
                    compress_kwargs: dict[str, Any] = {
                        "texts": parts,
                        "query": self._query,
                        "system_prompt_tokens": 0,
                    }
                    if "request_deadline_monotonic" in compress_parameters:
                        compress_kwargs["request_deadline_monotonic"] = request_deadline_monotonic
                    if "cancellation_token" in compress_parameters:
                        compress_kwargs["cancellation_token"] = cancellation_token
                    compressed = self.compressor.compress(**compress_kwargs)
                    logger.info(f"[M3] Context compressed: {total_chars} → {sum(len(c) for c in compressed)} chars")
                    return "\n".join(compressed)
                except Exception as e:
                    logger.warning(f"[M3] Compression failed: {e}, using raw context")

        return "\n".join(parts) if parts else ""

    def _build_task_context(self, subtask: SubTask) -> dict:
        """为单个 SubTask 构建执行上下文。"""
        ctx = dict(self._memory_store)
        ctx["query"] = self._query
        ctx["as_of_date"] = self._config.as_of_date
        ctx["_cancellation_token"] = self.cancellation_token
        # 注入依赖任务的结果
        for dep_id in subtask.dependencies:
            dep_key = f"result:{dep_id}"
            if dep_key in self._memory_store:
                ctx[f"dep:{dep_id}"] = self._memory_store[dep_key]
        return ctx

    def _build_failure_reason(self, results: list[AgentResult]) -> str:
        """分析失败原因，生成给 replanner 的描述。"""
        reasons = []
        timeout_count = sum(1 for r in results if r.status == AgentStatus.TIMEOUT)
        failed_count = sum(1 for r in results if r.status == AgentStatus.FAILED)
        if timeout_count > 0:
            reasons.append(f"{timeout_count} tasks timed out (may need simpler queries or longer timeout)")
        if failed_count > 0:
            reasons.append(f"{failed_count} tasks failed with errors")
        return "; ".join(reasons) if reasons else "Unknown failure"

    def _rebuild_task_map_from_dag(self) -> dict[str, SubTask]:
        """从 DAG 重建 task_map（当缺少原始 SubTask 信息时使用占位符）。

        实际场景中，planner 应返回完整的 SubTask 列表；
        这里作为降级：为 DAG 中每个节点创建默认 SubTask。
        """
        if self._dag is None:
            return {}

        task_map: dict[str, SubTask] = {}
        for node_id in self._dag:
            deps = self._dag.get_dependencies(node_id)
            if node_id not in self._task_map:
                # 新建占位 SubTask
                task_map[node_id] = SubTask(
                    task_id=node_id,
                    task_type=TaskType.SEARCH,
                    description=f"Auto-generated task for {node_id}",
                    dependencies=deps,
                )
            else:
                # 保留已有信息，更新依赖
                old = self._task_map[node_id]
                task_map[node_id] = SubTask(
                    task_id=old.task_id,
                    task_type=old.task_type,
                    description=old.description,
                    dependencies=deps,
                    context_keys=old.context_keys,
                    timeout_seconds=old.timeout_seconds,
                    priority=old.priority,
                    expected_type=old.expected_type,
                    search_hints=old.search_hints,
                )
        return task_map
