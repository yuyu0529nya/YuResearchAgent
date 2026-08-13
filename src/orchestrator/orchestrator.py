"""
YuResearchAgent — 核心编排器 (M1: Multi-Agent Orchestrator)

9 状态状态机驱动的异步任务编排引擎：
  IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING → ADVERSARIAL → DONE
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
import logging
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

        # 运行时状态（保留 dict 作为快速缓存，M4 提供持久化 + 语义检索）
        self._memory_store: dict[str, Any] = {}
        self._results: list[AgentResult] = []
        self._all_results: list[AgentResult] = []
        self._dag: DAG | None = None
        self._task_map: dict[str, SubTask] = {}
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
        self._current_state = OrchestratorState.IDLE
        if self.evidence_store is not None:
            self.evidence_store.reset(query)

        # 状态机主循环
        while self._current_state not in (OrchestratorState.DONE, OrchestratorState.FAILED):
            # 全局超时检查
            if self._is_global_timeout():
                if self._all_results or self._results:
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

        # 返回结果
        if self._current_state == OrchestratorState.DONE:
            # 最终报告应在 memory 中
            report = self._memory_store.get("final_report")
            if report is None:
                report = ResearchReport(query=query, content="Report generation failed unexpectedly.")
            report.num_replan = self._replan_count
            report.adversarial_rounds = self._adversarial_count
            report.evidence_gap_rounds = self._evidence_gap_rounds
            if self._config.enable_evidence:
                await self._finalize_evidence(report)

            # M4: 将最终报告存入 SharedMemoryStore
            if self.memory_store is not None:
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

            return report

        # FAILED 状态
        return ResearchReport(
            query=query,
            content="Research failed due to persistent errors or global timeout.",
            num_replan=self._replan_count,
            adversarial_rounds=self._adversarial_count,
            run_status="failed",
        )

    def get_evidence_snapshot(self) -> dict[str, Any]:
        """Return a read-only UI/observability snapshot of the current evidence graph."""
        if self._evidence_audit is None or self.evidence_store is None:
            return {}
        snapshot = self._evidence_audit.to_dict(self.evidence_store.evidence)
        snapshot["sources"] = self.evidence_store.to_source_dicts()
        snapshot["gap_rounds"] = self._evidence_gap_rounds
        report = self._memory_store.get("final_report")
        snapshot["artifact"] = getattr(report, "evidence_artifact", "") if report is not None else ""
        return snapshot

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
            memory_ctx = self._build_memory_context()
            self._dag = self.planner.generate_plan(self._query, memory_ctx)
            # 从 planner 获取完整的 SubTask 信息（包括 description、search_hints 等）
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                # 降级：如果解析失败，使用占位符
                self._task_map = self._rebuild_task_map_from_dag()
        except PlanParseError as e:
            logger.warning(f"[Planning] Failed: {e}")
            return OrchestratorState.FAILED
        except Exception as e:
            logger.warning(f"[Planning] Unexpected error: {e}")
            return OrchestratorState.FAILED

        n_tasks = len(self._dag)
        n_layers = len(self._dag.get_parallel_groups()) if self._dag else 0
        logger.info(f"[Planning] ✓ DAG 生成完成: {n_tasks} 个子任务, {n_layers} 个执行层")
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
            logger.info(f"[Dispatch] ▶ Layer {layer_idx + 1}/{len(parallel_groups)}: {group} (并行执行)")

            # 构建本层的 coroutine 列表
            async def _run_one(task_id: str) -> AgentResult:
                async with semaphore:
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
                        remaining = self._remaining_seconds()
                        if remaining <= 0:
                            return AgentResult(
                                task_id=task_id,
                                status=AgentStatus.TIMEOUT,
                                output="Global time budget exhausted before task execution",
                            )
                        agent = await self.agent_pool.get_agent(subtask.task_type)
                        try:
                            result = await asyncio.wait_for(
                                agent.run(subtask, context),
                                timeout=min(subtask.timeout_seconds, remaining),
                            )
                        except asyncio.TimeoutError:
                            result = AgentResult(
                                task_id=task_id,
                                status=AgentStatus.TIMEOUT,
                                output=f"Task timed out after {min(subtask.timeout_seconds, remaining):.1f}s",
                            )
                        except Exception as e:
                            result = AgentResult(
                                task_id=task_id,
                                status=AgentStatus.FAILED,
                                output=f"Exception: {type(e).__name__}: {e}",
                            )
                        finally:
                            await self.agent_pool.release_agent(agent)

                        if result.status == AgentStatus.SUCCESS or attempt >= attempts:
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

        success_count = sum(1 for r in self._results if r.status == AgentStatus.SUCCESS)
        total_count = len(self._results)
        fail_count = total_count - success_count
        status_icon = "✓" if success_count == total_count else "⚠"
        if fail_count > 0:
            logger.info(f"[Collect] {status_icon} 子任务完成: {success_count}/{total_count} 成功 ({fail_count} 失败)")
        else:
            logger.info(f"[Collect] {status_icon} 子任务完成: {success_count}/{total_count} 成功")

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
        # 创建合成任务
        synth_task = SubTask(
            task_id="synthesize_final",
            task_type=TaskType.ANALYZE,  # 使用 ANALYZE 类型，实际由 SummarizerAgent 处理
            description="Synthesize all sub-task results into a final research report.",
            timeout_seconds=300,
        )

        synthesis_results = self._prepare_synthesis_results(self._all_results or self._results)
        context = {
            "query": self._query,
            "results": synthesis_results,
            "evidence_audit": (
                self._evidence_audit.to_dict(self.evidence_store.evidence)
                if self._evidence_audit is not None and self.evidence_store is not None
                else {}
            ),
            "evidence_sources": (
                self.evidence_store.to_source_dicts() if self.evidence_store is not None else []
            ),
        }

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
        final_audit_reserve = (
            self._config.final_audit_reserve_seconds
            if self._config.enable_evidence
            else 8.0
        )
        synthesis_timeout = min(
            synth_task.timeout_seconds,
            max(1.0, remaining - final_audit_reserve),
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
            fallback = self._build_timeout_report()
            fallback.run_status = (
                "partial_timeout"
                if result.status == AgentStatus.TIMEOUT
                else "partial_failure"
            )
            self._memory_store["final_report"] = fallback

        if self._config.enable_adversarial:
            logger.info("[Synthesize] ✓ 报告合成完成，进入对抗优化")
            return OrchestratorState.ADVERSARIAL
        logger.info("[Synthesize] ✓ 报告合成完成")
        return OrchestratorState.DONE

    async def _do_adversarial(self) -> OrchestratorState:
        """M5: Red-Blue 对抗降噪循环。

        调用 AdversarialLoop 对报告进行 challenge-verify 迭代优化。
        仅在报告置信度低于阈值时触发，避免资源浪费。
        """
        report = self._memory_store.get("final_report")
        if report is None:
            return OrchestratorState.DONE

        # 置信度足够高时跳过对抗
        if report.confidence >= 0.8:
            logger.info("[Adversarial] ✓ 报告置信度已达标 (≥0.8)，跳过对抗优化")
            return OrchestratorState.DONE

        if self.adversarial_loop is None:
            logger.info("[Adversarial] AdversarialLoop 未配置，跳过")
            return OrchestratorState.DONE

        # 剩余全局时间预算：对抗循环是单个长状态，不限时会绕过状态机层面的全局超时检查，
        # 故用 asyncio.wait_for 显式套上剩余预算，避免限流重试时跑失控（实测曾达 1394s）。
        remaining = self._config.global_timeout_seconds - (time.monotonic() - self._start_time)
        if remaining <= 0:
            logger.warning("[Adversarial] 全局时间已耗尽，跳过对抗优化")
            return OrchestratorState.DONE

        try:
            logger.info(
                f"[Adversarial] ▶ 启动 Red-Blue 对抗优化 "
                f"(当前置信度={report.confidence:.2f}, 剩余预算={remaining:.0f}s)"
            )
            optimized_report, history = await asyncio.wait_for(
                self.adversarial_loop.run(report), timeout=remaining
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

        try:
            new_dag = self.planner.replan(
                query=self._query,
                failed_tasks=failed_tasks,
                existing_results=self._all_results or self._results,
                reason=reason,
            )
            self._dag = new_dag
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                self._task_map = self._rebuild_task_map_from_dag()
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
        if self._evidence_audit.coverage >= self._config.min_evidence_coverage:
            return False
        required_seconds = (
            self._config.synthesis_reserve_seconds
            + self._config.evidence_gap_min_seconds
        )
        if not self._evidence_audit.claims or self._remaining_seconds() < required_seconds:
            return False
        return any(
            claim.status.value in {"refuted", "not_enough_evidence"}
            for claim in self._evidence_audit.claims
        )

    def _prepare_evidence_gap_round(self) -> None:
        from ..evidence import build_evidence_gap_tasks

        round_index = self._evidence_gap_rounds + 1
        tasks = build_evidence_gap_tasks(
            self._evidence_audit,
            round_index=round_index,
            max_tasks=self._config.max_evidence_gap_tasks,
        )
        dag = DAG()
        for task in tasks:
            dag.add_node(task.task_id)
        self._dag = dag
        self._task_map = {task.task_id: task for task in tasks}
        self._results = []
        self._evidence_gap_rounds = round_index
        logger.info(
            "[Evidence] coverage %.1f%% < %.1f%%; launching gap round %d with %d verification tasks",
            self._evidence_audit.coverage * 100,
            self._config.min_evidence_coverage * 100,
            round_index,
            len(tasks),
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

    def _build_timeout_report(self) -> ResearchReport:
        successful = [
            result
            for result in (self._all_results or self._results)
            if result.status == AgentStatus.SUCCESS and result.output
        ]
        if successful:
            sections = [
                "## Partial results",
                "",
                "The global time budget expired before full synthesis. The following completed sub-task outputs are preserved.",
            ]
            for result in successful:
                sections.extend(["", f"### {result.task_id}", "", str(result.output)])
            content = "\n".join(sections)
            confidence = round(
                sum(result.confidence for result in successful) / len(successful) * 0.7,
                2,
            )
        else:
            content = "Research timed out before any sub-task completed successfully."
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
            run_status="partial_timeout",
        )

    def _prepare_synthesis_results(self, results: list[AgentResult]) -> list[AgentResult]:
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
            compressed = self.compressor.compress(texts=texts, query=self._query, system_prompt_tokens=1000)
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
        audit = None
        remaining = self._remaining_seconds()
        if report.content:
            try:
                # Always audit the text users actually receive. The heuristic
                # pass is local and bounded, so it remains valid even when the
                # network/LLM budget was exhausted during synthesis. Hybrid
                # refinement is enabled only when enough global budget remains.
                audit = await asyncio.to_thread(
                    self.evidence_verifier.audit_text,
                    report.content,
                    self.evidence_store,
                    "final_report",
                    [source.get("source_id", "") for source in report.sources],
                    use_llm=False,
                )
                if (
                    self.evidence_verifier.mode == "hybrid"
                    and self.evidence_verifier.policy is not None
                    and remaining > 5
                ):
                    hybrid_timeout = min(60.0, max(1.0, remaining - 2.0))
                    audit = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.evidence_verifier.audit_text,
                            report.content,
                            self.evidence_store,
                            "final_report",
                            [source.get("source_id", "") for source in report.sources],
                            use_llm=True,
                        ),
                        timeout=hybrid_timeout,
                    )
            except asyncio.TimeoutError:
                logger.warning("[Evidence] hybrid final audit timed out; using heuristic final audit")
            except Exception as exc:
                logger.warning("[Evidence] final report audit failed: %s", exc)
        if audit is None and self._evidence_audit is not None:
            logger.warning(
                "[Evidence] no final-text audit available; falling back to pre-synthesis audit"
            )
            audit = self._evidence_audit
        if audit is None:
            return
        self._evidence_audit = audit
        report.evidence_audit = audit.to_dict(self.evidence_store.evidence)
        if audit.claims:
            evidence_factor = 0.7 + 0.3 * audit.coverage
            report.confidence = round(report.confidence * evidence_factor, 2)
        try:
            report.evidence_artifact = self.evidence_store.persist(audit, query=self._query)
        except Exception as exc:
            logger.warning("[Evidence] failed to persist evidence artifact: %s", exc)
        logger.info(
            "[Evidence] final audit: coverage=%.1f%%, supported=%d/%d, artifact=%s",
            audit.coverage * 100,
            audit.supported_count,
            len(audit.claims),
            report.evidence_artifact or "disabled",
        )

    def _build_memory_context(self) -> str:
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
                    compressed = self.compressor.compress(
                        texts=parts,
                        query=self._query,
                        system_prompt_tokens=0,
                    )
                    logger.info(f"[M3] Context compressed: {total_chars} → {sum(len(c) for c in compressed)} chars")
                    return "\n".join(compressed)
                except Exception as e:
                    logger.warning(f"[M3] Compression failed: {e}, using raw context")

        return "\n".join(parts) if parts else ""

    def _build_task_context(self, subtask: SubTask) -> dict:
        """为单个 SubTask 构建执行上下文。"""
        ctx = dict(self._memory_store)
        ctx["query"] = self._query
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
