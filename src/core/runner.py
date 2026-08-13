#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/runner.py
================================================================================
YuResearchAgent 核心运行逻辑。

本模块包含初始化所有模块和执行完整研究流程的核心函数，
供 scripts/ 和 evaluation/ 统一调用，避免 evaluation/ 反向依赖 scripts/。

对外接口:
    - load_config(config_path) -> dict
    - initialize_modules(config) -> dict
    - run_research(query, config, modules) -> str
    - run_research_with_metadata(query, config, modules) -> tuple[str, dict]
    - save_report(report, query, output_dir) -> str
================================================================================
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# 将项目根目录加入 sys.path，确保 src 包可导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
def setup_logging(log_level: str = "INFO") -> None:
    """配置全局日志格式与级别。"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None) -> dict:
    """
    加载 YAML 配置文件。

    若未指定路径，默认加载 configs/default.yaml。
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ---------------------------------------------------------------------------
# 工具工厂
# ---------------------------------------------------------------------------
def _create_tools_factory(config: dict):
    """创建工具工厂函数，返回 Agent 可用的工具列表。"""
    tools_cfg = config.get("tools", {})
    mock_mode = tools_cfg.get("web_search", {}).get("mock_mode", True)

    from src.tools import (
        WebSearchTool,
        MockWebSearchTool,
        ArxivReaderTool,
        BrowserTool,
        MockBrowserTool,
        FileReaderTool,
        CodeSandboxTool,
        CalculatorTool,
        NotepadTool,
    )

    tools = {}

    # 1. web_search
    if mock_mode:
        tools["web_search"] = MockWebSearchTool()
    else:
        from src.utils.env_config import get_env

        configured_backend = tools_cfg.get("web_search", {}).get("backend", "auto")
        tools["web_search"] = WebSearchTool(
            backend=get_env("SEARCH_BACKEND", configured_backend)
        )

    # 2. browser
    if mock_mode:
        tools["browser"] = MockBrowserTool()
    else:
        tools["browser"] = BrowserTool()

    # 3. arxiv_reader
    tools["arxiv_reader"] = ArxivReaderTool(use_mock=mock_mode)

    # 4. file_reader（不限制目录）
    tools["file_reader"] = FileReaderTool(allowed_base_dir=None)

    # 5. code_sandbox
    tools["code_sandbox"] = CodeSandboxTool(use_mock=mock_mode)

    # 6. calculator
    tools["calculator"] = CalculatorTool()

    # 7. notepad
    tools["notepad"] = NotepadTool()

    # 返回列表形式（AgentPool 和 Agent 构造函数需要 list）
    return list(tools.values())


# ---------------------------------------------------------------------------
# 模块初始化
# ---------------------------------------------------------------------------
def initialize_modules(config: dict, session_id: str = "") -> dict[str, Any]:
    """
    根据配置初始化所有核心模块。

    Args:
        config: 全局配置字典。
        session_id: 会话 ID，用于 memory store 的 session 隔离。

    返回一个包含各模块实例的字典。
    """
    logger = logging.getLogger("runner")
    logger.info("正在初始化核心模块...")

    modules: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 多后端 LLM 初始化（从 .env + configs/default.yaml 读取配置）
    # ------------------------------------------------------------------
    from src.models.model_router import ModelRouter

    model_cfg = config.get("model", {})
    default_backend = model_cfg.get("backend", "vllm")
    backend_mapping = model_cfg.get("backend_mapping", {})
    backend_sampling = model_cfg.get("backend_sampling", {})

    # 辅助函数：根据模块名获取采样参数覆盖
    def _get_sampling_kwargs(module_name: str, backend_name: str) -> dict:
        """合并后端全局默认 + 模块级覆盖参数。"""
        kwargs = {}
        # 1. 后端全局默认
        if backend_name in backend_sampling:
            kwargs.update(backend_sampling[backend_name])
        # 2. 模块级覆盖（优先级更高）
        module_overrides = backend_sampling.get("modules", {}).get(module_name, {})
        kwargs.update(module_overrides)
        return kwargs

    # 默认后端（所有模块共用）
    default_kwargs = _get_sampling_kwargs("default", default_backend)
    default_policy = ModelRouter.create_backend(default_backend, **default_kwargs)
    modules["default_policy"] = default_policy
    logger.info(
        f"[LLM] 默认后端已加载: {default_backend} | 模型={default_policy.model_name} | "
        f"temperature={default_policy.temperature} | max_tokens={default_policy.max_tokens}"
    )

    # 多后端分工：不同模块用不同后端 + 不同采样参数
    for module_name, backend_name in backend_mapping.items():
        kwargs = _get_sampling_kwargs(module_name, backend_name)
        policy = ModelRouter.create_backend(backend_name, **kwargs)
        modules[f"{module_name}_policy"] = policy
        logger.info(
            f"[LLM] {module_name} → 后端={backend_name} | 模型={policy.model_name} | "
            f"temperature={policy.temperature} | max_tokens={policy.max_tokens}"
        )

    # 若未配置分工，所有模块回退到 default_policy
    # ------------------------------------------------------------------

    # M2: Adaptive Planner（Orchestrator 依赖 Planner，先初始化）
    from src.planner.planner import Planner
    from src.planner.budget_tracker import BudgetTracker

    planner_policy = modules.get("planner_policy", default_policy)
    budget_tracker = BudgetTracker()
    planner = Planner(
        policy=planner_policy,
        budget_tracker=budget_tracker,
        max_sub_questions=config.get("orchestrator", {}).get("max_sub_questions", 8),
    )
    modules["planner"] = planner
    logger.info("[M2] Planner 模块已初始化")

    # M3: Context Compressor
    from src.compressor.compressor import ContextCompressor
    from src.memory.embedder import Embedder

    compressor_policy = modules.get("compressor_policy", default_policy)
    compressor_cfg = config.get("compressor", {})
    compressor = None
    if compressor_cfg.get("enable_multilevel", True):
        compressor = ContextCompressor(
            llm_policy=compressor_policy,
            embedder=Embedder(compressor_cfg.get("embedding_model")),
            budget=compressor_cfg.get("max_context_length", 16000),
            output_reserve=compressor_cfg.get("output_reserve_tokens", 2048),
            l1_threshold=compressor_cfg.get("l1_threshold", 0.6),
            l2_threshold=compressor_cfg.get("l2_threshold", 0.8),
            l3_threshold=compressor_cfg.get("l3_threshold", 0.95),
        )
    modules["compressor"] = compressor
    logger.info("[M3] Compressor 模块%s", "已初始化" if compressor is not None else "已关闭")

    # M4: Shared Memory Store
    from src.memory.memory_store import SharedMemoryStore

    memory_cfg = config.get("memory", {})
    memory_store = None
    if memory_cfg.get("enabled", True):
        memory_store = SharedMemoryStore(
            db_path=memory_cfg.get("db_path", "data/memory.db"),
            session_id=session_id,
        )
    modules["memory_store"] = memory_store
    logger.info(
        "[M4] Memory Store 模块%s%s",
        "已初始化" if memory_store is not None else "已关闭",
        f" (session={session_id})" if memory_store is not None else "",
    )

    # Tools（真实工具或 Mock 工具）
    tools_list = _create_tools_factory(config)
    modules["tools"] = tools_list
    logger.info(f"Tools 模块已初始化（共 {len(tools_list)} 个工具）")

    # M5: Red-Blue Adversarial Loop（先创建，再注入 Orchestrator）
    from src.adversarial.loop import AdversarialLoop
    from src.adversarial.red_agent import RedAgent
    from src.adversarial.blue_agent import BlueAgent

    red_policy = modules.get("red_agent_policy", default_policy)
    blue_policy = modules.get("blue_agent_policy", default_policy)
    adversarial_cfg = config.get("adversarial", {})

    red_agent = RedAgent(policy=red_policy)
    blue_agent = BlueAgent(policy=blue_policy, tools=tools_list)
    adversarial_loop = AdversarialLoop(
        red_agent=red_agent,
        blue_agent=blue_agent,
        policy=modules.get("judge_policy", default_policy),
        max_rounds=adversarial_cfg.get("max_rounds", 3),
        score_threshold=adversarial_cfg.get("score_threshold", 8.0),
        delta_threshold=adversarial_cfg.get("delta_threshold", 0.3),
    )
    modules["adversarial"] = adversarial_loop
    logger.info("[M5] Adversarial 模块已初始化")

    # M1: Multi-Agent Orchestrator
    from src.orchestrator.orchestrator import Orchestrator
    from src.orchestrator.agent_pool import AgentPool

    solver_backend = backend_mapping.get("solver", default_backend)
    solver_kwargs = _get_sampling_kwargs("solver", solver_backend)
    agent_pool = AgentPool(
        # Worker policies contain mutable tool/truncation state. Each pooled agent
        # owns one policy instance so concurrent trajectories cannot contaminate
        # each other; AgentPool still reuses that instance across sequential jobs.
        policy_factory=lambda: ModelRouter.create_backend(
            solver_backend,
            use_cache=False,
            **solver_kwargs,
        ),
        tools_factory=lambda: list(modules["tools"]),
        max_idle=3,
        researcher_max_turns=max(
            6,
            config.get("planner", {}).get("max_search_rounds_per_subagent", 4) * 2 + 2,
        ),
        researcher_max_tool_calls=config.get("planner", {}).get("max_search_rounds_per_subagent", 4),
    )
    modules["agent_pool"] = agent_pool

    evidence_cfg = config.get("evidence", {})
    evidence_store = None
    evidence_verifier = None
    evidence_reviser = None
    if evidence_cfg.get("enabled", True):
        from src.evidence import ClaimVerifier, EvidenceReviser, EvidenceStore

        evidence_store = EvidenceStore(
            artifact_dir=evidence_cfg.get("artifact_dir", "outputs/evidence"),
            session_id=session_id,
            persist_enabled=evidence_cfg.get("persist", True),
        )
        evidence_verifier = ClaimVerifier(
            policy=modules.get("judge_policy", default_policy),
            mode=evidence_cfg.get("verification_mode", "heuristic"),
            support_threshold=evidence_cfg.get("support_threshold", 0.38),
            max_claims=evidence_cfg.get("max_claims", 60),
            max_llm_claims=evidence_cfg.get("max_llm_claims", 12),
        )
        revision_cfg = evidence_cfg.get("revision", {})
        if revision_cfg.get("enabled", True):
            evidence_reviser = EvidenceReviser(
                modules.get("summarizer_policy", default_policy),
                min_length_ratio=revision_cfg.get("min_length_ratio", 0.50),
                max_prompt_chars=revision_cfg.get("max_prompt_chars", 32_000),
            )
        logger.info("[Evidence] Claim-Evidence graph 已启用 (%s)", evidence_verifier.mode)

    orchestrator = Orchestrator(
        planner=planner,
        agent_pool=agent_pool,
        budget_tracker=budget_tracker,
        compressor=compressor,
        adversarial_loop=adversarial_loop,
        memory_store=memory_store,
        summarizer_policy=modules.get("summarizer_policy", default_policy),
        evidence_store=evidence_store,
        evidence_verifier=evidence_verifier,
        evidence_reviser=evidence_reviser,
    )
    modules["orchestrator"] = orchestrator
    logger.info("[M1] Orchestrator 模块已初始化")

    # M6: Self-Evolution Engine（预留，默认禁用）
    if config.get("evolution", {}).get("enabled", False):
        logger.info("[M6] Evolution 模块已启用（预留接口）")
    else:
        logger.info("[M6] Evolution 模块已禁用")

    return modules


# ---------------------------------------------------------------------------
# 研究流程主函数
# ---------------------------------------------------------------------------
def build_run_config(config: dict):
    """Build the typed runtime config from the public YAML-shaped mapping."""
    from src.orchestrator.schemas import RunConfig

    return RunConfig(
        max_concurrent=config.get("orchestrator", {}).get("max_concurrent", 5),
        global_timeout_seconds=config.get("orchestrator", {}).get("global_timeout_seconds", 600),
        max_replan_rounds=config.get("orchestrator", {}).get("max_replan_rounds", 3),
        max_sub_questions=config.get("orchestrator", {}).get("max_sub_questions", 8),
        max_subagent_retries=config.get("orchestrator", {}).get("max_subagent_retries", 1),
        enable_replan=config.get("planner", {}).get("enable_replan", True),
        enable_completeness_check=config.get("planner", {}).get("enable_completeness_check", True),
        enable_adversarial=config.get("adversarial", {}).get("enabled", True),
        enable_evolution=config.get("evolution", {}).get("enabled", False),
        enable_evidence=config.get("evidence", {}).get("enabled", True),
        max_evidence_gap_rounds=config.get("evidence", {}).get("max_gap_rounds", 1),
        max_evidence_gap_tasks=config.get("evidence", {}).get("max_gap_tasks", 2),
        min_evidence_coverage=config.get("evidence", {}).get("min_coverage", 0.55),
        synthesis_reserve_seconds=config.get("orchestrator", {}).get(
            "synthesis_reserve_seconds", 130
        ),
        final_audit_reserve_seconds=config.get("orchestrator", {}).get(
            "final_audit_reserve_seconds", 70
        ),
        evidence_gap_min_seconds=config.get("evidence", {}).get(
            "gap_min_seconds", 100
        ),
        enable_evidence_revision=config.get("evidence", {}).get("revision", {}).get(
            "enabled", True
        ),
        evidence_revision_trigger_coverage=config.get("evidence", {}).get(
            "revision", {}
        ).get("trigger_coverage", 0.80),
        evidence_revision_min_coverage_gain=config.get("evidence", {}).get(
            "revision", {}
        ).get("min_coverage_gain", 0.03),
        evidence_revision_min_claim_retention=config.get("evidence", {}).get(
            "revision", {}
        ).get("min_claim_retention", 0.60),
        evidence_revision_timeout_seconds=config.get("evidence", {}).get(
            "revision", {}
        ).get("timeout_seconds", 40.0),
    )


async def run_research_with_metadata(
    query: str,
    config: dict,
    modules: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    执行完整的研究流程。

    流程：
        1. Orchestrator 调用 Planner 拆解问题为子任务 DAG
        2. Orchestrator 调度 AgentPool 中的子 Agent 并行/串行执行
        3. 子 Agent 调用 Tools 检索信息并生成子报告
        4. Compressor 管理长上下文
        5. Memory 存储中间结果
        6. Adversarial Loop 对报告进行多轮对抗优化（若启用）
        7. 输出最终研究报告

    Args:
        query: 用户输入的研究问题。
        config: 全局配置字典。
        modules: 已初始化的模块实例字典。

    Returns:
        最终研究报告文本（Markdown 格式）。
    """
    logger = logging.getLogger("runner")
    logger.info(f"开始研究，查询: {query[:80]}...")

    start_time = time.time()

    # Step 1-3: Orchestrator 内部完成规划、调度、收集、合成
    orchestrator = modules["orchestrator"]
    run_cfg = build_run_config(config)

    try:
        report = await orchestrator.run(query, config=run_cfg)
    finally:
        # The search client is process-wide. Always close it after a top-level run,
        # including failures, so batch experiments do not leak connections.
        from src.tools.web_search import WebSearchTool

        await WebSearchTool.close_session()
    logger.info(
        f"[Orchestrator] 报告生成完成 | 置信度={report.confidence:.2f} | "
        f"搜索轮数={report.num_searches} | 重规划={report.num_replan} | 对抗轮数={report.adversarial_rounds}"
    )

    # Step 4/5: 进化优化（如启用且已训练）
    if run_cfg.enable_evolution:
        logger.info("[Evolution] 进化优化已启用（预留接口）")
    else:
        logger.info("[Evolution] 进化优化已跳过")

    elapsed = time.time() - start_time
    logger.info(f"研究完成，耗时: {elapsed:.2f} 秒")

    # 组装最终输出
    final_report = _format_report(report, elapsed)
    audit = report.evidence_audit or {}
    metadata = {
        "run_status": report.run_status,
        "elapsed_seconds": round(elapsed, 4),
        "confidence": report.confidence,
        "num_searches": report.num_searches,
        "num_replan": report.num_replan,
        "evidence_gap_rounds": report.evidence_gap_rounds,
        "adversarial_rounds": report.adversarial_rounds,
        "source_count": len(report.sources),
        "estimated_worker_tokens": sum(
            max(0, int(getattr(result, "token_usage", 0) or 0))
            for result in getattr(orchestrator, "_all_results", [])
        ),
        "evidence_artifact": report.evidence_artifact,
        "evidence_audit": audit,
        "evidence_revision": report.evidence_revision,
    }
    return final_report, metadata


async def run_research(query: str, config: dict, modules: dict[str, Any]) -> str:
    """Execute the full research workflow and return its Markdown report."""
    report, _ = await run_research_with_metadata(query, config, modules)
    return report


def _format_report(report, elapsed: float) -> str:
    """将 ResearchReport 格式化为 Markdown 文本。"""
    content = report.content or ""

    # 统一置信度：如果正文中有 LLM 自评的"整体置信度"，替换为实际计算值，避免不一致
    content = re.sub(
        r"(整体置信度|Overall Confidence|置信度)[:：]\s*0?\.\d+",
        f"\\1: {report.confidence:.2f}",
        content,
        flags=re.I,
    )

    lines = [
        f"# 研究报告：{report.query}",
        "",
        "---",
        "",
        content,
        "",
        "---",
        "",
        "## 元信息",
        "",
        f"- **置信度**: {report.confidence:.2f}",
        f"- **搜索轮数**: {report.num_searches}",
        f"- **重规划次数**: {report.num_replan}",
        f"- **证据补充轮数**: {report.evidence_gap_rounds}",
        f"- **对抗轮数**: {report.adversarial_rounds}",
        f"- **总耗时**: {elapsed:.2f} 秒",
        "",
    ]

    audit = report.evidence_audit or {}
    if audit:
        total_claims = len(audit.get("claims", []))
        lines.extend([
            "## 证据审计",
            "",
            f"- **Claim 覆盖率**: {audit.get('coverage', 0.0):.1%}",
            f"- **核验结果**: {audit.get('supported_count', 0)} supported / "
            f"{audit.get('refuted_count', 0)} refuted / "
            f"{audit.get('not_enough_evidence_count', 0)} NEI（共 {total_claims} 条）",
            f"- **原始/权威来源占比**: {audit.get('primary_source_ratio', 0.0):.1%}",
            f"- **全文证据来源占比**: {audit.get('fulltext_source_ratio', 0.0):.1%}",
        ])
        if report.evidence_artifact:
            lines.append(f"- **审计文件**: `{report.evidence_artifact}`")
        unresolved = [
            claim
            for claim in audit.get("claims", [])
            if claim.get("status") != "supported"
        ]
        if unresolved:
            lines.extend(["", "### 仍需谨慎的陈述", ""])
            for claim in unresolved[:5]:
                lines.append(
                    f"- `{claim.get('status', 'not_enough_evidence')}` "
                    f"{claim.get('text', '')}"
                )
        lines.append("")

    revision = report.evidence_revision or {}
    if revision.get("attempted"):
        before = revision.get("before", {})
        after = revision.get("after", {})
        verdict = "已采用" if revision.get("accepted") else "已回滚"
        lines.extend([
            "## 证据约束修订",
            "",
            f"- **验收结果**: {verdict}",
            f"- **覆盖率变化**: {before.get('coverage', 0.0):.1%} → "
            f"{after.get('coverage', 0.0):.1%}",
            f"- **Claim 数量**: {before.get('claim_count', 0)} → "
            f"{after.get('claim_count', 0)}",
            f"- **门控说明**: {revision.get('reason', '')}",
            "",
        ])

    has_reference_section = re.search(
        r"^#{1,4}\s*(参考来源|参考文献|引用|references|bibliography|sources)\s*$",
        content,
        flags=re.I | re.M,
    )
    if report.sources and not has_reference_section:
        lines.append("## 参考来源")
        lines.append("")
        for i, src in enumerate(report.sources, 1):
            title = src.get("title", "未知标题")
            url = src.get("url", "")
            snippet = src.get("snippet", "")
            authors = src.get("authors", "")
            year = src.get("year", "")
            attribution = " · ".join(str(value) for value in (authors, year) if value)
            suffix = f" — {attribution}" if attribution else ""
            if snippet:
                suffix += f" — {snippet}"
            lines.append(f"{i}. [{title}]({url}){suffix}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 报告保存
# ---------------------------------------------------------------------------
def save_report(report: str, query: str, output_dir: str = "outputs/reports") -> str:
    """
    将研究报告保存到文件。

    文件名格式：report_YYYYMMDD_HHMMSS_<query前20字>.md
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in query[:20])
    filename = f"report_{timestamp}_{safe_query}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return filepath
