"""
YuResearchAgent — 核心数据结构定义 (M1/M2 共享 Schema)

所有跨模块传递的数据结构集中定义于此，保证类型一致性和可维护性。
使用 Python 3.10+ 的 | 联合类型语法。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


__all__ = [
    "OrchestratorState",
    "TaskType",
    "AgentStatus",
    "SubTask",
    "AgentResult",
    "ResearchReport",
    "RunConfig",
]


# ============================================================================
# 枚举定义
# ============================================================================

class OrchestratorState(Enum):
    """M1 编排层 10 状态状态机。

    正常流: IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING
            → ADVERSARIAL（可选）→ EVIDENCE_REFINING → DONE
    异常流:
      - 局部失败 → REPLANNING (增量重规划) → DISPATCHING
      - 全局失败 / 超过最大重规划次数 → FAILED
    """
    IDLE = "idle"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    COLLECTING = "collecting"
    SYNTHESIZING = "synthesizing"
    ADVERSARIAL = "adversarial"
    EVIDENCE_REFINING = "evidence_refining"
    REPLANNING = "replanning"
    DONE = "done"
    FAILED = "failed"


class TaskType(Enum):
    """Sub-task 的任务类型，决定由哪类 Agent 执行。"""
    SEARCH = "search"
    ANALYZE = "analyze"
    VERIFY = "verify"


class AgentStatus(Enum):
    """单个 Sub-task 的执行结果状态。"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class SubTask:
    """规划器生成的原子任务单元。

    Attributes:
        task_id: 全局唯一标识，用于 DAG 依赖引用。
        task_type: 任务类型，决定调度到哪个 Agent。
        description: 自然语言描述，传给 Agent 的指令。
        dependencies: 依赖的 task_id 列表，这些任务完成后才能执行本任务。
        context_keys: 需要从共享 Memory 中读取的上下文键名。
        timeout_seconds: 单任务超时阈值（秒）。
        priority: 优先级，数值越小优先级越高。
        expected_type: 期望结果类型，辅助 Agent 调整输出格式。
        search_hints: 搜索类任务的额外关键词提示。
    """
    task_id: str
    task_type: TaskType
    description: str
    dependencies: list[str] = field(default_factory=list)
    context_keys: list[str] = field(default_factory=list)
    timeout_seconds: int = 300
    priority: int = 1
    expected_type: str = "factual"  # factual | analytical | comparative | temporal
    search_hints: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Agent 执行 SubTask 后的结果。

    Attributes:
        task_id: 对应 SubTask 的 task_id。
        status: 执行状态（成功/失败/超时）。
        output: 实际输出内容，类型由任务决定（str | dict | list）。
        trajectory: 多轮交互轨迹，用于日志和后续分析。
        token_usage: 本次任务消耗的 token 数。
        confidence: 结果置信度 [0.0, 1.0]。
    """
    task_id: str
    status: AgentStatus
    output: Any = None
    trajectory: list[dict] = field(default_factory=list)
    token_usage: int = 0
    confidence: float = 0.0


@dataclass
class ResearchReport:
    """最终交付给用户的研究报告。

    Attributes:
        query: 原始研究问题。
        content: 报告正文（Markdown 格式）。
        sources: 引用的信息源列表，每条包含 url/title/snippet。
        confidence: 整体置信度。
        num_searches: 实际执行的搜索/分析轮数。
        num_replan: 重规划次数。
        adversarial_rounds: 对抗验证轮数。
        final_score: 最终综合评分（由外部评测模块写入）。
        evidence_audit: claim 级证据覆盖与三态核验结果。
        evidence_artifact: 可复现的证据图 JSON 路径。
        evidence_revision: 终审驱动修订的前后指标、哈希和验收结果。
        evidence_gap_rounds: 因证据不足触发的补充检索轮数。
        run_status: complete / partial_timeout / partial_failure / failed，区分完整与降级输出。
    """
    query: str
    content: str
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    num_searches: int = 0
    num_replan: int = 0
    adversarial_rounds: int = 0
    final_score: float = 0.0
    evidence_audit: dict[str, Any] = field(default_factory=dict)
    evidence_artifact: str = ""
    evidence_revision: dict[str, Any] = field(default_factory=dict)
    evidence_gap_rounds: int = 0
    run_status: str = "complete"


@dataclass
class RunConfig:
    """单次运行的全局配置。

    Attributes:
        max_concurrent: 最大并发 Sub-agent 数。
        global_timeout_seconds: 全局硬超时（秒）。
        max_replan_rounds: 最大重规划轮数。
        max_sub_questions: 单次规划最多子问题数。
        enable_adversarial: 是否启用对抗验证。
        enable_evolution: 是否启用自我进化（预留 M6 接口）。
    """
    max_concurrent: int = 5
    global_timeout_seconds: int = 600
    max_replan_rounds: int = 3
    max_sub_questions: int = 8
    max_subagent_retries: int = 1
    enable_replan: bool = True
    enable_completeness_check: bool = True
    enable_adversarial: bool = True
    enable_evolution: bool = False
    enable_evidence: bool = True
    max_evidence_gap_rounds: int = 1
    max_evidence_gap_tasks: int = 2
    min_evidence_coverage: float = 0.55
    synthesis_reserve_seconds: float = 130.0
    evidence_gap_min_seconds: float = 100.0
    final_audit_reserve_seconds: float = 70.0
    enable_evidence_revision: bool = True
    evidence_revision_trigger_coverage: float = 0.80
    evidence_revision_min_coverage_gain: float = 0.03
    evidence_revision_min_claim_retention: float = 0.60
    evidence_revision_timeout_seconds: float = 40.0
