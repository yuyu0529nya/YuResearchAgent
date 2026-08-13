"""
tests/unit/test_orchestrator_logic.py
src/orchestrator/orchestrator.py 的纯决策逻辑单元测试。

只测不依赖异步状态机的纯方法（重规划判定、全局超时、失败原因归纳）；
用 planner=None / agent_pool=None 构造，因为这些方法不触及它们。
"""
import asyncio
import time
from types import SimpleNamespace

from src.evidence import ClaimVerifier, EvidenceKind, EvidenceStore, RevisionDraft
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import AgentResult, AgentStatus, ResearchReport, RunConfig

S, F, T = AgentStatus.SUCCESS, AgentStatus.FAILED, AgentStatus.TIMEOUT


def _orch(max_replan=3, timeout=600) -> Orchestrator:
    o = Orchestrator(planner=None, agent_pool=None)
    o._config = RunConfig(max_replan_rounds=max_replan, global_timeout_seconds=timeout)
    return o


def _res(status, tid="t") -> AgentResult:
    return AgentResult(task_id=tid, status=status)


def test_should_replan_when_failure_rate_above_half():
    assert _orch()._should_replan([_res(F), _res(F), _res(F), _res(S)]) is True  # 75%


def test_should_not_replan_at_exactly_half():
    assert _orch()._should_replan([_res(F), _res(S)]) is False  # 恰 50%，非 >50%


def test_should_not_replan_all_success():
    assert _orch()._should_replan([_res(S), _res(S)]) is False


def test_should_not_replan_empty():
    assert _orch()._should_replan([]) is False


def test_timeout_counts_as_failure_for_replan():
    assert _orch()._should_replan([_res(T), _res(T), _res(S)]) is True  # 2/3 > 0.5


def test_is_global_timeout():
    o = _orch(timeout=10)
    o._start_time = time.monotonic() - 100  # 100s 前启动，限 10s
    assert o._is_global_timeout() is True
    o._start_time = time.monotonic()
    assert o._is_global_timeout() is False


def test_build_failure_reason_mentions_timeout_and_failure():
    reason = _orch()._build_failure_reason([_res(T), _res(F), _res(S)])
    assert "timed out" in reason
    assert "failed" in reason


def test_build_failure_reason_unknown_when_no_failures():
    assert _orch()._build_failure_reason([_res(S)]) == "Unknown failure"


def test_adversarial_respects_global_time_budget():
    # 回归：对抗循环必须受剩余全局时间预算约束，不能跑失控（曾达 1394s / 480s 预算）
    o = _orch(timeout=1)
    o._start_time = time.monotonic()  # 刚开始，约 1s 剩余预算

    class _SlowLoop:
        async def run(self, report):
            await asyncio.sleep(10)  # 远超预算
            return report, [{}]

    o.adversarial_loop = _SlowLoop()
    o._memory_store["final_report"] = ResearchReport(query="q", content="原始报告", confidence=0.5)

    t0 = time.monotonic()
    asyncio.run(o._do_adversarial())
    assert time.monotonic() - t0 < 3                              # 被预算掐断，没等满 10s
    assert o._memory_store["final_report"].content == "原始报告"   # 保留原报告


def test_adversarial_keeps_original_when_optimized_is_truncated():
    # 回归：Blue 重写被截断导致报告显著变短时，应保留更完整的原报告
    o = _orch(timeout=60)
    o._start_time = time.monotonic()
    orig = ResearchReport(query="q", content="完整报告内容" * 50, confidence=0.5)
    truncated = ResearchReport(query="q", content="短", confidence=0.5)

    class _TruncLoop:
        async def run(self, report):
            return truncated, [{}]

    o.adversarial_loop = _TruncLoop()
    o._memory_store["final_report"] = orig
    asyncio.run(o._do_adversarial())
    assert o._memory_store["final_report"].content == orig.content  # 没用截断版


def test_looks_like_error_filters_llm_and_tool_errors():
    from src.orchestrator.orchestrator import _looks_like_error
    assert _looks_like_error("Error: Connection error.") is True
    assert _looks_like_error("Error: Error code: 429 - 速率限制") is True
    assert _looks_like_error("[Calculator Error] Division by zero") is True
    assert _looks_like_error("2024 年 LLM Agent 的关键趋势是工具使用与多智能体协作。") is False
    assert _looks_like_error("") is False


def test_timeout_report_preserves_successful_partial_results():
    o = _orch(timeout=1)
    o._query = "q"
    o._all_results = [
        AgentResult(
            task_id="t1",
            status=S,
            output="A completed evidence-based finding.",
            confidence=0.8,
            trajectory=[{"role": "tool"}],
        ),
        AgentResult(task_id="t2", status=T, output="timeout"),
    ]

    report = o._build_timeout_report()

    assert "A completed evidence-based finding" in report.content
    assert report.confidence == 0.56
    assert report.num_searches == 1
    assert report.run_status == "partial_timeout"


def test_gap_research_preserves_synthesis_time_budget():
    o = _orch(timeout=480)
    o._config.synthesis_reserve_seconds = 110
    o._config.evidence_gap_min_seconds = 100
    o._config.enable_evidence = True
    o._config.enable_completeness_check = True
    o._config.max_evidence_gap_rounds = 1
    o._config.min_evidence_coverage = 0.55
    o._evidence_audit = SimpleNamespace(
        coverage=0.1,
        claims=[SimpleNamespace(status=SimpleNamespace(value="not_enough_evidence"))],
    )
    o.evidence_store = object()

    o._start_time = time.monotonic() - 330  # 150 seconds remain; synthesis wins.
    assert o._should_fill_evidence_gaps() is False

    o._start_time = time.monotonic() - 180  # 300 seconds remain; one gap round fits.
    assert o._should_fill_evidence_gaps() is True


def test_synthesis_compressor_changes_long_context_without_mutating_raw_results():
    class _Compressor:
        available_budget = 10
        l1_threshold = 0.6

        @staticmethod
        def calculate_tokens(texts):
            return 100

        @staticmethod
        def compress(texts, query, system_prompt_tokens):
            return ["compressed"]

    o = Orchestrator(planner=None, agent_pool=None, compressor=_Compressor())
    o._query = "q"
    raw = AgentResult(task_id="t1", status=S, output="raw evidence " * 20, confidence=0.8)

    prepared = o._prepare_synthesis_results([raw])

    assert prepared[0].output == "compressed"
    assert raw.output.startswith("raw evidence")


def test_final_evidence_audits_report_even_after_global_budget_expires():
    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(url="https://example.com/model", title="Model card")
    store.add_evidence(
        source.source_id,
        "The model supports a context window of 128K tokens.",
        EvidenceKind.FULL_TEXT,
    )
    o = Orchestrator(
        planner=None,
        agent_pool=None,
        evidence_store=store,
        evidence_verifier=ClaimVerifier(support_threshold=0.2),
    )
    o._query = "q"
    o._config = RunConfig(global_timeout_seconds=1, enable_evidence=True)
    o._start_time = time.monotonic() - 10
    o._evidence_audit = SimpleNamespace(
        coverage=0.0,
        claims=[],
        to_dict=lambda _: {"coverage": 0.0, "claims": []},
    )
    report = ResearchReport(
        query="q",
        content="The model supports a context window of 128K tokens [1].",
        sources=[{"source_id": source.source_id, "url": source.url}],
        confidence=0.8,
    )

    asyncio.run(o._finalize_evidence(report))

    assert report.evidence_audit["coverage"] == 1.0
    assert report.evidence_audit["claims"][0]["cited_indices"] == [1]


def test_run_config_has_final_audit_reserve() -> None:
    from src.core.runner import build_run_config

    run_config = build_run_config(
        {
            "orchestrator": {"final_audit_reserve_seconds": 27},
            "evidence": {"enabled": True},
        }
    )

    assert run_config.final_audit_reserve_seconds == 27


def test_run_config_maps_evidence_revision_controls() -> None:
    from src.core.runner import build_run_config

    run_config = build_run_config(
        {
            "evidence": {
                "enabled": True,
                "revision": {
                    "enabled": False,
                    "trigger_coverage": 0.7,
                    "min_coverage_gain": 0.08,
                    "min_claim_retention": 0.75,
                    "timeout_seconds": 23,
                },
            }
        }
    )

    assert run_config.enable_evidence_revision is False
    assert run_config.evidence_revision_trigger_coverage == 0.7
    assert run_config.evidence_revision_min_coverage_gain == 0.08
    assert run_config.evidence_revision_min_claim_retention == 0.75
    assert run_config.evidence_revision_timeout_seconds == 23


def test_evidence_refinement_accepts_supported_revision() -> None:
    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(url="https://example.com/model", title="Model card")
    store.add_evidence(
        source.source_id,
        "The model supports a context window of 128K tokens.",
        EvidenceKind.FULL_TEXT,
    )
    original = (
        "# Findings\n\n"
        "The model supports a context window of 128K tokens [1].\n\n"
        "An unrelated product claim has no supporting source [1].\n\n"
        "## References\n\n[1] Model card - https://example.com/model"
    )
    revised = (
        "# Findings\n\n"
        "The model supports a context window of 128K tokens [1].\n\n"
        "## References\n\n[1] Model card - https://example.com/model"
    )

    class _Reviser:
        @staticmethod
        def revise(**_kwargs):
            return RevisionDraft(content=revised, valid=True, reason="ok")

    o = Orchestrator(
        planner=None,
        agent_pool=None,
        evidence_store=store,
        evidence_verifier=ClaimVerifier(support_threshold=0.2),
        evidence_reviser=_Reviser(),
    )
    o._query = "q"
    o._config = RunConfig(
        global_timeout_seconds=60,
        enable_evidence=True,
        enable_evidence_revision=True,
        evidence_revision_trigger_coverage=0.9,
        evidence_revision_min_coverage_gain=0.01,
        evidence_revision_min_claim_retention=0.5,
    )
    o._start_time = time.monotonic()
    report = ResearchReport(
        query="q",
        content=original,
        sources=[{"source_id": source.source_id, "url": source.url}],
        confidence=0.8,
    )
    o._memory_store["final_report"] = report

    state = asyncio.run(o._do_evidence_refining())

    assert state.value == "done"
    assert report.content == revised
    assert report.evidence_revision["accepted"] is True
    assert report.evidence_revision["after"]["coverage"] > report.evidence_revision["before"]["coverage"]


def test_evidence_refinement_rolls_back_invalid_revision() -> None:
    store = EvidenceStore(persist_enabled=False)
    source = store.upsert_source(url="https://example.com/model", title="Model card")
    store.add_evidence(
        source.source_id,
        "The model supports a context window of 128K tokens.",
        EvidenceKind.FULL_TEXT,
    )
    original = (
        "# Findings\n\n"
        "The model supports a context window of 128K tokens [1].\n\n"
        "An unrelated product claim has no supporting source [1].\n\n"
        "## References\n\n[1] Model card - https://example.com/model"
    )

    class _Reviser:
        @staticmethod
        def revise(**_kwargs):
            return RevisionDraft(content="short", valid=False, reason="too short")

    o = Orchestrator(
        planner=None,
        agent_pool=None,
        evidence_store=store,
        evidence_verifier=ClaimVerifier(support_threshold=0.2),
        evidence_reviser=_Reviser(),
    )
    o._query = "q"
    o._config = RunConfig(
        global_timeout_seconds=60,
        enable_evidence=True,
        enable_evidence_revision=True,
        evidence_revision_trigger_coverage=0.9,
    )
    o._start_time = time.monotonic()
    report = ResearchReport(
        query="q",
        content=original,
        sources=[{"source_id": source.source_id, "url": source.url}],
        confidence=0.8,
    )
    o._memory_store["final_report"] = report

    asyncio.run(o._do_evidence_refining())

    assert report.content == original
    assert report.evidence_revision["accepted"] is False
    assert report.evidence_revision["reason"] == "too short"
