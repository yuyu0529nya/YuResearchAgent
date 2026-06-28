"""
tests/unit/test_orchestrator_logic.py
src/orchestrator/orchestrator.py 的纯决策逻辑单元测试。

只测不依赖异步状态机的纯方法（重规划判定、全局超时、失败原因归纳）；
用 planner=None / agent_pool=None 构造，因为这些方法不触及它们。
"""
import asyncio
import time

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
