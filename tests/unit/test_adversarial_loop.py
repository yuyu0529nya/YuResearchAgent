"""
tests/unit/test_adversarial_loop.py
src/adversarial/loop.py 单元测试 —— 用 mock 的 Red/Blue agent 驱动真实的
AdversarialLoop，验证收敛判定与震荡检测（M5 头牌功能）。

不调用任何 LLM：MockRed 返回脚本化的 RedVerdict，MockBlue 标记 issue 已修复。
"""
import asyncio

import pytest

from src.adversarial.loop import AdversarialLoop
from src.adversarial.verdict import (
    Dimension,
    FixOperation,
    Issue,
    RedVerdict,
    Severity,
)
from src.orchestrator.schemas import ResearchReport


def _verdict(overall: float, factual: float | None = None, issues=None) -> RedVerdict:
    f = overall if factual is None else factual
    return RedVerdict(
        dimension_scores={Dimension.FACTUAL: f, Dimension.LOGICAL: f},
        overall_score=overall,
        issues=issues or [],
    )


class MockRed:
    """按脚本依次返回 verdict。"""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)

    async def attack(self, report):
        return self._verdicts.pop(0)


class MockBlue:
    """把 verdict 中的每个 issue 都标记为已修复（success=resolve）。"""

    def __init__(self, resolve: bool = True):
        self.resolve = resolve

    async def defend(self, report, verdict):
        ops = [FixOperation(issue=i, action="fix", success=self.resolve) for i in verdict.issues]
        return report, ops


def _run(verdicts, *, max_rounds=3, score_threshold=8.0, delta_threshold=0.3, resolve=True):
    loop = AdversarialLoop(
        red_agent=MockRed(verdicts),
        blue_agent=MockBlue(resolve),
        max_rounds=max_rounds,
        score_threshold=score_threshold,
        delta_threshold=delta_threshold,
    )
    report = ResearchReport(query="q", content="c")
    return asyncio.run(loop.run(report))


def test_converges_on_score_threshold():
    final, hist = _run([_verdict(8.5)], score_threshold=8.0)
    assert len(hist) == 1
    assert "score_threshold_met" in hist[-1]["stop_reason"]
    assert final.final_score == pytest.approx(8.5)


def test_stops_at_max_rounds():
    # 分数始终偏低、维度分恒定 → 不达标也不收敛 → 跑满 max_rounds
    final, hist = _run([_verdict(5.0), _verdict(5.0), _verdict(5.0)],
                       max_rounds=3, delta_threshold=0.0)
    assert len(hist) == 3
    assert "max_rounds_reached" in hist[-1]["stop_reason"]


def test_converges_on_small_delta():
    # 第 2 轮分数几乎不变 → Δ < 阈值 → 收敛
    final, hist = _run([_verdict(5.0, factual=5.0), _verdict(5.05, factual=5.05), _verdict(5.0)],
                       max_rounds=5, delta_threshold=0.3)
    assert len(hist) == 2
    assert "delta_converged" in hist[-1]["stop_reason"]


def test_detects_oscillation_when_resolved_issue_reappears():
    issue = Issue(Severity.MAJOR, Dimension.FACTUAL, "X 数据错误", location="p1")
    # 第 1 轮发现并"修复"该 issue；第 2 轮同一 issue 重新出现 → 震荡
    final, hist = _run(
        [_verdict(5.0, issues=[issue]), _verdict(5.0, issues=[issue]), _verdict(5.0)],
        max_rounds=5,
        resolve=True,
    )
    assert hist[-1]["oscillation_detected"] is True
    assert "oscillation_at_round_2" in hist[-1]["stop_reason"]


def test_no_oscillation_if_issue_not_resolved():
    # Blue 未成功修复（resolve=False）→ issue 不进入 resolved 集合 → 不判震荡
    issue = Issue(Severity.MAJOR, Dimension.FACTUAL, "X 数据错误", location="p1")
    final, hist = _run(
        [_verdict(5.0, issues=[issue]), _verdict(5.0, issues=[issue]), _verdict(5.0)],
        max_rounds=3,
        delta_threshold=0.0,
        resolve=False,
    )
    assert all(not h["oscillation_detected"] for h in hist)
    assert "max_rounds_reached" in hist[-1]["stop_reason"]
