"""
tests/unit/test_verdict.py
src/adversarial/verdict.py 单元测试：评分引擎、Issue 去重、序列化往返。

verdict.py 仅依赖 stdlib，故直接走包导入即可。
"""
import pytest

from src.adversarial.verdict import (
    Dimension,
    FixOperation,
    FixType,
    Issue,
    RedVerdict,
    Severity,
    VerdictEngine,
)


def test_compute_overall_weighted_average():
    scores = {
        Dimension.FACTUAL: 8.0,            # 0.30
        Dimension.HALLUCINATION: 6.0,      # 0.25
        Dimension.LOGICAL: 10.0,           # 0.20
        Dimension.SOURCE_CREDIBILITY: 4.0, # 0.15
        Dimension.COVERAGE: 2.0,           # 0.10
    }
    # 8*.3 + 6*.25 + 10*.2 + 4*.15 + 2*.1 = 6.7
    assert VerdictEngine.compute_overall(scores) == pytest.approx(6.7)


def test_compute_overall_empty_is_zero():
    assert VerdictEngine.compute_overall({}) == 0.0


def test_compute_overall_clamps_to_0_10():
    assert VerdictEngine.compute_overall({Dimension.FACTUAL: 999}) == pytest.approx(10.0)
    assert VerdictEngine.compute_overall({Dimension.FACTUAL: -5}) == pytest.approx(0.0)


def test_compute_delta_euclidean_distance():
    prev = {Dimension.FACTUAL: 5.0, Dimension.LOGICAL: 5.0}
    curr = {Dimension.FACTUAL: 8.0, Dimension.LOGICAL: 9.0}
    assert VerdictEngine.compute_delta(prev, curr) == pytest.approx(5.0)  # sqrt(3²+4²)


def test_compute_delta_missing_dimension_treated_as_zero():
    prev = {Dimension.FACTUAL: 3.0}
    curr = {Dimension.LOGICAL: 4.0}
    assert VerdictEngine.compute_delta(prev, curr) == pytest.approx(5.0)  # sqrt(3²+4²)


def test_compute_priority_critical_beats_minor():
    crit = Issue(Severity.CRITICAL, Dimension.FACTUAL, "核心事实错误")
    minor = Issue(Severity.MINOR, Dimension.COVERAGE, "措辞偏差")
    assert VerdictEngine.compute_priority(crit) > VerdictEngine.compute_priority(minor)


def test_issue_equality_and_hash_ignore_evidence():
    # 震荡检测依赖 Issue 的集合去重：内容相同但证据不同应视为同一 issue
    a = Issue(Severity.MAJOR, Dimension.FACTUAL, "同样的问题", location="p1")
    b = Issue(Severity.MAJOR, Dimension.FACTUAL, "同样的问题", location="p1", evidence="不同证据")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_issue_distinct_when_core_field_differs():
    a = Issue(Severity.MAJOR, Dimension.FACTUAL, "问题 A", location="p1")
    b = Issue(Severity.MAJOR, Dimension.FACTUAL, "问题 B", location="p1")
    assert a != b
    assert len({a, b}) == 2


def test_verdict_json_roundtrip():
    v = RedVerdict(
        dimension_scores={Dimension.FACTUAL: 7.5, Dimension.LOGICAL: 6.0},
        overall_score=7.0,
        issues=[Issue(Severity.MINOR, Dimension.LOGICAL, "d", location="p2")],
        raw_feedback="原始反馈",
    )
    restored = VerdictEngine.from_json(VerdictEngine.to_json(v))
    assert restored.overall_score == 7.0
    assert restored.dimension_scores[Dimension.FACTUAL] == 7.5
    assert restored.issues[0].description == "d"
    assert restored.raw_feedback == "原始反馈"


def test_fix_operation_to_dict():
    op = FixOperation(
        issue=Issue(Severity.MAJOR, Dimension.FACTUAL, "x"),
        action="in_place_fix",
        success=True,
    )
    d = op.to_dict()
    assert d["success"] is True
    assert d["action"] == "in_place_fix"
    assert d["issue"]["dimension"] == "fact_check"
