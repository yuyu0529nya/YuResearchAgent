"""
tests/unit/test_rule_based.py
evaluation/metrics/rule_based.py 的纯规则指标单元测试。

直接按文件路径加载，避免 evaluation/metrics/__init__ 牵连 numpy 等依赖。
仅覆盖纯文本指标（不含需要 embedding 的 semantic_fact_accuracy）。
"""
import importlib.util
import os

import pytest

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "evaluation", "metrics", "rule_based.py",
)
_spec = importlib.util.spec_from_file_location("rule_based_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
M = _mod.RuleBasedMetrics


def test_logical_consistency_negation_not_false_positive():
    # 回归：单句否定（"不是"含"是"、"不会"含"会"）不应被误判为自相矛盾
    assert M.logical_consistency("这不是一个事实。") == 1.0
    assert M.logical_consistency("我不会去。") == 1.0
    assert M.logical_consistency("该方案不可以采用。") == 1.0


def test_logical_consistency_detects_real_contradiction():
    # 同句内既"支持"又"反对"是真矛盾，应被检出
    assert M.logical_consistency("他说支持这个方案，但马上又反对了。") < 1.0


def test_citation_coverage_partial():
    text = "结论 A [1]\n无引用的一段\nhttps://example.com 的数据"
    assert M.citation_coverage(text) == pytest.approx(2 / 3)


def test_hallucination_rate_empty_is_one():
    assert M.hallucination_rate("") == 1.0


def test_hallucination_rate_cited_sentence_is_low():
    # 含引用标记的句子被排除在幻觉嫌疑之外
    assert M.hallucination_rate("根据 [1] 的数据，该指标保持稳定。") == 0.0


def test_comprehensiveness_topic_coverage():
    assert M.comprehensiveness("讨论了 A 和 B", ["A", "B", "C"]) == pytest.approx(2 / 3)


def test_efficiency_score_rewards_fewer_turns():
    # 轮数越少，效率奖励越高（sigmoid 单调）
    assert M.efficiency_score(2) > M.efficiency_score(12)
