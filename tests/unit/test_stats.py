"""
tests/unit/test_stats.py
evaluation/metrics/stats.py 的单元测试：bootstrap CI / Cohen's d / 配对 t 检验。

直接按文件路径加载（避免 evaluation/metrics/__init__ 的连带导入）。需要 numpy。
所有 bootstrap 用例都传 seed，保证确定性。
"""
import importlib.util
import os

import pytest

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "evaluation", "metrics", "stats.py",
)
_spec = importlib.util.spec_from_file_location("stats_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
S = _mod

_DIFFS = [1.1, 0.5, 1.5, 0.3, 1.6, 0.7, 0.8]
_A = [8.2, 7.9, 8.5, 8.1]
_B = [7.1, 7.0, 6.9, 7.2]


def test_bootstrap_reproducible_with_seed():
    assert S.bootstrap_ci_paired(_DIFFS, seed=7) == S.bootstrap_ci_paired(_DIFFS, seed=7)


def test_bootstrap_significant_for_clear_positive_diff():
    r = S.bootstrap_ci_paired(_DIFFS, seed=1)
    assert r["mean_diff"] > 0
    assert r["ci_lower"] > 0
    assert r["significant"] is True
    assert r["n"] == len(_DIFFS)


def test_bootstrap_empty_input():
    r = S.bootstrap_ci_paired([])
    assert r["significant"] is False
    assert r["n"] == 0


def test_two_sample_reproducible_with_seed():
    assert S.bootstrap_ci_two_sample(_A, _B, seed=3) == S.bootstrap_ci_two_sample(_A, _B, seed=3)


def test_cohens_d_large_effect():
    assert S.cohens_d(_A, _B) > 0.8  # 远超 Cohen 的 "large" 阈值 0.8


def test_cohens_d_degenerate_returns_zero_not_nan():
    assert S.cohens_d([5.0], [3.0]) == 0.0          # n < 2
    assert S.cohens_d([], []) == 0.0                # 空
    assert S.cohens_d([5, 5, 5], [5, 5, 5]) == 0.0  # 零方差


def test_paired_t_test_consistent_schema():
    r = S.paired_t_test(_A, _B)
    assert {"t_statistic", "p_value", "mean_diff", "n", "method"}.issubset(r)
    assert r["t_statistic"] > 0
    assert 0.0 <= r["p_value"] <= 1.0


def test_paired_t_test_insufficient_n():
    r = S.paired_t_test([1.0], [0.5])
    assert r["method"] == "insufficient_n"
    assert r["t_statistic"] == 0.0
