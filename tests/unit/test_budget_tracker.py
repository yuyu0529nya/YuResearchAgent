"""
tests/unit/test_budget_tracker.py
src/planner/budget_tracker.py 的单元测试：累计、阈值、快照、重置。
"""
import pytest

from src.planner.budget_tracker import BudgetTracker


def test_track_accumulates_and_history():
    b = BudgetTracker(budget_limit=1000)
    b.track(100)
    b.track(250)
    assert b.get_usage() == 350
    assert b.get_history() == [100, 250]


def test_negative_track_raises():
    b = BudgetTracker()
    with pytest.raises(ValueError):
        b.track(-1)


def test_usage_ratio_near_and_over_budget():
    b = BudgetTracker(budget_limit=1000)
    b.track(800)
    assert b.get_usage_ratio() == pytest.approx(0.8)
    assert b.is_near_budget()        # >= 0.8
    assert not b.is_over_budget()
    b.track(200)
    assert b.is_over_budget()         # 1000 >= 1000


def test_reset_clears_state():
    b = BudgetTracker()
    b.track(500)
    b.reset()
    assert b.get_usage() == 0
    assert b.get_history() == []


def test_budget_limit_clamped_to_min_one():
    b = BudgetTracker()
    b.set_budget_limit(0)   # 被夹到 1
    b.track(1)
    assert b.is_over_budget()


def test_snapshot_fields():
    b = BudgetTracker(budget_limit=200)
    b.track(50)
    s = b.snapshot()
    assert s.total_tokens == 50
    assert s.budget_limit == 200
    assert s.usage_ratio == pytest.approx(0.25)
    assert s.is_over_budget is False
