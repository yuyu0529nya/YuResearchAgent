"""
tests/unit/test_planner_parse.py
src/planner/planner.py 的解析路径单元测试。

同时作为集成测试，验证 Planner 复用统一 JSON 解析器（extract_json_object）
后，仍能正确从噪声输出构建 DAG，并保留 SubTask 信息。
policy 传 None —— 这些路径不触发任何 LLM 调用。
"""
import pytest

from src.planner.planner import Planner, PlanParseError
from src.planner.budget_tracker import BudgetTracker


def _planner() -> Planner:
    return Planner(policy=None, budget_tracker=BudgetTracker())


def test_parse_plan_from_fenced_json_builds_dag():
    raw = (
        "规划如下：\n```json\n"
        '{"sub_tasks":['
        '{"task_id":"t1","task_type":"search","description":"d1","dependencies":[]},'
        '{"task_id":"t2","task_type":"analyze","description":"d2","dependencies":["t1"]}'
        "]}\n```"
    )
    dag = _planner()._parse_plan(raw)
    assert len(dag) == 2
    order = dag.topological_sort()
    assert order.index("t1") < order.index("t2")


def test_parse_plan_missing_subtasks_key_raises():
    with pytest.raises(PlanParseError):
        _planner()._parse_plan('{"foo": 1}')


def test_parse_plan_unparseable_raises():
    with pytest.raises(PlanParseError):
        _planner()._parse_plan("完全不是 JSON 的一段话")


def test_parse_plan_tolerates_trailing_comma():
    raw = '{"sub_tasks":[{"task_id":"t1","task_type":"search","description":"d","dependencies":[]},]}'
    dag = _planner()._parse_plan(raw)
    assert len(dag) == 1


def test_get_task_map_preserves_subtask_fields():
    raw = '{"sub_tasks":[{"task_id":"t1","task_type":"search","description":"desc-1","dependencies":[]}]}'
    p = _planner()
    dag = p._parse_plan(raw)
    task_map = p.get_task_map_from_dag(dag, raw)
    assert "t1" in task_map
    assert task_map["t1"].description == "desc-1"


def test_build_prompt_uses_configured_subtask_limit():
    planner = Planner(policy=None, max_sub_questions=5)
    prompt = planner._build_prompt("test query", "")
    assert "no more than 5 sub_tasks" in prompt
    assert "up to 5 focused sub_tasks" in prompt


def test_build_prompt_reserves_explicit_dimensions_for_research_tasks():
    planner = Planner(policy=None, max_sub_questions=4)

    prompt = planner._build_prompt(
        "Analyze world models, tactile sensing, and task planning.",
        "",
    )

    assert "dedicated final Summarizer" in prompt
    assert "distinct evidence-gathering sub-task" in prompt
    assert "Do not merge two named dimensions" in prompt


def test_planner_forwards_provider_deadline():
    class _Policy:
        timeout = None

        def __call__(self, _messages):
            raise AssertionError("bounded planning must use call_with_timeout")

        def call_with_timeout(self, _messages, timeout_seconds):
            self.timeout = timeout_seconds
            return {
                "content": (
                    '{"sub_tasks":[{"task_id":"t1","task_type":"search",'
                    '"description":"find","dependencies":[]}]}'
                )
            }

    policy = _Policy()
    planner = Planner(policy=policy)

    dag = planner.generate_plan_with_timeout("q", "", 4.5)

    assert len(dag) == 1
    assert policy.timeout == 4.5
