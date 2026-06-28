"""
tests/unit/test_schemas.py
src/orchestrator/schemas.py 的核心数据结构测试（枚举、默认值）。
"""
from src.orchestrator.schemas import (
    AgentResult,
    AgentStatus,
    OrchestratorState,
    ResearchReport,
    RunConfig,
    SubTask,
    TaskType,
)


def test_orchestrator_has_nine_states():
    # 坐实"9 状态状态机"
    assert len(list(OrchestratorState)) == 9
    assert OrchestratorState.REPLANNING in OrchestratorState
    assert OrchestratorState.ADVERSARIAL in OrchestratorState


def test_task_type_values():
    assert {t.value for t in TaskType} == {"search", "analyze", "verify"}


def test_agent_status_values():
    assert {s.value for s in AgentStatus} == {"success", "failed", "timeout"}


def test_run_config_defaults():
    c = RunConfig()
    assert c.max_concurrent == 5
    assert c.max_replan_rounds == 3
    assert c.enable_adversarial is True
    assert c.enable_evolution is False


def test_subtask_defaults():
    t = SubTask(task_id="t1", task_type=TaskType.SEARCH, description="d")
    assert t.dependencies == []
    assert t.context_keys == []
    assert t.timeout_seconds == 300
    assert t.priority == 1


def test_research_report_defaults():
    r = ResearchReport(query="q", content="c")
    assert r.confidence == 0.0
    assert r.adversarial_rounds == 0
    assert r.num_replan == 0
    assert r.sources == []


def test_agent_result_defaults():
    a = AgentResult(task_id="t", status=AgentStatus.SUCCESS)
    assert a.confidence == 0.0
    assert a.trajectory == []
    assert a.token_usage == 0
