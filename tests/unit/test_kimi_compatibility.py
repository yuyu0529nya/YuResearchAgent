"""Kimi K3 compatibility regressions for sampling and tool schemas."""

from src.agents.researcher import ResearcherAgent
from src.orchestrator.schemas import AgentResult, AgentStatus, SubTask, TaskType
from src.agents.summarizer import SummarizerAgent
from src.tools.arxiv_reader import ArxivReaderTool


def test_arxiv_schema_avoids_unsupported_combinators():
    parameters = ArxivReaderTool(use_mock=True).get_openai_tool_schema()["function"]["parameters"]
    assert "anyOf" not in parameters
    assert "oneOf" not in parameters
    assert parameters["type"] == "object"


def test_model_api_error_is_not_treated_as_success():
    agent = ResearcherAgent(name="test", policy=None, tools=[])
    assert agent._is_tool_failure_explanation("Error: Error code: 400 - invalid schema") is True


def test_summarizer_restores_tools_and_marks_api_error_failed():
    class ErrorPolicy:
        def __init__(self):
            self.tools = ["original"]

        def __call__(self, messages):
            return {"content": "Error: Request timed out.", "tool_calls": []}

    policy = ErrorPolicy()
    agent = SummarizerAgent(name="summarizer", policy=policy)
    task = SubTask(task_id="s", task_type=TaskType.ANALYZE, description="synthesize")
    context = {
        "query": "q",
        "results": [AgentResult(task_id="r", status=AgentStatus.SUCCESS, output="evidence")],
    }

    import asyncio

    result = asyncio.run(agent.run(task, context))
    assert result.status == AgentStatus.FAILED
    assert policy.tools == ["original"]
