import asyncio

from src.agents.researcher import ResearcherAgent
from src.orchestrator.schemas import AgentResult, AgentStatus, SubTask, TaskType


class _Policy:
    def __init__(self) -> None:
        self.tools = ["existing"]
        self.messages = []

    def __call__(self, messages):
        self.messages = messages
        return {"content": "基于上游证据完成分析。置信度: 0.80", "tool_calls": []}


def test_analyze_task_consumes_dependency_without_new_search() -> None:
    policy = _Policy()
    agent = ResearcherAgent(name="analyst", policy=policy, tools=[])
    task = SubTask(
        task_id="analysis",
        task_type=TaskType.ANALYZE,
        description="比较指定的四款模型",
        dependencies=["search"],
    )
    context = {
        "query": "比较 GPT-4o、Claude 3.5、Gemini 1.5、Qwen2.5",
        "dep:search": AgentResult(
            task_id="search",
            status=AgentStatus.SUCCESS,
            output="Qwen2.5 Technical Report: 18T tokens.",
        ),
    }

    result = asyncio.run(agent.run(task, context))

    assert result.status == AgentStatus.SUCCESS
    assert policy.tools == ["existing"]
    prompt = policy.messages[-1]["content"]
    assert "GPT-4o" in prompt
    assert "Qwen2.5 Technical Report" in prompt
    assert "dep:search" in prompt


def test_mislabeled_search_synthesis_consumes_dependency_without_tools() -> None:
    policy = _Policy()
    agent = ResearcherAgent(name="analyst", policy=policy, tools=[])
    task = SubTask(
        task_id="analysis",
        task_type=TaskType.SEARCH,
        description="基于前序检索结果，对比分析四款模型并归因差异",
        dependencies=["search"],
    )
    context = {
        "query": "比较四款模型",
        "dep:search": AgentResult(
            task_id="search",
            status=AgentStatus.SUCCESS,
            output="上游已提供来源与模型事实。",
        ),
    }

    result = asyncio.run(agent.run(task, context))

    assert result.status == AgentStatus.SUCCESS
    assert policy.tools == ["existing"]
    assert "上游已提供来源" in policy.messages[-1]["content"]


def test_dependency_verification_remains_tool_backed() -> None:
    agent = ResearcherAgent(name="verifier", policy=_Policy(), tools=[])
    task = SubTask(
        task_id="verify",
        task_type=TaskType.SEARCH,
        description="核验前序报告中的参数数字",
        dependencies=["search"],
    )
    context = {"dep:search": "未经核验的参数数字"}

    assert agent._is_dependency_analysis_task(task, context) is False
