from __future__ import annotations

import asyncio
import json

from src.agents.researcher import ResearcherAgent
from src.orchestrator.schemas import AgentStatus, SubTask, TaskType


class _SearchTool:
    name = "web_search"

    @staticmethod
    def get_openai_tool_schema() -> dict:
        return {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}

    async def execute(self, query: str) -> dict:
        return {
            "query": query,
            "results": [
                {
                    "title": "Republished explanation",
                    "url": "https://www.sohu.com/a/123",
                    "snippet": "secondary account",
                },
                {
                    "title": "Official policy",
                    "url": "https://www.gov.cn/zhengce/example",
                    "snippet": "primary policy text",
                },
            ],
        }


class _BrowserTool:
    name = "browser"

    def __init__(self) -> None:
        self.urls: list[str] = []

    @staticmethod
    def get_openai_tool_schema() -> dict:
        return {"type": "function", "function": {"name": "browser", "parameters": {"type": "object"}}}

    async def execute(self, url: str, max_chars: int = 8000) -> str:
        self.urls.append(url)
        return "Official full policy text with sufficient evidence."[:max_chars]


class _AcademicTool:
    name = "arxiv_reader"

    @staticmethod
    def get_openai_tool_schema() -> dict:
        return {"type": "function", "function": {"name": "arxiv_reader", "parameters": {"type": "object"}}}

    async def execute(self, query: str, max_results: int = 3) -> dict:
        return {
            "query": query,
            "papers": [
                {
                    "title": "Robotics benchmark survey",
                    "url": "https://arxiv.org/abs/2501.00001",
                    "summary": "A survey of robotics benchmarks.",
                }
            ][:max_results],
        }


class _Policy:
    def __init__(self) -> None:
        self.calls = 0
        self.tools = None

    def set_tools(self, tools) -> None:
        self.tools = tools

    def __call__(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "policy"}),
                        },
                    }
                ],
            }
        return {"content": "Evidence-backed final summary. Confidence: 0.8", "tool_calls": []}


class _SearchUntilBudgetPolicy(_Policy):
    def __call__(self, _messages):
        self.calls += 1
        if self.calls <= 3:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"search_{self.calls}",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": f"policy {self.calls}"}),
                        },
                    }
                ],
            }
        return {"content": "Evidence-backed final summary. Confidence: 0.8", "tool_calls": []}


class _WeakBrowserFirstPolicy(_Policy):
    def __call__(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [{
                    "id": "search_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": "policy"})},
                }],
            }
        if self.calls == 2:
            return {
                "content": "",
                "tool_calls": [{
                    "id": "browser_weak",
                    "type": "function",
                    "function": {
                        "name": "browser",
                        "arguments": json.dumps({"url": "https://www.sohu.com/a/123"}),
                    },
                }],
            }
        return {"content": "Evidence-backed final summary. Confidence: 0.8", "tool_calls": []}


class _SkipsInitialToolPolicy(_Policy):
    def __call__(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return {"content": "Unsourced answer.", "tool_calls": []}
        return {"content": "Evidence-backed final summary. Confidence: 0.8", "tool_calls": []}


def test_researcher_forces_initial_retrieval_when_model_skips_tool_use() -> None:
    policy = _SkipsInitialToolPolicy()
    agent = ResearcherAgent(
        name="researcher",
        policy=policy,
        tools=[_SearchTool()],
        max_turns=4,
        max_tool_calls=1,
    )

    result = asyncio.run(
        agent.run(
            SubTask(task_id="search", task_type=TaskType.SEARCH, description="research policy"),
            {"query": "research policy"},
        )
    )

    assert result.status == AgentStatus.SUCCESS
    assert policy.calls == 2
    assert [step.get("name") for step in result.trajectory if step.get("role") == "tool"] == [
        "web_search"
    ]


def test_researcher_reads_best_primary_source_before_finishing() -> None:
    browser = _BrowserTool()
    policy = _Policy()
    agent = ResearcherAgent(
        name="researcher",
        policy=policy,
        tools=[_SearchTool(), browser],
        max_turns=5,
        max_tool_calls=3,
    )

    result = asyncio.run(
        agent.run(
            SubTask(task_id="search", task_type=TaskType.SEARCH, description="research policy"),
            {"query": "research policy"},
        )
    )

    assert result.status == AgentStatus.SUCCESS
    assert browser.urls == ["https://www.gov.cn/zhengce/example"]
    assert [step.get("name") for step in result.trajectory if step.get("role") == "tool"] == [
        "web_search",
        "browser",
    ]


def test_researcher_reserves_last_tool_slot_for_fulltext() -> None:
    browser = _BrowserTool()
    agent = ResearcherAgent(
        name="researcher",
        policy=_SearchUntilBudgetPolicy(),
        tools=[_SearchTool(), browser],
        max_turns=6,
        max_tool_calls=3,
    )

    result = asyncio.run(
        agent.run(
            SubTask(task_id="search", task_type=TaskType.SEARCH, description="research policy"),
            {"query": "research policy"},
        )
    )

    assert result.status == AgentStatus.SUCCESS
    assert [step.get("name") for step in result.trajectory if step.get("role") == "tool"] == [
        "web_search",
        "web_search",
        "browser",
    ]


def test_researcher_reserves_slot_when_model_proposes_parallel_searches() -> None:
    class _ParallelPolicy(_Policy):
        def __call__(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"search_{index}",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": f"policy {index}"}),
                            },
                        }
                        for index in range(3)
                    ],
                }
            return {"content": "Evidence-backed final summary. Confidence: 0.8", "tool_calls": []}

    browser = _BrowserTool()
    agent = ResearcherAgent(
        name="researcher",
        policy=_ParallelPolicy(),
        tools=[_SearchTool(), browser],
        max_turns=5,
        max_tool_calls=3,
    )

    result = asyncio.run(
        agent.run(
            SubTask(task_id="search", task_type=TaskType.SEARCH, description="research policy"),
            {"query": "research policy"},
        )
    )

    assert result.status == AgentStatus.SUCCESS
    assert [step.get("name") for step in result.trajectory if step.get("role") == "tool"] == [
        "web_search",
        "web_search",
        "browser",
    ]


def test_weak_fulltext_does_not_satisfy_authoritative_read_gate() -> None:
    browser = _BrowserTool()
    agent = ResearcherAgent(
        name="researcher",
        policy=_WeakBrowserFirstPolicy(),
        tools=[_SearchTool(), browser],
        max_turns=5,
        max_tool_calls=3,
    )

    result = asyncio.run(
        agent.run(
            SubTask(task_id="search", task_type=TaskType.SEARCH, description="research policy"),
            {"query": "research policy"},
        )
    )

    assert result.status == AgentStatus.SUCCESS
    assert browser.urls == [
        "https://www.sohu.com/a/123",
        "https://www.gov.cn/zhengce/example",
    ]


def test_technical_research_reserves_academic_and_fulltext_calls() -> None:
    browser = _BrowserTool()
    agent = ResearcherAgent(
        name="researcher",
        policy=_SearchUntilBudgetPolicy(),
        tools=[_SearchTool(), _AcademicTool(), browser],
        max_turns=7,
        max_tool_calls=4,
    )

    result = asyncio.run(
        agent.run(
            SubTask(
                task_id="search",
                task_type=TaskType.SEARCH,
                description="research robot world model benchmarks",
            ),
            {"query": "latest Embodied AI robot progress"},
        )
    )

    names = [
        step.get("name")
        for step in result.trajectory
        if step.get("role") == "tool"
    ]
    assert result.status == AgentStatus.SUCCESS
    assert "web_search" in names
    assert "arxiv_reader" in names
    assert "browser" in names
