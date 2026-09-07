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


class _EmptySearchTool(_SearchTool):
    async def execute(self, query: str) -> dict:
        return {"query": query, "results": [{"title": "No excerpt", "url": "https://example.com"}]}


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


def test_researcher_keeps_last_slot_for_initial_search_when_browser_is_available() -> None:
    policy = _SkipsInitialToolPolicy()
    agent = ResearcherAgent(
        name="researcher",
        policy=policy,
        tools=[_SearchTool(), _BrowserTool()],
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
    assert [step.get("name") for step in result.trajectory if step.get("role") == "tool"] == [
        "web_search"
    ]


def test_researcher_rejects_empty_search_as_success() -> None:
    policy = _SkipsInitialToolPolicy()
    agent = ResearcherAgent(
        name="researcher",
        policy=policy,
        tools=[_EmptySearchTool()],
        max_turns=4,
        max_tool_calls=1,
    )

    result = asyncio.run(
        agent.run(
            SubTask(task_id="search", task_type=TaskType.SEARCH, description="research policy"),
            {"query": "research policy"},
        )
    )

    assert result.status == AgentStatus.FAILED
    assert [step.get("name") for step in result.trajectory if step.get("role") == "tool"] == [
        "web_search"
    ]


def test_researcher_parses_markdown_and_fullwidth_confidence() -> None:
    agent = ResearcherAgent(name="researcher", policy=_Policy())
    assert agent._extract_confidence("**置信度：0.25**") == 0.25


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


def test_early_read_leaves_budget_to_recover_from_blocked_source() -> None:
    class Browser(_BrowserTool):
        async def execute(self, url, max_chars=8000):
            self.urls.append(url)
            if len(self.urls) == 1:
                return "[Browser Error] HTTP 403"
            return "Republished explanation of the policy with sufficient evidence to read."

    browser = Browser()
    agent = ResearcherAgent(
        name="researcher", policy=_SearchUntilBudgetPolicy(),
        tools=[_SearchTool(), browser], max_turns=6, max_tool_calls=4,
    )
    result = asyncio.run(agent.run(
        SubTask(task_id="search", task_type=TaskType.SEARCH, description="research policy"),
        {"query": "research policy"},
    ))
    assert result.status == AgentStatus.SUCCESS
    assert browser.urls == ["https://www.gov.cn/zhengce/example", "https://www.sohu.com/a/123"]
    assert sum(step.get("role") == "tool" for step in result.trajectory) == 4


def test_paper_context_and_selection_use_same_fulltext_url() -> None:
    trajectory = [{"role": "tool", "name": "arxiv_reader", "result": {
        "query": "robotics benchmark", "papers": [{
            "title": "Robotics benchmark", "url": "https://example.org/landing",
            "pdf_url": "https://example.org/paper.pdf",
        }],
    }}]
    assert ResearcherAgent._best_unread_source_url(trajectory) == "https://example.org/paper.pdf"
    assert ResearcherAgent._source_context_for_url(
        trajectory, "https://example.org/paper.pdf"
    ) == ("Robotics benchmark", "robotics benchmark")


def test_researcher_closes_retrieval_before_summary_deadline(monkeypatch) -> None:
    import src.agents.researcher as module
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Search(_SearchTool):
        async def execute(self, query):
            result = await super().execute(query)
            clock[0] = 96.0
            return result

    class Policy(_Policy):
        def __call__(self, messages):
            if self.calls:
                assert self.tools is None
                assert "Retrieval is closed" in messages[-1]["content"]
            return super().__call__(messages)

    browser = _BrowserTool()
    policy = Policy()
    agent = ResearcherAgent("researcher", policy, [Search(), browser], max_tool_calls=4)
    result = asyncio.run(agent.run(
        SubTask(task_id="search", task_type=TaskType.SEARCH, description="research policy"),
        {"query": "research policy", "_request_deadline_monotonic": 120.0},
    ))
    assert result.status == AgentStatus.SUCCESS
    assert browser.urls == []
    assert policy.tools is not None


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
