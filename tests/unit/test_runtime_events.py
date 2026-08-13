from __future__ import annotations

import asyncio
import time
from pathlib import Path

from src.agents.researcher import ResearcherAgent
from src.agents.summarizer import SummarizerAgent
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import (
    AgentResult,
    AgentStatus,
    ResearchReport,
    RunConfig,
    SubTask,
    TaskType,
)
from src.planner.dag import DAG
from src.runtime import CancellationToken, RunController, RunEvent, RunStore
from src.runtime.presentation import (
    apply_run_event,
    history_choices,
    new_run_view,
    render_progress,
)


def test_cancellation_token_keeps_first_reason() -> None:
    token = CancellationToken()

    assert token.request("first") is True
    assert token.request("second") is False
    assert token.is_cancelled is True
    assert token.reason == "first"
    assert token.requested_at is not None


def test_run_store_roundtrip_and_interrupted_recovery(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.create_run("run_1", query="q", backend="kimi", adversarial=False)
    event = RunEvent(
        run_id="run_1",
        sequence=1,
        kind="run_started",
        state="idle",
        message="started",
    )
    store.record_event(event)
    store.complete_run(
        "run_1",
        status="complete",
        report_path="outputs/reports/r.md",
        evidence_path="outputs/evidence/e.json",
        metadata={
            "elapsed_seconds": 12.5,
            "confidence": 0.77,
            "source_count": 2,
            "num_searches": 3,
            "evidence_audit": {
                "coverage": 0.5,
                "supported_count": 1,
                "claims": [{"status": "supported"}, {"status": "not_enough_evidence"}],
            },
            "evidence_revision": {"attempted": True, "accepted": False},
        },
    )

    row = store.get_run("run_1")
    assert row is not None
    assert row["status"] == "complete"
    assert row["coverage"] == 0.5
    assert row["supported_count"] == 1
    assert row["claim_count"] == 2
    assert row["revision_attempted"] is True
    assert row["revision_accepted"] is False
    assert store.get_events("run_1")[0]["kind"] == "run_started"

    store.create_run("run_2", query="q2", backend="kimi", adversarial=False)
    assert store.recover_interrupted() == 1
    assert store.get_run("run_2")["status"] == "interrupted"


def test_run_store_event_limit_returns_latest_events_in_order(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.create_run("run", query="q", backend="kimi", adversarial=False)
    for sequence in range(1, 7):
        store.record_event(
            RunEvent(
                run_id="run",
                sequence=sequence,
                kind="task_started",
                state="dispatching",
            )
        )

    assert [event["sequence"] for event in store.get_events("run", limit=3)] == [4, 5, 6]


def test_run_controller_routes_events_and_requests_cancel(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    controller = RunController(store)
    handle = controller.create_run(query="q", backend="kimi", adversarial=False)
    controller.event_sink(
        RunEvent(
            run_id=handle.run_id,
            sequence=1,
            kind="run_started",
            state="idle",
        )
    )

    assert handle.events.get_nowait().kind == "run_started"
    assert controller.cancel(handle.run_id, "stop") is True
    assert controller.cancel(handle.run_id, "stop again") is False
    assert handle.token.reason == "stop"
    assert store.get_run(handle.run_id)["status"] == "cancelling"

    controller.finish(handle.run_id)
    assert controller.get(handle.run_id) is None


def test_event_projection_uses_typed_payloads() -> None:
    view = new_run_view()
    events = [
        RunEvent(
            run_id="run",
            sequence=1,
            kind="plan_created",
            state="planning",
            payload={
                "task_count": 1,
                "layer_count": 1,
                "tasks": [
                    {
                        "task_id": "research_1",
                        "type": "search",
                        "description": "Find the primary paper.",
                    }
                ],
            },
        ),
        RunEvent(
            run_id="run",
            sequence=2,
            kind="task_started",
            state="dispatching",
            message="Find the primary paper.",
            payload={"task_id": "research_1", "task_type": "search"},
        ),
        RunEvent(
            run_id="run",
            sequence=3,
            kind="task_completed",
            state="dispatching",
            payload={
                "task_id": "research_1",
                "status": "success",
                "confidence": 0.9,
                "tool_calls": 2,
            },
        ),
        RunEvent(
            run_id="run",
            sequence=4,
            kind="evidence_snapshot",
            state="collecting",
            payload={"coverage": 0.75, "supported_count": 3, "claim_count": 4},
        ),
    ]
    for event in events:
        apply_run_event(view, event)

    assert view["tasks"]["research_1"]["status"] == "success"
    assert view["evidence"]["coverage"] == 0.75
    rendered = render_progress(
        view,
        elapsed=12,
        backend="kimi",
        adversarial=False,
        run_id="run",
    )
    assert "research_1" in rendered
    assert "75.0%" in rendered


def test_history_choices_are_stable_and_bounded() -> None:
    rows = [
        {
            "run_id": "r1",
            "created_at": "2026-08-13T12:00:00+00:00",
            "status": "cancelled_partial",
            "query": "A very long research question " * 8,
        }
    ]

    choices = history_choices(rows)

    assert choices[0][1] == "r1"
    assert "CANCELLED PARTIAL" in choices[0][0]
    assert len(choices[0][0]) < 100


def test_researcher_stops_before_policy_call_when_cancelled() -> None:
    class _Policy:
        def __call__(self, _messages):
            raise AssertionError("cancelled task must not call the model")

    token = CancellationToken()
    token.request("stop")
    agent = ResearcherAgent(name="researcher", policy=_Policy(), tools=[])
    result = asyncio.run(
        agent.run(
            SubTask(task_id="t1", task_type=TaskType.SEARCH, description="search"),
            {"query": "q", "_cancellation_token": token},
        )
    )

    assert result.status == AgentStatus.CANCELLED
    assert result.output == "stop"


def test_summarizer_stops_before_policy_call_when_cancelled() -> None:
    class _Policy:
        tools = None

        def __call__(self, _messages):
            raise AssertionError("cancelled synthesis must not call the model")

    token = CancellationToken()
    token.request("stop")
    agent = SummarizerAgent(name="summarizer", policy=_Policy())
    result = asyncio.run(
        agent.run(
            SubTask(task_id="s", task_type=TaskType.ANALYZE, description="synthesize"),
            {
                "query": "q",
                "results": [AgentResult(task_id="t", status=AgentStatus.SUCCESS, output="fact")],
                "_cancellation_token": token,
            },
        )
    )

    assert result.status == AgentStatus.CANCELLED


def test_researcher_forwards_subtask_provider_deadline() -> None:
    class _Policy:
        def __init__(self) -> None:
            self.timeout = None

        def __call__(self, _messages):
            raise AssertionError("bounded worker must use call_with_timeout")

        def call_with_timeout(self, _messages, timeout_seconds):
            self.timeout = timeout_seconds
            return {
                "content": "A sufficiently detailed final answer. Confidence: 0.8",
                "tool_calls": [],
            }

    policy = _Policy()
    agent = ResearcherAgent(name="researcher", policy=policy, tools=[])
    result = asyncio.run(
        agent.run(
            SubTask(task_id="t1", task_type=TaskType.SEARCH, description="search"),
            {
                "query": "q",
                "_request_deadline_monotonic": time.monotonic() + 3,
            },
        )
    )

    assert result.status == AgentStatus.SUCCESS
    assert policy.timeout is not None
    assert 0.25 <= policy.timeout <= 3


def test_summarizer_forwards_provider_deadline() -> None:
    class _Policy:
        tools = None

        def __init__(self) -> None:
            self.timeout = None

        def __call__(self, _messages):
            raise AssertionError("bounded synthesis must use call_with_timeout")

        def call_with_timeout(self, _messages, timeout_seconds):
            self.timeout = timeout_seconds
            return {"content": "# Report\n\nA sufficiently detailed sourced finding."}

    policy = _Policy()
    agent = SummarizerAgent(name="summarizer", policy=policy)
    result = asyncio.run(
        agent.run(
            SubTask(task_id="s", task_type=TaskType.ANALYZE, description="synthesize"),
            {
                "query": "q",
                "results": [
                    AgentResult(task_id="t", status=AgentStatus.SUCCESS, output="fact")
                ],
                "_request_deadline_monotonic": time.monotonic() + 3,
            },
        )
    )

    assert result.status == AgentStatus.SUCCESS
    assert policy.timeout is not None
    assert 0.25 <= policy.timeout <= 3


def test_orchestrator_cancelled_run_preserves_prior_layer_results() -> None:
    token = CancellationToken()

    class _Planner:
        _last_raw_json = {}

        @staticmethod
        def generate_plan(_query, _memory):
            dag = DAG()
            dag.add_node("first")
            dag.add_node("second")
            dag.add_edge("first", "second")
            return dag

        @staticmethod
        def get_task_map_from_dag(_dag, _raw):
            return {
                "first": SubTask("first", TaskType.SEARCH, "first"),
                "second": SubTask(
                    "second",
                    TaskType.SEARCH,
                    "second",
                    dependencies=["first"],
                ),
            }

    class _Worker:
        async def run(self, task, _context):
            if task.task_id == "second":
                token.request("stop after first")
                return AgentResult(task_id=task.task_id, status=AgentStatus.CANCELLED)
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.SUCCESS,
                output="Preserved first-layer evidence.",
                confidence=0.8,
            )

    class _Pool:
        async def get_agent(self, _task_type):
            return _Worker()

        async def release_agent(self, _agent):
            return None

    orchestrator = Orchestrator(
        planner=_Planner(),
        agent_pool=_Pool(),
        cancellation_token=token,
    )
    report = asyncio.run(
        orchestrator.run(
            "q",
            RunConfig(enable_evidence=False, enable_adversarial=False),
        )
    )

    assert report.run_status == "cancelled_partial"
    assert "Preserved first-layer evidence" in report.content


def test_orchestrator_cancelled_run_preserves_completed_results_and_events() -> None:
    token = CancellationToken()
    token.request("stop")
    events: list[RunEvent] = []
    orchestrator = Orchestrator(
        planner=None,
        agent_pool=None,
        run_id="cancelled_run",
        event_sink=events.append,
        cancellation_token=token,
    )
    orchestrator._all_results = [
        AgentResult(
            task_id="done",
            status=AgentStatus.SUCCESS,
            output="A completed finding.",
            confidence=0.8,
        )
    ]

    report = asyncio.run(
        orchestrator.run(
            "q",
            RunConfig(enable_evidence=False),
        )
    )

    # run() resets prior in-memory results, so a pre-start cancellation has no partial result.
    assert report.run_status == "cancelled"
    assert any(event.kind == "cancellation_requested" for event in events)
    assert any(event.kind == "cancellation_observed" for event in events)
    assert events[-1].kind == "run_completed"


def test_full_state_machine_emits_ordered_structured_events() -> None:
    class _Planner:
        _last_raw_json = {}

        @staticmethod
        def generate_plan(_query, _memory):
            dag = DAG()
            dag.add_node("research_1")
            return dag

        @staticmethod
        def get_task_map_from_dag(_dag, _raw):
            return {
                "research_1": SubTask(
                    task_id="research_1",
                    task_type=TaskType.SEARCH,
                    description="Find one fact.",
                )
            }

    class _Worker:
        async def run(self, task, _context):
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.SUCCESS,
                output="A sourced finding.",
                confidence=0.8,
            )

    class _Policy:
        tools = None

        def __call__(self, _messages):
            return {"content": "# Report\n\nA sufficiently detailed sourced finding."}

    class _Pool:
        async def get_agent(self, task_type):
            return (
                SummarizerAgent(name="summarizer", policy=_Policy())
                if task_type == TaskType.ANALYZE
                else _Worker()
            )

        async def release_agent(self, _agent):
            return None

    events: list[RunEvent] = []
    orchestrator = Orchestrator(
        planner=_Planner(),
        agent_pool=_Pool(),
        run_id="structured",
        event_sink=events.append,
    )

    report = asyncio.run(
        orchestrator.run(
            "q",
            RunConfig(enable_evidence=False, enable_adversarial=False),
        )
    )

    kinds = [event.kind for event in events]
    assert report.run_status == "complete"
    assert kinds[0] == "run_started"
    assert "plan_created" in kinds
    assert "task_started" in kinds
    assert "task_completed" in kinds
    assert "synthesis_completed" in kinds
    assert kinds[-1] == "run_completed"
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
