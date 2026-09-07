#!/usr/bin/env python3
"""Run a single retrieval-equipped Researcher as a fair comparison baseline.

This intentionally does not call the multi-agent planner, evidence reviser, or
adversarial loop. It uses the same configured worker model and tools, with an
explicit tool/timeout budget, so future head-to-head studies can separate the
value of orchestration from the value of retrieval itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runner import initialize_modules, load_config
from src.orchestrator.schemas import AgentResult, AgentStatus, SubTask, TaskType


async def run_baseline(query: str, config: dict, timeout_seconds: float) -> dict:
    modules = initialize_modules(
        config,
        session_id=f"matched_baseline_{time.time_ns()}",
    )
    pool = modules["agent_pool"]
    agent = await pool.get_agent(TaskType.SEARCH)
    task = SubTask(
        task_id="single_retrieval_agent",
        task_type=TaskType.SEARCH,
        description=query,
        timeout_seconds=timeout_seconds,
        expected_type="factual",
    )
    context = {
        "query": query,
        "_request_deadline_monotonic": time.monotonic() + timeout_seconds - 0.5,
    }
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(
            agent.run(task, context),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        result = AgentResult(
            task_id=task.task_id, status=AgentStatus.TIMEOUT,
            output="Diagnostic timed out; completed tool evidence retained.",
            trajectory=list(getattr(agent, "last_trajectory", [])),
            token_usage=int(getattr(agent, "last_token_usage", 0)),
        )
    finally:
        await pool.release_agent(agent)
        for tool in modules.get("tools", []):
            close_session = getattr(tool, "close_session", None)
            if callable(close_session):
                await close_session()
    return {
        "baseline": "single_retrieval_agent",
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "status": result.status.value,
        "output": result.output,
        "trajectory": result.trajectory,
        "token_usage": result.token_usage,
        "estimated_worker_tokens": result.token_usage,
        "model_usage": modules["usage_tracker"].snapshot(),
        "not_a_quality_benchmark": True,
        "tool_calls": sum(step.get("role") == "tool" for step in result.trajectory),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument("--output", type=Path, help="Persist the diagnostic, including provider usage")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_tool_calls is not None:
        config.setdefault("planner", {})["max_search_rounds_per_subagent"] = max(
            1, int(args.max_tool_calls)
        )
    # Keep the helper's effective configuration visible in the output without
    # serializing credentials or the full runtime object graph.
    config.setdefault("orchestrator", {})["global_timeout_seconds"] = max(
        1, int(args.timeout)
    )
    result = asyncio.run(run_baseline(args.query, config, max(1.0, args.timeout)))
    result["effective_worker_budget"] = {
        "max_tool_calls": config.get("planner", {}).get("max_search_rounds_per_subagent"),
        "timeout_seconds": args.timeout,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key not in {"output", "trajectory"}}, indent=2))
    else:
        print(serialized)


if __name__ == "__main__":
    main()
