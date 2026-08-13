#!/usr/bin/env python3
"""Auditable paired evaluation: full research agent vs one-call LLM baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmarks.research_bench import (
    RESEARCHBENCH_EVALUATION_VERSION,
    RESEARCHBENCH_METRIC_WEIGHTS,
    ResearchBench,
)
from evaluation.evidence_metrics import evidence_quality_metrics
from evaluation.protocol import (
    atomic_write_json,
    load_report_artifact,
    normalize_counterbalanced_judgments,
    paired_summary,
    save_report_artifact,
    sha256_text,
    usage_delta,
)
from src.core.judge import LLMJudge
from src.core.runner import (
    initialize_modules,
    load_config,
    run_research_with_metadata,
    setup_logging,
)
from src.models.model_router import ModelRouter
from src.models.vllm_policy import VLLMPolicy


ARTIFACT_SCHEMA = "headtohead-v3-auditable"
_BASELINE_SYSTEM = (
    "你是一位研究助手。针对用户问题，直接写一份全面、结构化的 Markdown 研究报告。"
    "回答必须独立完成，不得声称已经联网或调用外部工具；只引用你确实知道的来源。"
    "结尾给出 Overall Confidence: X.XX。"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sampling_for_baseline(config: dict[str, Any], backend: str) -> dict[str, Any]:
    sampling = config.get("model", {}).get("backend_sampling", {})
    resolved = dict(sampling.get(backend, {}))
    resolved.update(sampling.get("modules", {}).get("summarizer", {}))
    return {
        key: resolved[key]
        for key in ("temperature", "top_p", "max_tokens")
        if key in resolved
    }


def run_baseline(query: str, policy: Any) -> str:
    response = policy(
        [
            {"role": "system", "content": _BASELINE_SYSTEM},
            {"role": "user", "content": query},
        ]
    )
    content = response.get("content", "") if isinstance(response, dict) else ""
    if not content or content.lstrip().lower().startswith("error:"):
        raise RuntimeError(content or "baseline returned empty content")
    return content


def _select_questions(bench: ResearchBench, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.question_ids:
        by_id = {question["id"]: question for question in bench.questions}
        missing = [question_id for question_id in args.question_ids if question_id not in by_id]
        if missing:
            raise ValueError(f"Unknown ResearchBench question IDs: {', '.join(missing)}")
        return [by_id[question_id] for question_id in args.question_ids]
    return bench.get_questions(n=args.num_questions)


def _copy_evidence_artifact(
    source_path: str,
    reports_dir: Path,
    question_id: str,
) -> dict[str, Any] | None:
    source = Path(source_path)
    if not source.is_file():
        return None
    target = reports_dir / "evidence" / f"{question_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    payload = target.read_text(encoding="utf-8")
    try:
        portable_path = target.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        portable_path = target.resolve().as_posix()
    return {"path": portable_path, "sha256": sha256_text(payload)}


def _counterbalanced_judge(
    judge: LLMJudge,
    *,
    agent_report: str,
    baseline_report: str,
    query: str,
    ground_truth: dict[str, Any],
    orders: int,
    reverse_first: bool,
) -> dict[str, Any]:
    presentations = [
        ({"A": "agent", "B": "baseline"}, agent_report, baseline_report),
        ({"A": "baseline", "B": "agent"}, baseline_report, agent_report),
    ]
    if orders == 1 and reverse_first:
        presentations.reverse()
    judgments = []
    for order, report_a, report_b in presentations[:orders]:
        judgments.append(
            {
                "order": order,
                "result": judge.compare_two(report_a, report_b, query, ground_truth),
            }
        )
    return normalize_counterbalanced_judgments(judgments)


def _has_requested_judgments(
    row: dict[str, Any],
    backend: str,
    orders: int,
) -> bool:
    judge_result = row.get("judge", {})
    if judge_result.get("orders_completed", 0) < orders:
        return False
    successful = [
        item.get("result", {})
        for item in judge_result.get("raw", [])
        if not item.get("result", {}).get("error")
    ]
    return len(successful) >= orders and all(
        result.get("judge_backend") == backend for result in successful[:orders]
    )


def _values(rows: list[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        current: Any = row
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            values.append(float(current))
    return values


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _build_artifact(
    *,
    rows: list[dict[str, Any]],
    question_ids: list[str],
    backend: str,
    model_name: str,
    baseline_sampling: dict[str, Any],
    judge_backend: str | None,
    judge_orders: int,
    seed: int,
    started_at: str,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("status") == "complete"]
    metric_summaries = {
        metric: paired_summary(
            valid_rows,
            left_path=f"agent.rule.metrics.{metric}",
            right_path=f"baseline.rule.metrics.{metric}",
            seed=seed,
        )
        for metric in RESEARCHBENCH_METRIC_WEIGHTS
    }
    rule_summary = paired_summary(
        valid_rows,
        left_path="agent.rule.composite_score",
        right_path="baseline.rule.composite_score",
        seed=seed,
    )
    judge_summary = paired_summary(
        valid_rows,
        left_path="judge.agent_mean",
        right_path="judge.baseline_mean",
        seed=seed,
    )
    completed = len(valid_rows)
    evidence_keys = (
        "claim_support_coverage",
        "claim_citation_rate",
        "cited_claim_support_precision",
        "attribution_coverage",
        "primary_source_ratio",
        "fulltext_source_ratio",
    )
    evidence_summary = {
        key: _mean(_values(valid_rows, "agent", "evidence_metrics", key))
        for key in evidence_keys
    }
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "status": "complete" if completed == len(question_ids) else "in_progress",
        "evaluation_version": RESEARCHBENCH_EVALUATION_VERSION,
        "metric_weights": RESEARCHBENCH_METRIC_WEIGHTS,
        "protocol": {
            "pairing": "same ResearchBench questions",
            "agent": "full orchestrated retrieval and evidence pipeline",
            "baseline": "same base model, one API call, no tools or retrieval",
            "report_retention": "exact Markdown reports and SHA-256 hashes",
            "execution_failures": "excluded from quality pairs and reported separately",
            "judge": (
                f"{judge_backend}, {judge_orders} counterbalanced presentation order(s)"
                if judge_backend and judge_orders
                else "disabled"
            ),
            "bootstrap_seed": seed,
        },
        "models": {
            "agent_backend": backend,
            "agent_model": model_name,
            "baseline_backend": backend,
            "baseline_model": model_name,
            "baseline_sampling": baseline_sampling,
            "judge_backend": judge_backend,
        },
        "question_ids": question_ids,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "completed_questions": completed,
        "num_questions": len(question_ids),
        "rows": rows,
        "summary": {
            "rule_composite": rule_summary,
            "rule_metrics": metric_summaries,
            "llm_judge": judge_summary if judge_summary["n_pairs"] else None,
            "agent_evidence": evidence_summary,
            "latency_seconds": {
                "agent_mean": _mean(
                    _values(valid_rows, "agent", "runtime", "elapsed_seconds")
                ),
                "baseline_mean": _mean(
                    _values(valid_rows, "baseline", "runtime", "elapsed_seconds")
                ),
            },
            "api_total_tokens": {
                "agent_mean": _mean(_values(valid_rows, "agent", "usage", "total_tokens")),
                "baseline_mean": _mean(
                    _values(valid_rows, "baseline", "usage", "total_tokens")
                ),
                "judge_mean": _mean(_values(valid_rows, "judge_usage", "total_tokens")),
            },
            "failed_questions": [row["qid"] for row in rows if row.get("status") != "complete"],
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Agent vs one-call LLM auditable evaluation")
    parser.add_argument("--num_questions", type=int, default=15)
    parser.add_argument("--question_ids", nargs="*", default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/evaluation/headtohead_v3.json")
    parser.add_argument("--reports_dir", type=str, default=None)
    parser.add_argument("--judge_backend", type=str, default=None)
    parser.add_argument("--judge_orders", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)
    backend = config.get("model", {}).get("backend", "kimi")
    baseline_sampling = _sampling_for_baseline(config, backend)
    baseline_policy = ModelRouter.create_backend(
        backend,
        use_cache=False,
        **baseline_sampling,
    )
    judge = LLMJudge(args.judge_backend) if args.judge_backend and args.judge_orders else None

    bench = ResearchBench()
    questions = _select_questions(bench, args)
    question_ids = [question["id"] for question in questions]
    output_path = Path(args.output)
    reports_dir = Path(args.reports_dir or output_path.with_suffix(""))
    started_at = _utc_now()
    rows_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("artifact_schema") != ARTIFACT_SCHEMA:
            raise ValueError(f"Cannot resume incompatible artifact: {output_path}")
        started_at = existing.get("started_at", started_at)
        rows_by_id = {row["qid"]: row for row in existing.get("rows", [])}

    print(
        f"[H2H] {len(questions)} questions | agent vs one-call {backend} | "
        f"judge={args.judge_backend or 'off'}"
    )

    for index, question in enumerate(questions, 1):
        qid, query = question["id"], question["query"]
        existing_row = rows_by_id.get(qid, {})
        if existing_row.get("status") == "complete":
            if judge is not None and not _has_requested_judgments(
                existing_row, args.judge_backend, args.judge_orders
            ):
                agent_report = load_report_artifact(existing_row["agent"]["report"])
                baseline_report = load_report_artifact(existing_row["baseline"]["report"])
                before = VLLMPolicy.global_usage_snapshot()
                existing_row["judge"] = await asyncio.to_thread(
                    _counterbalanced_judge,
                    judge,
                    agent_report=agent_report,
                    baseline_report=baseline_report,
                    query=query,
                    ground_truth=question.get("ground_truth", {}),
                    orders=args.judge_orders,
                    reverse_first=(index + args.seed) % 2 == 0,
                )
                existing_row["judge_usage"] = usage_delta(
                    before, VLLMPolicy.global_usage_snapshot()
                )
                rows_by_id[qid] = existing_row
                ordered_rows = [rows_by_id[item] for item in question_ids if item in rows_by_id]
                atomic_write_json(
                    output_path,
                    _build_artifact(
                        rows=ordered_rows,
                        question_ids=question_ids,
                        backend=backend,
                        model_name=baseline_policy.model_name,
                        baseline_sampling=baseline_sampling,
                        judge_backend=args.judge_backend,
                        judge_orders=args.judge_orders,
                        seed=args.seed,
                        started_at=started_at,
                    ),
                )
                print(
                    f"[{index}/{len(questions)}] {qid} reports resumed; "
                    f"judge orders={existing_row['judge']['orders_completed']}"
                )
            else:
                print(f"[{index}/{len(questions)}] {qid} resumed")
            continue

        print(f"[{index}/{len(questions)}] {qid}")
        row: dict[str, Any] = {
            "qid": qid,
            "domain": question.get("domain", ""),
            "query": query,
            "generation_order": ["agent", "baseline"]
            if (index + args.seed) % 2
            else ["baseline", "agent"],
            "errors": [],
        }
        agent_report = ""
        baseline_report = ""

        for system_name in row["generation_order"]:
            before = VLLMPolicy.global_usage_snapshot()
            started = time.monotonic()
            try:
                if system_name == "agent":
                    modules = initialize_modules(
                        config,
                        session_id=f"h2h_v3_{qid}_{time.time_ns()}",
                    )
                    agent_report, metadata = await run_research_with_metadata(query, config, modules)
                    audit = metadata.pop("evidence_audit", {})
                    evidence_path = metadata.pop("evidence_artifact", "")
                    row["agent"] = {
                        "rule": bench.evaluate_report(agent_report, qid),
                        "runtime": metadata,
                        "usage": usage_delta(before, VLLMPolicy.global_usage_snapshot()),
                        "evidence_metrics": evidence_quality_metrics(audit),
                        "report": save_report_artifact(
                            agent_report,
                            reports_dir=reports_dir,
                            question_id=qid,
                            system_name="agent",
                        ),
                        "evidence_artifact": _copy_evidence_artifact(
                            evidence_path, reports_dir, qid
                        ),
                    }
                else:
                    baseline_report = await asyncio.to_thread(run_baseline, query, baseline_policy)
                    row["baseline"] = {
                        "rule": bench.evaluate_report(baseline_report, qid),
                        "runtime": {"elapsed_seconds": round(time.monotonic() - started, 4)},
                        "usage": usage_delta(before, VLLMPolicy.global_usage_snapshot()),
                        "report": save_report_artifact(
                            baseline_report,
                            reports_dir=reports_dir,
                            question_id=qid,
                            system_name="baseline",
                        ),
                    }
            except Exception as exc:
                row["errors"].append(f"{system_name}: {type(exc).__name__}: {exc}")
                row[system_name] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime": {"elapsed_seconds": round(time.monotonic() - started, 4)},
                    "usage": usage_delta(before, VLLMPolicy.global_usage_snapshot()),
                }

        if judge is not None and agent_report and baseline_report:
            before = VLLMPolicy.global_usage_snapshot()
            row["judge"] = await asyncio.to_thread(
                _counterbalanced_judge,
                judge,
                agent_report=agent_report,
                baseline_report=baseline_report,
                query=query,
                ground_truth=question.get("ground_truth", {}),
                orders=args.judge_orders,
                reverse_first=(index + args.seed) % 2 == 0,
            )
            row["judge_usage"] = usage_delta(before, VLLMPolicy.global_usage_snapshot())

        agent_run_status = row.get("agent", {}).get("runtime", {}).get("run_status")
        if not agent_report or not baseline_report:
            row["status"] = "failed"
        elif agent_run_status != "complete":
            row["status"] = agent_run_status or "partial"
        else:
            row["status"] = "complete"
        rows_by_id[qid] = row
        ordered_rows = [rows_by_id[item] for item in question_ids if item in rows_by_id]
        artifact = _build_artifact(
            rows=ordered_rows,
            question_ids=question_ids,
            backend=backend,
            model_name=baseline_policy.model_name,
            baseline_sampling=baseline_sampling,
            judge_backend=args.judge_backend,
            judge_orders=args.judge_orders if judge is not None else 0,
            seed=args.seed,
            started_at=started_at,
        )
        atomic_write_json(output_path, artifact)
        agent_score = row.get("agent", {}).get("rule", {}).get("composite_score")
        baseline_score = row.get("baseline", {}).get("rule", {}).get("composite_score")
        print(f"    rule agent={agent_score} baseline={baseline_score} status={row['status']}")

    final_rows = [rows_by_id[item] for item in question_ids if item in rows_by_id]
    artifact = _build_artifact(
        rows=final_rows,
        question_ids=question_ids,
        backend=backend,
        model_name=baseline_policy.model_name,
        baseline_sampling=baseline_sampling,
        judge_backend=args.judge_backend,
        judge_orders=args.judge_orders if judge is not None else 0,
        seed=args.seed,
        started_at=started_at,
    )
    atomic_write_json(output_path, artifact)
    summary = artifact["summary"]["rule_composite"]
    print(
        f"[H2H] complete={artifact['completed_questions']}/{artifact['num_questions']} | "
        f"delta={summary['bootstrap']['mean_diff']:+.4f} | "
        f"95% CI=[{summary['bootstrap']['ci_lower']:+.4f}, "
        f"{summary['bootstrap']['ci_upper']:+.4f}] | {output_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
