#!/usr/bin/env python3
"""Run the frozen Agent-vs-one-call protocol with resumable checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import copy
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
from evaluation.preregistration import (
    BASELINE_SYSTEM_PROMPT,
    HEADTOHEAD_ARTIFACT_SCHEMA,
    TEMPORAL_CONTEXT_TEMPLATE,
    audit_headtohead_artifact,
    canonical_json,
    evaluation_config_snapshot,
    load_preregistration,
)
from evaluation.protocol import (
    atomic_write_json,
    load_report_artifact,
    normalize_counterbalanced_judgments,
    paired_summary,
    save_report_artifact,
    sha256_text,
    usage_delta,
)
from evaluation.runtime_identity import (
    configure_single_backend,
    create_policy,
    effective_module_policies,
    policy_identity,
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_baseline(query: str, policy: Any, as_of_date: str = "") -> str:
    system_prompt = BASELINE_SYSTEM_PROMPT
    if as_of_date:
        system_prompt += "\n" + TEMPORAL_CONTEXT_TEMPLATE.format(as_of_date=as_of_date)
    response = policy(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    )
    content = response.get("content", "") if isinstance(response, dict) else ""
    if not content or content.lstrip().lower().startswith("error:"):
        raise RuntimeError(content or "baseline returned empty content")
    return content


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
        portable_path = target.resolve().relative_to(PROJECT_ROOT).as_posix()
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
    presentation_orders: list[dict[str, str]],
    as_of_date: str = "",
) -> dict[str, Any]:
    reports = {"agent": agent_report, "baseline": baseline_report}
    judgments = []
    for order in presentation_orders:
        judgments.append(
            {
                "order": dict(order),
                "result": judge.compare_two(
                    reports[order["A"]],
                    reports[order["B"]],
                    query,
                    ground_truth,
                    as_of_date,
                ),
            }
        )
    return normalize_counterbalanced_judgments(judgments)


def _has_complete_judgments(
    row: dict[str, Any],
    backend: str,
    expected_orders: list[dict[str, str]],
) -> bool:
    judge_result = row.get("judge", {})
    if [item.get("order") for item in judge_result.get("raw", [])] != expected_orders:
        return False
    successful = [
        item.get("result", {}) for item in judge_result.get("raw", []) if not item.get("result", {}).get("error")
    ]
    return len(successful) == len(expected_orders) and all(
        result.get("judge_backend") == backend for result in successful
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
    preregistration: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    question_ids = list(preregistration["question_ids"])
    systems = preregistration["systems"]
    judge_spec = preregistration["judge"]
    analysis = preregistration["analysis"]
    valid_rows = [row for row in rows if row.get("status") == "complete"]
    seed = int(analysis["bootstrap_seed"])
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
    return {
        "artifact_schema": HEADTOHEAD_ARTIFACT_SCHEMA,
        "status": "complete" if completed == len(question_ids) else "in_progress",
        "preregistration_sha256": preregistration["fingerprint_sha256"],
        "agent_configuration_sha256": systems["agent_configuration_sha256"],
        "evaluation_version": RESEARCHBENCH_EVALUATION_VERSION,
        "metric_weights": RESEARCHBENCH_METRIC_WEIGHTS,
        "protocol": {
            "pairing": "same frozen ResearchBench questions",
            "agent": systems["agent"],
            "baseline": systems["baseline"],
            "baseline_prompt_sha256": systems["baseline_prompt_sha256"],
            "judge_implementation_sha256": judge_spec["implementation_sha256"],
            "report_retention": "exact Markdown reports and SHA-256 hashes",
            "execution_failures": analysis["missing_pair_policy"],
            "run_level_retry_policy": analysis["run_level_retry_policy"],
            "stopping_rule": analysis["stopping_rule"],
            "primary_endpoint": analysis["primary_endpoint"],
            "secondary_endpoint": analysis["secondary_endpoint"],
            "temporal_context": preregistration["temporal_context"],
        },
        "models": {
            "agent_backend": systems["backend"],
            "agent_model": systems["model"],
            "baseline_backend": systems["backend"],
            "baseline_model": systems["model"],
            "baseline_sampling": systems["effective_sampling"],
            "judge_backend": judge_spec["backend"],
            "judge_model": judge_spec["model"],
            "judge_effective_sampling": judge_spec["effective_sampling"],
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
            "agent_evidence": {
                key: _mean(_values(valid_rows, "agent", "evidence_metrics", key)) for key in evidence_keys
            },
            "latency_seconds": {
                "agent_mean": _mean(_values(valid_rows, "agent", "runtime", "elapsed_seconds")),
                "baseline_mean": _mean(_values(valid_rows, "baseline", "runtime", "elapsed_seconds")),
            },
            "api_total_tokens": {
                "agent_mean": _mean(_values(valid_rows, "agent", "usage", "total_tokens")),
                "baseline_mean": _mean(_values(valid_rows, "baseline", "usage", "total_tokens")),
                "judge_mean": _mean(_values(valid_rows, "judge_usage", "total_tokens")),
            },
            "failed_questions": [row["qid"] for row in rows if row.get("status") != "complete"],
        },
    }


def _write_checkpoint(
    output_path: Path,
    rows: list[dict[str, Any]],
    preregistration: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    artifact = _build_artifact(
        rows=rows,
        preregistration=preregistration,
        started_at=started_at,
    )
    audit = audit_headtohead_artifact(
        artifact,
        preregistration,
        project_root=PROJECT_ROOT,
    )
    if not audit["valid"]:
        raise ValueError("Checkpoint audit failed: " + "; ".join(audit["errors"]))
    atomic_write_json(output_path, artifact)
    return artifact


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Agent-vs-one-call head-to-head protocol")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--preregistration",
        type=str,
        default="docs/evaluation/headtohead_v4_preregistration.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation/headtohead_v4.json",
    )
    parser.add_argument("--reports_dir", type=str, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    preregistration = load_preregistration(args.preregistration)
    systems = preregistration["systems"]
    judge_spec = preregistration["judge"]
    backend = str(systems["backend"])

    config = load_config(args.config)
    configure_single_backend(config, backend)
    effective_modules = effective_module_policies(config, backend)
    agent_config = evaluation_config_snapshot(
        config,
        backend,
        effective_module_policies=effective_modules,
    )
    if sha256_text(canonical_json(agent_config)) != systems["agent_configuration_sha256"]:
        raise ValueError("Current Agent configuration differs from preregistration")

    baseline_policy = create_policy(config, backend, "summarizer")
    if baseline_policy.model_name != systems["model"]:
        raise ValueError("Current baseline model differs from preregistration")
    if policy_identity(baseline_policy) != systems["effective_sampling"]:
        raise ValueError("Current baseline sampling differs from preregistration")

    judge_policy = ModelRouter.create_backend(
        str(judge_spec["backend"]),
        use_cache=False,
    )
    if judge_policy.model_name != judge_spec["model"]:
        raise ValueError("Current Judge model differs from preregistration")
    if policy_identity(judge_policy) != judge_spec["effective_sampling"]:
        raise ValueError("Current Judge sampling differs from preregistration")
    judge = LLMJudge(str(judge_spec["backend"]), policy=judge_policy)
    as_of_date = str(preregistration.get("temporal_context", {}).get("as_of_date", ""))

    bench = ResearchBench()
    bench.questions = copy.deepcopy(preregistration["questions"])
    questions = bench.questions
    question_ids = list(preregistration["question_ids"])
    output_path = Path(args.output)
    reports_dir = Path(args.reports_dir or output_path.with_suffix(""))
    started_at = _utc_now()
    rows_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        audit = audit_headtohead_artifact(
            existing,
            preregistration,
            project_root=PROJECT_ROOT,
        )
        if not audit["valid"]:
            raise ValueError("Cannot resume invalid artifact: " + "; ".join(audit["errors"]))
        started_at = existing.get("started_at", started_at)
        rows_by_id = {row["qid"]: row for row in existing.get("rows", [])}

    print(
        f"[H2H v5] {len(questions)} frozen questions | {backend}/{systems['model']} | "
        f"judge={judge_spec['backend']}/{judge_spec['model']}"
    )

    for index, question in enumerate(questions, 1):
        qid, query = question["id"], question["query"]
        schedule = preregistration["schedule"][qid]
        existing_row = rows_by_id.get(qid, {})
        has_retained_pair = all(existing_row.get(role, {}).get("report") for role in ("agent", "baseline"))
        if existing_row and has_retained_pair:
            if _has_complete_judgments(
                existing_row,
                str(judge_spec["backend"]),
                schedule["judge_orders"],
            ):
                print(f"[{index}/{len(questions)}] {qid} resumed")
                continue
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
                presentation_orders=schedule["judge_orders"],
                as_of_date=as_of_date,
            )
            existing_row["judge_usage"] = usage_delta(before, VLLMPolicy.global_usage_snapshot())
            agent_status = existing_row.get("agent", {}).get("runtime", {}).get("run_status")
            existing_row["status"] = (
                "complete"
                if agent_status == "complete"
                and _has_complete_judgments(
                    existing_row,
                    str(judge_spec["backend"]),
                    schedule["judge_orders"],
                )
                else agent_status or "partial"
            )
            rows_by_id[qid] = existing_row
            ordered = [rows_by_id[item] for item in question_ids if item in rows_by_id]
            _write_checkpoint(output_path, ordered, preregistration, started_at)
            print(
                f"[{index}/{len(questions)}] {qid} reports resumed; "
                f"judge orders={existing_row['judge']['orders_completed']}"
            )
            continue
        if existing_row:
            print(
                f"[{index}/{len(questions)}] {qid} retained as "
                f"{existing_row.get('status', 'failed')}; frozen pair is not replaced"
            )
            continue

        print(f"[{index}/{len(questions)}] {qid}")
        row: dict[str, Any] = {
            "qid": qid,
            "domain": question.get("domain", ""),
            "query": query,
            "generation_order": list(schedule["generation_order"]),
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
                        session_id=f"h2h_v4_{qid}_{time.time_ns()}",
                    )
                    agent_report, metadata = await run_research_with_metadata(
                        query,
                        config,
                        modules,
                        as_of_date=as_of_date,
                    )
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
                        "evidence_artifact": _copy_evidence_artifact(evidence_path, reports_dir, qid),
                    }
                else:
                    baseline_report = await asyncio.to_thread(
                        run_baseline,
                        query,
                        baseline_policy,
                        as_of_date,
                    )
                    row["baseline"] = {
                        "rule": bench.evaluate_report(baseline_report, qid),
                        "runtime": {
                            "elapsed_seconds": round(time.monotonic() - started, 4),
                            "as_of_date": as_of_date,
                        },
                        "usage": usage_delta(before, VLLMPolicy.global_usage_snapshot()),
                        "report": save_report_artifact(
                            baseline_report,
                            reports_dir=reports_dir,
                            question_id=qid,
                            system_name="baseline",
                        ),
                    }
            except Exception as exc:  # noqa: BLE001 - retain failed frozen pairs
                message = f"{system_name}: {type(exc).__name__}: {exc}"
                row["errors"].append(message)
                row[system_name] = {
                    "error": message,
                    "runtime": {"elapsed_seconds": round(time.monotonic() - started, 4)},
                    "usage": usage_delta(before, VLLMPolicy.global_usage_snapshot()),
                }

        if agent_report and baseline_report:
            before = VLLMPolicy.global_usage_snapshot()
            row["judge"] = await asyncio.to_thread(
                _counterbalanced_judge,
                judge,
                agent_report=agent_report,
                baseline_report=baseline_report,
                query=query,
                ground_truth=question.get("ground_truth", {}),
                presentation_orders=schedule["judge_orders"],
                as_of_date=as_of_date,
            )
            row["judge_usage"] = usage_delta(before, VLLMPolicy.global_usage_snapshot())

        agent_status = row.get("agent", {}).get("runtime", {}).get("run_status")
        if not agent_report or not baseline_report:
            row["status"] = "failed"
        elif agent_status != "complete":
            row["status"] = agent_status or "partial"
        elif not _has_complete_judgments(
            row,
            str(judge_spec["backend"]),
            schedule["judge_orders"],
        ):
            row["status"] = "judge_incomplete"
        else:
            row["status"] = "complete"

        rows_by_id[qid] = row
        ordered = [rows_by_id[item] for item in question_ids if item in rows_by_id]
        _write_checkpoint(output_path, ordered, preregistration, started_at)
        print(
            f"    rule agent={row.get('agent', {}).get('rule', {}).get('composite_score')} "
            f"baseline={row.get('baseline', {}).get('rule', {}).get('composite_score')} "
            f"status={row['status']}"
        )

    final_rows = [rows_by_id[item] for item in question_ids if item in rows_by_id]
    artifact = _write_checkpoint(output_path, final_rows, preregistration, started_at)
    audit = audit_headtohead_artifact(
        artifact,
        preregistration,
        project_root=PROJECT_ROOT,
        require_complete=True,
    )
    summary = artifact["summary"]["rule_composite"]
    randomization = summary["paired_randomization_test"]
    print(
        f"[H2H v5] complete={artifact['completed_questions']}/{artifact['num_questions']} | "
        f"delta={summary['bootstrap']['mean_diff']:+.4f} | "
        f"95% CI=[{summary['bootstrap']['ci_lower']:+.4f}, "
        f"{summary['bootstrap']['ci_upper']:+.4f}] | "
        f"exact p={randomization['p_value']:.6f} | audit={audit['valid']} | "
        f"{output_path}"
    )
    if not audit["valid"]:
        raise SystemExit("Final preregistered experiment is incomplete or invalid")


if __name__ == "__main__":
    asyncio.run(main())
