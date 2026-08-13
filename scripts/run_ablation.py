#!/usr/bin/env python3
"""Paired, resumable module ablations with retained reports and evidence metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
    paired_summary,
    save_report_artifact,
    sha256_text,
    usage_delta,
)
from src.core.ablation import AblationStudy
from src.core.runner import (
    initialize_modules,
    load_config,
    run_research_with_metadata,
    setup_logging,
)
from src.models.vllm_policy import VLLMPolicy


ARTIFACT_SCHEMA = "ablation-v3-auditable"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_evidence_artifact(
    source_path: str,
    reports_dir: Path,
    system_name: str,
    question_id: str,
) -> dict[str, Any] | None:
    source = Path(source_path)
    if not source.is_file():
        return None
    target = reports_dir / system_name / "evidence" / f"{question_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    payload = target.read_text(encoding="utf-8")
    try:
        portable_path = target.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        portable_path = target.resolve().as_posix()
    return {"path": portable_path, "sha256": sha256_text(payload)}


def _system_map(args: argparse.Namespace) -> dict[str, tuple[str, dict[str, Any]]]:
    if args.mode == "rounds":
        return {
            f"adv_{rounds}": (
                f"对抗轮数={rounds}",
                {"adversarial": {"max_rounds": rounds, "enabled": rounds > 0}},
            )
            for rounds in range(args.max_rounds + 1)
        }

    requested = [name.strip() for name in args.systems.split(",") if name.strip()]
    if "full" not in requested:
        requested.insert(0, "full")
    unknown = [name for name in requested if name not in AblationStudy.DEFAULT_MODULE_ABLATIONS]
    if unknown:
        available = ", ".join(AblationStudy.DEFAULT_MODULE_ABLATIONS)
        raise ValueError(f"Unknown systems: {', '.join(unknown)}. Available: {available}")
    return {name: AblationStudy.DEFAULT_MODULE_ABLATIONS[name] for name in requested}


def _run_system(
    *,
    name: str,
    description: str,
    overrides: dict[str, Any],
    config: dict[str, Any],
    questions: list[dict[str, Any]],
    bench: ResearchBench,
    reports_dir: Path,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = AblationStudy.override_config(config, overrides)
    details_by_id = {
        detail["question_id"]: detail for detail in (existing or {}).get("details", [])
    }
    print(f"\n[Ablation] {name}: {description}")

    for question in questions:
        qid, query = question["id"], question["query"]
        if details_by_id.get(qid, {}).get("status") == "complete":
            print(f"  [{qid}] resumed")
            continue
        before = VLLMPolicy.global_usage_snapshot()
        try:
            # One module graph and one memory namespace per question prevents
            # cross-question memory leakage from contaminating paired scores.
            modules = initialize_modules(
                cfg,
                session_id=f"ablation_{name}_{qid}_{time.time_ns()}",
            )
            report, metadata = asyncio.run(run_research_with_metadata(query, cfg, modules))
            audit = metadata.pop("evidence_audit", {})
            evidence_path = metadata.pop("evidence_artifact", "")
            rule = bench.evaluate_report(report, qid)
            run_status = metadata.get("run_status", "complete")
            details_by_id[qid] = {
                "question_id": qid,
                "query": query,
                "status": run_status,
                "rule": rule,
                "runtime": metadata,
                "usage": usage_delta(before, VLLMPolicy.global_usage_snapshot()),
                "evidence_metrics": evidence_quality_metrics(audit),
                "report": save_report_artifact(
                    report,
                    reports_dir=reports_dir / name,
                    question_id=qid,
                    system_name=name,
                ),
                "evidence_artifact": _copy_evidence_artifact(
                    evidence_path,
                    reports_dir,
                    name,
                    qid,
                ),
            }
            print(
                f"  [{qid}] score={rule['composite_score']:.4f} "
                f"confidence={metadata.get('confidence', 0):.2f} "
                f"status={run_status} "
                f"time={metadata['elapsed_seconds']:.1f}s"
            )
        except Exception as exc:
            details_by_id[qid] = {
                "question_id": qid,
                "query": query,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "usage": usage_delta(before, VLLMPolicy.global_usage_snapshot()),
            }
            print(f"  [{qid}] FAILED: {exc}")

    details = [details_by_id[q["id"]] for q in questions if q["id"] in details_by_id]
    scores = [
        detail["rule"]["composite_score"]
        for detail in details
        if detail.get("status") == "complete"
    ]
    return {
        "system_name": name,
        "description": description,
        "overrides": overrides,
        "execution_success_rate": round(
            sum(detail.get("status") == "complete" for detail in details) / len(questions), 4
        )
        if questions
        else 0.0,
        "average_composite_score": round(sum(scores) / len(scores), 4) if scores else None,
        "details": details,
    }


def _paired_rows(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    left = {detail["question_id"]: detail for detail in reference["details"]}
    right = {detail["question_id"]: detail for detail in candidate["details"]}
    return [
        {"reference": left[qid], "candidate": right[qid]}
        for qid in left
        if qid in right
        and left[qid].get("status") == "complete"
        and right[qid].get("status") == "complete"
    ]


def _statistics(
    systems: dict[str, dict[str, Any]],
    reference_name: str,
    seed: int,
) -> dict[str, Any]:
    reference = systems[reference_name]
    output: dict[str, Any] = {}
    evidence_metrics = (
        "claim_support_coverage",
        "claim_citation_rate",
        "cited_claim_support_precision",
        "attribution_coverage",
        "primary_source_ratio",
        "fulltext_source_ratio",
    )
    for name, candidate in systems.items():
        if name == reference_name:
            continue
        pairs = _paired_rows(reference, candidate)
        output[name] = {
            "rule_composite": paired_summary(
                pairs,
                left_path="reference.rule.composite_score",
                right_path="candidate.rule.composite_score",
                seed=seed,
            ),
            "rule_metrics": {
                metric: paired_summary(
                    pairs,
                    left_path=f"reference.rule.metrics.{metric}",
                    right_path=f"candidate.rule.metrics.{metric}",
                    seed=seed,
                )
                for metric in RESEARCHBENCH_METRIC_WEIGHTS
            },
            "evidence_metrics": {
                metric: paired_summary(
                    pairs,
                    left_path=f"reference.evidence_metrics.{metric}",
                    right_path=f"candidate.evidence_metrics.{metric}",
                    seed=seed,
                )
                for metric in evidence_metrics
            },
            "latency": paired_summary(
                pairs,
                left_path="reference.runtime.elapsed_seconds",
                right_path="candidate.runtime.elapsed_seconds",
                seed=seed,
            ),
            "api_tokens": paired_summary(
                pairs,
                left_path="reference.usage.total_tokens",
                right_path="candidate.usage.total_tokens",
                seed=seed,
            ),
        }
    return output


def _build_artifact(
    *,
    mode: str,
    systems: dict[str, dict[str, Any]],
    questions: list[dict[str, Any]],
    reference_name: str,
    seed: int,
    started_at: str,
) -> dict[str, Any]:
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "status": "complete"
        if all(system["execution_success_rate"] == 1.0 for system in systems.values())
        else "in_progress",
        "evaluation_name": "YuResearchAgent paired module ablation",
        "evaluation_version": RESEARCHBENCH_EVALUATION_VERSION,
        "metric_weights": RESEARCHBENCH_METRIC_WEIGHTS,
        "mode": mode,
        "reference_system": reference_name,
        "question_ids": [question["id"] for question in questions],
        "num_questions": len(questions),
        "started_at": started_at,
        "updated_at": _utc_now(),
        "protocol": {
            "pairing": "same question set for every system",
            "memory_isolation": "fresh module graph and session per question",
            "report_retention": "exact Markdown and SHA-256",
            "execution_failures": "excluded from quality statistics",
            "bootstrap_seed": seed,
        },
        "systems": list(systems.values()),
        "summary": {
            name: {
                "average_composite_score": system["average_composite_score"],
                "execution_success_rate": system["execution_success_rate"],
            }
            for name, system in systems.items()
        },
        "paired_statistics": _statistics(systems, reference_name, seed)
        if reference_name in systems
        else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditable YuResearchAgent ablation study")
    parser.add_argument("--mode", choices=("module", "rounds"), default="module")
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--systems", default="full,heuristic_verifier,no_gap_research,no_evidence")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", default="outputs/evaluation/ablation_v3.json")
    parser.add_argument("--reports_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("ablation")
    config = load_config(args.config)
    bench = ResearchBench()
    questions = bench.get_questions(domain=args.domain, n=args.questions)
    definitions = _system_map(args)
    reference_name = "full" if args.mode == "module" else "adv_0"
    output_path = Path(args.output)
    reports_dir = Path(args.reports_dir or output_path.with_suffix(""))
    started_at = _utc_now()
    systems: dict[str, dict[str, Any]] = {}

    if args.resume and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("artifact_schema") != ARTIFACT_SCHEMA:
            raise ValueError(f"Cannot resume incompatible artifact: {output_path}")
        started_at = previous.get("started_at", started_at)
        systems = {system["system_name"]: system for system in previous.get("systems", [])}

    logger.info("Running %d systems on %d paired questions", len(definitions), len(questions))
    for name, (description, overrides) in definitions.items():
        systems[name] = _run_system(
            name=name,
            description=description,
            overrides=overrides,
            config=config,
            questions=questions,
            bench=bench,
            reports_dir=reports_dir,
            existing=systems.get(name),
        )
        artifact = _build_artifact(
            mode=args.mode,
            systems=systems,
            questions=questions,
            reference_name=reference_name,
            seed=args.seed,
            started_at=started_at,
        )
        atomic_write_json(output_path, artifact)

    artifact = _build_artifact(
        mode=args.mode,
        systems=systems,
        questions=questions,
        reference_name=reference_name,
        seed=args.seed,
        started_at=started_at,
    )
    atomic_write_json(output_path, artifact)
    print(f"\n[Ablation] saved: {output_path}")
    for name, summary in artifact["summary"].items():
        print(
            f"  {name}: score={summary['average_composite_score']} "
            f"success={summary['execution_success_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
