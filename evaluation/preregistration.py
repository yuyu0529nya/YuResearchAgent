"""Frozen head-to-head protocols and API-free artifact auditing."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.benchmarks.research_bench import (
    RESEARCHBENCH_EVALUATION_VERSION,
    RESEARCHBENCH_METRIC_WEIGHTS,
    ResearchBench,
)
from evaluation.evidence_metrics import evidence_quality_metrics
from evaluation.protocol import (
    normalize_counterbalanced_judgments,
    paired_summary,
    sha256_text,
)

PREREGISTRATION_SCHEMA = "headtohead-preregistration-v1"
HEADTOHEAD_ARTIFACT_SCHEMA = "headtohead-v4-preregistered"
BASELINE_SYSTEM_PROMPT = (
    "你是一位研究助手。针对用户问题，直接写一份全面、结构化的 Markdown 研究报告。"
    "回答必须独立完成，不得声称已经联网或调用外部工具；只引用你确实知道的来源。"
    "结尾给出 Overall Confidence: X.XX。"
)

# Allocate every domain at least one question, then allocate the four remaining
# slots to the three largest domain strata. IDs and final order are selected by
# SHA-256 rather than by their position in the benchmark file.
DEFAULT_DOMAIN_QUOTAS = {
    "科技": 3,
    "医疗": 2,
    "金融": 2,
    "教育": 1,
    "法律": 1,
    "能源": 1,
    "消费": 1,
    "汽车": 1,
    "游戏": 1,
    "传媒": 1,
    "交叉": 1,
}
DEFAULT_QUESTION_IDS = [
    "edu_002",
    "tech_001",
    "med_006",
    "fin_006",
    "auto_001",
    "cross_002",
    "tech_002",
    "tech_007",
    "game_002",
    "law_001",
    "fin_004",
    "retail_002",
    "media_001",
    "energy_002",
    "med_004",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def protocol_fingerprint(manifest: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in manifest.items() if key != "fingerprint_sha256"}
    return sha256_text(canonical_json(unsigned))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _benchmark_implementation_sha256() -> str:
    path = Path(__file__).resolve().parent / "benchmarks" / "research_bench.py"
    return _file_sha256(path)


def source_tree_sha256() -> str:
    """Fingerprint output-affecting Python source without including artifacts."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py"))
    paths.extend(sorted((root / "evaluation").rglob("*.py")))
    paths.append(root / "scripts" / "run_headtohead.py")
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def judge_implementation_sha256() -> str:
    """Fingerprint the exact pairwise Judge and report-sampling implementations."""
    from evaluation.report_sampling import balanced_report_excerpt
    from src.core.judge import LLMJudge

    source = inspect.getsource(LLMJudge.compare_two) + inspect.getsource(balanced_report_excerpt)
    return sha256_text(source)


def evaluation_config_snapshot(
    config: dict[str, Any],
    backend: str,
    *,
    effective_module_policies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return secret-free settings that can affect paired Agent outputs."""
    modules = [
        "solver",
        "planner",
        "summarizer",
        "judge",
        "red_agent",
        "blue_agent",
        "compressor",
    ]
    model = dict(config.get("model") or {})
    memory = dict(config.get("memory") or {})
    memory.pop("db_path", None)
    memory["evaluation_scope"] = "unique empty session per question"
    return {
        "model": {
            "backend": backend,
            "backend_mapping": {module: backend for module in modules},
            "backend_sampling": model.get("backend_sampling", {}),
            "effective_module_policies": dict(effective_module_policies or {}),
        },
        "orchestrator": config.get("orchestrator", {}),
        "planner": config.get("planner", {}),
        "compressor": config.get("compressor", {}),
        "memory": memory,
        "evidence": config.get("evidence", {}),
        "adversarial": config.get("adversarial", {}),
        "tools": config.get("tools", {}),
    }


def _question_snapshot(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": question["id"],
        "domain": question.get("domain", ""),
        "query": question.get("query", ""),
        "expected_topics": list(question.get("expected_topics") or []),
        "ground_truth": dict(question.get("ground_truth") or {}),
    }


def deterministic_stratified_question_ids(
    questions: list[dict[str, Any]],
    *,
    seed: int = 42,
    quotas: dict[str, int] | None = None,
) -> list[str]:
    """Select a reproducible, domain-stratified suite without mutable RNG state."""
    selected: list[str] = []
    requested = dict(quotas or DEFAULT_DOMAIN_QUOTAS)
    for domain, quota in requested.items():
        candidates = [str(question["id"]) for question in questions if question.get("domain") == domain]
        ordered = sorted(
            candidates,
            key=lambda question_id: sha256_text(f"{seed}:question:{question_id}"),
        )
        if quota < 0 or len(ordered) < quota:
            raise ValueError(f"Domain {domain!r} cannot satisfy quota {quota}")
        selected.extend(ordered[:quota])
    return sorted(
        selected,
        key=lambda question_id: sha256_text(f"{seed}:order:{question_id}"),
    )


def _schedule(question_ids: list[str], seed: int) -> dict[str, Any]:
    schedule: dict[str, Any] = {}
    for index, question_id in enumerate(question_ids, 1):
        agent_first = (index + seed) % 2 != 0
        schedule[question_id] = {
            "generation_order": (["agent", "baseline"] if agent_first else ["baseline", "agent"]),
            "judge_orders": (
                [
                    {"A": "agent", "B": "baseline"},
                    {"A": "baseline", "B": "agent"},
                ]
                if agent_first
                else [
                    {"A": "baseline", "B": "agent"},
                    {"A": "agent", "B": "baseline"},
                ]
            ),
        }
    return schedule


def build_preregistration(
    *,
    backend: str,
    model_name: str,
    sampling: dict[str, Any],
    judge_backend: str,
    judge_model_name: str,
    judge_sampling: dict[str, Any],
    question_ids: list[str] | None = None,
    seed: int = 42,
    agent_configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bench = ResearchBench()
    population = [_question_snapshot(question) for question in bench.questions]
    by_id = {question["id"]: question for question in population}
    selected_ids = list(question_ids or deterministic_stratified_question_ids(bench.questions, seed=seed))
    missing = [question_id for question_id in selected_ids if question_id not in by_id]
    if missing:
        raise ValueError(f"Unknown ResearchBench question IDs: {', '.join(missing)}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Preregistered question IDs must be unique")
    questions = [by_id[question_id] for question_id in selected_ids]
    agent_config = dict(agent_configuration or {})
    manifest = {
        "artifact_schema": PREREGISTRATION_SCHEMA,
        "evaluation_version": RESEARCHBENCH_EVALUATION_VERSION,
        "benchmark": {
            "population_size": len(population),
            "population_sha256": sha256_text(canonical_json(population)),
            "implementation_sha256": _benchmark_implementation_sha256(),
            "selection_method": "domain quotas plus SHA-256 ordering",
            "selection_seed": seed,
            "domain_quotas": DEFAULT_DOMAIN_QUOTAS,
        },
        "questions": questions,
        "question_ids": selected_ids,
        "domain_counts": dict(sorted(Counter(question["domain"] for question in questions).items())),
        "systems": {
            "agent": "full orchestrated retrieval and evidence pipeline",
            "baseline": "same model, one API call, no tools or retrieval",
            "backend": backend,
            "model": model_name,
            "implementation_sha256": source_tree_sha256(),
            "effective_sampling": dict(sampling),
            "baseline_prompt_sha256": sha256_text(BASELINE_SYSTEM_PROMPT),
            "agent_configuration": agent_config,
            "agent_configuration_sha256": sha256_text(canonical_json(agent_config)),
        },
        "judge": {
            "backend": judge_backend,
            "model": judge_model_name,
            "effective_sampling": dict(judge_sampling),
            "implementation_sha256": judge_implementation_sha256(),
            "orders_per_pair": 2,
            "presentation": "counterbalanced A/B order",
        },
        "schedule": _schedule(selected_ids, seed),
        "analysis": {
            "primary_endpoint": "rule_composite_agent_minus_baseline",
            "secondary_endpoint": "counterbalanced_judge_agent_minus_baseline",
            "metric_weights": RESEARCHBENCH_METRIC_WEIGHTS,
            "bootstrap_seed": seed,
            "bootstrap_confidence": 0.95,
            "effect_size": "paired Cohen's dz",
            "randomization_test": ("exact paired sign flip, one-sided agent > baseline"),
            "minimum_complete_pairs": len(selected_ids),
            "missing_pair_policy": ("report failures; do not impute or replace questions"),
            "run_level_retry_policy": ("do not regenerate a recorded pair; resume missing Judge orders only"),
            "stopping_rule": ("run every frozen question; no early stopping on significance"),
            "multiplicity": "secondary endpoints are descriptive",
        },
    }
    manifest["fingerprint_sha256"] = protocol_fingerprint(manifest)
    return manifest


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_schema") != PREREGISTRATION_SCHEMA:
        raise ValueError("Unsupported preregistration schema")
    expected = str(payload.get("fingerprint_sha256", ""))
    actual = protocol_fingerprint(payload)
    if not expected or expected != actual:
        raise ValueError("Preregistration fingerprint mismatch")
    return payload


def _verify_file(metadata: dict[str, Any], root: Path) -> Path:
    if not isinstance(metadata, dict):
        raise ValueError("Artifact metadata must be an object")
    raw = str(metadata.get("path", ""))
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe artifact path: {raw}")
    expected = str(metadata.get("sha256", ""))
    actual = _file_sha256(path)
    if not expected or actual != expected:
        raise ValueError(f"Artifact hash mismatch: {raw}")
    return path


def _same(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def audit_headtohead_artifact(
    artifact: dict[str, Any],
    preregistration: dict[str, Any],
    *,
    project_root: str | Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Independently verify protocol identity, retained reports, and primary scores."""
    errors: list[str] = []
    warnings: list[str] = []
    root = Path(project_root).resolve()
    expected_fingerprint = protocol_fingerprint(preregistration)
    if preregistration.get("fingerprint_sha256") != expected_fingerprint:
        errors.append("preregistration fingerprint is invalid")
    if artifact.get("artifact_schema") != HEADTOHEAD_ARTIFACT_SCHEMA:
        errors.append("artifact schema is not the preregistered v4 schema")
    if artifact.get("preregistration_sha256") != expected_fingerprint:
        errors.append("preregistration fingerprint does not match")
    if artifact.get("question_ids") != preregistration.get("question_ids"):
        errors.append("question order or membership differs from preregistration")
    if artifact.get("evaluation_version") != preregistration.get("evaluation_version"):
        errors.append("evaluation version differs from preregistration")
    if artifact.get("metric_weights") != preregistration.get("analysis", {}).get("metric_weights"):
        errors.append("metric weights differ from preregistration")
    if preregistration.get("benchmark", {}).get("implementation_sha256") != _benchmark_implementation_sha256():
        errors.append("ResearchBench implementation changed after preregistration")

    expected_systems = preregistration.get("systems", {})
    models = artifact.get("models", {})
    for role in ("agent", "baseline"):
        if models.get(f"{role}_backend") != expected_systems.get("backend"):
            errors.append(f"{role} backend differs from preregistration")
        if models.get(f"{role}_model") != expected_systems.get("model"):
            errors.append(f"{role} model differs from preregistration")
    if models.get("baseline_sampling") != expected_systems.get("effective_sampling"):
        errors.append("baseline effective sampling differs from preregistration")
    expected_judge = preregistration.get("judge", {})
    for key in ("backend", "model", "effective_sampling"):
        if models.get(f"judge_{key}") != expected_judge.get(key):
            errors.append(f"Judge {key.replace('_', ' ')} differs from preregistration")
    if artifact.get("agent_configuration_sha256") != expected_systems.get("agent_configuration_sha256"):
        errors.append("agent configuration fingerprint differs from preregistration")
    if expected_systems.get("implementation_sha256") != source_tree_sha256():
        errors.append("Agent source tree changed after preregistration")
    protocol = artifact.get("protocol", {})
    if protocol.get("baseline_prompt_sha256") != expected_systems.get("baseline_prompt_sha256"):
        errors.append("baseline prompt differs from preregistration")
    if protocol.get("judge_implementation_sha256") != expected_judge.get("implementation_sha256"):
        errors.append("Judge implementation differs from preregistration")

    rows = artifact.get("rows") if isinstance(artifact.get("rows"), list) else []
    expected_ids = list(preregistration.get("question_ids") or [])
    actual_ids = [row.get("qid") for row in rows if isinstance(row, dict)]
    if len(actual_ids) != len(rows):
        errors.append("artifact rows must be objects with question IDs")
    if len(actual_ids) != len(set(actual_ids)):
        errors.append("artifact contains duplicate question rows")
    if actual_ids != expected_ids[: len(actual_ids)]:
        errors.append("artifact rows are not a prefix of the frozen schedule")

    bench = ResearchBench()
    bench.questions = list(preregistration.get("questions") or [])
    frozen_by_id = {question["id"]: question for question in bench.questions}
    complete_rows = [row for row in rows if row.get("status") == "complete"]
    for row in complete_rows:
        question_id = str(row.get("qid", ""))
        frozen_question = frozen_by_id.get(question_id, {})
        if row.get("query") != frozen_question.get("query") or row.get("domain") != frozen_question.get("domain"):
            errors.append(f"{question_id or '?'}: frozen question content differs")
        schedule = preregistration.get("schedule", {}).get(question_id, {})
        if row.get("generation_order") != schedule.get("generation_order"):
            errors.append(f"{question_id or '?'}: generation order differs")
        try:
            report_texts: dict[str, str] = {}
            for role in ("agent", "baseline"):
                path = _verify_file(row[role]["report"], root)
                report_texts[role] = path.read_text(encoding="utf-8")
                if len(report_texts[role]) != row[role]["report"].get("characters"):
                    raise ValueError(f"{role} report character count differs")
                recomputed = bench.evaluate_report(report_texts[role], question_id)
                if not _same(recomputed, row[role].get("rule")):
                    raise ValueError(f"{role} rule score cannot be reproduced")

            evidence_path = _verify_file(row["agent"]["evidence_artifact"], root)
            evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence_metrics = evidence_quality_metrics(dict(evidence_payload.get("audit") or {}))
            if not _same(evidence_metrics, row["agent"].get("evidence_metrics")):
                raise ValueError("agent evidence metrics cannot be reproduced")
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{question_id or '?'}: {exc}")

        judge = row.get("judge", {})
        if judge.get("orders_completed") != expected_judge.get("orders_per_pair"):
            errors.append(f"{question_id or '?'}: two valid Judge orders are required")
        observed_orders = [item.get("order") for item in judge.get("raw", [])]
        if observed_orders != schedule.get("judge_orders"):
            errors.append(f"{question_id or '?'}: Judge order schedule differs")
        if not _same(normalize_counterbalanced_judgments(judge.get("raw", [])), judge):
            errors.append(f"{question_id or '?'}: normalized Judge scores differ")

    seed = int(preregistration.get("analysis", {}).get("bootstrap_seed", 42))
    recomputed_primary = paired_summary(
        complete_rows,
        left_path="agent.rule.composite_score",
        right_path="baseline.rule.composite_score",
        seed=seed,
    )
    if not _same(
        recomputed_primary,
        artifact.get("summary", {}).get("rule_composite"),
    ):
        errors.append("primary summary cannot be reproduced from complete rows")
    recomputed_judge = paired_summary(
        complete_rows,
        left_path="judge.agent_mean",
        right_path="judge.baseline_mean",
        seed=seed,
    )
    recorded_judge = artifact.get("summary", {}).get("llm_judge")
    expected_judge_summary = recomputed_judge if recomputed_judge["n_pairs"] else None
    if not _same(recomputed_judge if recorded_judge else expected_judge_summary, recorded_judge):
        errors.append("Judge summary cannot be reproduced from complete rows")

    minimum = int(preregistration.get("analysis", {}).get("minimum_complete_pairs", 0) or 0)
    if require_complete and len(complete_rows) < minimum:
        errors.append(f"only {len(complete_rows)}/{minimum} preregistered pairs are complete")
    elif len(complete_rows) < minimum:
        warnings.append(f"experiment is in progress: {len(complete_rows)}/{minimum} complete")
    if artifact.get("num_questions") != len(expected_ids):
        errors.append("num_questions does not match preregistration")
    if artifact.get("completed_questions") != len(complete_rows):
        errors.append("completed_questions does not match complete rows")
    expected_status = "complete" if len(complete_rows) == len(expected_ids) else "in_progress"
    if artifact.get("status") != expected_status:
        errors.append(f"status must be {expected_status!r}")

    return {
        "valid": not errors,
        "complete": len(complete_rows) == len(expected_ids) and not errors,
        "completed_pairs": len(complete_rows),
        "expected_pairs": len(expected_ids),
        "errors": errors,
        "warnings": warnings,
    }


__all__ = [
    "BASELINE_SYSTEM_PROMPT",
    "DEFAULT_DOMAIN_QUOTAS",
    "DEFAULT_QUESTION_IDS",
    "HEADTOHEAD_ARTIFACT_SCHEMA",
    "PREREGISTRATION_SCHEMA",
    "audit_headtohead_artifact",
    "build_preregistration",
    "canonical_json",
    "deterministic_stratified_question_ids",
    "evaluation_config_snapshot",
    "judge_implementation_sha256",
    "load_preregistration",
    "protocol_fingerprint",
    "source_tree_sha256",
]
