"""Shared, versioned helpers for auditable paired evaluations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from evaluation.metrics.stats import (
    bootstrap_ci_paired,
    cohens_d,
    cohens_dz,
    paired_randomization_test,
    paired_t_test,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a checkpoint without leaving a partially-written JSON artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def save_report_artifact(
    report: str,
    *,
    reports_dir: str | Path,
    question_id: str,
    system_name: str,
) -> dict[str, Any]:
    """Persist an exact evaluated report and return portable integrity metadata."""
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{question_id}_{system_name}.md"
    path.write_text(report, encoding="utf-8")
    try:
        portable_path = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        portable_path = path.resolve().as_posix()
    return {
        "path": portable_path,
        "sha256": sha256_text(report),
        "characters": len(report),
    }


def load_report_artifact(metadata: dict[str, Any]) -> str:
    """Load a retained report and fail if its integrity hash no longer matches."""
    path_value = metadata.get("path") if isinstance(metadata, dict) else None
    if not path_value:
        raise ValueError("Report artifact metadata does not contain a path")
    path = Path(str(path_value))
    if not path.is_file():
        raise FileNotFoundError(f"Retained report is missing: {path}")
    report = path.read_text(encoding="utf-8")
    expected_hash = str(metadata.get("sha256", ""))
    actual_hash = sha256_text(report)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(f"Retained report hash mismatch: {path}")
    return report


def usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in sorted(keys)}


def _nested_number(row: dict[str, Any], path: str) -> float | None:
    value: Any = row
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def paired_summary(
    rows: Iterable[dict[str, Any]],
    *,
    left_path: str,
    right_path: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize only valid pairs; execution failures are reported separately."""
    left: list[float] = []
    right: list[float] = []
    for row in rows:
        left_value = _nested_number(row, left_path)
        right_value = _nested_number(row, right_path)
        if left_value is None or right_value is None:
            continue
        left.append(left_value)
        right.append(right_value)

    differences = [a - b for a, b in zip(left, right)]
    bootstrap = bootstrap_ci_paired(differences, seed=seed)
    return {
        "left_mean": round(sum(left) / len(left), 4) if left else None,
        "right_mean": round(sum(right) / len(right), 4) if right else None,
        "cohens_d": round(cohens_d(left, right), 4),
        "cohens_dz": round(cohens_dz(left, right), 4),
        "bootstrap": bootstrap,
        "paired_t_test": paired_t_test(left, right),
        "paired_randomization_test": paired_randomization_test(differences, seed=seed),
        "n_pairs": len(left),
    }


def normalize_counterbalanced_judgments(
    judgments: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Map A/B scores back to system names and average both presentation orders."""
    dimensions: dict[str, dict[str, list[float]]] = {}
    raw: list[dict[str, Any]] = []
    for item in judgments:
        order = item.get("order", {})
        result = item.get("result", {})
        raw.append(item)
        if not isinstance(order, dict) or not isinstance(result, dict) or result.get("error"):
            continue
        for dimension, scores in result.items():
            if dimension == "judge_backend" or not isinstance(scores, dict):
                continue
            bucket = dimensions.setdefault(dimension, {"agent": [], "baseline": []})
            for label in ("A", "B"):
                system = order.get(label)
                score = scores.get(label)
                if system in bucket and isinstance(score, (int, float)) and not isinstance(score, bool):
                    bucket[system].append(float(score))

    normalized: dict[str, dict[str, float | None]] = {}
    all_agent: list[float] = []
    all_baseline: list[float] = []
    for dimension, systems in dimensions.items():
        agent_values = systems["agent"]
        baseline_values = systems["baseline"]
        agent_mean = sum(agent_values) / len(agent_values) if agent_values else None
        baseline_mean = sum(baseline_values) / len(baseline_values) if baseline_values else None
        normalized[dimension] = {
            "agent": round(agent_mean, 4) if agent_mean is not None else None,
            "baseline": round(baseline_mean, 4) if baseline_mean is not None else None,
            "delta": (
                round(agent_mean - baseline_mean, 4) if agent_mean is not None and baseline_mean is not None else None
            ),
        }
        all_agent.extend(agent_values)
        all_baseline.extend(baseline_values)

    agent_overall = sum(all_agent) / len(all_agent) if all_agent else None
    baseline_overall = sum(all_baseline) / len(all_baseline) if all_baseline else None
    return {
        "dimensions": normalized,
        "agent_mean": round(agent_overall, 4) if agent_overall is not None else None,
        "baseline_mean": round(baseline_overall, 4) if baseline_overall is not None else None,
        "delta": (
            round(agent_overall - baseline_overall, 4)
            if agent_overall is not None and baseline_overall is not None
            else None
        ),
        "orders_completed": sum(1 for item in raw if not item.get("result", {}).get("error")),
        "raw": raw,
    }
