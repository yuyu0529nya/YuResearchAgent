from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.benchmarks.research_bench import ResearchBench
from evaluation.evidence_metrics import evidence_quality_metrics
from evaluation.preregistration import (
    DEFAULT_QUESTION_IDS,
    audit_headtohead_artifact,
    build_preregistration,
    deterministic_stratified_question_ids,
    evaluation_config_snapshot,
    load_preregistration,
)
from evaluation.protocol import (
    normalize_counterbalanced_judgments,
    save_report_artifact,
)
from scripts.run_headtohead import _build_artifact


def _manifest(question_id: str) -> dict:
    return build_preregistration(
        backend="test-backend",
        model_name="test-model",
        sampling={
            "model": "test-model",
            "endpoint": "https://example.invalid/v1",
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 100,
            "max_input_chars": 35000,
            "extra_body": {},
        },
        judge_backend="test-judge",
        judge_model_name="judge-model",
        judge_sampling={
            "model": "judge-model",
            "endpoint": "https://judge.invalid/v1",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 100,
            "max_input_chars": 35000,
            "extra_body": {},
        },
        question_ids=[question_id],
        agent_configuration={"frozen": True},
        as_of_date="2026-08-14",
    )


def _valid_artifact(tmp_path: Path) -> tuple[dict, dict, Path]:
    manifest = _manifest("tech_001")
    question = manifest["questions"][0]
    bench = ResearchBench()
    bench.questions = [question]
    agent_text = "GPT-4o Claude 3.5 Gemini 1.5 Qwen2.5 中文推理 代码生成 长上下文 技术路线 [1]"
    baseline_text = "GPT-4o 与 Qwen2.5 的中文推理。"
    agent_report = save_report_artifact(
        agent_text,
        reports_dir=tmp_path / "reports",
        question_id="tech_001",
        system_name="agent",
    )
    baseline_report = save_report_artifact(
        baseline_text,
        reports_dir=tmp_path / "reports",
        question_id="tech_001",
        system_name="baseline",
    )
    audit_payload = {
        "claims": [],
        "source_count": 0,
        "evidence_count": 0,
        "primary_source_ratio": 0.0,
        "fulltext_source_ratio": 0.0,
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps({"audit": audit_payload}, ensure_ascii=False), encoding="utf-8")
    evidence = {
        "path": str(evidence_path),
        "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    }
    raw = []
    for order in manifest["schedule"]["tech_001"]["judge_orders"]:
        result = {
            dimension: {"A": 4.0, "B": 3.0, "reason": "fixed"}
            for dimension in (
                "comprehensiveness",
                "accuracy",
                "structure",
                "sources",
            )
        }
        result["judge_backend"] = "test-judge"
        raw.append({"order": order, "result": result})
    row = {
        "qid": "tech_001",
        "domain": question["domain"],
        "query": question["query"],
        "generation_order": manifest["schedule"]["tech_001"]["generation_order"],
        "errors": [],
        "agent": {
            "rule": bench.evaluate_report(agent_text, "tech_001"),
            "runtime": {
                "run_status": "complete",
                "elapsed_seconds": 1.0,
                "as_of_date": manifest["temporal_context"]["as_of_date"],
            },
            "usage": {"total_tokens": 10},
            "evidence_metrics": evidence_quality_metrics(audit_payload),
            "report": agent_report,
            "evidence_artifact": evidence,
        },
        "baseline": {
            "rule": bench.evaluate_report(baseline_text, "tech_001"),
            "runtime": {
                "elapsed_seconds": 0.5,
                "as_of_date": manifest["temporal_context"]["as_of_date"],
            },
            "usage": {"total_tokens": 5},
            "report": baseline_report,
        },
        "judge": normalize_counterbalanced_judgments(raw),
        "judge_usage": {"total_tokens": 3},
        "status": "complete",
    }
    return (
        _build_artifact(rows=[row], preregistration=manifest, started_at="fixed"),
        manifest,
        Path(agent_report["path"]),
    )


def test_default_suite_is_deterministic_and_domain_stratified() -> None:
    bench = ResearchBench()
    selected = deterministic_stratified_question_ids(bench.questions, seed=42)

    assert selected == DEFAULT_QUESTION_IDS
    assert len(selected) == 15
    domains = {question["domain"] for question in bench.questions if question["id"] in selected}
    assert len(domains) == 11


def test_preregistration_fingerprint_rejects_tampering(tmp_path: Path) -> None:
    manifest = _manifest("tech_001")
    manifest["questions"][0]["query"] = "changed after registration"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    try:
        load_preregistration(path)
    except ValueError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("tampered preregistration should fail")


def test_preregistration_freezes_shared_temporal_context(tmp_path: Path) -> None:
    manifest = _manifest("tech_001")
    assert manifest["temporal_context"]["as_of_date"]

    manifest["temporal_context"]["as_of_date"] = "2099-01-01"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    try:
        load_preregistration(path)
    except ValueError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("tampered temporal context should fail")


def test_evaluation_snapshot_records_effective_search_backend(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_BACKEND", "openrouter")

    snapshot = evaluation_config_snapshot(
        {"tools": {"web_search": {"backend": "auto"}}},
        "kimi",
    )

    assert snapshot["tools"]["web_search"]["effective_backend"] == "openrouter"


def test_offline_audit_recomputes_scores_and_detects_report_tampering(
    tmp_path: Path,
) -> None:
    artifact, manifest, report_path = _valid_artifact(tmp_path)

    valid = audit_headtohead_artifact(artifact, manifest, project_root=tmp_path, require_complete=True)
    assert valid["valid"] is True
    assert valid["complete"] is True

    report_path.write_text("tampered", encoding="utf-8")
    invalid = audit_headtohead_artifact(artifact, manifest, project_root=tmp_path, require_complete=True)
    assert invalid["valid"] is False
    assert any("hash mismatch" in error for error in invalid["errors"])


def test_offline_audit_detects_judge_order_tampering(tmp_path: Path) -> None:
    artifact, manifest, _ = _valid_artifact(tmp_path)
    artifact["rows"][0]["judge"]["raw"].reverse()

    result = audit_headtohead_artifact(artifact, manifest, project_root=tmp_path, require_complete=True)

    assert result["valid"] is False
    assert any("Judge order" in error for error in result["errors"])


def test_offline_audit_detects_runtime_temporal_context_tampering(tmp_path: Path) -> None:
    artifact, manifest, _ = _valid_artifact(tmp_path)
    artifact["rows"][0]["baseline"]["runtime"]["as_of_date"] = "2099-01-01"

    result = audit_headtohead_artifact(
        artifact,
        manifest,
        project_root=tmp_path,
        require_complete=True,
    )

    assert result["valid"] is False
    assert any("baseline temporal context differs" in error for error in result["errors"])
