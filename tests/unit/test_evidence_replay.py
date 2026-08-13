from __future__ import annotations

import json

from scripts.replay_evidence import replay_evidence
from src.evidence import EvidenceKind, EvidenceStore


def test_replay_evidence_records_verified_inputs_and_audit(tmp_path) -> None:
    report_path = tmp_path / "report.md"
    report_path.write_text(
        "The Transformer architecture relies entirely on attention mechanisms [1].\n\n"
        "## References\n[1] Attention Is All You Need.",
        encoding="utf-8",
    )
    store = EvidenceStore(artifact_dir=str(tmp_path), session_id="replay", persist_enabled=True)
    source = store.upsert_source(
        url="https://arxiv.org/abs/1706.03762",
        title="Attention Is All You Need",
        source_type="paper",
    )
    store.add_evidence(
        source.source_id,
        "The Transformer architecture relies entirely on attention mechanisms.",
        EvidenceKind.ABSTRACT,
    )
    evidence_path = store.persist(query="How does Transformer work?")

    result = replay_evidence(report_path, evidence_path)

    assert result["artifact_schema"] == "evidence-replay-v1"
    assert result["verification"]["mode"] == "heuristic"
    assert result["audit"]["supported_count"] == 1
    assert result["inputs"]["report"]["sha256"]
    assert result["inputs"]["evidence"]["source_count"] == 1


def test_replay_evidence_rejects_wrong_expected_hash(tmp_path) -> None:
    report_path = tmp_path / "report.md"
    report_path.write_text("A sufficiently long factual report statement.", encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps({"schema_version": "1.0", "sources": [], "evidence": []}),
        encoding="utf-8",
    )

    try:
        replay_evidence(report_path, evidence_path, expected_report_sha256="0" * 64)
    except ValueError as exc:
        assert "Report SHA-256" in str(exc)
    else:
        raise AssertionError("a mismatched expected hash should fail")
