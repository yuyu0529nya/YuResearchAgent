from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.evidence import EvidenceStore


ROOT = Path(__file__).resolve().parents[2]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _assert_hash(metadata: dict) -> None:
    path = ROOT / metadata["path"]
    assert path.is_file(), metadata["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]


def test_committed_headtohead_reports_and_evidence_match_hashes() -> None:
    artifact = _load("docs/evaluation/artifacts/headtohead_v3/headtohead_v3_pilot.json")

    for row in artifact["rows"]:
        _assert_hash(row["agent"]["report"])
        _assert_hash(row["agent"]["evidence_artifact"])
        _assert_hash(row["baseline"]["report"])


def test_committed_ablation_reports_and_evidence_match_hashes() -> None:
    artifact = _load("docs/evaluation/artifacts/ablation_v3/ablation_v3_smoke.json")

    for system in artifact["systems"]:
        for row in system["details"]:
            _assert_hash(row["report"])
            if row["evidence_artifact"]:
                _assert_hash(row["evidence_artifact"])


def test_committed_evidence_runtime_manifest_is_self_consistent() -> None:
    manifest = _load("docs/evaluation/artifacts/evidence_v2/runtime_manifest.json")

    _assert_hash(manifest["immutable_inputs"]["report"])
    _assert_hash(manifest["immutable_inputs"]["evidence"])
    _assert_hash(manifest["current_replays"]["heuristic"])
    _assert_hash(manifest["current_replays"]["hybrid"])
    store = EvidenceStore.load_artifact(ROOT / manifest["immutable_inputs"]["evidence"]["path"])
    assert len(store.sources) == manifest["runtime"]["source_count"]
    assert len(store.evidence) == manifest["runtime"]["evidence_count"]
