#!/usr/bin/env python3
"""Replay claim verification over an immutable report and evidence artifact."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.protocol import atomic_write_json, sha256_text, usage_delta
from src.evidence import ClaimVerifier, EvidenceStore
from src.models.model_router import ModelRouter
from src.models.vllm_policy import VLLMPolicy


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def replay_evidence(
    report_path: str | Path,
    evidence_path: str | Path,
    *,
    verifier_backend: str | None = None,
    support_threshold: float = 0.38,
    max_claims: int = 60,
    max_llm_claims: int = 12,
    expected_report_sha256: str = "",
    expected_evidence_sha256: str = "",
) -> dict[str, Any]:
    """Recompute an audit while retaining input hashes and verifier telemetry."""
    report_file = Path(report_path)
    evidence_file = Path(evidence_path)
    report = report_file.read_text(encoding="utf-8")
    report_hash = sha256_text(report)
    evidence_hash = _sha256_file(evidence_file)
    if expected_report_sha256 and report_hash != expected_report_sha256:
        raise ValueError("Report SHA-256 does not match --expect-report-sha256")
    if expected_evidence_sha256 and evidence_hash != expected_evidence_sha256:
        raise ValueError("Evidence SHA-256 does not match --expect-evidence-sha256")

    store = EvidenceStore.load_artifact(evidence_file)
    policy = None
    mode = "heuristic"
    if verifier_backend:
        mode = "hybrid"
        policy = ModelRouter.create_backend(
            verifier_backend,
            use_cache=False,
            temperature=0.1,
            max_tokens=4000,
        )

    verifier = ClaimVerifier(
        policy=policy,
        mode=mode,
        support_threshold=support_threshold,
        max_claims=max_claims,
        max_llm_claims=max_llm_claims,
    )
    usage_before = VLLMPolicy.global_usage_snapshot()
    started = time.perf_counter()
    audit = verifier.audit_text(
        report,
        store,
        citation_source_ids=list(store.sources),
        use_llm=policy is not None,
    )
    elapsed = time.perf_counter() - started

    return {
        "artifact_schema": "evidence-replay-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "immutable_evidence_replay",
        "not_a_quality_benchmark": True,
        "inputs": {
            "report": {
                "path": _portable_path(report_file),
                "sha256": report_hash,
                "characters": len(report),
            },
            "evidence": {
                "path": _portable_path(evidence_file),
                "sha256": evidence_hash,
                "schema_version": "1.0",
                "source_count": len(store.sources),
                "evidence_count": len(store.evidence),
            },
        },
        "verification": {
            "mode": mode,
            "backend": verifier_backend,
            "support_threshold": support_threshold,
            "max_claims": max_claims,
            "max_llm_claims": max_llm_claims if policy is not None else 0,
            "elapsed_seconds": round(elapsed, 4),
            "usage": usage_delta(usage_before, VLLMPolicy.global_usage_snapshot()),
        },
        "audit": audit.to_dict(store.evidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay claim verification over retained report/evidence artifacts."
    )
    parser.add_argument("--report", required=True, help="Exact Markdown report to audit")
    parser.add_argument("--evidence", required=True, help="Persisted EvidenceStore JSON")
    parser.add_argument("--output", required=True, help="Replay JSON output path")
    parser.add_argument(
        "--verifier-backend",
        default=None,
        help="Optional configured ModelRouter backend; omit for deterministic heuristic mode",
    )
    parser.add_argument("--support-threshold", type=float, default=0.38)
    parser.add_argument("--max-claims", type=int, default=60)
    parser.add_argument("--max-llm-claims", type=int, default=12)
    parser.add_argument("--expect-report-sha256", default="")
    parser.add_argument("--expect-evidence-sha256", default="")
    args = parser.parse_args()

    result = replay_evidence(
        args.report,
        args.evidence,
        verifier_backend=args.verifier_backend,
        support_threshold=args.support_threshold,
        max_claims=args.max_claims,
        max_llm_claims=args.max_llm_claims,
        expected_report_sha256=args.expect_report_sha256,
        expected_evidence_sha256=args.expect_evidence_sha256,
    )
    atomic_write_json(args.output, result)
    audit = result["audit"]
    print(
        f"{result['verification']['mode']} replay: "
        f"{audit['supported_count']}/{len(audit['claims'])} supported, "
        f"coverage={audit['coverage']:.1%}; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
