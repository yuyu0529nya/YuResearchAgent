"""Deterministic metrics derived from a persisted claim-evidence audit."""

from __future__ import annotations

from typing import Any


def evidence_quality_metrics(audit: dict[str, Any] | None) -> dict[str, float | int]:
    if not audit:
        return {}

    claims = [claim for claim in audit.get("claims", []) if isinstance(claim, dict)]
    total = len(claims)
    cited = [claim for claim in claims if claim.get("cited_indices")]
    supported = [claim for claim in claims if claim.get("status") == "supported"]
    supported_cited = [claim for claim in cited if claim.get("status") == "supported"]
    attributed = [
        claim
        for claim in supported
        if claim.get("support_evidence_ids") and claim.get("source_ids")
    ]

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "claim_count": total,
        "claim_support_coverage": ratio(len(supported), total),
        "claim_citation_rate": ratio(len(cited), total),
        "cited_claim_support_precision": ratio(len(supported_cited), len(cited)),
        "supported_claim_citation_rate": ratio(len(supported_cited), len(supported)),
        "attribution_coverage": ratio(len(attributed), total),
        "refuted_rate": ratio(
            sum(claim.get("status") == "refuted" for claim in claims), total
        ),
        "nei_rate": ratio(
            sum(claim.get("status") == "not_enough_evidence" for claim in claims), total
        ),
        "source_count": int(audit.get("source_count", 0)),
        "evidence_count": int(audit.get("evidence_count", 0)),
        "primary_source_ratio": float(audit.get("primary_source_ratio", 0.0)),
        "fulltext_source_ratio": float(audit.get("fulltext_source_ratio", 0.0)),
    }
