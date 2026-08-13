"""Typed schemas for the claim-evidence-source graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"


class EvidenceKind(str, Enum):
    SEARCH_SNIPPET = "search_snippet"
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"
    FILE = "file"


@dataclass
class SourceRecord:
    source_id: str
    url: str = ""
    title: str = ""
    source_type: str = "web"
    task_ids: list[str] = field(default_factory=list)
    authors: str = ""
    year: str = ""
    publisher: str = ""
    quality_score: float = 0.5
    is_primary: bool = False
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceChunk:
    evidence_id: str
    source_id: str
    text: str
    kind: EvidenceKind
    task_id: str = ""
    locator: str = ""
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass
class ClaimRecord:
    claim_id: str
    text: str
    task_id: str = ""
    cited_indices: list[int] = field(default_factory=list)
    status: VerificationStatus = VerificationStatus.NOT_ENOUGH_EVIDENCE
    verification_score: float = 0.0
    support_evidence_ids: list[str] = field(default_factory=list)
    contradiction_evidence_ids: list[str] = field(default_factory=list)
    candidate_evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class EvidenceAudit:
    claims: list[ClaimRecord] = field(default_factory=list)
    source_count: int = 0
    evidence_count: int = 0
    supported_count: int = 0
    refuted_count: int = 0
    nei_count: int = 0
    coverage: float = 0.0
    primary_source_ratio: float = 0.0
    fulltext_source_ratio: float = 0.0
    verification_mode: str = "heuristic"

    def to_dict(self, evidence_lookup: dict[str, EvidenceChunk] | None = None) -> dict[str, Any]:
        lookup = evidence_lookup or {}
        claim_dicts: list[dict[str, Any]] = []
        for claim in self.claims:
            data = claim.to_dict()
            evidence_ids = claim.support_evidence_ids or claim.candidate_evidence_ids[:2]
            data["evidence_excerpts"] = [
                {
                    "evidence_id": evidence_id,
                    "source_id": lookup[evidence_id].source_id,
                    "text": lookup[evidence_id].text[:500],
                    "kind": lookup[evidence_id].kind.value,
                }
                for evidence_id in evidence_ids
                if evidence_id in lookup
            ]
            claim_dicts.append(data)
        return {
            "claims": claim_dicts,
            "source_count": self.source_count,
            "evidence_count": self.evidence_count,
            "supported_count": self.supported_count,
            "refuted_count": self.refuted_count,
            "not_enough_evidence_count": self.nei_count,
            "coverage": self.coverage,
            "primary_source_ratio": self.primary_source_ratio,
            "fulltext_source_ratio": self.fulltext_source_ratio,
            "verification_mode": self.verification_mode,
        }
