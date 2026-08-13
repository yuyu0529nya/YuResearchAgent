"""Claim-level evidence tracking and verification."""

from .gap_planner import build_evidence_gap_tasks
from .schemas import (
    ClaimRecord,
    EvidenceAudit,
    EvidenceChunk,
    EvidenceKind,
    SourceRecord,
    VerificationStatus,
)
from .store import EvidenceStore
from .verifier import ClaimVerifier

__all__ = [
    "ClaimRecord",
    "ClaimVerifier",
    "EvidenceAudit",
    "EvidenceChunk",
    "EvidenceKind",
    "EvidenceStore",
    "SourceRecord",
    "VerificationStatus",
    "build_evidence_gap_tasks",
]
