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
from .task_coverage import audit_task_coverage
from .verifier import ClaimVerifier
from .reviser import (
    EvidenceReviser,
    RevisionDecision,
    RevisionDraft,
    audit_summary,
    evaluate_revision,
)

__all__ = [
    "ClaimRecord",
    "ClaimVerifier",
    "EvidenceAudit",
    "EvidenceChunk",
    "EvidenceKind",
    "EvidenceStore",
    "EvidenceReviser",
    "RevisionDecision",
    "RevisionDraft",
    "SourceRecord",
    "VerificationStatus",
    "audit_summary",
    "audit_task_coverage",
    "build_evidence_gap_tasks",
    "evaluate_revision",
]
