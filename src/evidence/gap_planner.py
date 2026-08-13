"""Convert unresolved claims into targeted verification tasks."""

from __future__ import annotations

import re

from ..orchestrator.schemas import SubTask, TaskType
from .schemas import EvidenceAudit, VerificationStatus
from .store import source_relevance


def _search_hints(text: str) -> list[str]:
    english = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,}", text)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
    return list(dict.fromkeys(english + chinese))[:6]


def _claim_tokens(text: str) -> set[str]:
    english = set(re.findall(r"[a-z][a-z0-9_.+-]{2,}", text.lower()))
    chinese: set[str] = set()
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        chinese.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return english | chinese


def _redundancy(text: str, selected: list[str]) -> float:
    tokens = _claim_tokens(text)
    if not tokens or not selected:
        return 0.0
    overlaps = []
    for previous in selected:
        previous_tokens = _claim_tokens(previous)
        union = tokens | previous_tokens
        overlaps.append(len(tokens & previous_tokens) / len(union) if union else 0.0)
    return max(overlaps, default=0.0)


def build_evidence_gap_tasks(
    audit: EvidenceAudit,
    round_index: int,
    max_tasks: int = 2,
    query: str = "",
) -> list[SubTask]:
    unresolved = [
        claim
        for claim in audit.claims
        if claim.status in (VerificationStatus.NOT_ENOUGH_EVIDENCE, VerificationStatus.REFUTED)
    ]
    # MMR-style selection favors claims central to the original question while
    # avoiding near-duplicate verification tasks. Numeric density is not an
    # importance signal: it previously over-selected sensational side claims.
    selected_claims = []
    selected_texts: list[str] = []
    remaining = list(unresolved)
    while remaining and len(selected_claims) < max_tasks:
        best = max(
            remaining,
            key=lambda claim: (
                0.30 * (claim.status == VerificationStatus.REFUTED)
                + 0.55 * source_relevance(query, claim.text)[0]
                + 0.15 * (1.0 - claim.verification_score)
                - 0.35 * _redundancy(claim.text, selected_texts),
                -len(claim.text),
            ),
        )
        selected_claims.append(best)
        selected_texts.append(best.text)
        remaining.remove(best)
    tasks: list[SubTask] = []
    for index, claim in enumerate(selected_claims, 1):
        tasks.append(
            SubTask(
                task_id=f"evidence_gap_r{round_index}_{index}",
                task_type=TaskType.VERIFY,
                description=(
                    "核验下列 claim，优先查找原始论文、官方文档或政府/机构数据；必须打开至少一个原文页面，"
                    "记录支持、反驳或证据不足，并保留精确数字与发布日期：\n"
                    f"{claim.text}"
                ),
                timeout_seconds=240,
                priority=0,
                expected_type="verification",
                search_hints=_search_hints(claim.text),
            )
        )
    return tasks
