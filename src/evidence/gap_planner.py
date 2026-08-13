"""Convert unresolved claims into targeted verification tasks."""

from __future__ import annotations

import re

from ..orchestrator.schemas import SubTask, TaskType
from .schemas import EvidenceAudit, VerificationStatus


def _search_hints(text: str) -> list[str]:
    english = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,}", text)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
    return list(dict.fromkeys(english + chinese))[:6]


def build_evidence_gap_tasks(
    audit: EvidenceAudit,
    round_index: int,
    max_tasks: int = 2,
) -> list[SubTask]:
    unresolved = [
        claim
        for claim in audit.claims
        if claim.status in (VerificationStatus.NOT_ENOUGH_EVIDENCE, VerificationStatus.REFUTED)
    ]
    unresolved.sort(
        key=lambda claim: (
            claim.status != VerificationStatus.REFUTED,
            -len(re.findall(r"\d", claim.text)),
            claim.verification_score,
        )
    )
    tasks: list[SubTask] = []
    for index, claim in enumerate(unresolved[:max_tasks], 1):
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
