"""Deterministic delivery coverage diagnostics for planned research dimensions."""

from __future__ import annotations

import re
from typing import Any


__all__ = ["audit_task_coverage"]


_REFERENCE_HEADING = re.compile(
    r"^#{1,4}\s*(?:参考来源|参考文献|引用|references|bibliography|sources)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ENGLISH_WORD = re.compile(r"[a-z][a-z0-9_.+-]{2,}")
_CHINESE_SEQUENCE = re.compile(r"[\u4e00-\u9fff]{2,}")
_STOP_TERMS = {
    "about", "analysis", "and", "are", "compare", "comparison", "for", "from",
    "how", "into", "research", "that", "the", "this", "with", "what", "which",
    "分析", "对比", "比较", "影响", "情况", "方面", "相关", "研究", "问题", "最新",
    "系统", "内容", "方法", "趋势", "进行", "以及", "不同", "通过", "一个", "什么",
}


def _body_blocks(content: str) -> list[str]:
    """Return report blocks before the bibliography, excluding Markdown scaffolding."""

    body = _REFERENCE_HEADING.split(content or "", maxsplit=1)[0]
    blocks: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)、]\s*", "", line)
        line = re.sub(r"\[\d{1,3}\]", "", line)
        if line:
            blocks.append(line)
    return blocks


def _terms(text: str) -> set[str]:
    lowered = (text or "").lower()
    terms = {
        match.group(0)
        for match in _ENGLISH_WORD.finditer(lowered)
        if match.group(0) not in _STOP_TERMS
    }
    for match in _CHINESE_SEQUENCE.finditer(lowered):
        sequence = match.group(0)
        terms.update(
            sequence[index : index + 2]
            for index in range(len(sequence) - 1)
            if sequence[index : index + 2] not in _STOP_TERMS
        )
    return terms


def _is_covered(description: str, blocks: list[str]) -> tuple[bool, list[str]]:
    anchors = _terms(description)
    if not anchors:
        return False, []
    required_overlap = 1 if len(anchors) <= 3 else 2
    best: set[str] = set()
    for block in blocks:
        overlap = anchors & _terms(block)
        if len(overlap) > len(best):
            best = overlap
    return len(best) >= required_overlap, sorted(best)


def audit_task_coverage(
    content: str,
    requirements: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Diagnose whether planned dimensions reached the report with source backing.

    This is intentionally a lexical, deterministic diagnostic. It distinguishes
    an absent task result from a synthesis omission, but does not claim semantic
    entailment or factual correctness.
    """

    blocks = _body_blocks(content)
    sources_by_task: dict[str, int] = {}
    for source in sources or []:
        task_ids = source.get("task_ids") or []
        for task_id in task_ids:
            normalized = str(task_id or "")
            if normalized:
                sources_by_task[normalized] = sources_by_task.get(normalized, 0) + 1

    items: list[dict[str, Any]] = []
    for requirement in requirements or []:
        task_id = str(requirement.get("task_id") or "")
        description = str(requirement.get("description") or "").strip()
        if not task_id or not description:
            continue
        status = str(requirement.get("status") or "unknown")
        source_count = int(sources_by_task.get(task_id, 0))
        covered, matched_terms = _is_covered(description, blocks)
        if covered and source_count > 0:
            outcome = "covered"
        elif source_count > 0:
            outcome = "synthesis_gap"
        elif status != "success":
            outcome = "research_gap"
        else:
            outcome = "source_gap"
        items.append(
            {
                "task_id": task_id,
                "description": description,
                "task_status": status,
                "source_count": source_count,
                "covered_in_report": covered,
                "matched_terms": matched_terms,
                "outcome": outcome,
            }
        )

    total = len(items)
    covered_count = sum(item["outcome"] == "covered" for item in items)
    source_backed_count = sum(item["source_count"] > 0 for item in items)
    return {
        "method": "deterministic_task_coverage_v1",
        "required_count": total,
        "covered_count": covered_count,
        "coverage": covered_count / total if total else 0.0,
        "source_backed_count": source_backed_count,
        "source_backed_coverage": source_backed_count / total if total else 0.0,
        "synthesis_gap_count": sum(item["outcome"] == "synthesis_gap" for item in items),
        "research_gap_count": sum(item["outcome"] == "research_gap" for item in items),
        "source_gap_count": sum(item["outcome"] == "source_gap" for item in items),
        "items": items,
    }
