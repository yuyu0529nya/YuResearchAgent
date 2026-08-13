"""Deterministic long-report sampling for LLM-as-Judge prompts."""

from __future__ import annotations


def balanced_report_excerpt(report: str, max_chars: int = 12000, segments: int = 4) -> str:
    """Return a length-bounded excerpt sampled across the entire report.

    Prefix-only truncation systematically hides conclusions and bibliographies. This
    sampler keeps evenly spaced windows, including both the beginning and the end,
    while making omissions explicit to the judge.
    """
    text = report or ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    segments = max(2, segments)
    marker_template = "\n\n[... omitted; excerpt {}/{} ...]\n\n"
    marker_budget = sum(len(marker_template.format(i, segments)) for i in range(2, segments + 1))
    content_budget = max(max_chars - marker_budget, segments)
    chunk_size = max(1, content_budget // segments)
    max_start = max(0, len(text) - chunk_size)

    fractions = [i / (segments - 1) for i in range(segments)]
    middle_index = min(range(1, segments - 1), key=lambda i: abs(fractions[i] - 0.5))
    fractions[middle_index] = 0.5
    starts = [round(fraction * max_start) for fraction in fractions]
    chunks: list[str] = []
    for index, start in enumerate(starts):
        end = min(len(text), start + chunk_size)
        if index > 0:
            next_break = text.find("\n", start, min(end, start + 200))
            if next_break != -1:
                start = next_break + 1
        if index < segments - 1:
            previous_break = text.rfind("\n", max(start, end - 200), end)
            if previous_break > start:
                end = previous_break
        chunks.append(text[start:end].strip())

    excerpt = chunks[0]
    for index, chunk in enumerate(chunks[1:], 2):
        excerpt += marker_template.format(index, segments) + chunk
    return excerpt[:max_chars]


__all__ = ["balanced_report_excerpt"]
