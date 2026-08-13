"""Claim extraction and support/refute/NEI verification."""

from __future__ import annotations

import json
import re
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from ..orchestrator.schemas import AgentResult, AgentStatus
from ..utils.json_parsing import extract_json_object
from .schemas import ClaimRecord, EvidenceAudit, EvidenceChunk, EvidenceKind, VerificationStatus
from .store import EvidenceStore, _stable_id, source_relevance


_REFERENCE_HEADING = re.compile(
    r"^#{1,4}\s*(参考来源|参考文献|引用|references|bibliography|sources)\s*$",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)*)(?:\s*(万亿|万|亿|trillion|billion|million|thousand|[KMBT]|%|％|倍|年|月|日))?",
    re.IGNORECASE,
)
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "were", "was", "are", "is",
    "of", "to", "in", "on", "as", "by", "or", "an", "a", "it", "its", "be", "can",
    "以及", "一个", "一种", "这些", "这种", "其中", "通过", "对于", "由于", "因此", "同时",
    "可以", "进行", "主要", "已经", "研究", "结果", "报告", "显示", "表明",
}
_PROCESS_CLAIM_PATTERNS = (
    r"^(?:本报告|本文|本次(?:调研|检索|研究)|研究问题|两份?子任务|子任务\s*\d*|证据审计)",
    r"(?:不讨论其他模型|不得混用其他模型|无与其他模型混淆)",
    r"(?:证据审计中|子任务报告中|两份?子任务(?:报告|结果)|当前证据(?:审计|粒度))",
)
_PROCESS_WITHOUT_CITATION = re.compile(
    r"(?:检索工具|检索渠道|网络超时|自评置信度|未能当场|本次未能|当前证据(?:审计|粒度)|"
    r"子任务报告|核验结果|未被逐条独立证实)",
    re.IGNORECASE,
)
_DISCOURSE_PREFIX = re.compile(
    r"^(?:以下为|结论如下|原始表述可直接引用|需要说明的是|需要指出)\s*[:：]\s*",
    re.IGNORECASE,
)
_CITATION_FRAGMENT = re.compile(
    r"^(?:[A-Z][A-Za-z'.-]+(?:\s+et\s+al\.)?\s*[,，].*(?:19|20)\d{2}"
    r"|[^。！？.!?]{1,60}《[^》]{3,160}》(?:教程|综述|论文|报告)?)"
    r"[）)]?[。.]?$",
    re.IGNORECASE,
)
_TEMPORAL_LABEL = re.compile(
    r"^[（(].*(?:信息截至|截至|as[- ]of|updated|报告日期|report date).*[）)]$",
    re.IGNORECASE,
)


def _normalize_numeric_spacing(text: str) -> str:
    """Repair decimal/thousands separators split by HTML or PDF extraction."""
    normalized = re.sub(r"(?<=\d)\s*([.,])\s*(?=\d)", r"\1", text or "")
    return re.sub(r"(?<=\d)\s+(?=[%％万亿倍年月日])", "", normalized)


def _tokens(text: str) -> set[str]:
    lowered = _normalize_numeric_spacing(text).lower()
    tokens = {
        token
        for token in re.findall(r"[a-z][a-z0-9_.+-]{2,}|\d+(?:\.\d+)?", lowered)
        if token not in _STOPWORDS
    }
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
        tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _numbers(text: str) -> set[str]:
    normalized = _normalize_numeric_spacing(text)
    values: set[str] = set()
    multipliers = {
        "万": Decimal("1e4"),
        "亿": Decimal("1e8"),
        "万亿": Decimal("1e12"),
        "thousand": Decimal("1e3"),
        "k": Decimal("1e3"),
        "million": Decimal("1e6"),
        "m": Decimal("1e6"),
        "billion": Decimal("1e9"),
        "b": Decimal("1e9"),
        "trillion": Decimal("1e12"),
        "t": Decimal("1e12"),
    }
    for raw_number, raw_suffix in _NUMBER_RE.findall(normalized):
        number = raw_number.replace(",", "")
        suffix = raw_suffix.replace("％", "%").lower()
        try:
            value = Decimal(number)
        except InvalidOperation:
            continue
        if suffix in multipliers:
            scaled = value * multipliers[suffix]
            values.add(format(scaled.normalize(), "f"))
        elif suffix:
            values.add(f"{format(value.normalize(), 'f')}{suffix}")
        else:
            values.add(format(value.normalize(), "f"))
    return values


def _negated(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"(?:并非|不是|未能|没有|无法|不支持|no |not |never |failed to)", lowered))


class ClaimVerifier:
    """Build a CAGE-style claim-document attribution graph.

    A deterministic lexical pass provides reproducible coverage. Optional hybrid
    mode asks an injected LLM only about ambiguous claim/evidence pairs and keeps
    NEI as the required fallback when the supplied evidence is insufficient.
    """

    def __init__(
        self,
        policy: Any | None = None,
        mode: str = "heuristic",
        support_threshold: float = 0.38,
        max_claims: int = 60,
        max_llm_claims: int = 12,
    ) -> None:
        self.policy = policy
        self.mode = mode if mode in {"heuristic", "hybrid"} else "heuristic"
        self.support_threshold = support_threshold
        self.max_claims = max_claims
        self.max_llm_claims = max_llm_claims

    def audit_results(
        self,
        results: Iterable[AgentResult],
        store: EvidenceStore,
        *,
        use_llm: bool = True,
        timeout_seconds: float | None = None,
    ) -> EvidenceAudit:
        claims: list[ClaimRecord] = []
        seen: set[str] = set()
        for result in results:
            if result.status != AgentStatus.SUCCESS or not isinstance(result.output, str):
                continue
            for claim in self.extract_claims(result.output, result.task_id):
                normalized = re.sub(r"\W+", "", claim.text).lower()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    claims.append(claim)
                if len(claims) >= self.max_claims:
                    break
            if len(claims) >= self.max_claims:
                break
        return self._verify(
            claims,
            store,
            use_llm=use_llm,
            timeout_seconds=timeout_seconds,
        )

    def audit_text(
        self,
        text: str,
        store: EvidenceStore,
        task_id: str = "final_report",
        citation_source_ids: list[str] | None = None,
        *,
        use_llm: bool = True,
        timeout_seconds: float | None = None,
    ) -> EvidenceAudit:
        return self._verify(
            self.extract_claims(text, task_id),
            store,
            citation_source_ids=citation_source_ids,
            use_llm=use_llm,
            timeout_seconds=timeout_seconds,
        )

    def extract_claims(self, text: str, task_id: str = "") -> list[ClaimRecord]:
        claims: list[ClaimRecord] = []
        body_lines: list[str] = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if _REFERENCE_HEADING.match(line):
                break
            if not line or line.startswith("#") or line.startswith("|") or line.startswith("```"):
                continue
            if re.match(r"^(overall confidence|整体置信度|置信度)\s*[:：]", line, re.I):
                continue
            if _TEMPORAL_LABEL.match(line):
                continue
            unbulleted = re.sub(r"^(?:[-*+] |\d+[.)]\s*)", "", line)
            if (
                not _CITATION_RE.search(unbulleted)
                and re.fullmatch(r"\*{1,2}[^*]{1,100}\*{1,2}[。.!]?", unbulleted)
            ):
                continue
            body_lines.append(unbulleted)

        sentences = re.split(
            r"(?<=[。！？；;])|(?<=[.!?])\s+|\n+",
            "\n".join(body_lines),
        )
        for sentence in sentences:
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            plain = _CITATION_RE.sub("", cleaned).strip()
            plain = _DISCOURSE_PREFIX.sub("", plain)
            plain = re.sub(
                r"^[*_`>\s]*(?:\(\d+\)|（\d+）|\d+[.)、])[*_`\s]*",
                "",
                plain,
            )
            plain = re.sub(r"[*_`]", "", plain).strip()
            cited = sorted({int(value) for value in _CITATION_RE.findall(cleaned)})
            if (
                len(plain) < 12
                or len(plain) > 500
                or plain.endswith(("?", "？", ":", "："))
            ):
                continue
            if len(_tokens(plain)) < 3 and not _numbers(plain):
                continue
            if self._is_process_claim(plain, cited=bool(cited)):
                continue
            claim_id = _stable_id("claim", task_id, plain.lower())
            claims.append(
                ClaimRecord(
                    claim_id=claim_id,
                    text=plain,
                    task_id=task_id,
                    cited_indices=cited,
                )
            )
            if len(claims) >= self.max_claims:
                break
        return claims

    @staticmethod
    def _is_process_claim(text: str, *, cited: bool = False) -> bool:
        """Exclude run/report narration that external sources cannot entail."""
        normalized = re.sub(r"^[>*\s]+", "", text).strip()
        if _CITATION_FRAGMENT.match(normalized):
            return True
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _PROCESS_CLAIM_PATTERNS):
            return True
        return not cited and bool(_PROCESS_WITHOUT_CITATION.search(normalized))

    def _verify(
        self,
        claims: list[ClaimRecord],
        store: EvidenceStore,
        citation_source_ids: list[str] | None = None,
        use_llm: bool = True,
        timeout_seconds: float | None = None,
    ) -> EvidenceAudit:
        evidence = list(store.evidence.values())
        for claim in claims:
            has_global_citation_map = citation_source_ids is not None
            if not has_global_citation_map:
                # Sub-agent reports do not share a global citation numbering scheme.
                # Apply citation constraints only for final synthesis, where the
                # orchestrator supplies the exact ordered source list.
                cited_source_ids = []
            else:
                cited_source_ids = [
                    citation_source_ids[index - 1]
                    for index in claim.cited_indices
                    if 0 < index <= len(citation_source_ids)
                ]
            candidates = [
                chunk
                for chunk in evidence
                if not cited_source_ids or chunk.source_id in cited_source_ids
            ]
            ranked_all = sorted(
                (
                    (self._candidate_rank(claim.text, chunk, store), chunk)
                    for chunk in candidates
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            support_grade = [
                item for item in ranked_all if self._is_support_grade(item[1], store)
            ]
            # A numbered citation is an explicit attribution edge. Preserve its
            # evidence candidates even when a Chinese claim and English paper
            # have zero lexical overlap; hybrid verification can adjudicate the
            # cross-language entailment. Uncited claims still require overlap.
            if cited_source_ids:
                ranked = (support_grade or ranked_all)[:3]
            else:
                ranked = [(score, chunk) for score, chunk in ranked_all if score > 0][:3]
                if not has_global_citation_map and claim.task_id:
                    # Research-agent prose and its tool evidence share a task ID.
                    # Preserve that provenance edge as a *candidate* when lexical
                    # overlap is zero (for example, a Chinese claim backed by an
                    # English abstract). Only the strict hybrid verifier may turn
                    # this edge into support.
                    provenance = sorted(
                        (
                            (self._provenance_rank(claim.text, chunk, store), chunk)
                            for _, chunk in support_grade
                            if self._shares_task_provenance(claim, chunk, store)
                        ),
                        key=lambda item: item[0],
                        reverse=True,
                    )
                    lexical = list(ranked)
                    ranked = lexical[:1]
                    seen_evidence = {chunk.evidence_id for _, chunk in ranked}
                    for item in provenance[:2]:
                        if item[1].evidence_id not in seen_evidence:
                            ranked.append(item)
                            seen_evidence.add(item[1].evidence_id)
                    for item in lexical[1:]:
                        if item[1].evidence_id not in seen_evidence:
                            ranked.append(item)
                            seen_evidence.add(item[1].evidence_id)
                        if len(ranked) >= 3:
                            break
            claim.candidate_evidence_ids = [chunk.evidence_id for _, chunk in ranked]
            best_chunk = ranked[0][1] if ranked else None
            best_score = self._score_pair(claim.text, best_chunk, store) if best_chunk else 0.0
            claim.verification_score = round(best_score, 4)

            if not ranked:
                claim.reason = "No matching evidence chunk was retrieved."
                continue

            claim.source_ids = list(dict.fromkeys(chunk.source_id for _, chunk in ranked))
            number_conflict = self._number_conflict(claim.text, best_chunk.text)
            negation_conflict = _negated(claim.text) != _negated(best_chunk.text)
            if best_score >= 0.62 and negation_conflict and not number_conflict:
                # Surface polarity as an ambiguous pair, not a deterministic
                # contradiction. Lexical negation is brittle ("not only",
                # quoted denials, and unresolved-problem wording all caused
                # false REFUTED labels in real reports). The hybrid verifier may
                # still establish a contradiction from the complete clauses.
                claim.reason = "A possible polarity conflict requires semantic verification."
            elif best_score >= self.support_threshold and not number_conflict:
                if self._is_support_grade(best_chunk, store):
                    claim.status = VerificationStatus.SUPPORTED
                    claim.support_evidence_ids = [
                        chunk.evidence_id
                        for _, chunk in ranked
                        if self._is_support_grade(chunk, store)
                        and self._score_pair(claim.text, chunk, store)
                        >= max(self.support_threshold * 0.85, best_score - 0.12)
                    ][:2]
                    claim.reason = "Claim terms and numeric details are covered by retrieved evidence."
                    claim.source_ids = list(
                        dict.fromkeys(
                            store.evidence[evidence_id].source_id
                            for evidence_id in claim.support_evidence_ids
                            if evidence_id in store.evidence
                        )
                    )
                else:
                    claim.reason = (
                        "Only a secondary search snippet matched; full text, a paper abstract, "
                        "or a primary-source snippet is required for support."
                    )
            elif number_conflict:
                claim.reason = "Related evidence was found, but numeric details do not match."
            else:
                claim.reason = "Related evidence was found, but entailment was too weak."

        verification_mode = "heuristic"
        semantic_reviewed_count = 0
        semantic_candidate_count = 0
        if self.mode == "hybrid" and use_llm:
            if self.policy is None:
                verification_mode = "hybrid_unavailable"
            else:
                hybrid_result = self._apply_llm_verdicts(
                    claims,
                    store,
                    timeout_seconds=timeout_seconds,
                )
                if hybrid_result is None:
                    verification_mode = "hybrid"
                else:
                    semantic_reviewed_count, semantic_candidate_count = hybrid_result
                    if semantic_reviewed_count == 0:
                        verification_mode = "hybrid_unavailable"
                    elif semantic_reviewed_count < semantic_candidate_count:
                        verification_mode = "hybrid_partial"
                    else:
                        verification_mode = "hybrid"
        elif use_llm:
            verification_mode = self.mode
        return self._build_audit(
            claims,
            store,
            verification_mode=verification_mode,
            semantic_reviewed_count=semantic_reviewed_count,
            semantic_candidate_count=semantic_candidate_count,
        )

    def _score_pair(self, claim: str, chunk: EvidenceChunk, store: EvidenceStore) -> float:
        claim_tokens = _tokens(claim)
        evidence_tokens = _tokens(chunk.text)
        if not claim_tokens or not evidence_tokens:
            return 0.0
        coverage = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        source = store.sources.get(chunk.source_id)
        quality = source.quality_score if source else 0.4
        kind_bonus = {
            EvidenceKind.SEARCH_SNIPPET: 0.82,
            EvidenceKind.ABSTRACT: 0.95,
            EvidenceKind.FULL_TEXT: 1.0,
            EvidenceKind.FILE: 1.0,
        }[chunk.kind]
        return min(1.0, coverage * (0.75 + 0.25 * quality) * kind_bonus)

    def _candidate_rank(self, claim: str, chunk: EvidenceChunk, store: EvidenceStore) -> float:
        """Rank candidates before entailment without conflating rank and support."""
        lexical = self._score_pair(claim, chunk, store)
        claim_numbers = _numbers(claim)
        evidence_numbers = _numbers(chunk.text)
        numeric = (
            len(claim_numbers & evidence_numbers) / len(claim_numbers)
            if claim_numbers
            else 0.0
        )
        claim_entities = {
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,}", claim.lower())
            if token not in _STOPWORDS
        }
        evidence_entities = _tokens(chunk.text)
        entity = (
            len(claim_entities & evidence_entities) / len(claim_entities)
            if claim_entities
            else 0.0
        )
        return lexical + 0.55 * numeric + 0.20 * entity

    def _provenance_rank(
        self,
        claim: str,
        chunk: EvidenceChunk,
        store: EvidenceStore,
    ) -> float:
        """Rank same-task candidates without treating provenance as entailment."""
        source = store.sources.get(chunk.source_id)
        if source is None:
            return self._candidate_rank(claim, chunk, store)
        title_tokens = _tokens(source.title)
        claim_tokens = _tokens(claim)
        title_overlap = (
            len(title_tokens & claim_tokens) / len(claim_tokens)
            if claim_tokens
            else 0.0
        )
        try:
            retrieval_relevance = float(source.metadata.get("retrieval_relevance", 0.0) or 0.0)
        except (TypeError, ValueError):
            retrieval_relevance = 0.0
        retrieval_query = str(source.metadata.get("retrieval_query", ""))
        context_relevance = max(
            source_relevance(claim, f"{source.title} {source.authors}")[0],
            source_relevance(claim, retrieval_query)[0],
        )
        kind_bonus = {
            EvidenceKind.SEARCH_SNIPPET: 0.0,
            EvidenceKind.ABSTRACT: 0.02,
            EvidenceKind.FULL_TEXT: 0.04,
            EvidenceKind.FILE: 0.04,
        }[chunk.kind]
        return (
            self._candidate_rank(claim, chunk, store)
            + 0.20 * title_overlap
            + 0.35 * context_relevance
            + 0.10
            * max(0.0, min(1.0, retrieval_relevance))
            * max(context_relevance, title_overlap)
            + 0.02 * source.quality_score
            + kind_bonus
        )

    @staticmethod
    def _shares_task_provenance(
        claim: ClaimRecord,
        chunk: EvidenceChunk,
        store: EvidenceStore,
    ) -> bool:
        if not claim.task_id:
            return False
        source = store.sources.get(chunk.source_id)
        return bool(
            chunk.task_id == claim.task_id
            or (source is not None and claim.task_id in source.task_ids)
        )

    @staticmethod
    def _is_support_grade(chunk: EvidenceChunk, store: EvidenceStore) -> bool:
        """Require inspectable evidence before a lexical match can establish a claim."""
        source = store.sources.get(chunk.source_id)
        quality = source.quality_score if source else 0.0
        if chunk.kind == EvidenceKind.FILE:
            return True
        if chunk.kind == EvidenceKind.FULL_TEXT:
            return quality >= 0.5
        if chunk.kind == EvidenceKind.ABSTRACT:
            return bool(
                source
                and (
                    source.source_type == "paper"
                    or (source.is_primary and source.quality_score >= 0.85)
                )
            )
        return bool(source and source.is_primary and quality >= 0.85)

    @staticmethod
    def _number_conflict(claim: str, evidence: str) -> bool:
        claim_numbers = _numbers(claim)
        evidence_numbers = _numbers(evidence)
        return bool(claim_numbers and evidence_numbers and not claim_numbers.issubset(evidence_numbers))

    def _apply_llm_verdicts(
        self,
        claims: list[ClaimRecord],
        store: EvidenceStore,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[int, int] | None:
        ambiguous = [
            claim
            for claim in claims
            if claim.status == VerificationStatus.NOT_ENOUGH_EVIDENCE
            and any(
                evidence_id in store.evidence
                and self._is_support_grade(store.evidence[evidence_id], store)
                for evidence_id in claim.candidate_evidence_ids
            )
        ]
        ambiguous.sort(
            key=lambda claim: (
                bool(claim.cited_indices),
                self._has_cross_language_candidate(claim, store),
                self._hybrid_candidate_priority(claim, store),
            ),
            reverse=True,
        )
        semantic_candidate_count = len(ambiguous)
        # Give each research task one audit slot before filling the remainder.
        # This avoids spending every hybrid verdict on the first sub-agent.
        selected: list[ClaimRecord] = []
        seen_tasks: set[str] = set()
        for claim in ambiguous:
            if claim.task_id and claim.task_id not in seen_tasks:
                selected.append(claim)
                seen_tasks.add(claim.task_id)
            if len(selected) >= self.max_llm_claims:
                break
        for claim in ambiguous:
            if claim not in selected:
                selected.append(claim)
            if len(selected) >= self.max_llm_claims:
                break
        ambiguous = selected
        if not ambiguous:
            return None
        items = []
        for claim in ambiguous:
            clauses = self._material_clauses(claim.text)
            items.append(
                {
                    "claim_id": claim.claim_id,
                    "claim": claim.text,
                    "clauses": clauses,
                    "evidence": [
                        {
                            "evidence_id": evidence_id,
                            "source": self._source_context(
                                store,
                                store.evidence[evidence_id].source_id,
                            ),
                            "text": store.evidence[evidence_id].text[:900],
                        }
                        for evidence_id in claim.candidate_evidence_ids[:2]
                        if evidence_id in store.evidence
                    ],
                }
            )
        verdicts: list[dict[str, Any]] = []
        deadline = (
            time.monotonic() + max(0.25, float(timeout_seconds))
            if timeout_seconds is not None
            else None
        )
        old_tools = getattr(self.policy, "tools", None)
        try:
            if hasattr(self.policy, "tools"):
                self.policy.tools = None
            # Keep each request below the policy's message truncation threshold.
            # A truncated JSON evidence payload cannot be audited reliably.
            for offset in range(0, len(items), 3):
                remaining = deadline - time.monotonic() if deadline is not None else None
                if remaining is not None and remaining <= 0.25:
                    break
                batch = items[offset : offset + 3]
                prompt = (
                    "Verify each claim using ONLY its supplied evidence. Labels: SUPPORTED when evidence directly "
                    "entails the claim; REFUTED when it directly contradicts it; NOT_ENOUGH_EVIDENCE otherwise. "
                    "Source metadata may verify title, author, year, DOI/arXiv ID, and URL facts. Numeric details "
                    "must match. Every material clause in a compound claim must be entailed; partial support is "
                    "NOT_ENOUGH_EVIDENCE. For every input clause, return one aligned clause_labels entry. Return "
                    "JSON as {\"verdicts\":[{\"claim_id\":...,\"label\":...,\"clause_labels\":[...],"
                    "\"score\":0-1,\"reason\":...}]}. Default to NOT_ENOUGH_EVIDENCE.\n\n"
                    + json.dumps(batch, ensure_ascii=False)
                )
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a strict claim-evidence verifier. Output JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                    call_with_timeout = getattr(self.policy, "call_with_timeout", None)
                    if remaining is not None and callable(call_with_timeout):
                        response = call_with_timeout(messages, max(0.25, remaining))
                    else:
                        response = self.policy(messages)
                except Exception:
                    continue
                parsed = extract_json_object(
                    response.get("content", "") if isinstance(response, dict) else ""
                ) or {}
                verdicts.extend(item for item in parsed.get("verdicts", []) if isinstance(item, dict))
        finally:
            if hasattr(self.policy, "tools"):
                self.policy.tools = old_tools

        by_id = {claim.claim_id: claim for claim in ambiguous}
        reviewed_ids: set[str] = set()
        for verdict in verdicts:
            if not isinstance(verdict, dict) or verdict.get("claim_id") not in by_id:
                continue
            claim = by_id[verdict["claim_id"]]
            label = str(verdict.get("label", "")).upper()
            status_map = {
                "SUPPORTED": VerificationStatus.SUPPORTED,
                "REFUTED": VerificationStatus.REFUTED,
                "NOT_ENOUGH_EVIDENCE": VerificationStatus.NOT_ENOUGH_EVIDENCE,
                "NEI": VerificationStatus.NOT_ENOUGH_EVIDENCE,
            }
            if label not in status_map:
                continue
            if claim.claim_id in reviewed_ids:
                continue
            reviewed_ids.add(claim.claim_id)
            claim.status = status_map[label]
            claim.verification_score = max(0.0, min(1.0, float(verdict.get("score", 0.0))))
            claim.reason = str(verdict.get("reason", ""))[:300]
            if claim.status == VerificationStatus.SUPPORTED:
                evidence_text = "\n".join(
                    json.dumps(
                        self._source_context(
                            store,
                            store.evidence[evidence_id].source_id,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                    + store.evidence[evidence_id].text
                    for evidence_id in claim.candidate_evidence_ids[:2]
                    if evidence_id in store.evidence
                )
                if not self._all_material_clauses_covered(
                    claim.text,
                    evidence_text,
                    verdict.get("clause_labels"),
                ):
                    claim.status = VerificationStatus.NOT_ENOUGH_EVIDENCE
                    claim.verification_score = min(claim.verification_score, 0.49)
                    claim.reason = (
                        "Hybrid SUPPORTED verdict was downgraded because at least one material "
                        "clause or numeric detail is absent from the supplied evidence."
                    )
                    claim.support_evidence_ids = []
                else:
                    claim.support_evidence_ids = [
                        evidence_id
                        for evidence_id in claim.candidate_evidence_ids
                        if evidence_id in store.evidence
                        and self._is_support_grade(store.evidence[evidence_id], store)
                    ][:2]
                    claim.source_ids = list(
                        dict.fromkeys(
                            store.evidence[evidence_id].source_id
                            for evidence_id in claim.support_evidence_ids
                            if evidence_id in store.evidence
                        )
                    )
            elif claim.status == VerificationStatus.REFUTED:
                claim.contradiction_evidence_ids = claim.candidate_evidence_ids[:1]
        return len(reviewed_ids), semantic_candidate_count

    @staticmethod
    def _source_context(store: EvidenceStore, source_id: str) -> dict[str, Any]:
        source = store.sources.get(source_id)
        if source is None:
            return {}
        return {
            "source_id": source.source_id,
            "title": source.title,
            "url": source.url,
            "authors": source.authors,
            "year": source.year,
            "source_type": source.source_type,
            "is_primary": source.is_primary,
        }

    @staticmethod
    def _has_cross_language_candidate(claim: ClaimRecord, store: EvidenceStore) -> bool:
        claim_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", claim.text))
        return any(
            evidence_id in store.evidence
            and claim_has_cjk
            != bool(re.search(r"[\u4e00-\u9fff]", store.evidence[evidence_id].text))
            for evidence_id in claim.candidate_evidence_ids
        )

    def _hybrid_candidate_priority(
        self,
        claim: ClaimRecord,
        store: EvidenceStore,
    ) -> float:
        scores = [
            self._provenance_rank(claim.text, store.evidence[evidence_id], store)
            for evidence_id in claim.candidate_evidence_ids
            if evidence_id in store.evidence
        ]
        return max([claim.verification_score, *scores])

    @staticmethod
    def _material_clauses(claim: str) -> list[str]:
        clauses = re.split(
            r"(?:[;；]|，(?:并且|且|同时|而且|但是|但|以及)|\b(?:and|but|while|whereas)\b)",
            claim,
            flags=re.IGNORECASE,
        )
        material = [clause.strip() for clause in clauses if len(_tokens(clause)) >= 2]
        return material or [claim.strip()]

    @classmethod
    def _all_material_clauses_covered(
        cls,
        claim: str,
        evidence: str,
        clause_labels: Any = None,
    ) -> bool:
        """Conservatively guard LLM support labels for compound claims."""
        evidence_tokens = _tokens(evidence)
        if not evidence_tokens:
            return False
        claim_numbers = _numbers(claim)
        if claim_numbers and not claim_numbers.issubset(_numbers(evidence)):
            return False
        clauses = cls._material_clauses(claim)
        if len(clauses) > 1:
            if not isinstance(clause_labels, list) or len(clause_labels) != len(clauses):
                return False
            if any(str(label).upper() != "SUPPORTED" for label in clause_labels):
                return False

        claim_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", claim))
        evidence_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", evidence))
        if claim_has_cjk != evidence_has_cjk:
            # Cross-language lexical overlap is not meaningful. Numeric checks
            # and the strict per-clause LLM verdict remain mandatory.
            return bool(
                isinstance(clause_labels, list)
                and len(clause_labels) == len(clauses)
                and all(str(label).upper() == "SUPPORTED" for label in clause_labels)
            )
        material = [_tokens(clause) for clause in clauses]
        return all(
            len(clause_tokens & evidence_tokens) / len(clause_tokens) >= 0.45
            for clause_tokens in material
        )

    def _build_audit(
        self,
        claims: list[ClaimRecord],
        store: EvidenceStore,
        *,
        verification_mode: str | None = None,
        semantic_reviewed_count: int = 0,
        semantic_candidate_count: int = 0,
    ) -> EvidenceAudit:
        supported = sum(claim.status == VerificationStatus.SUPPORTED for claim in claims)
        refuted = sum(claim.status == VerificationStatus.REFUTED for claim in claims)
        nei = len(claims) - supported - refuted
        sources = store.source_list()
        primary = sum(source.is_primary for source in sources)
        fulltext = len(store.fulltext_source_ids())
        return EvidenceAudit(
            claims=claims,
            source_count=len(sources),
            evidence_count=len(store.evidence),
            supported_count=supported,
            refuted_count=refuted,
            nei_count=nei,
            coverage=round(supported / len(claims), 4) if claims else 0.0,
            primary_source_ratio=round(primary / len(sources), 4) if sources else 0.0,
            fulltext_source_ratio=round(fulltext / len(sources), 4) if sources else 0.0,
            verification_mode=verification_mode or self.mode,
            semantic_reviewed_count=semantic_reviewed_count,
            semantic_candidate_count=semantic_candidate_count,
        )
