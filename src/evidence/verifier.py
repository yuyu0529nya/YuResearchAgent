"""Claim extraction and support/refute/NEI verification."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from ..orchestrator.schemas import AgentResult, AgentStatus
from ..utils.json_parsing import extract_json_object
from .schemas import ClaimRecord, EvidenceAudit, EvidenceChunk, EvidenceKind, VerificationStatus
from .store import EvidenceStore, _stable_id


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
        return self._verify(claims, store, use_llm=use_llm)

    def audit_text(
        self,
        text: str,
        store: EvidenceStore,
        task_id: str = "final_report",
        citation_source_ids: list[str] | None = None,
        *,
        use_llm: bool = True,
    ) -> EvidenceAudit:
        return self._verify(
            self.extract_claims(text, task_id),
            store,
            citation_source_ids=citation_source_ids,
            use_llm=use_llm,
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
            body_lines.append(re.sub(r"^(?:[-*+] |\d+[.)]\s*)", "", line))

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
            cited = sorted({int(value) for value in _CITATION_RE.findall(cleaned)})
            if (
                len(plain) < 12
                or len(plain) > 500
                or plain.endswith(("?", "？", ":", "："))
            ):
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
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _PROCESS_CLAIM_PATTERNS):
            return True
        return not cited and bool(_PROCESS_WITHOUT_CITATION.search(normalized))

    def _verify(
        self,
        claims: list[ClaimRecord],
        store: EvidenceStore,
        citation_source_ids: list[str] | None = None,
        use_llm: bool = True,
    ) -> EvidenceAudit:
        evidence = list(store.evidence.values())
        for claim in claims:
            if citation_source_ids is None:
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
            # A numbered citation is an explicit attribution edge. Preserve its
            # evidence candidates even when a Chinese claim and English paper
            # have zero lexical overlap; hybrid verification can adjudicate the
            # cross-language entailment. Uncited claims still require overlap.
            ranked = (
                ranked_all[:3]
                if cited_source_ids
                else [(score, chunk) for score, chunk in ranked_all if score > 0][:3]
            )
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
                claim.status = VerificationStatus.REFUTED
                claim.contradiction_evidence_ids = [best_chunk.evidence_id]
                claim.reason = "High-overlap evidence expresses the opposite polarity."
            elif best_score >= self.support_threshold and not number_conflict:
                claim.status = VerificationStatus.SUPPORTED
                claim.support_evidence_ids = [
                    chunk.evidence_id
                    for _, chunk in ranked
                    if self._score_pair(claim.text, chunk, store)
                    >= max(self.support_threshold * 0.85, best_score - 0.12)
                ][:2]
                claim.reason = "Claim terms and numeric details are covered by retrieved evidence."
            elif number_conflict:
                claim.reason = "Related evidence was found, but numeric details do not match."
            else:
                claim.reason = "Related evidence was found, but entailment was too weak."

        if self.mode == "hybrid" and use_llm and self.policy is not None:
            self._apply_llm_verdicts(claims, store)
        verification_mode = self.mode if use_llm else "heuristic"
        return self._build_audit(claims, store, verification_mode=verification_mode)

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

    @staticmethod
    def _number_conflict(claim: str, evidence: str) -> bool:
        claim_numbers = _numbers(claim)
        evidence_numbers = _numbers(evidence)
        return bool(claim_numbers and evidence_numbers and not claim_numbers.issubset(evidence_numbers))

    def _apply_llm_verdicts(self, claims: list[ClaimRecord], store: EvidenceStore) -> None:
        ambiguous = [
            claim
            for claim in claims
            if claim.status == VerificationStatus.NOT_ENOUGH_EVIDENCE and claim.candidate_evidence_ids
        ]
        ambiguous.sort(
            key=lambda claim: (bool(claim.cited_indices), claim.verification_score),
            reverse=True,
        )
        ambiguous = ambiguous[: self.max_llm_claims]
        if not ambiguous:
            return
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
        old_tools = getattr(self.policy, "tools", None)
        try:
            if hasattr(self.policy, "tools"):
                self.policy.tools = None
            # Keep each request below the policy's message truncation threshold.
            # A truncated JSON evidence payload cannot be audited reliably.
            for offset in range(0, len(items), 8):
                batch = items[offset : offset + 8]
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
                    response = self.policy(
                        [
                            {
                                "role": "system",
                                "content": "You are a strict claim-evidence verifier. Output JSON only.",
                            },
                            {"role": "user", "content": prompt},
                        ]
                    )
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
                    claim.support_evidence_ids = claim.candidate_evidence_ids[:2]
            elif claim.status == VerificationStatus.REFUTED:
                claim.contradiction_evidence_ids = claim.candidate_evidence_ids[:1]

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
            return True
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
        )
