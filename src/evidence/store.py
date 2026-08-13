"""Persistent source and evidence storage built from agent tool trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit, urlunsplit

from ..orchestrator.schemas import AgentResult
from .schemas import EvidenceAudit, EvidenceChunk, EvidenceKind, SourceRecord


_PRIMARY_DOMAINS = {
    "arxiv.org",
    "doi.org",
    "openreview.net",
    "acm.org",
    "ieee.org",
    "nature.com",
    "science.org",
    "who.int",
    "worldbank.org",
    "oecd.org",
    "un.org",
    "openai.com",
    "anthropic.com",
    "deepmind.google",
    "ai.google.dev",
    "qwenlm.github.io",
    "nextgenscience.org",
    "federalregister.gov",
    "eric.ed.gov",
    "europa.eu",
    "icmagroup.org",
    "dodstem.us",
    "ed.gov",
    "nist.gov",
    "nsf.gov",
    # First-party research/product announcements. These establish what the
    # issuing organization released; they do not independently validate vendor
    # performance claims.
    "nvidia.com",
    "pi.website",
    "dyna.co",
    "xenserobotics.com",
}
_HIGH_QUALITY_DOMAINS = {
    "docs.python.org",
    "semanticscholar.org",
    "openalex.org",
    "reuters.com",
    "apnews.com",
    "sinoss.net",
    "edu.cn",
}
_SECONDARY_REPOSITORY_DOMAINS = {
    "academia.edu",
    "github.com",
    "researchgate.net",
}
_LOW_QUALITY_DOMAINS = {
    "baidu.com",
    "baike.baidu.com",
    "blog.csdn.net",
    "chinabaogao.com",
    "medium.com",
    "sohu.com",
    "douyin.com",
    "t.me",
    "tom.com",
    "youtube.com",
    "doc88.com",
    "fenbi.com",
    "renrendoc.com",
    "bilibili.com",
    "docin.com",
    "wenku.baidu.com",
    "360doc.com",
}

_RELEVANCE_STOPWORDS = {
    "about", "analysis", "and", "china", "chinese", "compare", "comparison",
    "education", "effect", "evaluation", "impact", "policy", "report", "research",
    "study", "the", "united", "with", "official", "original", "or", "site",
    "gov.cn", "moe.gov.cn",
    "中国", "中美", "两国", "分析", "发展", "影响", "政策", "教育", "模式",
    "相关", "研究", "美国", "评估", "进行", "对比", "差异", "官网", "官方",
    "原文", "网站", "教育部",
}

_CANONICAL_INSTITUTION_DOMAINS = {
    "gov.cn",
    "moe.gov.cn",
    "ed.gov",
    "nist.gov",
    "nsf.gov",
}

_GOVERNMENT_MEDIA_PATH = re.compile(
    r"/(?:rmtzx|rmzx|rongmeiti(?:zhongxin)?|media[-_]?center)(?:/|$)",
    re.IGNORECASE,
)


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(part.strip() for part in parts if part is not None)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def canonicalize_source_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        domain = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.rstrip("/") or "/"
        if domain == "arxiv.org":
            match = re.match(r"/(?:abs|pdf|html)/([^/]+?)(?:\.pdf)?$", path, re.IGNORECASE)
            if match:
                arxiv_id = re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE)
                return f"https://arxiv.org/abs/{arxiv_id}"
        if domain in {"doi.org", "dx.doi.org"}:
            doi = unquote(path.lstrip("/")).lower()
            return f"https://doi.org/{doi}" if doi else "https://doi.org/"
        return urlunsplit((parsed.scheme.lower(), domain, path, parsed.query, ""))
    except ValueError:
        return raw


def _domain(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower().removeprefix("www.")
    except ValueError:
        return ""


def _domain_matches(domain: str, candidates: set[str]) -> bool:
    return any(domain == item or domain.endswith(f".{item}") for item in candidates)


def relevance_tokens(text: str) -> set[str]:
    """Return mixed-language lexical units for retrieval relevance checks."""
    lowered = (text or "").lower()
    tokens = set(re.findall(r"[a-z][a-z0-9_.+-]+|\d+(?:\.\d+)?", lowered))
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
        for size in (2, 3):
            if len(sequence) >= size:
                tokens.update(sequence[index : index + size] for index in range(len(sequence) - size + 1))
    return tokens


def source_relevance(query: str, text: str) -> tuple[float, int]:
    """Score lexical relevance and count discriminating query-term matches."""
    query_tokens = relevance_tokens(query)
    if not query_tokens:
        return 0.0, 0
    result_tokens = relevance_tokens(text)
    overlap = query_tokens & result_tokens
    discriminating = query_tokens - _RELEVANCE_STOPWORDS
    discriminating_overlap = discriminating & result_tokens
    broad_coverage = len(overlap) / len(query_tokens)
    focused_coverage = (
        len(discriminating_overlap) / len(discriminating)
        if discriminating
        else broad_coverage
    )
    # Sources normally cover one sub-question, not every term in a broad query.
    # Reward a small number of discriminating anchors while retaining query
    # coverage as a tie-breaker.
    anchor_strength = min(1.0, len(discriminating_overlap) / 3.0)
    score = min(
        1.0,
        anchor_strength * 0.65 + broad_coverage * 0.15 + focused_coverage * 0.20,
    )
    return score, len(discriminating_overlap)


def infer_source_year(url: str = "", *texts: str) -> str:
    """Infer a conservative publication year without confusing upload dates.

    A dated HTML path is usually the page publication date. For downloadable
    files, however, a parent directory often reflects a later migration or
    upload date, so only a year embedded in the filename is accepted.
    """
    upper_bound = datetime.now(timezone.utc).year + 1
    try:
        path = unquote(urlsplit(url or "").path)
    except ValueError:
        path = ""
    is_download = bool(
        re.search(r"\.(?:pdf|docx?|xlsx?|pptx?|zip|csv)$", path, re.IGNORECASE)
    )
    if not is_download:
        for raw in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", path):
            year = int(raw)
            if 1900 <= year <= upper_bound:
                return str(year)
    for value in texts:
        for raw in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", value or ""):
            year = int(raw)
            if 1900 <= year <= upper_bound:
                return str(year)
    filename = path.rsplit("/", 1)[-1]
    for raw in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", filename):
        year = int(raw)
        if 1900 <= year <= upper_bound:
            return str(year)
    if is_download:
        return ""
    return ""


def fulltext_matches_source(
    text: str,
    *,
    title: str = "",
    query: str = "",
) -> bool:
    """Reject successful HTTP fetches whose body is unrelated to the target.

    Redirects, anti-bot pages, and PDF viewer fallbacks can return substantial
    text while silently landing on a site homepage. When search context exists,
    require at least one discriminating title or retrieval-query anchor.
    """
    body = re.sub(r"\s+", " ", text or "").strip()
    if len(body) < 40:
        return False
    contexts = []
    clean_title = re.sub(r"https?://\S+", " ", title or "").strip()
    if clean_title:
        contexts.append(clean_title)
    if query:
        contexts.append(query)
    if not contexts:
        return True
    for context in contexts:
        relevance, anchor_hits = source_relevance(context, body)
        if anchor_hits > 0 and relevance >= 0.07:
            return True
        discriminating = relevance_tokens(context) - _RELEVANCE_STOPWORDS
        if not discriminating and relevance >= 0.20:
            return True
    return False


def source_quality(url: str, source_type: str = "web") -> tuple[float, bool, str]:
    """Classify a source using its registrable institutional domain.

    The checks intentionally cover exact public-sector roots such as ``gov.cn``
    as well as their subdomains. The previous suffix-only checks missed the
    Chinese central-government portal and most ``*.edu.cn`` institutions.
    """
    domain = _domain(url)
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        path = ""
    is_academic = source_type == "paper" or _domain_matches(domain, _PRIMARY_DOMAINS)
    is_government = any(
        domain == suffix or domain.endswith(f".{suffix}")
        for suffix in ("gov", "gov.cn", "gov.uk", "gc.ca", "gouv.fr")
    ) or _domain_matches(domain, {"europa.eu"})
    is_education = not _domain_matches(domain, _SECONDARY_REPOSITORY_DOMAINS) and any(
        domain == suffix or domain.endswith(f".{suffix}")
        for suffix in ("edu", "edu.cn", "ac.cn", "ac.uk", "edu.au")
    )
    is_official_docs = (
        domain.startswith(("docs.", "developer.", "developers.", "platform."))
        or any(segment in path for segment in ("/docs/", "/documentation/", "/developers/"))
    )
    # A government domain establishes who hosts a page, not that every page is
    # a primary document. Local integrated-media sections republish reporting
    # and should not make a search snippet sufficient to prove a factual claim.
    if is_government and _GOVERNMENT_MEDIA_PATH.search(path):
        return 0.75, False, domain
    if _domain_matches(domain, _CANONICAL_INSTITUTION_DOMAINS):
        return 0.95, True, domain
    if is_academic or is_government:
        return 0.9, True, domain
    if is_official_docs:
        return 0.85, True, domain
    if is_education:
        return 0.85, True, domain
    if _domain_matches(domain, _HIGH_QUALITY_DOMAINS):
        return 0.75, False, domain
    if _domain_matches(domain, _SECONDARY_REPOSITORY_DOMAINS):
        return 0.65, False, domain
    if domain.endswith("wikipedia.org"):
        return 0.6, False, domain
    if _domain_matches(domain, _LOW_QUALITY_DOMAINS):
        return 0.35, False, domain
    return (0.5 if domain else 0.35), False, domain


# Backward-compatible private alias for older integrations.
_source_quality = source_quality


class EvidenceStore:
    """In-memory claim graph backing store with deterministic JSON persistence."""

    def __init__(
        self,
        artifact_dir: str = "outputs/evidence",
        session_id: str = "",
        persist_enabled: bool = True,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.session_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id).strip("_")
        self.persist_enabled = persist_enabled
        self.query = ""
        self.sources: dict[str, SourceRecord] = {}
        self.evidence: dict[str, EvidenceChunk] = {}
        self._source_keys: dict[str, str] = {}

    def reset(self, query: str = "") -> None:
        self.query = query
        self.sources.clear()
        self.evidence.clear()
        self._source_keys.clear()

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        *,
        persist_enabled: bool = False,
    ) -> "EvidenceStore":
        """Load a persisted evidence graph without silently rewriting its IDs."""
        artifact_path = Path(path)
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not str(payload.get("schema_version", "")).startswith("1."):
            raise ValueError(f"Unsupported evidence artifact: {artifact_path}")

        store = cls(
            artifact_dir=str(artifact_path.parent),
            session_id=artifact_path.stem,
            persist_enabled=persist_enabled,
        )
        store.query = str(payload.get("query", ""))

        for raw_source in payload.get("sources", []):
            if not isinstance(raw_source, dict) or not raw_source.get("source_id"):
                raise ValueError(f"Invalid source record in {artifact_path}")
            source = SourceRecord(
                source_id=str(raw_source["source_id"]),
                url=str(raw_source.get("url", "")),
                title=str(raw_source.get("title", "")),
                source_type=str(raw_source.get("source_type", "web")),
                task_ids=[str(value) for value in raw_source.get("task_ids", [])],
                authors=str(raw_source.get("authors", "")),
                year=str(raw_source.get("year", "")),
                publisher=str(raw_source.get("publisher", "")),
                quality_score=float(raw_source.get("quality_score", 0.5)),
                is_primary=bool(raw_source.get("is_primary", False)),
                content_hash=str(raw_source.get("content_hash", "")),
                metadata=dict(raw_source.get("metadata") or {}),
            )
            if source.source_id in store.sources:
                raise ValueError(f"Duplicate source ID in {artifact_path}: {source.source_id}")
            store.sources[source.source_id] = source
            key = canonicalize_source_url(source.url) or source.title.strip().lower()
            if key:
                store._source_keys.setdefault(key, source.source_id)

        for raw_chunk in payload.get("evidence", []):
            if not isinstance(raw_chunk, dict) or not raw_chunk.get("evidence_id"):
                raise ValueError(f"Invalid evidence record in {artifact_path}")
            source_id = str(raw_chunk.get("source_id", ""))
            if source_id not in store.sources:
                raise ValueError(f"Evidence references an unknown source: {source_id}")
            text = str(raw_chunk.get("text", ""))
            actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            expected_hash = str(raw_chunk.get("content_hash", ""))
            if expected_hash and expected_hash != actual_hash:
                raise ValueError(
                    f"Evidence content hash mismatch in {artifact_path}: "
                    f"{raw_chunk['evidence_id']}"
                )
            chunk = EvidenceChunk(
                evidence_id=str(raw_chunk["evidence_id"]),
                source_id=source_id,
                text=text,
                kind=EvidenceKind(str(raw_chunk.get("kind", "search_snippet"))),
                task_id=str(raw_chunk.get("task_id", "")),
                locator=str(raw_chunk.get("locator", "")),
                content_hash=expected_hash or actual_hash,
            )
            if chunk.evidence_id in store.evidence:
                raise ValueError(f"Duplicate evidence ID in {artifact_path}: {chunk.evidence_id}")
            store.evidence[chunk.evidence_id] = chunk

        return store

    def ingest_results(self, results: Iterable[AgentResult]) -> None:
        for result in results:
            for step in result.trajectory:
                if step.get("role") != "tool":
                    continue
                name = str(step.get("name", ""))
                payload = step.get("result")
                arguments = step.get("arguments") or {}
                if name == "web_search" and isinstance(payload, dict):
                    self._ingest_web_search(result.task_id, payload)
                elif name == "arxiv_reader" and isinstance(payload, dict):
                    self._ingest_papers(result.task_id, payload)
                elif name == "browser" and isinstance(payload, str):
                    self._ingest_fulltext(result.task_id, arguments.get("url", ""), payload)
                elif name == "file_reader" and payload:
                    path = str(arguments.get("path") or arguments.get("file_path") or "")
                    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
                    self._ingest_file(result.task_id, path, text)

    def _ingest_web_search(self, task_id: str, payload: dict[str, Any]) -> None:
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            source = self.upsert_source(
                url=str(item.get("url", "")),
                title=str(item.get("title", "")),
                source_type="web",
                task_id=task_id,
                metadata={
                    "search_backend": payload.get("source", ""),
                    "retrieval_query": payload.get("query", ""),
                    "retrieval_relevance": item.get("_relevance_score", 0.0),
                },
            )
            snippet = str(item.get("snippet", "")).strip()
            if snippet:
                self.add_evidence(source.source_id, snippet, EvidenceKind.SEARCH_SNIPPET, task_id)

    def _ingest_papers(self, task_id: str, payload: dict[str, Any]) -> None:
        for paper in payload.get("papers", []):
            if not isinstance(paper, dict):
                continue
            authors = paper.get("authors", "")
            if isinstance(authors, list):
                authors = ", ".join(str(author) for author in authors if author)
            year = str(paper.get("published") or paper.get("year") or "")[:4]
            source = self.upsert_source(
                url=str(paper.get("pdf_url") or paper.get("url") or ""),
                title=str(paper.get("title", "")),
                source_type="paper",
                task_id=task_id,
                authors=str(authors),
                year=year,
                publisher=str(paper.get("publisher", "")),
                metadata={
                    "paper_id": paper.get("id", ""),
                    "citation_count": paper.get("citation_count"),
                    "provider": paper.get("source") or payload.get("source", ""),
                    "retrieval_query": payload.get("query", ""),
                    "retrieval_relevance": paper.get("_relevance_score", 0.0),
                },
            )
            abstract = str(paper.get("summary", "")).strip()
            if abstract:
                self.add_evidence(source.source_id, abstract, EvidenceKind.ABSTRACT, task_id, "abstract")

    def _ingest_fulltext(self, task_id: str, url: str, text: str) -> None:
        normalized = text.lstrip().lower()
        if not text.strip() or normalized.startswith(
            ("error", "[browser error]", "[browser warning]")
        ):
            return
        evidence_kind = EvidenceKind.FULL_TEXT
        if normalized.startswith("[abstract_only]"):
            text = re.sub(r"^\s*\[ABSTRACT_ONLY\]\s*", "", text, flags=re.IGNORECASE)
            evidence_kind = EvidenceKind.ABSTRACT
        source = self.upsert_source(url=url, source_type="web", task_id=task_id)
        retrieval_query = str(source.metadata.get("retrieval_query", ""))
        if not fulltext_matches_source(
            text,
            title=source.title,
            query=retrieval_query,
        ):
            source.metadata["fulltext_rejected"] = "content_mismatch"
            return
        source.metadata.pop("fulltext_rejected", None)
        source.content_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        for index, chunk in enumerate(self._chunk_text(text)):
            self.add_evidence(
                source.source_id,
                chunk,
                evidence_kind,
                task_id,
                locator=f"chunk-{index + 1}",
            )

    def _ingest_file(self, task_id: str, path: str, text: str) -> None:
        source = self.upsert_source(
            url=path,
            title=Path(path).name if path else "Local file",
            source_type="file",
            task_id=task_id,
        )
        source.content_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        for index, chunk in enumerate(self._chunk_text(text)):
            self.add_evidence(source.source_id, chunk, EvidenceKind.FILE, task_id, f"chunk-{index + 1}")

    def upsert_source(
        self,
        *,
        url: str = "",
        title: str = "",
        source_type: str = "web",
        task_id: str = "",
        authors: str = "",
        year: str = "",
        publisher: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SourceRecord:
        normalized_url = canonicalize_source_url(url)
        year = year or infer_source_year(normalized_url, title)
        quality, primary, host_domain = source_quality(normalized_url, source_type)
        source_metadata = {
            key: value for key, value in (metadata or {}).items() if value not in (None, "")
        }
        if host_domain:
            source_metadata.setdefault("host_domain", host_domain)
        key = normalized_url or title.strip().lower()
        source_id = self._source_keys.get(key) if key else None
        if source_id and source_id in self.sources:
            source = self.sources[source_id]
            source.title = source.title or title
            source.authors = source.authors or authors
            source.year = source.year or year
            source.publisher = source.publisher or publisher.strip()
            if task_id and task_id not in source.task_ids:
                source.task_ids.append(task_id)
            if source_type == "paper":
                source.source_type = source_type
                source.quality_score = max(source.quality_score, quality)
                source.is_primary = source.is_primary or primary
            if source_metadata:
                cleaned_metadata = dict(source_metadata)
                old_relevance = float(source.metadata.get("retrieval_relevance", 0.0) or 0.0)
                new_relevance = float(cleaned_metadata.get("retrieval_relevance", 0.0) or 0.0)
                if new_relevance < old_relevance:
                    cleaned_metadata.pop("retrieval_relevance", None)
                    cleaned_metadata.pop("retrieval_query", None)
                source.metadata.update(cleaned_metadata)
            return source

        source_id = _stable_id("src", normalized_url, title.lower())
        source = SourceRecord(
            source_id=source_id,
            url=normalized_url,
            title=title.strip(),
            source_type=source_type,
            task_ids=[task_id] if task_id else [],
            authors=authors,
            year=year,
            publisher=publisher.strip(),
            quality_score=quality,
            is_primary=primary,
            metadata=source_metadata,
        )
        self.sources[source_id] = source
        if key:
            self._source_keys[key] = source_id
        return source

    def add_evidence(
        self,
        source_id: str,
        text: str,
        kind: EvidenceKind,
        task_id: str = "",
        locator: str = "",
    ) -> EvidenceChunk:
        cleaned = re.sub(r"\s+", " ", text).strip()
        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        evidence_id = _stable_id("ev", source_id, content_hash)
        if evidence_id not in self.evidence:
            self.evidence[evidence_id] = EvidenceChunk(
                evidence_id=evidence_id,
                source_id=source_id,
                text=cleaned,
                kind=kind,
                task_id=task_id,
                locator=locator,
                content_hash=content_hash,
            )
        else:
            existing = self.evidence[evidence_id]
            strength = {
                EvidenceKind.SEARCH_SNIPPET: 0,
                EvidenceKind.ABSTRACT: 1,
                EvidenceKind.FULL_TEXT: 2,
                EvidenceKind.FILE: 3,
            }
            if strength[kind] > strength[existing.kind]:
                existing.kind = kind
                existing.task_id = task_id or existing.task_id
                existing.locator = locator or existing.locator
        return self.evidence[evidence_id]

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 1400) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        chunks: list[str] = []
        buffer = ""
        for paragraph in paragraphs:
            if len(paragraph) > max_chars:
                sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", paragraph) if s.strip()]
            else:
                sentences = [paragraph]
            for sentence in sentences:
                candidate = f"{buffer}\n{sentence}".strip()
                if buffer and len(candidate) > max_chars:
                    chunks.append(buffer)
                    buffer = sentence[:max_chars]
                else:
                    buffer = candidate[:max_chars]
        if buffer:
            chunks.append(buffer)
        return chunks[:20]

    def source_list(self) -> list[SourceRecord]:
        return list(self.sources.values())

    def source_ids_for_citations(self, indices: list[int]) -> list[str]:
        ordered = self.source_list()
        return [ordered[index - 1].source_id for index in indices if 0 < index <= len(ordered)]

    def to_source_dicts(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        first_evidence: dict[str, str] = {}
        evidence_by_source: dict[str, list[EvidenceChunk]] = {}
        evidence_kinds: dict[str, set[str]] = {}
        evidence_counts: dict[str, int] = {}
        for chunk in self.evidence.values():
            first_evidence.setdefault(chunk.source_id, chunk.text[:240])
            evidence_by_source.setdefault(chunk.source_id, []).append(chunk)
            evidence_kinds.setdefault(chunk.source_id, set()).add(chunk.kind.value)
            evidence_counts[chunk.source_id] = evidence_counts.get(chunk.source_id, 0) + 1
        for source in self.source_list():
            data = source.to_dict()
            data["snippet"] = first_evidence.get(source.source_id, "")
            chunks = evidence_by_source.get(source.source_id, [])
            if chunks:
                query = str(source.metadata.get("retrieval_query", ""))
                kind_strength = {
                    EvidenceKind.SEARCH_SNIPPET: 0,
                    EvidenceKind.ABSTRACT: 1,
                    EvidenceKind.FULL_TEXT: 2,
                    EvidenceKind.FILE: 3,
                }
                best = max(
                    chunks,
                    key=lambda chunk: (
                        kind_strength[chunk.kind],
                        source_relevance(
                            query or source.title,
                            chunk.text,
                        )[0],
                    ),
                )
                data["evidence_excerpt"] = best.text[:500]
                data["evidence_kind"] = best.kind.value
            kinds = sorted(evidence_kinds.get(source.source_id, set()))
            data["evidence_kinds"] = kinds
            data["evidence_count"] = evidence_counts.get(source.source_id, 0)
            data["has_fulltext"] = bool(
                {EvidenceKind.FULL_TEXT.value, EvidenceKind.FILE.value}.intersection(kinds)
            )
            result.append(data)
        return result

    def fulltext_source_ids(self) -> set[str]:
        return {
            chunk.source_id
            for chunk in self.evidence.values()
            if chunk.kind in (EvidenceKind.FULL_TEXT, EvidenceKind.FILE)
        }

    def persist(
        self,
        audit: EvidenceAudit | None = None,
        query: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not self.persist_enabled:
            return ""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = self.session_id or timestamp
        path = self.artifact_dir / f"evidence_{stem}_{timestamp}.json"
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "query": query or self.query,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": [source.to_dict() for source in self.source_list()],
            "evidence": [chunk.to_dict() for chunk in self.evidence.values()],
        }
        if audit is not None:
            payload["audit"] = audit.to_dict(self.evidence)
        if metadata:
            payload["run_metadata"] = metadata
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
