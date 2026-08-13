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
}
_HIGH_QUALITY_DOMAINS = {
    "docs.python.org",
    "github.com",
    "semanticscholar.org",
    "openalex.org",
    "reuters.com",
    "apnews.com",
}


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


def _source_quality(url: str, source_type: str) -> tuple[float, bool, str]:
    domain = _domain(url)
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        path = ""
    is_academic = source_type == "paper" or _domain_matches(domain, _PRIMARY_DOMAINS)
    is_government = domain.endswith(".gov") or domain.endswith(".gov.cn")
    is_education = domain.endswith(".edu") or domain.endswith(".ac.uk")
    is_official_docs = (
        domain.startswith(("docs.", "developer.", "developers.", "platform."))
        or any(segment in path for segment in ("/docs/", "/documentation/", "/developers/"))
    )
    if is_academic or is_government:
        return 0.9, True, domain
    if is_official_docs:
        return 0.85, True, domain
    if is_education:
        return 0.85, True, domain
    if domain == "github.com":
        return 0.8, True, domain
    if _domain_matches(domain, _HIGH_QUALITY_DOMAINS):
        return 0.75, False, domain
    if domain.endswith("wikipedia.org"):
        return 0.6, False, domain
    return (0.5 if domain else 0.35), False, domain


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
                metadata={"search_backend": payload.get("source", "")},
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
                metadata={
                    "paper_id": paper.get("id", ""),
                    "citation_count": paper.get("citation_count"),
                    "provider": paper.get("source") or payload.get("source", ""),
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
        metadata: dict[str, Any] | None = None,
    ) -> SourceRecord:
        normalized_url = canonicalize_source_url(url)
        key = normalized_url or title.strip().lower()
        source_id = self._source_keys.get(key) if key else None
        if source_id and source_id in self.sources:
            source = self.sources[source_id]
            source.title = source.title or title
            source.authors = source.authors or authors
            source.year = source.year or year
            if task_id and task_id not in source.task_ids:
                source.task_ids.append(task_id)
            if source_type == "paper":
                source.source_type = source_type
                quality, primary, publisher = _source_quality(normalized_url, source_type)
                source.quality_score = max(source.quality_score, quality)
                source.is_primary = source.is_primary or primary
                source.publisher = source.publisher or publisher
            if metadata:
                source.metadata.update({k: v for k, v in metadata.items() if v not in (None, "")})
            return source

        quality, primary, publisher = _source_quality(normalized_url, source_type)
        source_id = _stable_id("src", normalized_url, title.lower())
        source = SourceRecord(
            source_id=source_id,
            url=normalized_url,
            title=title.strip(),
            source_type=source_type,
            task_ids=[task_id] if task_id else [],
            authors=authors,
            year=year,
            publisher=publisher,
            quality_score=quality,
            is_primary=primary,
            metadata={k: v for k, v in (metadata or {}).items() if v not in (None, "")},
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
        for chunk in self.evidence.values():
            first_evidence.setdefault(chunk.source_id, chunk.text[:240])
        for source in self.source_list():
            data = source.to_dict()
            data["snippet"] = first_evidence.get(source.source_id, "")
            result.append(data)
        return result

    def fulltext_source_ids(self) -> set[str]:
        return {
            chunk.source_id
            for chunk in self.evidence.values()
            if chunk.kind in (EvidenceKind.FULL_TEXT, EvidenceKind.FILE)
        }

    def persist(self, audit: EvidenceAudit | None = None, query: str = "") -> str:
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
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
