"""Integrity-checked loading for committed, keyless demo artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ArtifactIntegrityError(ValueError):
    """Raised when a demo catalog or one of its artifacts is invalid."""


@dataclass(frozen=True)
class DemoArtifact:
    demo_id: str
    title: str
    query: str
    report: str
    report_path: str
    evidence: dict[str, Any]
    runtime: dict[str, Any]
    interpretation: str


class DemoCatalog:
    """Load immutable repository demos without requiring an API key."""

    SCHEMA = "research-demo-catalog-v1"

    def __init__(self, project_root: str | Path, catalog_path: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        candidate = Path(catalog_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        self.catalog_path = candidate.resolve()
        self._catalog = self._read_json(self.catalog_path)
        if self._catalog.get("artifact_schema") != self.SCHEMA:
            raise ArtifactIntegrityError(f"Unsupported demo catalog schema: {self._catalog.get('artifact_schema')!r}")
        demos = self._catalog.get("demos")
        if not isinstance(demos, list) or not demos:
            raise ArtifactIntegrityError("Demo catalog must contain at least one demo")
        self._entries: dict[str, dict[str, Any]] = {}
        for entry in demos:
            demo_id = str(entry.get("id", "")).strip() if isinstance(entry, dict) else ""
            if not demo_id or demo_id in self._entries:
                raise ArtifactIntegrityError("Demo IDs must be non-empty and unique")
            self._entries[demo_id] = entry

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(f"Cannot read JSON artifact {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ArtifactIntegrityError(f"JSON artifact must contain an object: {path}")
        return payload

    def _verified_path(self, metadata: dict[str, Any]) -> Path:
        raw_path = str(metadata.get("path", "")).strip()
        expected = str(metadata.get("sha256", "")).strip().lower()
        if not raw_path or len(expected) != 64:
            raise ArtifactIntegrityError("Artifact metadata requires path and SHA-256")
        candidate = (self.project_root / raw_path).resolve()
        if not candidate.is_relative_to(self.project_root) or not candidate.is_file():
            raise ArtifactIntegrityError(f"Artifact path is unavailable or unsafe: {raw_path}")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            raise ArtifactIntegrityError(f"Artifact hash mismatch: {raw_path}")
        return candidate

    def choices(self) -> list[tuple[str, str]]:
        return [(str(entry.get("title") or demo_id), demo_id) for demo_id, entry in self._entries.items()]

    @property
    def default_id(self) -> str:
        preferred = str(self._catalog.get("default_demo", ""))
        return preferred if preferred in self._entries else next(iter(self._entries))

    def load(self, demo_id: str) -> DemoArtifact:
        if demo_id not in self._entries:
            raise KeyError(f"Unknown demo artifact: {demo_id}")
        entry = self._entries[demo_id]
        report_path = self._verified_path(dict(entry.get("report") or {}))
        evidence_path = self._verified_path(dict(entry.get("evidence") or {}))
        audit_path = self._verified_path(dict(entry.get("audit") or {}))

        report = report_path.read_text(encoding="utf-8")
        evidence_payload = self._read_json(evidence_path)
        audit_payload = self._read_json(audit_path)
        evidence = dict(audit_payload.get("audit") or evidence_payload.get("audit") or {})
        evidence["sources"] = list(evidence_payload.get("sources") or [])
        evidence["artifact"] = str(evidence_path)

        return DemoArtifact(
            demo_id=demo_id,
            title=str(entry.get("title") or demo_id),
            query=str(entry.get("query") or ""),
            report=report,
            report_path=str(report_path),
            evidence=evidence,
            runtime=dict(entry.get("runtime") or {}),
            interpretation=str(entry.get("interpretation") or ""),
        )


__all__ = ["ArtifactIntegrityError", "DemoArtifact", "DemoCatalog"]
