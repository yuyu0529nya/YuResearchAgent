"""Framework-neutral research run lifecycle and artifact access."""

from __future__ import annotations

import asyncio
import copy
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from src.core.runner import (
    initialize_modules,
    load_config,
    run_research_with_metadata,
    save_report,
)
from src.runtime import RunController, RunStore
from src.runtime.presentation import apply_run_event, new_run_view

_MODULE_NAMES = [
    "solver",
    "planner",
    "summarizer",
    "judge",
    "red_agent",
    "blue_agent",
    "compressor",
]


@dataclass(frozen=True)
class RunUpdate:
    run_id: str
    elapsed_seconds: float
    backend: str
    adversarial: bool
    view: dict[str, Any]
    report: str = ""
    evidence: dict[str, Any] | None = None
    download_path: str | None = None
    error: str = ""
    terminal: bool = False


@dataclass(frozen=True)
class HistoryArtifact:
    row: dict[str, Any]
    events: list[dict[str, Any]]
    report: str
    evidence: dict[str, Any]
    download_path: str | None


class ResearchRunService:
    """Own research threads and persistence without importing a UI framework."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        store: RunStore | None = None,
        controller: RunController | None = None,
        config_loader: Callable[..., dict[str, Any]] = load_config,
        module_initializer: Callable[..., dict[str, Any]] = initialize_modules,
        research_runner: Callable[..., Any] = run_research_with_metadata,
        report_saver: Callable[..., str] = save_report,
        poll_interval: float = 0.35,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.report_root = (self.project_root / "outputs" / "reports").resolve()
        self.evidence_root = (self.project_root / "outputs" / "evidence").resolve()
        self.store = store or RunStore(self.project_root / "outputs" / "runs" / "runs.db")
        self.controller = controller or RunController(self.store)
        self.config_loader = config_loader
        self.module_initializer = module_initializer
        self.research_runner = research_runner
        self.report_saver = report_saver
        self.poll_interval = max(0.01, float(poll_interval))

    def recover_interrupted(self) -> int:
        return self.store.recover_interrupted()

    def request_cancel(self, run_id: str, reason: str = "Cancelled from the Web UI.") -> str:
        if self.controller.cancel(str(run_id or ""), reason):
            return "requested"
        if run_id and self.store.get_run(run_id):
            return "terminal"
        return "missing"

    def _safe_artifact_path(self, raw_path: str, allowed_root: Path) -> Path | None:
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not resolved.is_file() or not resolved.is_relative_to(allowed_root):
            return None
        return resolved

    def _load_evidence(self, raw_path: str) -> dict[str, Any]:
        path = self._safe_artifact_path(raw_path, self.evidence_root)
        if path is None:
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        audit = dict(payload.get("audit") or {})
        audit["sources"] = list(payload.get("sources") or [])
        audit["artifact"] = str(path)
        audit["revision"] = dict(payload.get("run_metadata", {}).get("evidence_revision") or {})
        return audit

    def list_history(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.list_runs(limit=limit)

    def load_history(self, run_id: str) -> HistoryArtifact | None:
        row = self.store.get_run(str(run_id or ""))
        if not row:
            return None
        events = self.store.get_events(row["run_id"])
        report_path = self._safe_artifact_path(str(row.get("report_path") or ""), self.report_root)
        try:
            report = report_path.read_text(encoding="utf-8") if report_path else ""
        except OSError:
            report = ""
        evidence = self._load_evidence(str(row.get("evidence_path") or ""))
        if not evidence:
            evidence = dict(row.get("metadata", {}).get("evidence_summary") or {})
            evidence["source_count"] = int(row.get("source_count", 0) or 0)
            evidence["revision"] = dict(row.get("metadata", {}).get("evidence_revision") or {})
        return HistoryArtifact(
            row=row,
            events=events,
            report=report,
            evidence=evidence,
            download_path=str(report_path) if report_path else None,
        )

    def stream(self, query: str, backend: str, adversarial: bool) -> Iterator[RunUpdate]:
        query = str(query or "").strip()
        if not query:
            raise ValueError("Research query cannot be empty")

        config = self.config_loader()
        config.setdefault("model", {})["backend"] = backend
        config["model"]["backend_mapping"] = {module: backend for module in _MODULE_NAMES}
        config.setdefault("adversarial", {})["enabled"] = bool(adversarial)

        handle = self.controller.create_run(
            query=query,
            backend=backend,
            adversarial=bool(adversarial),
        )
        holder: dict[str, Any] = {}

        def _worker() -> None:
            try:
                modules = self.module_initializer(
                    config,
                    session_id=handle.run_id,
                    run_id=handle.run_id,
                    event_sink=self.controller.event_sink,
                    cancellation_token=handle.token,
                )
                report, metadata = asyncio.run(self.research_runner(query, config, modules))
                evidence = modules["orchestrator"].get_evidence_snapshot()
                report_path = self.report_saver(
                    report,
                    query,
                    output_dir=str(self.report_root),
                )
                status = str(metadata.get("run_status") or "complete")
                self.store.complete_run(
                    handle.run_id,
                    status=status,
                    report_path=report_path,
                    evidence_path=str(metadata.get("evidence_artifact") or ""),
                    metadata=metadata,
                )
                holder.update(
                    report=report,
                    metadata=metadata,
                    evidence=evidence,
                    download=report_path,
                )
            except Exception as exc:  # noqa: BLE001 - terminal failure belongs in the ledger
                error = f"{type(exc).__name__}: {exc}"
                holder["error"] = error
                self.store.fail_run(handle.run_id, error)
            finally:
                self.controller.finish(handle.run_id)

        worker = threading.Thread(
            target=_worker,
            name=f"research-{handle.run_id}",
            daemon=True,
        )
        started = time.monotonic()
        view = new_run_view()
        worker.start()
        yield RunUpdate(
            run_id=handle.run_id,
            elapsed_seconds=0.0,
            backend=backend,
            adversarial=bool(adversarial),
            view=copy.deepcopy(view),
        )

        while worker.is_alive() or not handle.events.empty():
            changed = False
            while True:
                try:
                    event = handle.events.get_nowait()
                except queue.Empty:
                    break
                apply_run_event(view, event)
                changed = True
            if handle.token.is_cancelled and view.get("status") == "running":
                view["status"] = "cancelling"
                changed = True
            if changed or worker.is_alive():
                yield RunUpdate(
                    run_id=handle.run_id,
                    elapsed_seconds=time.monotonic() - started,
                    backend=backend,
                    adversarial=bool(adversarial),
                    view=copy.deepcopy(view),
                    evidence=copy.deepcopy(view.get("evidence") or {}),
                )
            if worker.is_alive():
                time.sleep(self.poll_interval)

        worker.join(timeout=1.0)
        elapsed = time.monotonic() - started
        if holder.get("error"):
            view["status"] = "failed"
            yield RunUpdate(
                run_id=handle.run_id,
                elapsed_seconds=elapsed,
                backend=backend,
                adversarial=bool(adversarial),
                view=copy.deepcopy(view),
                evidence=copy.deepcopy(view.get("evidence") or {}),
                error=str(holder["error"]),
                terminal=True,
            )
            return

        metadata = dict(holder.get("metadata") or {})
        view["status"] = str(metadata.get("run_status") or view.get("status") or "complete")
        final_evidence = holder.get("evidence")
        if isinstance(final_evidence, dict) and final_evidence:
            view["evidence"] = final_evidence
        yield RunUpdate(
            run_id=handle.run_id,
            elapsed_seconds=elapsed,
            backend=backend,
            adversarial=bool(adversarial),
            view=copy.deepcopy(view),
            report=str(holder.get("report") or ""),
            evidence=copy.deepcopy(view.get("evidence") or {}),
            download_path=str(holder.get("download") or "") or None,
            terminal=True,
        )


__all__ = ["HistoryArtifact", "ResearchRunService", "RunUpdate"]
