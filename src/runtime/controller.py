"""Thread-safe coordination for active research runs."""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .events import CancellationToken, RunEvent
from .run_store import RunStore


@dataclass
class RunHandle:
    run_id: str
    token: CancellationToken
    events: "queue.Queue[RunEvent]"
    created_at: float


class RunController:
    """Own active cancellation tokens while RunStore owns durable history."""

    def __init__(self, store: RunStore) -> None:
        self.store = store
        self._lock = threading.RLock()
        self._active: dict[str, RunHandle] = {}

    def create_run(self, *, query: str, backend: str, adversarial: bool) -> RunHandle:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"web_{stamp}_{uuid.uuid4().hex[:8]}"
        handle = RunHandle(
            run_id=run_id,
            token=CancellationToken(),
            events=queue.Queue(),
            created_at=time.time(),
        )
        self.store.create_run(
            run_id,
            query=query,
            backend=backend,
            adversarial=adversarial,
        )
        with self._lock:
            self._active[run_id] = handle
        return handle

    def event_sink(self, event: RunEvent) -> None:
        with self._lock:
            handle = self._active.get(event.run_id)
        try:
            self.store.record_event(event)
        finally:
            if handle is not None:
                handle.events.put_nowait(event)

    def cancel(self, run_id: str, reason: str = "Cancelled by user.") -> bool:
        if not run_id:
            return False
        with self._lock:
            handle = self._active.get(run_id)
        if handle is None:
            return False
        requested = handle.token.request(reason)
        if requested:
            self.store.mark_cancelling(run_id)
        return requested

    def finish(self, run_id: str) -> None:
        with self._lock:
            self._active.pop(run_id, None)

    def get(self, run_id: str) -> RunHandle | None:
        with self._lock:
            return self._active.get(run_id)

    def active_run_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._active)


__all__ = ["RunController", "RunHandle"]
