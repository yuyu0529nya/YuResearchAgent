"""Typed runtime events and cooperative cancellation primitives."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunEvent:
    """One ordered, serializable observation emitted by a research run."""

    run_id: str
    sequence: int
    kind: str
    state: str
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "kind": self.kind,
            "state": self.state,
            "message": self.message,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }


class CancellationToken:
    """Thread-safe cooperative cancellation shared by UI, orchestrator, and agents.

    Synchronous provider calls cannot be interrupted safely from another thread.
    Components therefore check this token before and after every model/tool call,
    preserve completed work, and stop before starting the next paid operation.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""
        self._requested_at: float | None = None

    def request(self, reason: str = "Cancelled by user.") -> bool:
        """Request cancellation and return True only for the first request."""
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = str(reason or "Cancelled by user.")[:500]
            self._requested_at = time.time()
            self._event.set()
            return True

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    @property
    def requested_at(self) -> float | None:
        with self._lock:
            return self._requested_at

    def snapshot(self) -> dict[str, Any]:
        return {
            "is_cancelled": self.is_cancelled,
            "reason": self.reason,
            "requested_at": self.requested_at,
        }


class UsageTracker:
    """Aggregate provider usage across every policy instance in one run."""

    _KEYS = (
        "api_calls",
        "failed_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage = {key: 0 for key in self._KEYS}

    def record(self, **increments: int) -> None:
        with self._lock:
            for key, value in increments.items():
                if key in self._usage:
                    self._usage[key] += max(0, int(value or 0))

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._usage)


__all__ = ["CancellationToken", "RunEvent", "UsageTracker"]
