"""Coordination primitives for local, memory-heavy NextEcho tasks.

The ASR, diarization and local LLM stages all compete for the same small
machine memory budget.  This module intentionally contains no knowledge of
Flask or a particular worker implementation: callers acquire one lease for
the complete heavy workflow and use the lease heartbeat while they work.

The coordinator is process-local.  That is deliberate for the desktop app;
the job heartbeat is the source of truth exposed by the API, rather than
filesystem mtimes which can be stale or change for unrelated reasons.
"""

from __future__ import annotations

import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Optional


DEFAULT_HEARTBEAT_SECONDS = 15


@dataclass(frozen=True)
class CoordinatorSnapshot:
    """A small, JSON-friendly view of the current heavy-task lease."""

    busy: bool
    active_job_id: str | None
    active_stage: str | None
    queue_depth: int
    heartbeat_seconds: int
    heartbeat_at: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "busy": self.busy,
            "active_job_id": self.active_job_id,
            "active_stage": self.active_stage,
            "queue_depth": self.queue_depth,
            "heartbeat_seconds": self.heartbeat_seconds,
            "heartbeat_at": self.heartbeat_at,
        }


class HeavyTaskLease(AbstractContextManager["HeavyTaskLease"]):
    """A lease held by one job until its complete heavy workflow finishes."""

    def __init__(self, coordinator: "HeavyTaskCoordinator", job_id: str, stage: str = "model") -> None:
        self.coordinator = coordinator
        self.job_id = job_id
        self.stage = stage
        self._released = False

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.coordinator.set_stage(self.job_id, stage)

    def heartbeat(self, *, stage: str | None = None) -> None:
        if stage:
            self.set_stage(stage)
        self.coordinator.heartbeat(self.job_id, stage=stage or self.stage)

    def release(self) -> None:
        if not self._released:
            self._released = True
            self.coordinator.release(self.job_id)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
        return None


class HeavyTaskCoordinator:
    """Serialize all local model work and expose a regular heartbeat.

    ``on_wait`` is called once immediately and then at most once per
    ``heartbeat_seconds`` while a job is waiting.  ``on_heartbeat`` is called
    by the lease's background heartbeat thread once every interval.  Callback
    exceptions are intentionally swallowed so observability cannot interrupt
    a transcription task.
    """

    def __init__(self, heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS) -> None:
        self.heartbeat_seconds = max(1, int(heartbeat_seconds))
        self._condition = threading.Condition(threading.RLock())
        self._active_job_id: str | None = None
        self._active_stage: str | None = None
        self._active_heartbeat_at: float | None = None
        self._waiters: list[str] = []

    def acquire(
        self,
        job_id: str,
        *,
        stage: str = "model",
        on_wait: Optional[Callable[[int], None]] = None,
        timeout: float | None = None,
    ) -> HeavyTaskLease:
        """Wait for and acquire the single heavy-task lease.

        ``timeout`` is primarily useful to callers/tests that want bounded
        shutdown behavior.  Normal desktop jobs wait indefinitely.
        """

        started = time.monotonic()
        with self._condition:
            if self._active_job_id != job_id and job_id not in self._waiters:
                self._waiters.append(job_id)
            wait_notified = False
            while self._active_job_id is not None and self._active_job_id != job_id:
                position = self._waiters.index(job_id) + 1 if job_id in self._waiters else len(self._waiters)
                if on_wait is not None and not wait_notified:
                    self._safe_callback(on_wait, position)
                    wait_notified = True
                remaining = None
                if timeout is not None:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        if job_id in self._waiters:
                            self._waiters.remove(job_id)
                        raise TimeoutError("timed out waiting for the heavy-task coordinator")
                self._condition.wait(timeout=min(self.heartbeat_seconds, remaining) if remaining is not None else self.heartbeat_seconds)
                if on_wait is not None:
                    position = self._waiters.index(job_id) + 1 if job_id in self._waiters else len(self._waiters)
                    self._safe_callback(on_wait, position)

            if job_id in self._waiters:
                self._waiters.remove(job_id)
            self._active_job_id = job_id
            self._active_stage = stage
            self._active_heartbeat_at = time.time()
            self._condition.notify_all()
            return HeavyTaskLease(self, job_id, stage)

    def set_stage(self, job_id: str, stage: str) -> None:
        with self._condition:
            if self._active_job_id != job_id:
                return
            self._active_stage = str(stage)
            self._active_heartbeat_at = time.time()
            self._condition.notify_all()

    def heartbeat(self, job_id: str, *, stage: str | None = None) -> bool:
        """Record a lease heartbeat; return ``False`` for a stale owner."""

        with self._condition:
            if self._active_job_id != job_id:
                return False
            if stage:
                self._active_stage = str(stage)
            self._active_heartbeat_at = time.time()
            self._condition.notify_all()
            return True

    def release(self, job_id: str) -> None:
        with self._condition:
            if self._active_job_id == job_id:
                self._active_job_id = None
                self._active_stage = None
                self._active_heartbeat_at = None
                self._condition.notify_all()
            if job_id in self._waiters:
                self._waiters.remove(job_id)

    def owns(self, job_id: str) -> bool:
        with self._condition:
            return self._active_job_id == job_id

    def snapshot(self) -> CoordinatorSnapshot:
        with self._condition:
            return CoordinatorSnapshot(
                busy=self._active_job_id is not None,
                active_job_id=self._active_job_id,
                active_stage=self._active_stage,
                queue_depth=len(self._waiters),
                heartbeat_seconds=self.heartbeat_seconds,
                heartbeat_at=self._active_heartbeat_at,
            )

    def start_heartbeat(
        self,
        lease: HeavyTaskLease,
        callback: Optional[Callable[[CoordinatorSnapshot], None]] = None,
    ) -> threading.Event:
        """Start a daemon heartbeat loop and return its stop event."""

        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(self.heartbeat_seconds):
                if not self.owns(lease.job_id):
                    return
                lease.heartbeat()
                if callback is not None:
                    self._safe_callback(callback, self.snapshot())

        threading.Thread(target=loop, name=f"nextecho-heartbeat-{lease.job_id[:8]}", daemon=True).start()
        return stop

    @staticmethod
    def _safe_callback(callback: Callable, *args) -> None:
        try:
            callback(*args)
        except Exception:
            # Job progress reporting is best effort and must never release a
            # lease or turn a successful local task into a failed one.
            return


__all__ = [
    "DEFAULT_HEARTBEAT_SECONDS",
    "CoordinatorSnapshot",
    "HeavyTaskCoordinator",
    "HeavyTaskLease",
]
