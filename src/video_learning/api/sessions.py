"""Bounded in-memory report sessions: one uploaded artifact, one retained search index.

No database, no disk, no background sweeper. A session holds the ``SearchService`` built
at upload time, so N queries against one report cost one index — which is exactly the
shape :class:`~video_learning.services.search_service.SearchService` was written for.

Expiration is evaluated when a request touches the store (``get`` and ``create`` both
sweep). That is sufficient for a single-user local API and keeps every lifetime rule in
one place; the trade-off is that an expired session nobody asks about stays resident
until the next request or an eviction. ``MAX_SESSIONS`` bounds that residue, so the
store cannot grow without traffic.

**When every slot is in use, the least-recently-used session is evicted** and the upload
still succeeds. A single-user MVP has no reason to refuse a legitimate new report, and
the evicted client's next search returns 404 with a code that tells the UI to ask for the
file again. Eviction never cancels an in-flight query: the handler holds its own
reference to the session, and the index it reads is frozen.

``max_sessions * max_report_bytes`` is an **input bound, not a process-memory ceiling**.
The retained objects are the parsed JSON plus the derived evidence tuples and per-record
token lists, which are a multiple of the text size and depend on the artifact's content.
See ``tests/unit/test_api_sessions.py`` for measured sizes on representative artifacts.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from video_learning.api.errors import ApiError
from video_learning.services.search_service import SearchService

DEFAULT_MAX_SESSIONS = 8
DEFAULT_IDLE_TTL_SECONDS = 1800.0
DEFAULT_HARD_TTL_SECONDS = 28800.0


def _new_report_id() -> str:
    return secrets.token_urlsafe(16)


@dataclass
class ReportSession:
    """One uploaded report and the index built from it. Nothing here is persisted."""

    report_id: str
    service: SearchService
    filename: str
    segment_count: int
    bytes_in: int
    created_at: float
    last_seen: float

    def deadline(self, *, idle_ttl: float, hard_ttl: float) -> float:
        """Whichever comes first: the idle window from last use, or the hard cap."""
        return min(self.created_at + hard_ttl, self.last_seen + idle_ttl)


class ReportSessionStore:
    """A locked dict of sessions with sliding idle expiry, a hard cap, and LRU eviction."""

    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        idle_ttl_seconds: float = DEFAULT_IDLE_TTL_SECONDS,
        hard_ttl_seconds: float = DEFAULT_HARD_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        new_id: Callable[[], str] = _new_report_id,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if idle_ttl_seconds <= 0 or hard_ttl_seconds <= 0:
            raise ValueError("session TTLs must be positive")
        self.max_sessions = max_sessions
        self.idle_ttl_seconds = idle_ttl_seconds
        self.hard_ttl_seconds = hard_ttl_seconds
        self.evicted_count = 0
        self.expired_count = 0
        self._clock = clock
        self._new_id = new_id
        self._lock = threading.Lock()
        self._sessions: dict[str, ReportSession] = {}

    def now(self) -> float:
        return self._clock()

    def expires_in(self, session: ReportSession) -> float:
        return max(
            0.0,
            session.deadline(
                idle_ttl=self.idle_ttl_seconds, hard_ttl=self.hard_ttl_seconds
            )
            - self.now(),
        )

    def purge_expired(self) -> int:
        """Drop every session past its deadline; return how many were dropped."""
        with self._lock:
            return self._purge_expired_locked(self.now())

    def _purge_expired_locked(self, now: float) -> int:
        expired = [
            report_id
            for report_id, session in self._sessions.items()
            if now
            >= session.deadline(
                idle_ttl=self.idle_ttl_seconds, hard_ttl=self.hard_ttl_seconds
            )
        ]
        for report_id in expired:
            del self._sessions[report_id]
        self.expired_count += len(expired)
        return len(expired)

    def create(
        self,
        *,
        service: SearchService,
        filename: str,
        segment_count: int,
        bytes_in: int,
    ) -> ReportSession:
        with self._lock:
            now = self.now()
            self._purge_expired_locked(now)
            if len(self._sessions) >= self.max_sessions:
                victim = min(self._sessions.values(), key=lambda item: item.last_seen)
                del self._sessions[victim.report_id]
                self.evicted_count += 1
            session = ReportSession(
                report_id=self._new_id(),
                service=service,
                filename=filename,
                segment_count=segment_count,
                bytes_in=bytes_in,
                created_at=now,
                last_seen=now,
            )
            self._sessions[session.report_id] = session
            return session

    def get(self, report_id: str) -> ReportSession:
        """Return a live session and renew its idle window, or refuse with 404."""
        with self._lock:
            now = self.now()
            session = self._sessions.get(report_id)
            if session is None:
                raise ApiError(
                    404, "unknown_report", "No report is loaded under that id; upload it again."
                )
            if now >= session.deadline(
                idle_ttl=self.idle_ttl_seconds, hard_ttl=self.hard_ttl_seconds
            ):
                del self._sessions[report_id]
                self.expired_count += 1
                raise ApiError(
                    404, "report_expired", "That report session expired; upload it again."
                )
            session.last_seen = now
            return session

    def __len__(self) -> int:
        return len(self._sessions)
