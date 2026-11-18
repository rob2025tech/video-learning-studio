"""Session lifetime rules: bounded count, sliding idle expiry, hard cap, LRU eviction.

Every test drives an injected clock, so nothing sleeps and nothing depends on wall-clock
time. Expiration is request-time only — there is no background sweeper — and these tests
pin both halves of that decision: expiry is noticed when a request touches the store, and
an untouched expired session stays resident until then, bounded by ``max_sessions``.
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import synthetic_report
from video_learning.api.errors import ApiError
from video_learning.api.sessions import ReportSessionStore
from video_learning.services.report_service import ReportStore
from video_learning.services.search_service import SearchService


class FakeClock:
    """A monotonic stand-in the test advances by hand."""

    def __init__(self) -> None:
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _service(tmp_path: Path, *, speech_records: int = 6) -> SearchService:
    report = synthetic_report(tmp_path / "clip.mov", speech_records=speech_records)
    text = json.dumps(report.to_dict())
    return SearchService(ReportStore().load_evidence_from_text(text, label="report.json"))


def _store(tmp_path: Path, clock: FakeClock, **kwargs: Any) -> ReportSessionStore:
    defaults: dict[str, Any] = {
        "max_sessions": 2,
        "idle_ttl_seconds": 100.0,
        "hard_ttl_seconds": 1000.0,
        "clock": clock,
    }
    defaults.update(kwargs)
    return ReportSessionStore(**defaults)


def _add(store: ReportSessionStore, tmp_path: Path, name: str) -> str:
    session = store.create(
        service=_service(tmp_path), filename=name, segment_count=1, bytes_in=10
    )
    return session.report_id


def test_create_retains_one_index_and_reuses_it(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock)
    session = store.create(
        service=_service(tmp_path), filename="r.json", segment_count=1, bytes_in=10
    )

    first = store.get(session.report_id)
    clock.advance(10)
    second = store.get(session.report_id)

    assert first is second
    assert first.service.index is second.service.index


def test_get_renews_the_idle_window(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock)
    report_id = _add(store, tmp_path, "r.json")

    clock.advance(60)
    store.get(report_id)
    clock.advance(60)  # 120s since upload, but only 60s since last use

    assert store.get(report_id).report_id == report_id


def test_idle_expiry_refuses_with_a_distinct_code(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock)
    report_id = _add(store, tmp_path, "r.json")

    clock.advance(100)  # exactly the idle TTL: the deadline is reached, not passed

    with pytest.raises(ApiError) as exc:
        store.get(report_id)
    assert exc.value.status_code == 404
    assert exc.value.code == "report_expired"
    assert len(store) == 0
    assert store.expired_count == 1


def test_hard_cap_expires_a_session_that_is_still_being_used(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock, hard_ttl_seconds=200.0)
    report_id = _add(store, tmp_path, "r.json")

    for _ in range(3):
        clock.advance(50)
        # Each use renewing the idle window keeps it alive past idle_ttl (100s).
        assert store.get(report_id).report_id == report_id

    clock.advance(51)  # 201s since upload: the hard cap (200s) is now the binding one
    with pytest.raises(ApiError) as exc:
        store.get(report_id)
    assert exc.value.code == "report_expired"


def test_unknown_id_is_refused_separately_from_expiry(tmp_path: Path) -> None:
    store = _store(tmp_path, FakeClock())

    with pytest.raises(ApiError) as exc:
        store.get("never-issued")
    assert exc.value.status_code == 404
    assert exc.value.code == "unknown_report"


def test_a_full_store_evicts_the_least_recently_used(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock, max_sessions=2)
    first = _add(store, tmp_path, "first.json")
    second = _add(store, tmp_path, "second.json")

    clock.advance(5)
    store.get(first)  # first is now the most recently used
    clock.advance(5)
    third = _add(store, tmp_path, "third.json")

    assert len(store) == 2
    assert store.evicted_count == 1
    assert store.get(first).report_id == first
    assert store.get(third).report_id == third
    with pytest.raises(ApiError) as exc:
        store.get(second)
    assert exc.value.code == "unknown_report"


def test_upload_still_succeeds_when_every_slot_is_in_use(tmp_path: Path) -> None:
    """The documented full-store behavior: evict, never refuse a legitimate upload."""
    clock = FakeClock()
    store = _store(tmp_path, clock, max_sessions=1)
    _add(store, tmp_path, "first.json")

    clock.advance(1)
    second = _add(store, tmp_path, "second.json")

    assert store.get(second).report_id == second
    assert len(store) == 1


def test_eviction_does_not_cancel_a_query_already_holding_the_session(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock, max_sessions=1)
    report_id = _add(store, tmp_path, "first.json")
    held = store.get(report_id)

    _add(store, tmp_path, "second.json")  # evicts the session we still hold

    results = held.service.query("index locally")
    assert results.status == "ok"
    assert results.coverage.matched_record_count == 7
    with pytest.raises(ApiError):
        store.get(report_id)


def test_create_sweeps_expired_sessions_before_evicting_a_live_one(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock, max_sessions=2)
    stale = _add(store, tmp_path, "stale.json")
    clock.advance(100)
    live = _add(store, tmp_path, "live.json")  # also purges `stale`

    assert store.get(live).report_id == live
    assert len(store) == 1
    assert store.evicted_count == 0
    assert store.expired_count == 1
    with pytest.raises(ApiError):
        store.get(stale)


def test_purge_expired_reaps_without_any_search_traffic(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock)
    _add(store, tmp_path, "a.json")
    _add(store, tmp_path, "b.json")

    clock.advance(101)
    assert store.purge_expired() == 2
    assert len(store) == 0
    assert store.purge_expired() == 0


def test_expires_in_reports_the_nearer_deadline_and_never_goes_negative(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock, hard_ttl_seconds=150.0)
    session = store.create(
        service=_service(tmp_path), filename="r.json", segment_count=1, bytes_in=10
    )

    assert store.expires_in(session) == 100.0  # idle (100) is nearer than hard (150)
    clock.advance(500)
    assert store.expires_in(session) == 0.0


@pytest.mark.parametrize("kwargs", [{"max_sessions": 0}, {"idle_ttl_seconds": 0.0},
                                     {"hard_ttl_seconds": -1.0}])
def test_invalid_bounds_are_refused_at_construction(tmp_path: Path, kwargs: Any) -> None:
    with pytest.raises(ValueError):
        _store(tmp_path, FakeClock(), **kwargs)


@pytest.mark.parametrize("speech_records", [6, 200, 2000])
def test_representative_artifact_sizes_are_an_input_bound_not_a_memory_ceiling(
    tmp_path: Path, speech_records: int
) -> None:
    """Measure what one session actually retains against the bytes that arrived.

    ``max_sessions * max_report_bytes`` bounds the *input*, not the process: the retained
    objects are the parsed evidence plus per-record token lists, a content-dependent
    multiple of the text. The assertion below is a generous regression guard against a
    pathological blow-up, not a promise about peak RSS — see api/sessions.py.
    """
    report = synthetic_report(tmp_path / "clip.mov", speech_records=speech_records)
    text = json.dumps(report.to_dict())
    input_bytes = len(text.encode("utf-8"))

    tracemalloc.start()
    try:
        service = SearchService(ReportStore().load_evidence_from_text(text, label="r.json"))
        retained, _peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(service.index.records) == speech_records + 1
    assert retained < 40 * input_bytes, (retained, input_bytes)
    print(
        f"\n  records={speech_records + 1} input={input_bytes}B "
        f"retained={retained}B ratio={retained / input_bytes:.1f}x"
    )
