"""Which thread does the work, asserted by thread identity rather than by elapsed time.

The upload route is ``async def`` because reading a multipart stream is what an event loop is
for, but the artifact work behind it is not: ``ReportStore.load_evidence_from_text`` parses the
document, the UTF-8 transport check walks it again, and ``SearchService.__init__`` builds the
index. Measured on the pre-fix tree that cost scaled with the record count and stalled the loop
for 68 ms at 2 000 records and 758 ms at 20 000 — during which no other request, a health check
included, could be served. So the route awaits it through ``run_in_threadpool``.

A wall-clock assertion would be the wrong way to hold that: a threshold loose enough to survive a
loaded machine stops detecting the stall, and one tight enough goes red on a busy runner. Instead
these tests record ``threading.current_thread()`` where the work happens and compare it with the
thread the route body ran on. Two different threads is a fact about the dispatch, so a revert to
an inline call fails deterministically, and a fast or slow machine cannot change the verdict.

The search route keeps its own shape: it is a plain ``def``, which is how FastAPI hands the whole
handler to a worker thread without the route saying so. That is asserted structurally below,
because it is the arrangement these probes depend on rather than another thing to measure.
"""

from __future__ import annotations

import inspect
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.unit.api_support import client_for, upload
from video_learning.api import app as api_app
from video_learning.api import reports as api_reports
from video_learning.services.report_service import ReportStore
from video_learning.services.search_service import SearchService

#: Where the work happened, keyed by the name of the step, filled in by the probes below.
#: ``loop`` is recorded inside the awaited upload helper, which runs on the event loop.
ThreadProbe = dict[str, threading.Thread]


@pytest.fixture
def client() -> TestClient:
    return client_for()


def _route(client: TestClient, path: str) -> Any:
    """The endpoint function registered for one exact path."""
    for route in client.app.routes:  # type: ignore[attr-defined]
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"no route registered for {path}")


def test_the_upload_route_is_async_and_the_search_route_is_not(
    client: TestClient,
) -> None:
    """The two shapes are deliberate, so one is stated as an assertion, not a comment.

    ``async def`` is right for the stream read and wrong for the corpus walk: a plain ``def``
    route is how FastAPI is told to hand the whole handler to a worker thread. If the search
    route were ever made ``async def``, lexical work over a large corpus would return to the
    loop by that door, and the event-loop fix this file exists for would be undone in halves.
    """
    assert inspect.iscoroutinefunction(_route(client, "/api/reports"))
    assert not inspect.iscoroutinefunction(_route(client, "/api/search"))


def test_multipart_reading_stays_a_coroutine(client: TestClient) -> None:
    """Moving the parse off the loop must not move the *stream* read onto a thread.

    ``request.form()`` is a coroutine; wrapping it in a worker thread would block a worker on an
    awaitable instead of relieving the loop, which is the opposite of the fix.
    """
    assert inspect.iscoroutinefunction(api_app.read_report_upload)


def test_the_parse_and_the_index_build_run_off_the_event_loop(
    client: TestClient,
    report_artifact_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity, not duration: the loop thread and the working threads may not be one thread.

    Three steps are probed where they happen — the store's parse, the service constructor that
    indexes, and the retention call that owns both — each compared against the thread the route
    body ran on, captured inside the awaited upload helper. ``open_session`` is probed as the
    module attribute the route calls, so a revert to an inline call lands on the loop thread and
    fails the identity comparisons rather than quietly passing.
    """
    where: ThreadProbe = {}
    real_load = ReportStore.load_evidence_from_text
    real_service = SearchService
    real_open = api_app.open_session
    real_read = api_app.read_report_upload

    def probe_load(self: ReportStore, raw: str, *, label: str) -> Any:
        where["parse"] = threading.current_thread()
        return real_load(self, raw, label=label)

    class ProbedService(real_service):
        def __init__(self, evidence: Any) -> None:
            where["index"] = threading.current_thread()
            super().__init__(evidence)

    def probe_open(store: Any, **fields: Any) -> Any:
        where["session"] = threading.current_thread()
        return real_open(store, **fields)

    async def probe_read(request: Any, *, max_bytes: int) -> tuple[str, str, int]:
        where["loop"] = threading.current_thread()
        return await real_read(request, max_bytes=max_bytes)

    monkeypatch.setattr(ReportStore, "load_evidence_from_text", probe_load)
    monkeypatch.setattr(api_app, "read_report_upload", probe_read)
    monkeypatch.setattr(api_app, "open_session", probe_open)
    monkeypatch.setattr(api_reports, "SearchService", ProbedService)

    assert upload(client, report_artifact_bytes).status_code == 201
    assert {"loop", "parse", "index", "session"} == set(where), sorted(where)
    for step in ("parse", "index", "session"):
        assert where[step] is not where["loop"], step
    assert where["parse"] is where["session"] is where["index"]


def test_the_search_route_runs_off_the_loop_too(
    client: TestClient, report_artifact_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A synchronous handler is dispatched by the framework, proven by identity.

    The route's own ``def`` is the claim; a query that walked the index inline on the loop
    would satisfy that claim and still stall the server. So the assertion is on the thread the
    query ran on, against the loop thread captured moments earlier by the same upload.
    """
    where: ThreadProbe = {}
    real_query = SearchService.query
    real_read = api_app.read_report_upload

    def probe_query(self: SearchService, query: str, *, limit: int) -> Any:
        where["query"] = threading.current_thread()
        return real_query(self, query, limit=limit)

    async def probe_read(request: Any, *, max_bytes: int) -> tuple[str, str, int]:
        where["loop"] = threading.current_thread()
        return await real_read(request, max_bytes=max_bytes)

    monkeypatch.setattr(SearchService, "query", probe_query)
    monkeypatch.setattr(api_app, "read_report_upload", probe_read)

    uploaded = upload(client, report_artifact_bytes)
    assert uploaded.status_code == 201
    loop_thread = where["loop"]

    response = client.post(
        "/api/search", json={"report_id": uploaded.json()["report_id"], "query": "index"}
    )

    assert response.status_code == 200
    assert where["query"] is not loop_thread
