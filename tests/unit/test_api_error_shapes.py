"""Every refusal has one shape, and no refusal is a traceback or an echoed value.

``ApiError`` has always answered ``{"code": ..., "detail": ...}``, while the framework's own
404, 405 and 422 answered ``{"detail": ...}`` — one object for a route error, an array of
pydantic descriptors for a validation error, and the validation array's ``input`` field holds
the rejected value verbatim. Two shapes is a client having to parse both; an ``input`` echo is
worse, because a lone surrogate in that value used to make the *error response* unserializable
and turned a bad request into a 500.

So :func:`create_app` now registers handlers for ``StarletteHTTPException`` and
``RequestValidationError`` beside its own, and the assertions here are the ones that matter:
the exact key set for a refusal of every source, the stable code per case, that nothing the
client sent is quoted back, and that the body is strict-decodable UTF-8 — the serialized
answer, not the Python dict a handler returned.

What is deliberately *not* unified: a genuine unexpected exception is still a 500 with a
traceback on the server and a plain ``Internal Server Error`` to the client. :func:`test_an_
unexpected_exception_is_not_masqueraded` proves that by mounting a raising route on the same
app: a handler that swallowed ``Exception`` would make every bug look like this API's careful
refusals, which is the opposite of a consistent error shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from tests.unit.api_support import (
    client_for,
    multipart_body,
    post_multipart,
    report_id,
    search,
    store_of,
    upload,
    utf8_json,
)
from video_learning.api.app import create_app
from video_learning.api.security import DEFAULT_PORT

MARKER = "d0ebd9a6c1"
LONE_SURROGATE_ESCAPE = b"a\\ud800b"


@pytest.fixture
def client() -> TestClient:
    return client_for()


def _json_error(response: Any) -> dict[str, str]:
    """A refusal as this API shapes it: two keys, real UTF-8, no traceback, no echo list."""
    payload = utf8_json(response)

    assert set(payload) == {"code", "detail"}, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert "Traceback" not in response.text
    assert "Internal Server Error" not in response.text
    return payload


def _refusals(client: TestClient, report_artifact_bytes: bytes) -> list[tuple[str, int, str, Any]]:
    """One of every way this API can say no, from the framework and from our own code.

    The 413 and 403 rows need an app configured or headed differently, so each carries its
    own client built here rather than the shared fixture's.
    """
    rid = report_id(client, report_artifact_bytes)
    oversized = client_for(max_report_bytes=64)
    body, content_type = multipart_body(b'{"schema": "video-learning.report/v1"}' * 40)
    foreign_host = client_for()
    foreign_origin = client_for()

    return [
        ("unknown_route", 404, "route_not_found", client.get("/api/nothing-at-all-here")),
        ("wrong_method", 405, "method_not_allowed", client.get("/api/search")),
        (
            "wrong_field_type",
            422,
            "invalid_request",
            client.post("/api/search", json={"report_id": rid, "query": 1234567890}),
        ),
        (
            "unknown_field",
            422,
            "invalid_request",
            client.post("/api/search", json={"report_id": rid, "query": "index", MARKER: 1}),
        ),
        (
            "not_an_object",
            422,
            "invalid_request",
            client.post("/api/search", json=[1, 2]),
        ),
        (
            "malformed_json",
            422,
            "invalid_request",
            client.post(
                "/api/search", content=b"{not json", headers={"Content-Type": "application/json"}
            ),
        ),
        (
            "lone_surrogate_query",
            422,
            "invalid_request",
            client.post(
                "/api/search",
                content=b'{"report_id": "'
                + rid.encode()
                + b'", "query": "'
                + LONE_SURROGATE_ESCAPE
                + b'"}',
                headers={"Content-Type": "application/json"},
            ),
        ),
        ("no_file_field", 422, "missing_report", client.post("/api/reports")),
        (
            "broken_multipart",
            400,
            "invalid_multipart",
            client.post(
                "/api/reports",
                content=b"not a multipart body at all",
                headers={"Content-Type": "multipart/form-data; boundary=x"},
            ),
        ),
        (
            "not_utf8",
            400,
            "invalid_encoding",
            upload(client, b'{"schema": "\xff\xfe not utf-8"}'),
        ),
        (
            "not_an_artifact",
            400,
            "invalid_report",
            upload(client, b"{not json"),
        ),
        (
            "too_large",
            413,
            "request_too_large",
            post_multipart(oversized, body, content_type),
        ),
        (
            "foreign_host",
            403,
            "forbidden_host",
            foreign_host.post("/api/health", headers={"Host": "evil.example"}),
        ),
        (
            "foreign_origin",
            403,
            "forbidden_origin",
            foreign_origin.post(
                "/api/search",
                json={"report_id": "x", "query": "i"},
                headers={"Origin": "http://evil.example"},
            ),
        ),
        ("unknown_report", 404, "unknown_report", search(client, "no-such-session")),
    ]


def test_every_refusal_has_the_same_two_keys(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """404, 405, 422, 400, 403, 413 and 404 alike, whichever layer produced them.

    Collected rather than parametrized because the clients differ per row: an assertion
    inside the loop would stop at the first bad shape and hide the rest.
    """
    problems: list[str] = []
    for label, status, code, response in _refusals(client, report_artifact_bytes):
        try:
            payload = _json_error(response)
        except AssertionError as exc:
            problems.append(f"{label}: {exc}")
            continue
        if response.status_code != status:
            problems.append(f"{label}: status {response.status_code} != {status}")
        if payload["code"] != code:
            problems.append(f"{label}: code {payload['code']!r} != {code!r}")

    assert problems == []


def test_a_404_names_the_absence_not_the_path(client: TestClient) -> None:
    """The requested path is client text, and a 404 is reachable for any string invented."""
    response = client.get(f"/api/{MARKER}")

    assert _json_error(response)["code"] == "route_not_found"
    assert MARKER not in response.text


def test_a_422_never_quotes_the_rejected_value(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """``input`` held the offending text; a field name and pydantic's type id do not."""
    rid = report_id(client, report_artifact_bytes)
    cases = [
        client.post("/api/search", json={"report_id": rid, "query": 9876543210}),
        client.post("/api/search", json={"report_id": rid, MARKER: 1}),
        client.post(
            "/api/search",
            content=b'{"report_id": "'
            + rid.encode()
            + b'", "query": "'
            + LONE_SURROGATE_ESCAPE
            + b'"}',
            headers={"Content-Type": "application/json"},
        ),
    ]

    for response in cases:
        payload = _json_error(response)
        assert payload["code"] == "invalid_request"
        assert MARKER not in response.text
        assert "9876543210" not in response.text
        assert "\\ud800" not in response.text


def test_a_405_keeps_advertising_the_allowed_method(client: TestClient) -> None:
    """Replacing the handler must not drop the header that makes 405 useful."""
    response = client.get("/api/search")

    assert response.headers["allow"] == "POST"
    assert _json_error(response)["code"] == "method_not_allowed"


def test_a_framework_404_and_an_our_own_404_are_indistinguishable(
    client: TestClient,
) -> None:
    """A client branches on ``code`` and never on which layer answered — the point of the work."""
    framework = _json_error(client.get("/api/nothing-here"))
    ours = _json_error(search(client, "no-such-session"))

    assert set(framework) == set(ours) == {"code", "detail"}
    assert framework["code"] == "route_not_found"
    assert ours["code"] == "unknown_report"


def test_the_two_framework_handlers_are_registered(client: TestClient) -> None:
    """The behavior tests below prove the shape; this proves the wiring they depend on.

    A handler that stopped being registered would still show up as a framework shape in the
    behavior tests, so the registration is asserted directly rather than inferred.
    """
    handlers: dict[Any, Callable[..., Any]] = client.app.exception_handlers  # type: ignore[attr-defined]

    assert StarletteHTTPException in handlers
    assert RequestValidationError in handlers
    assert 500 not in handlers


def test_a_real_bug_is_still_a_500_and_not_masqueraded_as_a_refusal(
    tmp_path: Any,
) -> None:
    """No catch-all handler was added, which is the deliberate other half of this change.

    A route that raises is the app's own bug, not a client error: it must stay a 500 with the
    traceback on the server side. Swallowing ``Exception`` into ``invalid_request`` would make
    every defect in this API look like a careful refusal.
    """
    app = create_app()

    @app.get("/api/boom")
    def _boom() -> None:
        raise RuntimeError("a defect, not a refusal")

    client = TestClient(
        app, base_url=f"http://127.0.0.1:{DEFAULT_PORT}", raise_server_exceptions=False
    )
    response = client.get("/api/boom")

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert response.headers["content-type"].startswith("text/plain")


def test_successes_are_untouched_by_the_handlers(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """The payload of a route that succeeded is still the core's own, not an error shape."""
    rid = report_id(client, report_artifact_bytes)
    health = client.get("/api/health")
    payload = utf8_json(search(client, rid))

    assert health.status_code == 200
    assert "code" not in health.json()
    assert payload["schema"] == "video-learning.search/v1"
    assert client.get("/api/docs").status_code == 200
    assert client.get("/api/openapi.json").json()["openapi"]


def test_an_expired_session_keeps_the_same_shape(report_artifact_bytes: bytes) -> None:
    """The one refusal that needs a clock of its own, checked for the keys and the code."""
    now = [1000.0]
    client = client_for(idle_ttl_seconds=100.0, clock=lambda: now[0])
    rid = report_id(client, report_artifact_bytes)
    now[0] += 101.0

    response = search(client, rid)

    assert _json_error(response) == {
        "code": "report_expired",
        "detail": "That report session expired; upload it again.",
    }
    assert len(store_of(client)) == 0
