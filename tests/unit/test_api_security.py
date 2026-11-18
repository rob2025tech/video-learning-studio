"""The localhost contract, tested as a matrix rather than as a couple of happy paths.

Three layers are covered: the pure header policy (every allowed and refused spelling),
the middleware over real HTTP (status, body, and the fact that a refused request never
reaches a handler or creates a session), and the raw ASGI boundary, where header bytes
are latin-1 rather than the ASCII httpx will send. The body-limit tests include a chunked
request with no ``Content-Length``, because that is the case a header-only check misses.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from video_learning.api.app import create_app
from video_learning.api.security import DEFAULT_PORT, RequestPolicy

PORT = 8000


def _headers(**values: str) -> Headers:
    return Headers({"host": f"127.0.0.1:{PORT}", **values})


@pytest.mark.parametrize(
    "host",
    [
        f"127.0.0.1:{PORT}",
        f"localhost:{PORT}",
        f"LOCALHOST:{PORT}",
        f"LocalHost:{PORT}",
    ],
)
def test_loopback_hosts_are_allowed(host: str) -> None:
    assert RequestPolicy(port=PORT).rejection(Headers({"host": host})) is None


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        f"evil.example:{PORT}",
        f"127.0.0.1.evil.example:{PORT}",
        f"sub.localhost:{PORT}",
        f"localhost:{PORT + 1}",
        f"127.0.0.1:{PORT}:9",
        f"[::1]:{PORT}",
        "127.0.0.1",  # port-less while we listen on 8000
        "localhost",
        "",
        # latin-1 superscript: isdigit() is true, int() raises. Reachable over HTTP,
        # unlike non-ASCII digits such as "٨٠٠٠", which a latin-1 header cannot carry.
        f"127.0.0.1:{PORT}²",
        f"127.0.0.1:{PORT} ",
        f"127.0.0.1:+{PORT}",
    ],
)
def test_foreign_or_malformed_hosts_are_refused(host: str) -> None:
    rejection = RequestPolicy(port=PORT).rejection(Headers({"host": host}))
    assert rejection is not None
    assert rejection[0] == "forbidden_host"


def test_a_port_less_host_is_allowed_only_on_port_80() -> None:
    assert RequestPolicy(port=80).rejection(Headers({"host": "localhost"})) is None
    assert RequestPolicy(port=80).rejection(Headers({"host": "127.0.0.1"})) is None


def test_missing_host_is_refused() -> None:
    assert RequestPolicy(port=PORT).rejection(Headers({})) is not None


@pytest.mark.parametrize("origin", [f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"])
def test_own_origins_are_allowed(origin: str) -> None:
    assert RequestPolicy(port=PORT).rejection(_headers(origin=origin)) is None


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        f"http://evil.example:{PORT}",
        "null",
        f"https://127.0.0.1:{PORT}",
        f"http://127.0.0.1:{PORT + 1}",
        f"http://localhost:{PORT}.evil.example",
        f"http://127.0.0.1:{PORT}/",
    ],
)
def test_foreign_origins_are_refused(origin: str) -> None:
    rejection = RequestPolicy(port=PORT).rejection(_headers(origin=origin))
    assert rejection is not None
    assert rejection[0] == "forbidden_origin"


def test_missing_origin_is_allowed_for_curl_and_same_origin_navigation() -> None:
    """A browser always sends Origin on a cross-origin fetch, so absence is not a read."""
    assert RequestPolicy(port=PORT).rejection(_headers()) is None


@pytest.mark.parametrize("site", ["same-origin", "same-site", "none"])
def test_non_cross_site_fetch_metadata_is_allowed(site: str) -> None:
    assert RequestPolicy(port=PORT).rejection(_headers(**{"sec-fetch-site": site})) is None


def test_cross_site_fetch_metadata_is_refused_even_with_a_loopback_host() -> None:
    rejection = RequestPolicy(port=PORT).rejection(_headers(**{"sec-fetch-site": "cross-site"}))
    assert rejection is not None
    assert rejection[0] == "forbidden_origin"


def test_dns_rebinding_shape_is_refused_by_host_not_by_origin() -> None:
    """The rebinding case: a foreign Host with no Origin at all, as a navigation sends."""
    rejection = RequestPolicy(port=PORT).rejection(Headers({"host": "evil.example:8000"}))
    assert rejection is not None
    assert rejection[0] == "forbidden_host"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(port=PORT), base_url=f"http://127.0.0.1:{PORT}")


def test_foreign_host_gets_403_and_never_reaches_a_handler(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    response = client.post(
        "/api/reports",
        files={"report": ("r.json", report_artifact_bytes, "application/json")},
        headers={"Host": "evil.example"},
    )

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden_host"
    store: Any = client.app.state.store  # type: ignore[attr-defined]
    assert len(store) == 0


def test_foreign_origin_gets_403_on_search(client: TestClient) -> None:
    response = client.post(
        "/api/search",
        json={"report_id": "x", "query": "index"},
        headers={"Origin": "http://evil.example"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "forbidden_origin"


#: Distinctive values a refusal detail must never quote back: an error body is a response
#: this API writes, and Host/Origin are entirely client-chosen text (F-1 hardening).
HOST_SENTINEL = "sentinel-host-d4f19c3a.example"
ORIGIN_SENTINEL = "http://sentinel-origin-d4f19c3a.example:9999"


@pytest.mark.parametrize("host", [HOST_SENTINEL, f"{HOST_SENTINEL}:{PORT}", f"127.0.0.1:{PORT}2"])
def test_a_refused_host_is_never_echoed_in_the_response(
    client: TestClient, host: str
) -> None:
    response = client.get("/api/health", headers={"Host": host})

    assert response.status_code == 403
    payload = response.json()
    assert set(payload) == {"code", "detail"}
    assert payload["code"] == "forbidden_host"
    assert "Host" in payload["detail"] and "not allowed" in payload["detail"]
    assert HOST_SENTINEL not in response.text
    # The refused spelling carries digits (a bad or wrong port); a fixed detail has none.
    assert not any(character.isdigit() for character in payload["detail"])


def test_a_refused_origin_is_never_echoed_in_the_response(client: TestClient) -> None:
    response = client.post(
        "/api/search",
        json={"report_id": "x", "query": "index"},
        headers={"Origin": ORIGIN_SENTINEL},
    )

    assert response.status_code == 403
    payload = response.json()
    assert set(payload) == {"code", "detail"}
    assert payload["code"] == "forbidden_origin"
    assert "Origin" in payload["detail"] and "not allowed" in payload["detail"]
    assert ORIGIN_SENTINEL not in response.text


def test_curl_shaped_request_with_no_origin_succeeds(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    uploaded = client.post(
        "/api/reports", files={"report": ("r.json", report_artifact_bytes, "application/json")}
    )
    assert uploaded.status_code == 201
    report_id = uploaded.json()["report_id"]

    searched = client.post("/api/search", json={"report_id": report_id, "query": "index"})
    assert searched.status_code == 200


def test_oversized_body_is_413(client: TestClient) -> None:
    app = create_app(port=PORT, max_report_bytes=1024)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as small:
        response = small.post(
            "/api/reports", files={"report": ("r.json", b"x" * 4096, "application/json")}
        )
    assert response.status_code == 413
    assert response.json()["code"] == "request_too_large"


def test_oversized_chunked_body_without_content_length_is_still_413() -> None:
    """The stream counter, not the header, is what bounds the read."""
    app = create_app(port=PORT, max_report_bytes=1024)

    def chunks() -> Any:
        for _ in range(8):
            yield b"y" * 512

    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as small:
        response = small.post("/api/reports", content=chunks())
    assert response.status_code == 413
    assert response.json()["code"] == "request_too_large"


def test_the_body_bound_is_inclusive() -> None:
    """Exactly max bytes is accepted; one byte more is refused."""
    prefix, suffix = b'{"report_id":"', b'"}'
    body = prefix + b"a" * (64 - len(prefix) - len(suffix)) + suffix
    assert len(body) == 64

    app = create_app(port=PORT, max_report_bytes=64)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as small:
        at_limit = small.post("/api/search", content=body)
        over_limit = small.post("/api/search", content=body + b" ")
    assert at_limit.status_code != 413
    assert over_limit.status_code == 413


def test_default_port_is_the_documented_one() -> None:
    assert DEFAULT_PORT == PORT


def test_an_oversized_upload_creates_no_session() -> None:
    """The 413 is sent before the app runs, so nothing is parsed and nothing is retained."""
    app = create_app(port=PORT, max_report_bytes=1024)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as small:
        response = small.post(
            "/api/reports", files={"report": ("r.json", b"x" * 4096, "application/json")}
        )
        store: Any = small.app.state.store  # type: ignore[attr-defined]
        assert len(store) == 0
    assert response.status_code == 413


def _asgi_get(app: Any, raw_headers: list[tuple[bytes, bytes]]) -> tuple[int, bytes]:
    """Drive the ASGI app with header *bytes*, as a real server delivers them.

    httpx encodes outgoing header values as ASCII, so a latin-1 byte such as ``0xB2``
    cannot be sent through TestClient. uvicorn decodes request headers as latin-1, which
    is the spelling this exercises.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/health",
        "raw_path": b"/api/health",
        "query_string": b"",
        "root_path": "",
        "headers": raw_headers,
        "client": ("127.0.0.1", 43210),
        "server": ("127.0.0.1", PORT),
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return int(start["status"]), body


def test_a_latin1_superscript_port_is_403_not_a_server_error() -> None:
    """``"8000²".isdigit()`` is true and ``int()`` raises, so the ascii guard is load-bearing."""
    status, body = _asgi_get(create_app(port=PORT), [(b"host", b"127.0.0.1:8000\xb2")])
    assert status == 403
    assert json.loads(body)["code"] == "forbidden_host"


def test_the_asgi_boundary_accepts_a_plain_loopback_host() -> None:
    status, body = _asgi_get(create_app(port=PORT), [(b"host", b"127.0.0.1:8000")])
    assert status == 200
    assert json.loads(body)["status"] == "ok"
