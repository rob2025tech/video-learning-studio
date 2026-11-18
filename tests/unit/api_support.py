"""Shared helpers for the API tests. Not a conftest: importing FastAPI here would make
the whole unit suite uncollectable in an environment without the ``api`` extra, and
``tests/conftest.py`` must stay importable by the CLI-only tests.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from fastapi.testclient import TestClient
from httpx import Response

from video_learning.api.app import create_app

PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"
_BOUNDARY = "----LearnLensTestBoundary"


def client_for(**kwargs: Any) -> TestClient:
    """A client whose Host header the policy accepts (the default would be ``testserver``)."""
    return TestClient(create_app(port=PORT, **kwargs), base_url=BASE_URL)


def upload(client: TestClient, data: bytes, *, filename: str = "report.json") -> Response:
    return client.post("/api/reports", files={"report": (filename, data, "application/json")})


def report_id(client: TestClient, data: bytes) -> str:
    response = upload(client, data)
    assert response.status_code == 201, response.text
    payload: dict[str, Any] = response.json()
    return str(payload["report_id"])


def search(client: TestClient, report_id: str, query: str = "index", **fields: Any) -> Response:
    return client.post("/api/search", json={"report_id": report_id, "query": query, **fields})


def multipart_body_parts(parts: Sequence[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    """Build the exact request bytes, so a size-cap test can count them itself.

    ``files=`` lets httpx choose the boundary and part headers, which makes the total
    length unknowable from the test. Here it is ``len(body)`` by construction. Each part
    is ``(field_name, filename, data)``.
    """
    chunks: list[bytes] = []
    for field, filename, data in parts:
        chunks.append(
            (
                f"--{_BOUNDARY}\r\n"
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
                "Content-Type: application/json\r\n\r\n"
            ).encode()
            + data
            + b"\r\n"
        )
    chunks.append(f"--{_BOUNDARY}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={_BOUNDARY}"


def multipart_body(
    data: bytes, *, field: str = "report", filename: str = "report.json"
) -> tuple[bytes, str]:
    return multipart_body_parts([(field, filename, data)])


def post_multipart(client: TestClient, body: bytes, content_type: str) -> Response:
    return client.post("/api/reports", content=body, headers={"Content-Type": content_type})


def store_of(client: TestClient) -> Any:
    return client.app.state.store  # type: ignore[attr-defined]


def utf8_json(response: Response) -> dict[str, Any]:
    """The response body as a parsed object, after proving it really is UTF-8 JSON.

    An assertion that goes through here tests the *serialized* answer rather than the
    Python dict a route returned: ``response.json()`` would happily hand back a dict
    holding a lone surrogate that no client could ever receive, which is exactly the defect
    these tests exist to keep out. So the raw bytes are decoded strictly (a
    ``UnicodeDecodeError`` here is a failure, not a pass), compared against the text httpx
    decoded, re-encoded, and parsed.
    """
    text = response.content.decode("utf-8")
    assert response.text == text
    document = json.loads(text)
    assert json.dumps(document, ensure_ascii=False).encode("utf-8")
    return document
