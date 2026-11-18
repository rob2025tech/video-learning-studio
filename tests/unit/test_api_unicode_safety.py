"""Lone surrogates must never become a 500, and never become a rewritten answer either.

A JSON body can carry an unpaired surrogate while staying pure ASCII: ``\\ud800`` is six
characters — backslash, ``u``, four hex digits — so the request decodes as UTF-8 and the
surrogate exists only in the *parsed* string. That is the whole difficulty. Starlette renders
a response with ``ensure_ascii=False``, so the moment such a string is inside a payload the
encode to UTF-8 raises, and the client gets a 500 of ``text/plain`` instead of an answer. The
two defenses are the two that matter:

* **upload** — :func:`video_learning.api.reports._require_utf8_transport` refuses a document
  whose parsed strings could never be sent back, and it refuses before the session exists. A
  poisoned artifact is therefore one clean 400 rather than a 201 followed by ranking-
  dependent 500s, with a session nobody can search left resident.
* **query** — pydantic already refuses a lone surrogate, and the 422 that used to crash
  inside FastAPI's own validation handler now comes from
  :func:`video_learning.api.errors.validation_error`, which names fields and types and never
  the rejected value. The tests drive it with real request bytes rather than
  ``model_validate``, because the defect *was* in the response path.

Nothing here rewrites evidence text to make a response serializable, and the CLI is asserted
unaffected on the same poisoned bytes: ``search --json`` escapes the surrogate and exits 0,
because writing with ``ensure_ascii=True`` never needed a UTF-8-safe string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.unit.api_support import (
    client_for,
    report_id,
    search,
    store_of,
    upload,
    utf8_json,
)
from video_learning.cli.main import app as cli_app
from video_learning.services.report_service import ReportStore
from video_learning.services.search_service import SearchService

runner = CliRunner()

#: An unpaired high surrogate, and one correctly paired with a low surrogate to make an
#: astral character. The first cannot be encoded in UTF-8; the second is ordinary text.
LONE_SURROGATE = "\ud800"
SURROGATE_PAIR = "\U0001f600"
MARKER = "SENTINEL"


@pytest.fixture
def client() -> TestClient:
    return client_for()


def _artifact_with(data: bytes, kind: str, text: str) -> bytes:
    """The fixture artifact with one timeline event's text replaced, as pure-ASCII bytes.

    ``json.dumps`` escapes by default, which is the point: the bytes handed to the route
    contain no non-ASCII byte at all, so nothing in them can explain the refusal.
    """
    document = json.loads(data)
    event = next(item for item in document["timeline"] if item["type"] == kind)
    event["text"] = text
    encoded = json.dumps(document).encode("utf-8")
    assert all(byte < 0x80 for byte in encoded), "the poison must arrive as pure ASCII"
    return encoded


def _non_ascii_artifact(data: bytes, text: str) -> bytes:
    """The fixture artifact with OCR evidence replaced by genuinely non-ASCII text."""
    document = json.loads(data)
    event = next(item for item in document["timeline"] if item["type"] == "ocr")
    event["text"] = text
    encoded = json.dumps(document, ensure_ascii=False).encode("utf-8")
    assert any(byte >= 0x80 for byte in encoded), "the fixture must be real UTF-8 bytes"
    return encoded


def _search_cli(artifact: Path, query: str) -> dict[str, Any]:
    """``video-learning search --json`` on an artifact file, as the terminal runs it."""
    result = runner.invoke(cli_app, ["search", str(artifact), query, "--json"])

    assert result.exit_code == 0, result.output
    parsed: dict[str, Any] = json.loads(result.stdout)
    return parsed


@pytest.mark.parametrize("kind", ["ocr", "speech"])
def test_an_escaped_lone_surrogate_in_evidence_is_refused_at_upload(
    client: TestClient, report_artifact_bytes: bytes, kind: str
) -> None:
    """Both evidence kinds, because OCR is a point snapshot and speech an interval."""
    poisoned = _artifact_with(report_artifact_bytes, kind, f"{MARKER} {LONE_SURROGATE} tail")

    response = upload(client, poisoned, filename="poisoned.json")

    assert response.status_code == 400, response.text
    assert set(utf8_json(response)) == {"code", "detail"}
    assert utf8_json(response)["code"] == "invalid_report"
    assert MARKER not in utf8_json(response)["detail"]


@pytest.mark.parametrize("kind", ["ocr", "speech"])
def test_a_refused_upload_creates_no_session_and_frees_no_slot(
    report_artifact_bytes: bytes, kind: str
) -> None:
    """A rejected upload must not half-load: nothing searchable, and nothing evicted.

    With one slot available, a ``store.create`` that evicted before failing would have
    discarded the good session as well as leaving a broken one behind, so both halves are
    checked: the count, the survivor, and the absent report id.
    """
    client = client_for(max_sessions=1)
    good = report_id(client, report_artifact_bytes)

    refused = upload(client, _artifact_with(report_artifact_bytes, kind, LONE_SURROGATE))

    assert refused.status_code == 400
    assert "report_id" not in refused.json()
    assert len(store_of(client)) == 1
    assert search(client, good).status_code == 200
    assert search(client, "no-such-report").status_code == 404


def test_a_poisoned_artifact_never_becomes_searchable(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """No id is returned, so there is no session whose search could reach the 500."""
    response = upload(client, _artifact_with(report_artifact_bytes, "ocr", LONE_SURROGATE))
    payload = utf8_json(response)

    assert response.status_code == 400
    assert "report_id" not in payload
    assert len(store_of(client)) == 0


def test_an_escaped_surrogate_query_is_refused_over_the_real_wire(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """The regression for the 500 FastAPI's own validation handler used to raise.

    What is asserted is that a real HTTP request is answered in this API's shape: a plain
    ``model_validate`` call could have passed all along while the response path still
    crashed, because the offending value is echoed inside ``errors()[0]["input"]``.
    """
    rid = report_id(client, report_artifact_bytes)
    body = b'{"report_id": "' + rid.encode() + b'", "query": "a\\ud800b"}'
    assert all(byte < 0x80 for byte in body), "the escape must be ASCII on the wire"

    response = client.post(
        "/api/search", content=body, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422, response.text
    payload = utf8_json(response)
    assert payload["code"] == "invalid_request"
    assert "string_unicode" in payload["detail"]
    assert LONE_SURROGATE not in response.text


def test_a_lone_surrogate_never_becomes_a_500_on_the_query_path(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """Escaped in a value, escaped in a key, and a value the schema never declares.

    Each of these used to be a 500 or an unsendable body: the surrogate reached the response
    either through pydantic's ``input`` echo or through the payload it was queried with.
    """
    rid = report_id(client, report_artifact_bytes)
    bodies = [
        b'{"report_id": "' + rid.encode() + b'", "query": "a\\ud800b"}',
        b'{"\\ud800": 1, "report_id": "' + rid.encode() + b'", "query": "index"}',
        b'{"report_id": "' + rid.encode() + b'", "query": 123}',
    ]

    for body in bodies:
        response = client.post(
            "/api/search", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code != 500, body
        assert set(utf8_json(response)) == {"code", "detail"}
        assert LONE_SURROGATE not in response.text


def test_a_valid_surrogate_pair_is_ordinary_text(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """An astral character encoded as a pair is what text is made of, not a defect."""
    artifact = _artifact_with(report_artifact_bytes, "ocr", f"slide {SURROGATE_PAIR} {MARKER}")

    rid = report_id(client, artifact)

    payload = utf8_json(search(client, rid, MARKER))

    assert payload["status"] != "no_evidence"
    assert any(SURROGATE_PAIR in item["text"] for item in payload["results"])


@pytest.mark.parametrize(
    ("query", "needle"),
    [("café", "café"), ("日本語", "日本語"), (f"emoji {SURROGATE_PAIR}", SURROGATE_PAIR)],
)
def test_valid_non_ascii_still_uploads_and_searches(
    client: TestClient, report_artifact_bytes: bytes, query: str, needle: str
) -> None:
    """The transport guard must not quietly become an ASCII-only policy."""
    rid = report_id(client, _non_ascii_artifact(report_artifact_bytes, f"{needle} index record"))

    payload = utf8_json(search(client, rid, query))

    assert payload["query"] == query
    assert payload["source"] == "clip.mov"
    if payload["results"]:
        assert any(needle in item["text"] for item in payload["results"])


def test_the_cli_still_searches_the_poisoned_artifact(
    tmp_path: Path, report_artifact_bytes: bytes
) -> None:
    """Same bytes, unchanged CLI contract: it escapes on output, exits 0, answers normally.

    The API refuses the artifact because it could not serialize a response; the CLI writes
    with ``ensure_ascii=True`` and never had that problem. A "fix" that rewrote evidence text,
    or that shared the refusal downward, would show up here as a difference from the core's
    own answer on the same query.
    """
    poisoned = tmp_path / "poisoned.json"
    poisoned.write_bytes(_artifact_with(report_artifact_bytes, "ocr", f"{LONE_SURROGATE} index"))
    clean = tmp_path / "clean.json"
    clean.write_bytes(report_artifact_bytes)

    payload = _search_cli(poisoned, "index")
    expected = _search_cli(clean, "index")

    assert payload["status"] == "ok"
    assert any(LONE_SURROGATE in item["text"] for item in payload["results"])
    assert LONE_SURROGATE not in json.dumps(payload)
    assert payload["coverage"] == expected["coverage"]
    assert "\\ud800" in poisoned.read_text(encoding="utf-8")


def test_the_poisoned_evidence_reaches_the_core_unchanged(
    tmp_path: Path, report_artifact_bytes: bytes
) -> None:
    """The artifact rules and the text they carry belong to the store, not to us to adjust."""
    path = tmp_path / "poison.json"
    path.write_bytes(_artifact_with(report_artifact_bytes, "ocr", f"{LONE_SURROGATE} index"))

    evidence = ReportStore().load_evidence(path)
    results = SearchService(evidence).query("index", limit=5).to_dict()

    assert any(LONE_SURROGATE in item["text"] for item in results["results"])
    assert json.dumps(results, ensure_ascii=True)
