"""``POST /api/reports``: untrusted bytes in, one validated session out.

Validation is still entirely the store's — these tests prove it by re-raising the
``ReportArtifactError`` that ``ReportStore.load_evidence_from_text`` produces for the
*same* text — but the API no longer copies that message into the response: it quotes the
uploaded filename and artifact-derived values back, so the HTTP contract is one stable
detail (F-2 hardening). The size assertions build the multipart body themselves so the
byte count is known exactly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SYNTHETIC_OCR_TEXT, synthetic_report
from tests.unit.api_support import (
    client_for,
    multipart_body,
    multipart_body_parts,
    post_multipart,
    report_id,
    search,
    store_of,
    upload,
)
from video_learning.api.app import DEFAULT_MAX_REPORT_BYTES
from video_learning.api.reports import INVALID_REPORT_DETAIL
from video_learning.api.responses import MAX_DISPLAY_NAME_LENGTH
from video_learning.core.errors import ReportArtifactError
from video_learning.core.timeline import OcrEvent, build_timeline
from video_learning.services.report_service import REPORT_SCHEMA, ReportResult, ReportStore


def _bytes(tmp_path: Path, report: ReportResult) -> bytes:
    out = tmp_path / "report.json"
    ReportStore().save(report, out)
    return out.read_bytes()


def _edited(data: bytes, edit: Callable[[dict[str, Any]], None]) -> bytes:
    document = json.loads(data)
    edit(document)
    return json.dumps(document).encode("utf-8")


def _event(document: dict[str, Any], kind: str) -> dict[str, Any]:
    """The timeline is time-ordered, so position 0 is not necessarily the OCR record."""
    return next(event for event in document["timeline"] if event["type"] == kind)


@pytest.fixture
def client() -> TestClient:
    return client_for()


def test_a_valid_artifact_becomes_a_session(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    response = upload(client, report_artifact_bytes)

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["report_schema"] == REPORT_SCHEMA
    assert payload["filename"] == "report.json"
    assert payload["bytes_received"] == len(report_artifact_bytes)
    assert payload["expires_in_seconds"] > 0
    assert payload["video_duration_seconds"] == 24.0
    assert payload["transcription_status"] == "ok"
    assert payload["transcription_available"] is True
    assert payload["corpus"] == {
        "record_count": 7,
        "searchable_record_count": 7,
        "speech_record_count": 6,
        "ocr_record_count": 1,
        "segment_count": 1,
    }
    assert len(store_of(client)) == 1


def test_the_upload_response_leaks_no_absolute_path(
    client: TestClient, report_artifact_bytes: bytes, tmp_path: Path
) -> None:
    """The artifact records a real absolute ``source``; the response must not echo it."""
    assert str(tmp_path) in report_artifact_bytes.decode("utf-8")

    payload = upload(client, report_artifact_bytes).json()

    assert "source" not in payload
    assert str(tmp_path) not in repr(payload)
    assert "/Users/" not in repr(payload)


def test_each_upload_gets_its_own_id(client: TestClient, report_artifact_bytes: bytes) -> None:
    first = report_id(client, report_artifact_bytes)
    second = report_id(client, report_artifact_bytes)

    assert first != second
    assert len(store_of(client)) == 2
    assert "/" not in first and "\\" not in first


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.json", "report.json"),
        ("/Users/someone/private/report.json", "report.json"),
        ("C:\\Users\\me\\report.json", "report.json"),
        ("C:/Users/me/report.json", "report.json"),
        ("..\\..\\report.json", "report.json"),
        ("../../report.json", "report.json"),
        ("dir/./report.json", "report.json"),
        ("  report.json  ", "report.json"),
        ("讲义.json", "讲义.json"),
        (".", "report.json"),
        ("..", "report.json"),
        ("/", "report.json"),
        ("C:\\", "report.json"),
    ],
)
def test_the_upload_filename_is_display_metadata_only(
    client: TestClient, report_artifact_bytes: bytes, filename: str, expected: str
) -> None:
    payload = upload(client, report_artifact_bytes, filename=filename).json()

    assert payload["filename"] == expected
    assert "/" not in payload["filename"]
    assert "\\" not in payload["filename"]
    assert store_of(client).get(payload["report_id"]).filename == expected


@pytest.mark.parametrize("raw", ["report\x00.json", "report\t.json", "report\x1f.json"])
def test_raw_control_characters_in_a_filename_are_stripped(
    client: TestClient, report_artifact_bytes: bytes, raw: str
) -> None:
    """Sent as raw multipart bytes: httpx would percent-encode these instead (see below)."""
    body, content_type = multipart_body_parts([("report", raw, report_artifact_bytes)])

    payload = post_multipart(client, body, content_type).json()

    assert payload["filename"] == "report.json"


def test_a_percent_encoded_filename_stays_literal_display_text(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """httpx escapes control characters on the wire, so ``%00`` arrives as three characters.

    They are inert text, not a control byte: the sanitizer must not decode them, and the
    result still carries no separator and no control character.
    """
    payload = upload(client, report_artifact_bytes, filename="report\x00.json").json()

    assert payload["filename"] == "report%00.json"
    assert not any(ord(char) < 0x20 for char in payload["filename"])


def test_an_empty_filename_is_not_a_file_part(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """With no filename, the part is a form field, and this route accepts no form fields."""
    response = upload(client, report_artifact_bytes, filename="")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_multipart"


def test_a_raw_carriage_return_in_a_filename_breaks_multipart_framing(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    body, content_type = multipart_body_parts([("report", "report\r.json", report_artifact_bytes)])

    response = post_multipart(client, body, content_type)

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_multipart"


def test_a_long_filename_is_capped(client: TestClient, report_artifact_bytes: bytes) -> None:
    payload = upload(client, report_artifact_bytes, filename="a" * 400 + ".json").json()

    assert len(payload["filename"]) == MAX_DISPLAY_NAME_LENGTH


def test_the_filename_is_never_opened(
    client: TestClient, report_artifact_bytes: bytes, tmp_path: Path
) -> None:
    """A filename naming a real file of garbage must not change what was loaded.

    The bytes uploaded are the artifact; the file on disk is not. If the route resolved
    the name against the filesystem, this would either fail validation or load the junk.
    """
    decoy = tmp_path / "report.json"
    decoy.write_text("this is not the artifact", encoding="utf-8")

    payload = upload(client, report_artifact_bytes, filename=str(decoy)).json()

    assert payload["filename"] == "report.json"
    assert payload["corpus"]["record_count"] == 7
    assert decoy.read_text(encoding="utf-8") == "this is not the artifact"


def test_no_file_is_written_for_an_upload(
    client: TestClient,
    report_artifact_bytes: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fresh working directory: the artifact fixture already wrote report.json into
    # tmp_path, so asserting on tmp_path itself would only find the fixture's own file.
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    upload(client, report_artifact_bytes, filename="uploaded-report.json")

    assert list(work.iterdir()) == []


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(
            lambda doc: doc.update(schema="video-learning.report/v2"), id="foreign-schema"
        ),
        pytest.param(lambda doc: doc.pop("timeline"), id="missing-section"),
        pytest.param(
            lambda doc: doc["segments"]["segments"][0].update(boundary_reason="chapter"),
            id="malformed-boundary-reason",
        ),
        pytest.param(
            lambda doc: doc["segments"].update(merge_tolerance_seconds=4.0),
            id="disagreeing-assembly-constant",
        ),
        pytest.param(
            lambda doc: doc["segments"].update(video_duration_seconds="24"),
            id="quoted-duration",
        ),
        pytest.param(
            lambda doc: _event(doc, "ocr").update(timestamp_seconds=True),
            id="boolean-timestamp",
        ),
        pytest.param(
            lambda doc: doc["timeline"].append({"type": "caption", "text": "hi"}),
            id="unknown-timeline-event",
        ),
    ],
)
def test_an_invalid_artifact_is_400_with_a_stable_generic_detail(
    client: TestClient, report_artifact_bytes: bytes, edit: Callable[[dict[str, Any]], None]
) -> None:
    data = _edited(report_artifact_bytes, edit)

    response = upload(client, data, filename="broken.json")

    assert response.status_code == 400, response.text
    payload = response.json()
    assert set(payload) == {"code", "detail"}
    assert payload["code"] == "invalid_report"
    assert payload["detail"] == INVALID_REPORT_DETAIL
    # The refusal is still the store's decision on the store's rules — its own detailed
    # message exists, it just never travels to the client (kept for server-side use).
    with pytest.raises(ReportArtifactError) as exc:
        ReportStore().load_evidence_from_text(data.decode("utf-8"), label="broken.json")
    assert str(exc.value) != INVALID_REPORT_DETAIL
    assert len(store_of(client)) == 0


@pytest.mark.parametrize(
    "data",
    [
        b"{not json",
        b"[]",
        b"",
    ],
)
def test_malformed_json_is_400(client: TestClient, data: bytes) -> None:
    response = upload(client, data)

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_report"
    assert response.json()["detail"] == INVALID_REPORT_DETAIL


def test_an_invalid_artifact_detail_never_echoes_uploaded_content(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """A long attacker-controlled schema value (and filename) stays out of the response.

    The store's exception interpolates both into its message; the F-2 contract is that the
    HTTP detail is one fixed sentence no matter what the artifact says, and the body has a
    size bound a 4096-byte upload cannot stretch.
    """
    marker = "SENTINEL7f3a"
    sentinel = f"{marker}-" + "x" * 4000
    data = _edited(report_artifact_bytes, lambda doc: doc.update(schema=sentinel))
    with pytest.raises(ReportArtifactError) as store_exc:
        ReportStore().load_evidence_from_text(data.decode("utf-8"), label=f"{marker}.json")
    assert marker in str(store_exc.value), "the store message must still be the detailed one"

    response = upload(client, data, filename=f"{marker}.json")

    assert response.status_code == 400
    payload = response.json()
    assert payload == {"code": "invalid_report", "detail": INVALID_REPORT_DETAIL}
    assert marker not in response.text
    assert len(response.content) < 1024
    assert len(store_of(client)) == 0


def test_non_utf8_bytes_are_400_invalid_encoding(client: TestClient) -> None:
    response = upload(client, b'{"schema": "\xff\xfe"}')

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_encoding"
    assert len(store_of(client)) == 0


def test_a_missing_file_field_is_422(client: TestClient) -> None:
    response = client.post("/api/reports")

    assert response.status_code == 422
    assert response.json()["code"] == "missing_report"


def test_a_wrongly_named_file_field_is_422(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    response = client.post(
        "/api/reports", files={"artifact": ("r.json", report_artifact_bytes, "application/json")}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "missing_report"


def test_a_text_field_named_report_is_not_an_upload(client: TestClient) -> None:
    """A non-file part is a form field, and ``max_fields=0`` refuses it before the lookup."""
    response = client.post("/api/reports", data={"report": "{}"})

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_multipart"


def test_two_files_are_refused(client: TestClient, report_artifact_bytes: bytes) -> None:
    body, content_type = multipart_body_parts(
        [
            ("report", "r.json", report_artifact_bytes),
            ("report", "second.json", report_artifact_bytes),
        ]
    )

    response = post_multipart(client, body, content_type)

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_multipart"


def test_an_extra_form_field_is_refused(client: TestClient, report_artifact_bytes: bytes) -> None:
    response = client.post(
        "/api/reports",
        files={"report": ("r.json", report_artifact_bytes, "application/json")},
        data={"note": "hello"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_multipart"


def test_the_size_cap_is_inclusive_and_counts_the_whole_body(
    report_artifact_bytes: bytes,
) -> None:
    """Exactly ``max_report_bytes`` of request body is accepted; one byte less is not.

    The cap counts multipart framing too, so it is built by hand here rather than inferred
    from the file size.
    """
    body, content_type = multipart_body(report_artifact_bytes)
    at_limit = client_for(max_report_bytes=len(body))
    over_limit = client_for(max_report_bytes=len(body) - 1)

    accepted = post_multipart(at_limit, body, content_type)
    refused = post_multipart(over_limit, body, content_type)

    assert accepted.status_code == 201, accepted.text
    assert accepted.json()["bytes_received"] == len(report_artifact_bytes)
    assert refused.status_code == 413
    assert refused.json()["code"] == "request_too_large"
    assert len(store_of(over_limit)) == 0


def test_bytes_received_is_the_file_not_the_request(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    body, content_type = multipart_body(report_artifact_bytes)

    payload = post_multipart(client, body, content_type).json()

    assert payload["bytes_received"] == len(report_artifact_bytes)
    assert payload["bytes_received"] < len(body)


def test_the_default_cap_is_eight_mebibytes() -> None:
    assert DEFAULT_MAX_REPORT_BYTES == 8 * 1024 * 1024


def test_a_no_audio_report_reports_its_transcription_state(
    client: TestClient, tmp_path: Path
) -> None:
    report = synthetic_report(tmp_path / "clip.mov")
    report = replace(
        report,
        transcription=None,
        transcription_status="no-audio",
        transcription_detail=None,
        # A no-audio report has no speech events at all; clearing only the transcript
        # would leave a timeline the real pipeline could never have written.
        timeline=build_timeline([OcrEvent(0.5, SYNTHETIC_OCR_TEXT)], []),
    )

    payload = upload(client, _bytes(tmp_path, report)).json()

    assert payload["transcription_status"] == "no-audio"
    assert payload["transcription_available"] is False
    assert payload["corpus"]["speech_record_count"] == 0
    assert payload["corpus"]["ocr_record_count"] == 1
    assert payload["corpus"]["record_count"] == 1


def test_an_unknown_duration_report_keeps_its_evidence(
    client: TestClient, tmp_path: Path
) -> None:
    report = synthetic_report(tmp_path / "clip.mov")
    report = replace(
        report,
        segmentation=replace(report.segmentation, segments=(), video_duration_seconds=None),
    )

    payload = upload(client, _bytes(tmp_path, report)).json()

    assert payload["video_duration_seconds"] is None
    assert payload["corpus"]["segment_count"] == 0
    assert payload["corpus"]["record_count"] == 7


def test_the_store_is_bounded_and_evicts_the_least_recently_used(
    report_artifact_bytes: bytes,
) -> None:
    client = client_for(max_sessions=2)

    first = report_id(client, report_artifact_bytes)
    second = report_id(client, report_artifact_bytes)
    search(client, second)
    third = report_id(client, report_artifact_bytes)

    assert len(store_of(client)) == 2
    assert store_of(client).evicted_count == 1
    assert search(client, first).status_code == 404
    assert search(client, third).status_code == 200


def test_health_counts_sessions_and_purges_expired_ones(report_artifact_bytes: bytes) -> None:
    now = [1000.0]
    client = client_for(idle_ttl_seconds=100.0, clock=lambda: now[0])
    report_id(client, report_artifact_bytes)
    assert client.get("/api/health").json()["active_report_sessions"] == 1

    now[0] += 101.0

    assert client.get("/api/health").json()["active_report_sessions"] == 0
    assert store_of(client).expired_count == 1


def test_health_advertises_the_configured_bounds() -> None:
    client = client_for(max_sessions=3, max_report_bytes=4096)

    payload = client.get("/api/health").json()

    assert payload["status"] == "ok"
    assert payload["max_report_sessions"] == 3
    assert payload["max_report_bytes"] == 4096
    assert payload["report_schema"] == REPORT_SCHEMA
    assert payload["search_schema"] == "video-learning.search/v1"