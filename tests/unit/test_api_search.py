"""``POST /api/search``: the CLI's own answer, with the path redacted and nothing else.

The central assertion is a dict equality against ``video-learning search --json`` run on
the same artifact bytes. That is what makes "preserve existing serialization semantics"
a checked claim rather than an intention: any drift in ranking, coverage, span offsets or
key names shows up here as a difference, not as a passing test of a re-modelled schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.conftest import synthetic_report
from tests.unit.api_support import client_for, report_id, search, store_of
from video_learning.api.schemas import MAX_QUERY_LENGTH
from video_learning.cli.main import app as cli_app
from video_learning.core.search import DEFAULT_RESULT_LIMIT, SEARCH_SCHEMA
from video_learning.services.report_service import ReportStore
from video_learning.services.search_service import SearchService

runner = CliRunner()


@pytest.fixture
def client() -> TestClient:
    return client_for()


def _payload(client: TestClient, rid: str, query: str = "index", **fields: Any) -> dict[str, Any]:
    response = search(client, rid, query, **fields)
    assert response.status_code == 200, response.text
    payload: dict[str, Any] = response.json()
    return payload


def test_the_answer_is_the_core_payload_with_only_the_source_redacted(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    rid = report_id(client, report_artifact_bytes)
    evidence = ReportStore().load_evidence_from_text(
        report_artifact_bytes.decode("utf-8"), label="report.json"
    )
    expected = SearchService(evidence).query("index", limit=DEFAULT_RESULT_LIMIT).to_dict()

    payload = _payload(client, rid)

    assert payload == {**expected, "source": "clip.mov", "source_is_redacted": True}
    assert payload["schema"] == SEARCH_SCHEMA


@pytest.mark.parametrize(
    ("query", "limit"),
    [
        ("index", 5),
        ("index", 0),
        ("index", -3),
        ("index", 2),
        ("index", 100),
        ("", 5),
        ("the", 5),
        ("record 3", 2),
        ("search index", 100),
        ("zzzznotpresent", 5),
    ],
)
def test_the_api_and_the_cli_agree_exactly(
    client: TestClient,
    report_artifact_bytes: bytes,
    tmp_path: Path,
    query: str,
    limit: int,
) -> None:
    artifact = tmp_path / "cli-report.json"
    artifact.write_bytes(report_artifact_bytes)
    rid = report_id(client, report_artifact_bytes)

    cli = runner.invoke(cli_app, ["search", str(artifact), query, "--limit", str(limit), "--json"])
    api = search(client, rid, query, limit=limit)

    assert cli.exit_code == 0, cli.output
    assert api.status_code == 200, api.text
    from_cli = json.loads(cli.stdout)
    assert api.json() == {**from_cli, "source": "clip.mov", "source_is_redacted": True}


@pytest.mark.parametrize("limit", [0, -3, -100])
def test_a_non_positive_limit_still_reports_status_and_coverage(
    client: TestClient, report_artifact_bytes: bytes, limit: int
) -> None:
    """Requirement: the API must not validate away the CLI's ``--limit 0`` behavior."""
    rid = report_id(client, report_artifact_bytes)

    payload = _payload(client, rid, limit=limit)

    assert payload["status"] == "ok"
    assert payload["limit"] == limit
    assert payload["results"] == []
    assert payload["result_count"] == 0
    assert payload["coverage"]["matched_record_count"] == 7
    assert payload["coverage"]["returned_record_count"] == 0
    assert payload["coverage"]["record_count"] == 7


def test_the_default_limit_is_the_cores_own(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    rid = report_id(client, report_artifact_bytes)

    payload = _payload(client, rid)

    assert payload["limit"] == DEFAULT_RESULT_LIMIT
    assert payload["result_count"] == DEFAULT_RESULT_LIMIT


def test_a_limit_caps_output_not_consideration(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    rid = report_id(client, report_artifact_bytes)

    payload = _payload(client, rid, limit=2)

    assert payload["result_count"] == 2
    assert payload["coverage"]["matched_record_count"] == 7
    scores = [item["score"] for item in payload["results"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    ("query", "meaningful"),
    [("", 0), ("the", 0), ("zzzznotpresent", 1)],
)
def test_an_unanswerable_query_is_no_confident_match_not_an_error(
    client: TestClient, report_artifact_bytes: bytes, query: str, meaningful: int
) -> None:
    rid = report_id(client, report_artifact_bytes)

    payload = _payload(client, rid, query)

    assert payload["status"] == "no_confident_match"
    assert payload["results"] == []
    assert payload["coverage"]["meaningful_query_term_count"] == meaningful
    assert payload["coverage"]["matched_record_count"] == 0


def test_an_empty_corpus_is_no_evidence(client: TestClient, tmp_path: Path) -> None:
    report = synthetic_report(tmp_path / "clip.mov", speech_records=0, ocr_text="")
    artifact = tmp_path / "empty.json"
    ReportStore().save(report, artifact)
    rid = report_id(client, artifact.read_bytes())

    payload = _payload(client, rid)

    assert payload["status"] == "no_evidence"
    # An empty OCR snapshot is still a record; what is absent is anything searchable.
    assert payload["coverage"]["record_count"] == 1
    assert payload["coverage"]["searchable_record_count"] == 0


def test_matched_spans_index_the_text_they_ship_with(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """The highlighting contract: offsets into the returned text, never into a rewrite."""
    rid = report_id(client, report_artifact_bytes)

    payload = _payload(client, rid, "index")

    assert payload["results"]
    for item in payload["results"]:
        for span in item["matched_spans"]:
            assert item["text"][span["start"] : span["end"]] == span["surface"]
        assert item["matched_terms"]


def test_evidence_shapes_survive_the_wire(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    rid = report_id(client, report_artifact_bytes)

    payload = _payload(client, rid, "index", limit=100)

    kinds = {item["source_type"] for item in payload["results"]}
    assert kinds == {"ocr", "speech"}
    for item in payload["results"]:
        assert set(item) == {
            "evidence_id",
            "source_type",
            "text",
            "start_seconds",
            "end_seconds",
            "segment_id",
            "score",
            "matched_terms",
            "matched_spans",
        }
        if item["source_type"] == "ocr":
            assert item["end_seconds"] is None
        else:
            assert item["end_seconds"] > item["start_seconds"]
        assert item["segment_id"] == 0


def test_no_absolute_path_appears_anywhere_in_the_answer(
    client: TestClient, report_artifact_bytes: bytes, tmp_path: Path
) -> None:
    rid = report_id(client, report_artifact_bytes)

    payload = _payload(client, rid, "index", limit=100)

    assert payload["source"] == "clip.mov"
    assert payload["source_is_redacted"] is True
    assert str(tmp_path) not in repr(payload)
    assert "/Users/" not in repr(payload)
    assert "/private/var/" not in repr(payload)


def test_queries_reuse_the_index_built_at_upload(
    client: TestClient, report_artifact_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After upload, re-reading the artifact would be a bug: make it raise and search anyway."""
    rid = report_id(client, report_artifact_bytes)
    service = store_of(client).get(rid).service

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("search must not reload the artifact")

    monkeypatch.setattr(ReportStore, "load_evidence_from_text", _refuse)
    monkeypatch.setattr(ReportStore, "load_evidence", _refuse)

    for _ in range(3):
        assert _payload(client, rid)["result_count"] == 5
    assert store_of(client).get(rid).service is service


def test_two_sessions_do_not_share_answers(client: TestClient, tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    ReportStore().save(synthetic_report(tmp_path / "a.mov", ocr_text="alpha widget"), first)
    ReportStore().save(synthetic_report(tmp_path / "b.mov", ocr_text="beta gadget"), second)
    first_id = report_id(client, first.read_bytes())
    second_id = report_id(client, second.read_bytes())

    assert _payload(client, first_id, "widget")["source"] == "a.mov"
    assert _payload(client, second_id, "gadget")["source"] == "b.mov"
    assert _payload(client, first_id, "gadget")["status"] == "no_confident_match"
    assert _payload(client, second_id, "widget")["status"] == "no_confident_match"


def test_an_unknown_report_id_is_404(client: TestClient) -> None:
    response = search(client, "not-a-session")

    assert response.status_code == 404
    assert response.json()["code"] == "unknown_report"


def test_an_expired_session_is_404_report_expired(report_artifact_bytes: bytes) -> None:
    now = [1000.0]
    client = client_for(idle_ttl_seconds=100.0, clock=lambda: now[0])
    rid = report_id(client, report_artifact_bytes)
    assert search(client, rid).status_code == 200

    now[0] += 101.0

    response = search(client, rid)
    assert response.status_code == 404
    assert response.json()["code"] == "report_expired"
    assert len(store_of(client)) == 0


def test_an_evicted_session_asks_for_the_file_again(report_artifact_bytes: bytes) -> None:
    client = client_for(max_sessions=1)
    first = report_id(client, report_artifact_bytes)
    report_id(client, report_artifact_bytes)

    response = search(client, first)

    assert response.status_code == 404
    assert response.json()["code"] == "unknown_report"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"query": "index"}, id="missing-report-id"),
        pytest.param({"report_id": "", "query": "index"}, id="empty-report-id"),
        pytest.param({"report_id": "x" * 129, "query": "index"}, id="overlong-report-id"),
        pytest.param(
            {"report_id": "x", "query": "q" * (MAX_QUERY_LENGTH + 1)}, id="overlong-query"
        ),
        pytest.param({"report_id": "x", "query": 123}, id="non-string-query"),
        pytest.param({"report_id": "x", "query": "index", "limit": 2.5}, id="fractional-limit"),
        pytest.param({"report_id": "x", "query": "index", "extra": 1}, id="unknown-field"),
    ],
)
def test_invalid_requests_are_422(client: TestClient, body: dict[str, Any]) -> None:
    response = client.post("/api/search", json=body)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_a_query_at_the_length_bound_is_accepted(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    rid = report_id(client, report_artifact_bytes)

    response = search(client, rid, "q" * MAX_QUERY_LENGTH)

    assert response.status_code == 200


def test_a_non_json_body_is_422(client: TestClient) -> None:
    response = client.post("/api/search", content=b"not json")

    assert response.status_code == 422


def test_a_json_array_body_is_422(client: TestClient) -> None:
    response = client.post("/api/search", json=[])

    assert response.status_code == 422


def test_search_is_not_a_get_route(client: TestClient) -> None:
    assert client.get("/api/search").status_code == 405


def test_there_is_no_media_or_report_build_route(client: TestClient) -> None:
    """Decision 1 and Decision 3: no streaming, no upload of video, no pipeline over HTTP."""
    paths = {route.path for route in client.app.routes}  # type: ignore[attr-defined]

    assert paths == {"/api/health", "/api/reports", "/api/search", "/api/docs", "/api/openapi.json"}
    assert not any("media" in path or "video" in path for path in paths)


def test_the_openapi_document_describes_only_these_routes(client: TestClient) -> None:
    document = client.get("/api/openapi.json").json()

    assert set(document["paths"]) == {"/api/health", "/api/reports", "/api/search"}
    assert document["info"]["title"] == "LearnLens local evidence API"


def test_uploading_a_report_whose_video_never_existed_still_searches(
    client: TestClient, report_artifact_bytes: bytes, tmp_path: Path
) -> None:
    assert not (tmp_path / "clip.mov").exists()
    rid = report_id(client, report_artifact_bytes)

    assert _payload(client, rid)["coverage"]["matched_record_count"] == 7
    assert not (tmp_path / "clip.mov").exists()