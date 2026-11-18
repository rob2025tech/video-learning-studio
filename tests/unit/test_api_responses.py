"""API output rules: untrusted names become display basenames, evidence stays verbatim.

These are pure-function tests. The redaction contract is checked against the core's own
``SearchResults.to_dict()`` so a future change to the core shape shows up here as a
difference rather than being silently absorbed by a hand-written expectation.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from video_learning.api.responses import (
    MAX_DISPLAY_NAME_LENGTH,
    UNKNOWN_VIDEO_NAME,
    safe_display_name,
    search_payload,
)
from video_learning.core.search import SearchResults
from video_learning.services.report_service import ReportStore
from video_learning.services.search_service import SearchService


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.json", "report.json"),
        ("/Users/someone/Movies/report.json", "report.json"),
        ("C:\\Users\\someone\\report.json", "report.json"),
        ("C:/Users/someone/report.json", "report.json"),
        ("..\\..\\Windows\\report.json", "report.json"),
        ("../../.ssh/id_rsa", "id_rsa"),
        ("dir/sub\\deep/report.json", "report.json"),
        ("/report.json", "report.json"),
        ("report.json\x00.png", "report.json.png"),
        ("report\n.json", "report.json"),
        ("  report.json  ", "report.json"),
        ("rapport été.json", "rapport été.json"),
    ],
)
def test_safe_display_name_reduces_to_a_basename(raw: str, expected: str) -> None:
    assert safe_display_name(raw, fallback="fallback.json") == expected


@pytest.mark.parametrize("raw", [None, "", "   ", ".", "..", "/", "\\", "C:\\", "dir/.."])
def test_safe_display_name_falls_back_when_nothing_usable_remains(raw: str | None) -> None:
    assert safe_display_name(raw, fallback="fallback.json") == "fallback.json"


def test_safe_display_name_caps_length() -> None:
    assert len(safe_display_name("x" * 900, fallback="f")) == MAX_DISPLAY_NAME_LENGTH


@pytest.mark.parametrize(
    "raw", ["/etc/passwd", "C:\\Windows\\System32\\config\\SAM", "../../secrets.json"]
)
def test_safe_display_name_never_returns_a_path(raw: str) -> None:
    """The result is display metadata: no separator survives, so it cannot name a directory."""
    name = safe_display_name(raw, fallback="f")
    assert "/" not in name
    assert "\\" not in name
    assert not Path(name).is_absolute()


def _results(text: str) -> SearchResults:
    evidence = ReportStore().load_evidence_from_text(text, label="report.json")
    return SearchService(evidence).query("index locally")


def test_search_payload_is_the_core_dict_with_one_provenance_change(
    report_artifact_text: str,
) -> None:
    results = _results(report_artifact_text)
    core = results.to_dict()

    payload = search_payload(results)

    assert payload["source"] == "clip.mov"
    assert payload["source_is_redacted"] is True
    assert {key: value for key, value in payload.items() if key != "source_is_redacted"} == {
        key: (value if key != "source" else "clip.mov") for key, value in core.items()
    }


def test_search_payload_leaks_no_absolute_path(report_artifact_text: str) -> None:
    payload = search_payload(_results(report_artifact_text))
    serialized = repr(payload)
    assert "/Users/" not in serialized
    assert "/private/var/" not in serialized
    assert "/tmp/" not in serialized
    assert str(Path(payload["source"]).parent) == "."


def test_search_payload_keeps_evidence_text_and_span_offsets_verbatim(
    report_artifact_text: str,
) -> None:
    """Redaction must not touch evidence: the spans index the text they arrive with."""
    results = _results(report_artifact_text)
    payload = search_payload(results)

    assert payload["results"]
    for record, original in zip(payload["results"], results.results, strict=True):
        assert record["text"] == original.evidence.text
        for span in record["matched_spans"]:
            assert record["text"][span["start"] : span["end"]] == span["surface"]


def test_search_payload_preserves_coverage_and_counts(report_artifact_text: str) -> None:
    results = _results(report_artifact_text)
    payload = search_payload(results)

    assert payload["coverage"] == results.to_dict()["coverage"]
    assert payload["result_count"] == len(results.results)
    assert payload["coverage"]["matched_record_count"] == 7
    assert payload["coverage"]["returned_record_count"] == 5


def test_search_payload_names_an_unknown_video_without_inventing_a_path(
    report_artifact_text: str,
) -> None:
    """A provenance path with no usable basename must not become an empty string."""
    document = json.loads(report_artifact_text)
    document["source"] = "/"
    evidence = ReportStore().load_evidence_from_text(json.dumps(document), label="report.json")
    results = replace(SearchService(evidence).query("index"), source=Path("/"))

    assert search_payload(results)["source"] == UNKNOWN_VIDEO_NAME
