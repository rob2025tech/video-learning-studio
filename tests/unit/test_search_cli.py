"""CLI tests for the `search` command (real artifact written, no media processing).

The report artifact is written by the production store from an in-memory report, so the
command is exercised against the real on-disk shape. The video the artifact names is
never created: reading an artifact must not need it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from video_learning.cli.main import app
from video_learning.core.keywords import KeywordSuggestion, PhraseSuggestion
from video_learning.core.models import AudioStreamInfo, MediaInfo, VideoStreamInfo
from video_learning.core.search import DEFAULT_RESULT_LIMIT, SEARCH_SCHEMA
from video_learning.core.segments import Segment, Segmentation
from video_learning.core.timeline import OcrEvent, SpeechEvent, build_timeline
from video_learning.core.transcript import TranscriptResult, TranscriptSegment
from video_learning.services.analyze_service import AnalyzeResult, FrameSnapshot
from video_learning.services.report_service import ReportResult, ReportStore
from video_learning.services.search_service import SearchService

runner = CliRunner()

# Every adapter in this project shells out through subprocess.run, so patching that one
# function turns "the search command ran no pipeline" into a failure instead of a claim.
_NO_SUBPROCESS = patch.object(
    subprocess, "run", side_effect=AssertionError("search must not run a subprocess")
)


def _report(video: Path, ocr_text: str = "build a search index locally") -> ReportResult:
    """A report whose evidence shares one distinctive term, so ranking and limits show."""
    speech = tuple(
        TranscriptSegment(index * 4.0, index * 4.0 + 3.0, f"the index ranks record {index}")
        for index in range(6)
    )
    analysis = AnalyzeResult(
        source=video,
        keywords=[KeywordSuggestion("index", 7)],
        phrases=[PhraseSuggestion("search index", 2)],
        snapshots=[FrameSnapshot(0.5, ocr_text)],
        frames_analyzed=1,
        ocr_text_chars=len(ocr_text),
    )
    transcription = TranscriptResult(
        source=str(video),
        backend="whisper.cpp",
        model="ggml-base.en",
        language="en",
        segments=speech,
    )
    timeline = build_timeline(
        [OcrEvent(0.5, ocr_text)],
        [SpeechEvent(s.start_seconds, s.end_seconds, s.text) for s in speech],
    )
    segmentation = Segmentation(
        visual_changes=(),
        speech_boundaries=(),
        boundaries=(),
        segments=(Segment(start_seconds=0.0, end_seconds=24.0, boundary_reason=None),),
        video_duration_seconds=24.0,
    )
    return ReportResult(
        source=video,
        media=MediaInfo(
            path=video,
            filename=video.name,
            container_format="mov",
            duration_seconds=24.0,
            video=VideoStreamInfo(
                codec="h264", width=1920, height=1080, frame_rate=30.0, pixel_format="yuv420p"
            ),
            audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2, language="und"),
        ),
        analysis=analysis,
        transcription=transcription,
        transcription_status="ok",
        transcription_detail=None,
        segmentation=segmentation,
        timeline=timeline,
    )


def _artifact(tmp_path: Path) -> Path:
    report = _report(tmp_path / "clip.mov")
    out = tmp_path / "report.json"
    ReportStore().save(report, out)
    return out


def _payload(result: Any) -> dict[str, Any]:
    return json.loads(result.output)


def test_search_lists_matches_with_timestamps(tmp_path: Path) -> None:
    out = _artifact(tmp_path)

    with _NO_SUBPROCESS:
        result = runner.invoke(app, ["search", str(out), "index"])

    assert result.exit_code == 0, result.output
    assert "SEARCH" in result.output
    assert "index" in result.output
    # Speech matches print their range, OCR matches their single point.
    assert "00:00:04.0" in result.output
    assert "SPEECH" in result.output
    assert "OCR" in result.output
    assert "segment 0" in result.output
    assert "matched index" in result.output
    assert "the video was not opened" in result.output


def test_search_ocr_line_shows_the_match_not_the_record_opening(tmp_path: Path) -> None:
    """A long OCR record must surface its matched phrase, not its opening words.

    The shared fixture's OCR record is too short to ever be truncated, so it cannot
    exercise this; here the only match sits past offset 280.
    """
    ocr_text = (
        "Preferences\n\n"
        + "General   Appearance   Advanced\n" * 8
        + "\nAdjust the histogram until the exposure looks right.\n"
    )
    report = _report(tmp_path / "clip.mov", ocr_text)
    out = tmp_path / "report.json"
    ReportStore().save(report, out)

    with _NO_SUBPROCESS:
        result = runner.invoke(app, ["search", str(out), "histogram"])

    assert result.exit_code == 0, result.output
    # Whole words, because rich wraps the evidence line at the terminal width.
    assert "Adjust" in result.output
    assert "exposure" in result.output
    assert "Preferences" not in result.output


def test_search_json_is_exactly_the_result_model(tmp_path: Path) -> None:
    out = _artifact(tmp_path)
    expected = SearchService(ReportStore().load_evidence(out)).query("index")

    with _NO_SUBPROCESS:
        result = runner.invoke(app, ["search", str(out), "index", "--json"])

    assert result.exit_code == 0, result.output
    payload = _payload(result)
    assert payload == expected.to_dict()
    assert payload["schema"] == SEARCH_SCHEMA
    assert payload["status"] == "ok"
    assert payload["limit"] == DEFAULT_RESULT_LIMIT
    # Verbatim evidence text and its provenance survive into the JSON.
    texts = [record["text"] for record in payload["results"]]
    assert "build a search index locally" in texts


def test_search_limit_caps_results_without_caping_candidates(tmp_path: Path) -> None:
    out = _artifact(tmp_path)

    with _NO_SUBPROCESS:
        result = runner.invoke(app, ["search", str(out), "index", "--limit", "2", "--json"])

    assert result.exit_code == 0, result.output
    payload = _payload(result)
    coverage = payload["coverage"]
    assert len(payload["results"]) == 2
    assert coverage["returned_record_count"] == 2
    assert coverage["matched_record_count"] == 7
    assert payload["status"] == "ok"


def test_search_zero_limit_reports_counts_only(tmp_path: Path) -> None:
    out = _artifact(tmp_path)

    with _NO_SUBPROCESS:
        result = runner.invoke(app, ["search", str(out), "index", "--limit", "0"])

    assert result.exit_code == 0, result.output
    assert "(no matches)" in result.output
    assert "asked for none" in result.output
    assert "7 records matched" in result.output


def test_search_explains_a_query_that_matches_nothing(tmp_path: Path) -> None:
    out = _artifact(tmp_path)

    with _NO_SUBPROCESS:
        result = runner.invoke(app, ["search", str(out), "histogram"])

    assert result.exit_code == 0, result.output
    assert "Status: no_confident_match" in result.output
    assert "(no matches)" in result.output
    assert "appears in the 7 searchable records" in result.output


def test_search_stops_at_the_artifact_never_the_video(tmp_path: Path) -> None:
    out = _artifact(tmp_path)
    # The recorded source was never created, so any media access would fail outright.
    assert not (tmp_path / "clip.mov").exists()

    with _NO_SUBPROCESS:
        result = runner.invoke(app, ["search", str(out), "index"])

    assert result.exit_code == 0, result.output


def test_search_missing_artifact_exits_1_without_a_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", str(tmp_path / "nope.json"), "index"])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Could not read report artifact" in result.output
    assert "Traceback" not in result.output


def test_search_malformed_artifact_exits_1_without_a_traceback(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    out.write_text("{not json", encoding="utf-8")

    result = runner.invoke(app, ["search", str(out), "index"])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "is not valid JSON" in result.output
    assert "Traceback" not in result.output


def test_search_invalid_utf8_exits_1_without_a_traceback(tmp_path: Path) -> None:
    """A byte-level corruption is an artifact error, not an escaped ``UnicodeDecodeError``.

    ``catch_exceptions=False`` is what makes this test mean something: the runner would
    otherwise swallow an escaping exception into exit code 1 and a stored traceback, and
    every assertion below would still pass.
    """
    out = _artifact(tmp_path)
    out.write_bytes(out.read_bytes().replace(b"clip.mov", b"clip\xff.mov", 1))

    result = runner.invoke(app, ["search", str(out), "index"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Could not read report artifact" in result.output
    assert "utf-8" in result.output
    assert "Traceback" not in result.output


def test_search_foreign_schema_exits_1_without_a_traceback(tmp_path: Path) -> None:
    out = _artifact(tmp_path)
    document = json.loads(out.read_text(encoding="utf-8"))
    document["schema"] = "video-learning.report/v0"
    out.write_text(json.dumps(document), encoding="utf-8")

    result = runner.invoke(app, ["search", str(out), "index"])

    assert result.exit_code == 1
    assert "Unsupported report artifact schema" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["{artifact}", "index", "--limit", "abc"], "'abc' is not a valid int"),
        (["{artifact}"], "Missing argument 'query'"),
        ([], "Missing argument 'artifact'"),
        (["{artifact}", "index", "extra"], "Got unexpected extra argument"),
    ],
    ids=["non-integer-limit", "missing-query", "missing-artifact", "extra-positional"],
)
def test_search_rejects_malformed_arguments(tmp_path: Path, args: list[str], expected: str) -> None:
    """Usage errors exit 2 with an explanation, before the artifact is ever read.

    Exit 2 is the parser refusing the command line; exit 1 means the command ran and the
    artifact was unusable. ``--limit -1`` is deliberately not in this list: it is a valid
    int, and the service reports the counts for it.
    """
    out = _artifact(tmp_path)
    argv = [str(out) if part == "{artifact}" else part for part in args]

    result = runner.invoke(app, ["search", *argv], catch_exceptions=False)

    assert result.exit_code == 2, result.output
    assert "Usage:" in result.output
    assert expected in result.output
    assert "Traceback" not in result.output
