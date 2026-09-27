"""CLI tests for the `report` command (services mocked; real artifact written).

The report model is built directly and ``ReportService.build`` is patched, so no
ffmpeg/tesseract/whisper.cpp is needed. The ``--out`` artifact write is real.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from video_learning.cli.main import app
from video_learning.core.keywords import KeywordSuggestion, PhraseSuggestion
from video_learning.core.models import AudioStreamInfo, MediaInfo, VideoStreamInfo
from video_learning.core.timeline import OcrEvent, SpeechEvent, build_timeline
from video_learning.core.transcript import TranscriptResult, TranscriptSegment
from video_learning.services.analyze_service import AnalyzeResult, FrameSnapshot
from video_learning.services.report_service import REPORT_SCHEMA, ReportResult, ReportService

runner = CliRunner()


def _report(video: Path) -> ReportResult:
    media = MediaInfo(
        path=video,
        filename=video.name,
        container_format="mov",
        duration_seconds=100.0,
        video=VideoStreamInfo(
            codec="h264", width=1920, height=1080, frame_rate=30.0, pixel_format="yuv420p"
        ),
        audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2, language="und"),
    )
    analysis = AnalyzeResult(
        source=video,
        keywords=[KeywordSuggestion("Qoder", 3)],
        phrases=[PhraseSuggestion("pyproject.toml", 2)],
        snapshots=[FrameSnapshot(15.2, "video-learning-studio pyproject.toml")],
        frames_analyzed=1,
        ocr_text_chars=34,
    )
    transcription = TranscriptResult(
        source=str(video),
        backend="whisper.cpp",
        model="ggml-base.en",
        language="en",
        segments=(TranscriptSegment(17.4, 23.1, "Today we build"),),
    )
    timeline = build_timeline(
        [OcrEvent(15.2, "video-learning-studio pyproject.toml")],
        [SpeechEvent(17.4, 23.1, "Today we build")],
    )
    return ReportResult(
        source=video,
        media=media,
        analysis=analysis,
        transcription=transcription,
        transcription_status="ok",
        transcription_detail=None,
        timeline=timeline,
    )


def test_report_renders_human_readable_summary(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")

    with patch.object(ReportService, "build", return_value=_report(video)):
        result = runner.invoke(app, ["report", str(video)])

    assert result.exit_code == 0, result.output
    assert "VIDEO REPORT" in result.output
    assert "Metadata" in result.output
    assert "UNIFIED TIMELINE" in result.output
    assert "OCR" in result.output
    assert "SPEECH" in result.output
    assert "pyproject.toml" in result.output


def test_report_writes_artifact_with_out(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    out = tmp_path / "reports" / "report.json"

    with patch.object(ReportService, "build", return_value=_report(video)):
        result = runner.invoke(app, ["report", str(video), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == REPORT_SCHEMA
    assert data["transcription"]["status"] == "ok"
    assert len(data["timeline"]) == 2
    # The complete OCR text is preserved in the artifact.
    assert data["ocr"]["snapshots"][0]["ocr_text"] == "video-learning-studio pyproject.toml"
    assert "Artifact written" in result.output


def test_report_json_is_exactly_result_to_dict(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    report = _report(video)

    with patch.object(ReportService, "build", return_value=report):
        result = runner.invoke(app, ["report", str(video), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == report.to_dict()
    # No CLI-only fields leak into --json.
    assert "artifact_path" not in payload


def test_report_json_still_writes_out(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    out = tmp_path / "report.json"

    with patch.object(ReportService, "build", return_value=_report(video)):
        result = runner.invoke(app, ["report", str(video), "--json", "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = json.loads(result.output)
    # --json stays pure even when the artifact is written.
    assert "artifact_path" not in payload
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == REPORT_SCHEMA


def test_report_forwards_model_option(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    model = tmp_path / "ggml-base.en.bin"
    model.write_bytes(b"x")

    with patch.object(ReportService, "build", return_value=_report(video)) as mocked:
        result = runner.invoke(app, ["report", str(video), "--model", str(model)])

    assert result.exit_code == 0, result.output
    assert mocked.call_args is not None
    assert mocked.call_args.kwargs["model"] == model


def test_report_missing_video_exits_1(tmp_path: Path) -> None:
    # No mocking: InspectService validates existence before any subprocess runs.
    out = tmp_path / "report.json"

    result = runner.invoke(app, ["report", str(tmp_path / "nope.mov"), "--out", str(out)])

    assert result.exit_code == 1
    assert "File not found" in result.output
    # The report failed before writing, so no artifact exists.
    assert not out.exists()
