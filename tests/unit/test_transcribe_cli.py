"""CLI tests for the print-only `transcribe-audio` command (service mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from video_learning.cli.main import app
from video_learning.core.transcript import (
    CapabilityItem,
    TranscriptionCapability,
    TranscriptResult,
    TranscriptSegment,
)
from video_learning.services.transcribe_service import TranscribeService

runner = CliRunner()

_READY = TranscriptionCapability(
    backend=CapabilityItem("whisper.cpp", "available", "/opt/bin/whisper-cli"),
    model=CapabilityItem("ggml-base.en", "available", "/models/ggml-base.en.bin"),
    recommended="whisper.cpp + ggml-base.en",
    recommended_detail="Local • English",
    using_backend="whisper.cpp",
    using_model="ggml-base.en",
    using_language="English",
    ready=True,
)

_NOT_READY = TranscriptionCapability(
    backend=CapabilityItem(
        "whisper.cpp",
        "missing",
        "whisper-cli not found on PATH — install whisper.cpp (brew install whisper-cpp)",
    ),
    model=CapabilityItem(
        "ggml-base.en", "missing", "model not found — pass --model PATH or set VLS_WHISPER_MODEL"
    ),
    recommended="whisper.cpp + ggml-base.en",
    recommended_detail="Local • English",
    using_backend="whisper.cpp",
    using_model="—",
    using_language="English",
    ready=False,
)


def _result(source: Path) -> TranscriptResult:
    return TranscriptResult(
        source=str(source),
        backend="whisper.cpp",
        model="ggml-base.en",
        language="en",
        segments=(
            TranscriptSegment(0.0, 3.84, "Today we build a tool."),
            TranscriptSegment(3.84, 7.2, "It reads a video."),
        ),
    )


def test_human_output_shows_capability_and_timeline(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    with (
        patch("video_learning.cli.main.WhisperCpp") as mock_whisper,
        patch.object(TranscribeService, "transcribe", return_value=_result(video)),
    ):
        mock_whisper.return_value.detect_capability.return_value = _READY
        result = runner.invoke(app, ["transcribe-audio", str(video)])

    assert result.exit_code == 0, result.output
    # Capability panel.
    assert "Transcription backends detected" in result.output
    assert "whisper.cpp" in result.output
    assert "ggml-base.en" in result.output
    assert "Recommended" in result.output
    # Transcript provenance + timestamped timeline.
    assert "Transcription" in result.output
    assert "Backend: whisper.cpp" in result.output
    assert "Language: en" in result.output
    assert "00:00:00.0" in result.output
    assert "00:00:03.8" in result.output
    assert "Today we build a tool." in result.output


def test_json_output_shape(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    with (
        patch("video_learning.cli.main.WhisperCpp") as mock_whisper,
        patch.object(TranscribeService, "transcribe", return_value=_result(video)),
    ):
        mock_whisper.return_value.detect_capability.return_value = _READY
        result = runner.invoke(app, ["transcribe-audio", str(video), "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["capability"]["ready"] is True
    transcription = data["transcription"]
    assert transcription["backend"] == "whisper.cpp"
    assert transcription["model"] == "ggml-base.en"
    assert transcription["language"] == "en"
    assert transcription["segments"] == [
        {"start_seconds": 0.0, "end_seconds": 3.84, "text": "Today we build a tool."},
        {"start_seconds": 3.84, "end_seconds": 7.2, "text": "It reads a video."},
    ]
    assert transcription["segment_count"] == 2
    assert transcription["source_modified"] is False
    assert transcription["renamed"] is False


def test_model_option_is_passed_through(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    model_path = tmp_path / "ggml-base.en.bin"
    model_path.write_bytes(b"x")
    with (
        patch("video_learning.cli.main.WhisperCpp") as mock_whisper,
        patch.object(
            TranscribeService, "transcribe", return_value=_result(video)
        ) as mock_tx,
    ):
        mock_whisper.return_value.detect_capability.return_value = _READY
        result = runner.invoke(
            app, ["transcribe-audio", str(video), "--model", str(model_path)]
        )

    assert result.exit_code == 0, result.output
    assert mock_tx.call_args is not None
    assert mock_tx.call_args.kwargs["model"] == model_path


def test_not_ready_human_shows_capability_and_exits_1(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    with patch("video_learning.cli.main.WhisperCpp") as mock_whisper:
        mock_whisper.return_value.detect_capability.return_value = _NOT_READY
        result = runner.invoke(app, ["transcribe-audio", str(video)])

    assert result.exit_code == 1
    assert "Transcription backends detected" in result.output
    assert "Error:" in result.output
    assert "brew install whisper-cpp" in result.output


def test_not_ready_json_exits_1(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    with patch("video_learning.cli.main.WhisperCpp") as mock_whisper:
        mock_whisper.return_value.detect_capability.return_value = _NOT_READY
        result = runner.invoke(app, ["transcribe-audio", str(video), "--json"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["transcription"] is None
    assert data["capability"]["ready"] is False
    assert "error" in data
