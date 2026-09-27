"""Regression guards: Stage 0F must not disturb analyze/OCR or Stage 0E audio.

These lock the pre-existing serialization contracts so the persistence layer is
provably additive rather than a redesign of earlier stages.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from video_learning.cli.main import app
from video_learning.core.keywords import KeywordSuggestion, PhraseSuggestion
from video_learning.core.transcript import TranscriptResult, TranscriptSegment
from video_learning.services.analyze_service import AnalyzeResult, FrameSnapshot

runner = CliRunner()

_ANALYZE_KEYS = {
    "source",
    "snapshots",
    "keywords",
    "phrases",
    "frames_analyzed",
    "ocr_text_chars",
    "renamed",
    "applied",
    "source_modified",
}


def test_analyze_to_dict_contract_unchanged(tmp_path: Path) -> None:
    result = AnalyzeResult(
        source=tmp_path / "clip.mov",
        keywords=[KeywordSuggestion("Qoder", 2)],
        phrases=[PhraseSuggestion("Release Notes", 2)],
        snapshots=[FrameSnapshot(1.0, "Release Notes")],
        frames_analyzed=1,
        ocr_text_chars=13,
    )

    payload = result.to_dict()

    assert set(payload) == _ANALYZE_KEYS
    assert payload["snapshots"] == [{"timestamp_seconds": 1.0, "ocr_text": "Release Notes"}]
    assert payload["renamed"] is False
    assert payload["source_modified"] is False


def test_analyze_result_dict_round_trip_is_symmetric(tmp_path: Path) -> None:
    original = AnalyzeResult(
        source=tmp_path / "clip.mov",
        keywords=[KeywordSuggestion("Qoder", 2)],
        phrases=[PhraseSuggestion("Release Notes", 2)],
        snapshots=[FrameSnapshot(1.0, "Release Notes"), FrameSnapshot(2.0, "Qoder")],
        frames_analyzed=2,
        ocr_text_chars=26,
    )

    assert AnalyzeResult.from_dict(original.to_dict()) == original


def test_frame_snapshot_still_binds_timestamp_to_text() -> None:
    snapshot = FrameSnapshot(3.5, "Qoder Model Usage")

    assert snapshot.to_dict() == {"timestamp_seconds": 3.5, "ocr_text": "Qoder Model Usage"}


def test_stage0e_transcript_serialization_unchanged() -> None:
    result = TranscriptResult(
        source="clip.mov",
        backend="whisper.cpp",
        model="ggml-base.en",
        language="en",
        segments=(
            TranscriptSegment(0.0, 3.84, "Hello"),
            TranscriptSegment(3.84, 7.2, "world"),
        ),
    )

    payload = result.to_dict()

    assert payload["backend"] == "whisper.cpp"
    assert payload["model"] == "ggml-base.en"
    assert payload["language"] == "en"
    assert payload["segments"] == [
        {"start_seconds": 0.0, "end_seconds": 3.84, "text": "Hello"},
        {"start_seconds": 3.84, "end_seconds": 7.2, "text": "world"},
    ]
    assert payload["source_modified"] is False


def test_existing_and_new_commands_all_registered() -> None:
    # A missing command would exit non-zero on --help, proving save-ocr was added
    # without displacing inspect/analyze/transcribe-audio.
    for name in ("inspect", "analyze", "transcribe-audio", "save-ocr"):
        res = runner.invoke(app, [name, "--help"])
        assert res.exit_code == 0, f"{name} missing: {res.output}"
