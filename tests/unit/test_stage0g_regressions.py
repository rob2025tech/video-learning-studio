"""Regression guards: Stage 0G must not disturb Stage 0E audio or Stage 0F OCR.

The report layer is additive orchestration that *reuses* the earlier result
models; these lock their serialization contracts and prove the new ``report``
command did not displace the existing ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from video_learning.cli.main import app
from video_learning.core.keywords import KeywordSuggestion, PhraseSuggestion
from video_learning.core.transcript import TranscriptResult, TranscriptSegment
from video_learning.services.analyze_service import AnalyzeResult, FrameSnapshot
from video_learning.services.ocr_snapshot_store import OCR_ARTIFACT_SCHEMA, OcrSnapshotStore

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
    assert payload["segments"] == [
        {"start_seconds": 0.0, "end_seconds": 3.84, "text": "Hello"},
        {"start_seconds": 3.84, "end_seconds": 7.2, "text": "world"},
    ]
    assert payload["segment_count"] == 2
    assert payload["source_modified"] is False


def test_stage0f_analyze_to_dict_contract_unchanged(tmp_path: Path) -> None:
    result = AnalyzeResult(
        source=tmp_path / "clip.mov",
        keywords=[KeywordSuggestion("Qoder", 2)],
        phrases=[PhraseSuggestion("pyproject.toml", 2)],
        snapshots=[FrameSnapshot(1.0, "pyproject.toml")],
        frames_analyzed=1,
        ocr_text_chars=14,
    )

    payload = result.to_dict()

    assert set(payload) == _ANALYZE_KEYS
    assert payload["snapshots"] == [{"timestamp_seconds": 1.0, "ocr_text": "pyproject.toml"}]


def test_stage0f_ocr_store_round_trip_still_works(tmp_path: Path) -> None:
    original = AnalyzeResult(
        source=tmp_path / "clip.mov",
        keywords=[KeywordSuggestion("Qoder", 2)],
        phrases=[PhraseSuggestion("pyproject.toml", 2)],
        snapshots=[FrameSnapshot(1.0, "pyproject.toml")],
        frames_analyzed=1,
        ocr_text_chars=14,
    )
    out = tmp_path / "ocr.json"

    OcrSnapshotStore().save(original, out)
    reloaded = OcrSnapshotStore().load(out)

    assert reloaded == original
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == OCR_ARTIFACT_SCHEMA


def test_all_commands_registered() -> None:
    # A missing command would exit non-zero on --help, proving `report` was added
    # without displacing inspect/analyze/transcribe-audio/save-ocr.
    for name in ("inspect", "analyze", "transcribe-audio", "save-ocr", "report"):
        res = runner.invoke(app, [name, "--help"])
        assert res.exit_code == 0, f"{name} missing: {res.output}"


def test_save_ocr_still_requires_out(tmp_path: Path) -> None:
    # Stage 0F behaviour is unchanged by the addition of `report`.
    res = runner.invoke(app, ["save-ocr", str(tmp_path / "clip.mov")])
    assert res.exit_code != 0
