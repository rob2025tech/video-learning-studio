"""CLI tests for the `save-ocr` command (analyze mocked; real artifact written)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from video_learning.cli.main import app
from video_learning.core.keywords import KeywordSuggestion, PhraseSuggestion
from video_learning.services.analyze_service import AnalyzeResult, AnalyzeService, FrameSnapshot
from video_learning.services.ocr_snapshot_store import OCR_ARTIFACT_SCHEMA

runner = CliRunner()


def _result(source: Path) -> AnalyzeResult:
    return AnalyzeResult(
        source=source,
        keywords=[KeywordSuggestion("Qoder", 4)],
        phrases=[PhraseSuggestion("Qoder Model Usage", 3)],
        snapshots=[
            FrameSnapshot(12.4, "Projects Templates video-learning-studio"),
            FrameSnapshot(227.1, "Qoder Model Usage\nRelease Notes"),
        ],
        frames_analyzed=2,
        ocr_text_chars=64,
    )


def test_save_ocr_writes_artifact_and_reports(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    out = tmp_path / "evidence" / "ocr.json"
    with patch.object(AnalyzeService, "analyze", return_value=_result(video)):
        result = runner.invoke(app, ["save-ocr", str(video), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == OCR_ARTIFACT_SCHEMA
    assert data["source"] == str(video)
    assert len(data["snapshots"]) == 2
    # Human summary confirms the save.
    assert "OCR evidence saved" in result.output


def test_save_ocr_json_receipt(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    out = tmp_path / "ocr.json"
    with patch.object(AnalyzeService, "analyze", return_value=_result(video)):
        result = runner.invoke(app, ["save-ocr", str(video), "--out", str(out), "--json"])

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["schema"] == OCR_ARTIFACT_SCHEMA
    assert receipt["artifact_path"] == str(out)
    assert receipt["source"] == str(video)
    assert receipt["snapshots_saved"] == 2
    assert receipt["phrases"] == 1
    assert receipt["keywords"] == 1
    assert receipt["source_modified"] is False
    # The artifact itself is still written when --json is used.
    assert out.exists()


def test_save_ocr_requires_out_option(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")

    result = runner.invoke(app, ["save-ocr", str(video)])

    # Typer reports a missing required option with a usage error (exit code 2).
    assert result.exit_code != 0


def test_save_ocr_missing_video_exits_1_and_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "ocr.json"

    result = runner.invoke(app, ["save-ocr", str(tmp_path / "nope.mov"), "--out", str(out)])

    assert result.exit_code == 1
    assert "File not found" in result.output
    # Analysis fails first, so no artifact is written.
    assert not out.exists()
