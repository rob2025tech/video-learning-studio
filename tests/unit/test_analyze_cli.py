"""CLI tests for the print-only `analyze` command (service mocked at its boundary)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from video_learning.cli.main import app
from video_learning.core.errors import AnalysisToolMissingError
from video_learning.core.keywords import KeywordSuggestion
from video_learning.services.analyze_service import AnalyzeResult, AnalyzeService

runner = CliRunner()


def _result(source: Path) -> AnalyzeResult:
    return AnalyzeResult(
        source=source,
        keywords=[KeywordSuggestion("Qoder", 4), KeywordSuggestion("terminal", 2)],
        frames_analyzed=5,
        ocr_text_chars=128,
    )


def test_analyze_human_readable(tmp_path: Path) -> None:
    video = tmp_path / "sample.mov"
    video.write_bytes(b"x")
    with patch.object(AnalyzeService, "analyze", return_value=_result(video)):
        result = runner.invoke(app, ["analyze", str(video)])

    assert result.exit_code == 0, result.output
    assert "Video: sample.mov" in result.output
    assert "Suggested keywords:" in result.output
    assert "- Qoder" in result.output
    assert "- terminal" in result.output


def test_analyze_json_output(tmp_path: Path) -> None:
    video = tmp_path / "sample.mov"
    video.write_bytes(b"x")
    with patch.object(AnalyzeService, "analyze", return_value=_result(video)):
        result = runner.invoke(app, ["analyze", str(video), "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["source"] == str(video)
    assert data["keywords"] == [
        {"term": "Qoder", "count": 4},
        {"term": "terminal", "count": 2},
    ]
    assert data["renamed"] is False
    assert data["applied"] is False
    assert data["source_modified"] is False


def test_analyze_missing_file_exits_1(tmp_path: Path) -> None:
    # Real service: inspection runs first and fails before any binary is invoked.
    result = runner.invoke(app, ["analyze", str(tmp_path / "nope.mov")])

    assert result.exit_code == 1
    assert "File not found" in result.output


def test_analyze_tool_missing_exits_1(tmp_path: Path) -> None:
    video = tmp_path / "sample.mov"
    video.write_bytes(b"x")
    with patch.object(
        AnalyzeService,
        "analyze",
        side_effect=AnalysisToolMissingError("tesseract was not found on PATH."),
    ):
        result = runner.invoke(app, ["analyze", str(video)])

    assert result.exit_code == 1
    assert "tesseract" in result.output
