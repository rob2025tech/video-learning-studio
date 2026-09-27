"""Unit tests for the CLI layer, with the probe mocked below the service."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tests.conftest import FFPROBE_JSON_MOV
from video_learning.adapters.ffprobe_media import FfprobeMediaProbe
from video_learning.cli.main import app

runner = CliRunner()


@pytest.fixture
def media_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mov"
    path.write_bytes(b"fake media bytes")
    return path


def test_inspect_human_readable(media_file: Path) -> None:
    with patch.object(FfprobeMediaProbe, "_run_ffprobe", return_value=FFPROBE_JSON_MOV):
        result = runner.invoke(app, ["inspect", str(media_file)])

    assert result.exit_code == 0
    out = result.output
    assert "sample.mov" in out
    assert "1920x1080" in out
    assert "h264" in out
    assert "aac" in out
    assert "48000 Hz" in out
    assert "unknown" not in out.lower()


def test_inspect_json_output_is_valid(media_file: Path) -> None:
    with patch.object(FfprobeMediaProbe, "_run_ffprobe", return_value=FFPROBE_JSON_MOV):
        result = runner.invoke(app, ["inspect", str(media_file), "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["filename"] == "sample.mov"
    assert data["duration_seconds"] == 62.5
    assert data["has_video"] is True
    assert data["has_audio"] is True
    assert data["video"]["width"] == 1920
    assert data["audio"]["sample_rate"] == 48000


def test_inspect_missing_file_exits_1(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    result = runner.invoke(app, ["inspect", str(missing)])

    assert result.exit_code == 1
    assert "File not found" in result.output


def test_inspect_directory_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tmp_path)])

    assert result.exit_code == 1
    assert "Not a file" in result.output


def test_inspect_unreadable_media_exits_1(media_file: Path) -> None:
    from video_learning.core.errors import UnreadableMediaError

    with patch.object(
        FfprobeMediaProbe,
        "_run_ffprobe",
        side_effect=UnreadableMediaError("ffprobe could not read 'sample.mov' as media"),
    ):
        result = runner.invoke(app, ["inspect", str(media_file)])

    assert result.exit_code == 1
    assert "could not read" in result.output
