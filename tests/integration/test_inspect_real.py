"""Integration test: real ffprobe against a tiny generated fixture.

The fixture is created on the fly (~1s of test pattern + sine audio) so no
binary video is ever committed. Skipped when ffmpeg is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from video_learning.cli.main import app

runner = CliRunner()

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="requires ffmpeg/ffprobe installed on PATH",
)


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("media") / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def test_inspect_real_generated_video(tiny_video: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tiny_video), "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)

    assert data["filename"] == "fixture.mp4"
    assert data["container_format"] == "mov"
    assert data["duration_seconds"] == pytest.approx(1.0, abs=0.2)
    assert data["has_video"] is True
    assert data["video"]["codec"] == "h264"
    assert data["video"]["width"] == 320
    assert data["video"]["height"] == 240
    assert data["video"]["frame_rate"] == pytest.approx(25.0, abs=0.1)
    assert data["has_audio"] is True
    assert data["audio"]["codec"] == "aac"
    assert data["audio"]["sample_rate"] == 44100


def test_inspect_real_generated_video_human_readable(tiny_video: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tiny_video)])

    assert result.exit_code == 0, result.output
    assert "fixture.mp4" in result.output
    assert "320x240" in result.output
