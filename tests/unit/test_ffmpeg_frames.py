"""Unit tests for FfmpegFrameExtractor: pure helpers + mocked subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from video_learning.adapters.ffmpeg_frames import FfmpegFrameExtractor
from video_learning.core.errors import AnalysisToolMissingError, FrameExtractionError
from video_learning.core.models import MediaInfo, VideoStreamInfo

WHICH = "video_learning.adapters.ffmpeg_frames.shutil.which"
RUN = "video_learning.adapters.ffmpeg_frames.subprocess.run"


def _media(
    tmp_path: Path,
    *,
    width: int | None = 1280,
    duration: float | None = 10.0,
    has_video: bool = True,
) -> MediaInfo:
    return MediaInfo(
        path=tmp_path / "clip.mov",
        filename="clip.mov",
        duration_seconds=duration,
        video=VideoStreamInfo(width=width, height=720) if has_video else None,
    )


def test_timestamps_are_evenly_spaced_midpoints() -> None:
    ts = FfmpegFrameExtractor(max_frames=5)._timestamps(10.0)
    assert len(ts) == 5
    assert ts == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0])
    assert all(0.0 <= t < 10.0 for t in ts)


def test_timestamps_unknown_duration_falls_back_to_start() -> None:
    assert FfmpegFrameExtractor()._timestamps(None) == [0.0]
    assert FfmpegFrameExtractor()._timestamps(0.0) == [0.0]


def test_scale_filter_downscales_only_when_wide(tmp_path: Path) -> None:
    extractor = FfmpegFrameExtractor(max_width=1600)
    assert extractor._scale_filter(_media(tmp_path, width=1728)) == "scale=1600:-2"
    assert extractor._scale_filter(_media(tmp_path, width=1280)) is None
    assert extractor._scale_filter(_media(tmp_path, width=None)) is None


def test_no_video_stream_raises(tmp_path: Path) -> None:
    with pytest.raises(FrameExtractionError, match="no video stream"):
        FfmpegFrameExtractor().extract(_media(tmp_path, has_video=False), tmp_path)


def test_missing_ffmpeg_raises(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value=None),
        pytest.raises(AnalysisToolMissingError, match="brew install ffmpeg"),
    ):
        FfmpegFrameExtractor().extract(_media(tmp_path), tmp_path)


def test_extract_happy_path_creates_and_returns_frames(tmp_path: Path) -> None:
    out_dir = tmp_path / "frames"
    out_dir.mkdir()

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        # Last argv element is the output PNG path; emulate ffmpeg writing it.
        Path(command[-1]).write_bytes(b"\x89PNG")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    with (
        patch(WHICH, return_value="/usr/local/bin/ffmpeg"),
        patch(RUN, side_effect=fake_run),
    ):
        frames = FfmpegFrameExtractor(max_frames=3).extract(_media(tmp_path), out_dir)

    assert len(frames) == 3
    assert all(f.exists() and f.suffix == ".png" for f in frames)


def test_extract_ffmpeg_failure_raises(tmp_path: Path) -> None:
    out_dir = tmp_path / "frames"
    out_dir.mkdir()
    failure = subprocess.CompletedProcess(
        args=["ffmpeg"], returncode=1, stdout="", stderr="Invalid data found"
    )
    with (
        patch(WHICH, return_value="/usr/local/bin/ffmpeg"),
        patch(RUN, return_value=failure),
        pytest.raises(FrameExtractionError, match="Invalid data found"),
    ):
        FfmpegFrameExtractor().extract(_media(tmp_path), out_dir)


def test_extract_no_frames_written_raises(tmp_path: Path) -> None:
    out_dir = tmp_path / "frames"
    out_dir.mkdir()
    ok = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout="", stderr="")
    with (
        patch(WHICH, return_value="/usr/local/bin/ffmpeg"),
        patch(RUN, return_value=ok),  # succeeds but writes nothing
        pytest.raises(FrameExtractionError, match="no frames"),
    ):
        FfmpegFrameExtractor().extract(_media(tmp_path), out_dir)
