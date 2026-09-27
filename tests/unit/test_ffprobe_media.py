"""Unit tests for FfprobeMediaProbe: mocked ffprobe output and failures."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.conftest import FFPROBE_JSON_MOV, FFPROBE_JSON_VIDEO_ONLY
from video_learning.adapters.ffprobe_media import FfprobeMediaProbe
from video_learning.core.errors import (
    MalformedProbeOutputError,
    ProbeToolMissingError,
    UnreadableMediaError,
)

WHICH = "video_learning.adapters.ffprobe_media.shutil.which"
RUN = "video_learning.adapters.ffprobe_media.subprocess.run"


def _completed(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ffprobe"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@contextmanager
def _mock_ffprobe(
    result: subprocess.CompletedProcess[str] | None = None,
    side_effect: Exception | None = None,
    installed: bool = True,
) -> Iterator[None]:
    """Patch shutil.which + subprocess.run below the adapter boundary."""
    run_kwargs: dict[str, Any] = (
        {"side_effect": side_effect} if side_effect is not None else {"return_value": result}
    )
    with (
        patch(WHICH, return_value="/usr/bin/ffprobe" if installed else None),
        patch(RUN, **run_kwargs),
    ):
        yield


@pytest.fixture
def media_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mov"
    path.write_bytes(b"fake media bytes")
    return path


def _probe_with_stdout(payload: dict[str, Any], path: Path) -> Any:
    with _mock_ffprobe(result=_completed(json.dumps(payload))):
        return FfprobeMediaProbe().probe(path)


def test_parses_full_mov_metadata(media_file: Path) -> None:
    info = _probe_with_stdout(FFPROBE_JSON_MOV, media_file)

    assert info.filename == "sample.mov"
    assert info.path == media_file
    assert info.container_format == "mov"
    assert info.duration_seconds == 62.5
    assert info.size_bytes == 12345678
    assert info.modified_at is not None  # real file on disk

    assert info.has_video
    assert info.video is not None
    assert info.video.codec == "h264"
    assert info.video.width == 1920
    assert info.video.height == 1080
    assert info.video.frame_rate == pytest.approx(29.97, abs=0.001)
    assert info.video.pixel_format == "yuv420p"

    assert info.has_audio
    assert info.audio is not None
    assert info.audio.codec == "aac"
    assert info.audio.sample_rate == 48000
    assert info.audio.channels == 2
    assert info.audio.language == "eng"


def test_video_only_file_has_no_audio(media_file: Path) -> None:
    info = _probe_with_stdout(FFPROBE_JSON_VIDEO_ONLY, media_file)

    assert info.has_video
    assert not info.has_audio
    assert info.audio is None
    # avg_frame_rate "0/0" is unusable; falls back to r_frame_rate 60/1.
    assert info.video is not None
    assert info.video.frame_rate == 60.0
    # No format-level duration and no stream duration -> unknown, never invented.
    assert info.duration_seconds is None


def test_duration_falls_back_to_stream(media_file: Path) -> None:
    payload = {
        "streams": [{"codec_type": "video", "duration": "12.25"}],
        "format": {"format_name": "rawvideo"},
    }
    info = _probe_with_stdout(payload, media_file)
    assert info.duration_seconds == 12.25


def test_missing_values_stay_none(media_file: Path) -> None:
    info = _probe_with_stdout({"streams": [], "format": {}}, media_file)

    assert info.container_format is None
    assert info.duration_seconds is None
    assert info.size_bytes is None
    assert info.video is None
    assert info.audio is None


def test_ffprobe_missing_from_path(media_file: Path) -> None:
    with (
        _mock_ffprobe(installed=False),
        pytest.raises(ProbeToolMissingError, match="brew install ffmpeg"),
    ):
        FfprobeMediaProbe().probe(media_file)


def test_ffprobe_nonzero_exit_is_unreadable_media(media_file: Path) -> None:
    failure = _completed("", returncode=1, stderr="moov atom not found")
    with (
        _mock_ffprobe(result=failure),
        pytest.raises(UnreadableMediaError, match="moov atom not found"),
    ):
        FfprobeMediaProbe().probe(media_file)


def test_ffprobe_timeout_is_unreadable_media(media_file: Path) -> None:
    timeout = subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=60)
    with (
        _mock_ffprobe(side_effect=timeout),
        pytest.raises(UnreadableMediaError, match="timed out"),
    ):
        FfprobeMediaProbe().probe(media_file)


def test_malformed_json_output(media_file: Path) -> None:
    with (
        _mock_ffprobe(result=_completed("{not json")),
        pytest.raises(MalformedProbeOutputError, match="not valid JSON"),
    ):
        FfprobeMediaProbe().probe(media_file)


def test_empty_output(media_file: Path) -> None:
    with (
        _mock_ffprobe(result=_completed("")),
        pytest.raises(MalformedProbeOutputError, match="no output"),
    ):
        FfprobeMediaProbe().probe(media_file)


def test_frame_rate_parser_edge_cases() -> None:
    from video_learning.adapters.ffprobe_media import _parse_frame_rate

    assert _parse_frame_rate("30000/1001") == pytest.approx(29.97, abs=0.001)
    assert _parse_frame_rate("0/0") is None
    assert _parse_frame_rate("25/1") == 25.0
    assert _parse_frame_rate("nonsense") is None
    assert _parse_frame_rate(None) is None
