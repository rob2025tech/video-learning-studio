"""Unit tests for FfmpegSceneDetector: authoritative scdet parsing + errors.

All subprocess/PATH access is mocked (fast, deterministic). The tested stderr is
the verified FFmpeg 8.0 atomic scdet line::

    [scdet @ 0x7fe4c2e0f880] lavfi.scd.score: 15.625, lavfi.scd.time: 3

The obsolete ``select='gt(scene,0.30)'`` approach is NOT used and is asserted
absent from the constructed command.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from video_learning.adapters.ffmpeg_scene import _SCENE_TIMEOUT_SECONDS, FfmpegSceneDetector
from video_learning.core.errors import AnalysisToolMissingError, SceneDetectionError
from video_learning.core.models import MediaInfo, VideoStreamInfo
from video_learning.core.scene import VISUAL_CHANGE_THRESHOLD, VisualChange

WHICH = "video_learning.adapters.ffmpeg_scene.shutil.which"
RUN = "video_learning.adapters.ffmpeg_scene.subprocess.run"

# The verified FFmpeg 8.0 atomic scdet line (with its arbitrary pointer prefix).
SCDET_LINE = "[scdet @ 0x7fe4c2e0f880] lavfi.scd.score: 15.625, lavfi.scd.time: 3"


def _media(tmp_path: Path, *, has_video: bool = True, duration: float | None = 10.0) -> MediaInfo:
    return MediaInfo(
        path=tmp_path / "clip.mov",
        filename="clip.mov",
        duration_seconds=duration,
        video=VideoStreamInfo(width=64, height=64) if has_video else None,
    )


def _ok(stderr: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout="", stderr=stderr)


def _detect_with_stderr(tmp_path: Path, stderr: str) -> list[VisualChange]:
    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(RUN, return_value=_ok(stderr)),
    ):
        return FfmpegSceneDetector().detect(_media(tmp_path))


# -- authoritative constants -------------------------------------------------


def test_timeout_constant_is_600_seconds() -> None:
    assert _SCENE_TIMEOUT_SECONDS == 600


def test_threshold_constant_is_10() -> None:
    assert VISUAL_CHANGE_THRESHOLD == 10.0


# -- scdet parsing -----------------------------------------------------------


def test_parses_score_and_timestamp_from_atomic_line(tmp_path: Path) -> None:
    changes = _detect_with_stderr(tmp_path, SCDET_LINE + "\n")
    assert changes == [VisualChange(timestamp_seconds=3.0, change_score=15.625)]


def test_arbitrary_ffmpeg_prefix_is_ignored(tmp_path: Path) -> None:
    # The "[scdet @ 0x...]" pointer prefix (any value) must not affect parsing.
    stderr = "[scdet @ 0xdeadbeef] lavfi.scd.score: 42.5, lavfi.scd.time: 7.5\n"
    assert _detect_with_stderr(tmp_path, stderr) == [
        VisualChange(timestamp_seconds=7.5, change_score=42.5)
    ]


def test_multiple_scdet_lines_parse_in_order(tmp_path: Path) -> None:
    stderr = (
        "[scdet @ 0x1] lavfi.scd.score: 15.625, lavfi.scd.time: 3\n"
        "[scdet @ 0x1] lavfi.scd.score: 60.0, lavfi.scd.time: 8\n"
        "[scdet @ 0x1] lavfi.scd.score: 22.5, lavfi.scd.time: 12.5\n"
    )
    changes = _detect_with_stderr(tmp_path, stderr)
    assert [c.timestamp_seconds for c in changes] == pytest.approx([3.0, 8.0, 12.5])
    assert [c.change_score for c in changes] == pytest.approx([15.625, 60.0, 22.5])


def test_unrelated_stderr_lines_are_ignored(tmp_path: Path) -> None:
    stderr = (
        "ffmpeg version 8.0 Copyright (c) 2000-2025 the FFmpeg developers\n"
        "  built with Apple clang version 17.0.0\n"
        "Input #0, mov,mp4, from 'clip.mov':\n"
        "    Stream #0:0(und): Video: h264, yuv420p, 64x64, 5 fps\n"
        "frame=   30 fps=0.0 q=-0.0 Lsize=0kB time=00:00:06.00 bitrate=0.0kbits/s\n"
        + SCDET_LINE
        + "\n"
        "video:0kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB\n"
    )
    assert _detect_with_stderr(tmp_path, stderr) == [
        VisualChange(timestamp_seconds=3.0, change_score=15.625)
    ]


def test_no_scdet_lines_returns_empty_list(tmp_path: Path) -> None:
    stderr = (
        "ffmpeg version 8.0\n"
        "frame=   30 fps=0.0 q=-0.0 Lsize=0kB time=00:00:06.00 bitrate=0.0kbits/s\n"
    )
    assert _detect_with_stderr(tmp_path, stderr) == []


def test_empty_stderr_returns_empty_list(tmp_path: Path) -> None:
    assert _detect_with_stderr(tmp_path, "") == []


# -- command construction ----------------------------------------------------


def test_command_uses_scdet_threshold_and_is_read_only(tmp_path: Path) -> None:
    media = _media(tmp_path)
    captured_command: list[str] = []
    captured_kwargs: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_command.extend(command)
        captured_kwargs.update(kwargs)
        return _ok("")

    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(RUN, side_effect=fake_run),
    ):
        FfmpegSceneDetector().detect(media)

    # Authoritative filter: scdet at the frozen threshold (0-100 scale).
    vf_value = captured_command[captured_command.index("-vf") + 1]
    assert vf_value == f"scdet=threshold={VISUAL_CHANGE_THRESHOLD}"
    assert vf_value == "scdet=threshold=10.0"
    # The obsolete select-based scene filter must NOT be used.
    assert not any("select=" in part or "gt(scene" in part for part in captured_command)
    # Read-only: the source is only an input; output is discarded to null.
    assert captured_command[captured_command.index("-i") + 1] == str(media.path)
    assert captured_command[-3:] == ["-f", "null", "-"]
    # Bounded by the authoritative timeout.
    assert captured_kwargs["timeout"] == 600


# -- no-video short circuit --------------------------------------------------


def test_no_video_returns_empty_without_invoking_ffmpeg(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg") as which_mock,
        patch(RUN, return_value=_ok("")) as run_mock,
    ):
        result = FfmpegSceneDetector().detect(_media(tmp_path, has_video=False))
    assert result == []
    run_mock.assert_not_called()
    which_mock.assert_not_called()


# -- error behavior ----------------------------------------------------------


def test_missing_ffmpeg_raises_analysis_tool_missing(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value=None),
        patch(RUN, return_value=_ok("")) as run_mock,
        pytest.raises(AnalysisToolMissingError, match="brew install ffmpeg"),
    ):
        FfmpegSceneDetector().detect(_media(tmp_path))
    run_mock.assert_not_called()


def test_timeout_raises_scene_detection_error(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(RUN, side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=600)),
        pytest.raises(SceneDetectionError, match="timed out"),
    ):
        FfmpegSceneDetector().detect(_media(tmp_path))


def test_nonzero_exit_raises_with_last_stderr_line(tmp_path: Path) -> None:
    failure = subprocess.CompletedProcess(
        args=["ffmpeg"], returncode=1, stdout="", stderr="boom line one\nInvalid data found"
    )
    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(RUN, return_value=failure),
        pytest.raises(SceneDetectionError, match="Invalid data found"),
    ):
        FfmpegSceneDetector().detect(_media(tmp_path))


def test_oserror_raises_scene_detection_error(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(RUN, side_effect=OSError("exec failed")),
        pytest.raises(SceneDetectionError, match="Could not execute ffmpeg"),
    ):
        FfmpegSceneDetector().detect(_media(tmp_path))


def test_malformed_score_raises_scene_detection_error(tmp_path: Path) -> None:
    stderr = "[scdet @ 0x1] lavfi.scd.score: not_a_number, lavfi.scd.time: 3\n"
    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(RUN, return_value=_ok(stderr)),
        pytest.raises(SceneDetectionError, match="malformed scdet"),
    ):
        FfmpegSceneDetector().detect(_media(tmp_path))


def test_malformed_time_raises_scene_detection_error(tmp_path: Path) -> None:
    stderr = "[scdet @ 0x1] lavfi.scd.score: 15.6, lavfi.scd.time: xyz\n"
    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(RUN, return_value=_ok(stderr)),
        pytest.raises(SceneDetectionError, match="malformed scdet"),
    ):
        FfmpegSceneDetector().detect(_media(tmp_path))
