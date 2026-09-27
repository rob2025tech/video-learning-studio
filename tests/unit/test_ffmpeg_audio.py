"""Unit tests for FfmpegAudioExtractor: pure behavior + mocked subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from video_learning.adapters.ffmpeg_audio import FfmpegAudioExtractor
from video_learning.core.errors import AnalysisToolMissingError, AudioExtractionError
from video_learning.core.models import AudioStreamInfo, MediaInfo

WHICH = "video_learning.adapters.ffmpeg_audio.shutil.which"
RUN = "video_learning.adapters.ffmpeg_audio.subprocess.run"


def _media(tmp_path: Path, *, has_audio: bool = True) -> MediaInfo:
    return MediaInfo(
        path=tmp_path / "clip.mov",
        filename="clip.mov",
        duration_seconds=10.0,
        audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2)
        if has_audio
        else None,
    )


def test_no_audio_stream_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioExtractionError, match="no audio stream"):
        FfmpegAudioExtractor().extract(_media(tmp_path, has_audio=False), tmp_path)


def test_missing_ffmpeg_raises(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value=None),
        pytest.raises(AnalysisToolMissingError, match="brew install ffmpeg"),
    ):
        FfmpegAudioExtractor().extract(_media(tmp_path), tmp_path)


def test_command_construction_writes_wav_into_out_dir(tmp_path: Path) -> None:
    media = _media(tmp_path)
    captured: dict[str, list[str]] = {}

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        Path(command[-1]).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")  # emulate ffmpeg
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="", stderr=""
        )

    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(RUN, side_effect=fake_run),
    ):
        wav = FfmpegAudioExtractor().extract(media, tmp_path)

    command = captured["command"]
    assert command[:5] == ["ffmpeg", "-nostdin", "-y", "-v", "error"]
    # The source is only ever an input (-i); the WAV is written inside out_dir.
    assert command[command.index("-i") + 1] == str(media.path)
    assert command[-1] == str(tmp_path / "audio.wav") == str(wav)
    for flag in ("-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav"):
        assert flag in command
    assert wav.exists()


def test_ffmpeg_failure_raises(tmp_path: Path) -> None:
    failure = subprocess.CompletedProcess(
        args=["ffmpeg"], returncode=1, stdout="", stderr="Invalid data found"
    )
    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(RUN, return_value=failure),
        pytest.raises(AudioExtractionError, match="Invalid data found"),
    ):
        FfmpegAudioExtractor().extract(_media(tmp_path), tmp_path)


def test_timeout_raises(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(
            RUN,
            side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=600),
        ),
        pytest.raises(AudioExtractionError, match="timed out"),
    ):
        FfmpegAudioExtractor().extract(_media(tmp_path), tmp_path)


def test_empty_wav_raises(tmp_path: Path) -> None:
    ok = subprocess.CompletedProcess(
        args=["ffmpeg"], returncode=0, stdout="", stderr=""
    )
    with (
        patch(WHICH, return_value="/usr/bin/ffmpeg"),
        patch(RUN, return_value=ok),  # succeeds but writes nothing
        pytest.raises(AudioExtractionError, match="no audio WAV"),
    ):
        FfmpegAudioExtractor().extract(_media(tmp_path), tmp_path)
