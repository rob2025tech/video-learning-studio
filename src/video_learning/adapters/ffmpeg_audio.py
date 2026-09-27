"""Adapter: extract a video's audio to a temporary WAV using FFmpeg.

whisper.cpp requires 16 kHz mono 16-bit PCM WAV input and does not decode
container formats itself, so the source audio is converted first. FFmpeg only
reads the source and only writes into the caller-provided temporary directory
(a ``TemporaryDirectory``); the source video is never modified.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from video_learning.core.errors import AnalysisToolMissingError, AudioExtractionError
from video_learning.core.models import MediaInfo

_AUDIO_TIMEOUT_SECONDS = 600


class FfmpegAudioExtractor:
    """Converts a video's audio track into a 16 kHz mono 16-bit PCM WAV."""

    def extract(self, media: MediaInfo, out_dir: Path) -> Path:
        if not media.has_audio:
            raise AudioExtractionError(
                f"'{media.filename}' has no audio stream to transcribe."
            )
        if shutil.which("ffmpeg") is None:
            raise AnalysisToolMissingError(
                "ffmpeg was not found on PATH. "
                "Install FFmpeg first, e.g. `brew install ffmpeg`."
            )
        wav_path = out_dir / "audio.wav"
        command = [
            "ffmpeg", "-nostdin", "-y", "-v", "error",
            "-i", str(media.path),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-f", "wav", str(wav_path),
        ]
        self._run(command, media.filename)
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise AudioExtractionError(
                f"ffmpeg produced no audio WAV for '{media.filename}'."
            )
        return wav_path

    def _run(self, command: list[str], filename: str) -> None:
        try:
            result = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                timeout=_AUDIO_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioExtractionError(
                f"ffmpeg timed out after {_AUDIO_TIMEOUT_SECONDS}s extracting "
                f"audio from '{filename}'."
            ) from exc
        except OSError as exc:
            raise AudioExtractionError(f"Could not execute ffmpeg: {exc}") from exc
        if result.returncode != 0:
            lines = (result.stderr or "").strip().splitlines()
            detail = lines[-1] if lines else f"exit code {result.returncode}"
            raise AudioExtractionError(
                f"ffmpeg could not extract audio from '{filename}': {detail}"
            )
