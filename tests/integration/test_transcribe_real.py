"""Integration test: real whisper.cpp transcription over generated audio.

Nothing is downloaded and no network call is made. The whole module skips
unless a *usable* ``whisper-cli`` is on PATH, a ggml model is configured
(``VLS_WHISPER_MODEL``), and ffmpeg/ffprobe are present — so CI (and this
environment, where whisper.cpp is not installed) simply skips.

The generated clip carries a sine tone rather than speech, so the assertions
validate the *pipeline* (probe -> temporary WAV -> whisper.cpp -> JSON parse ->
auditable result, source untouched) rather than specific recognised words.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from video_learning.adapters.ffmpeg_audio import FfmpegAudioExtractor
from video_learning.adapters.ffprobe_media import FfprobeMediaProbe
from video_learning.adapters.whisper_cpp import WhisperCpp, resolve_model
from video_learning.services.inspect_service import InspectService
from video_learning.services.transcribe_service import TranscribeService

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
_MODEL = resolve_model(None)  # reads VLS_WHISPER_MODEL; never downloads


def _whisper_usable() -> bool:
    """True only if ``whisper-cli`` actually runs (guards against a broken shim)."""
    if shutil.which("whisper-cli") is None:
        return False
    try:
        completed = subprocess.run(
            ["whisper-cli", "--help"], capture_output=True, text=True, timeout=15
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


pytestmark = pytest.mark.skipif(
    not (_HAS_FFMPEG and _MODEL is not None and _whisper_usable()),
    reason="requires a usable whisper-cli on PATH, a configured ggml model "
    "(VLS_WHISPER_MODEL), and ffmpeg/ffprobe",
)


@pytest.fixture(scope="module")
def audio_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("media") / "tone.mov"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2:r=5",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        pytest.skip(f"could not generate an audio fixture with ffmpeg: {exc}")
    return out


def test_real_transcription_is_auditable_and_read_only(audio_video: Path) -> None:
    assert _MODEL is not None  # guaranteed by the module-level skipif
    service = TranscribeService(
        inspect_service=InspectService(probe=FfprobeMediaProbe()),
        audio_extractor=FfmpegAudioExtractor(),
        whisper=WhisperCpp(),
    )

    before = audio_video.stat()
    result = service.transcribe(audio_video)
    after = audio_video.stat()

    # Provenance is recorded so the transcript is auditable, not anonymous text.
    assert result.backend == "whisper.cpp"
    assert result.model == _MODEL.stem
    # Segments are ordered, non-negative intervals (a tone may yield none).
    starts = [s.start_seconds for s in result.segments]
    ends = [s.end_seconds for s in result.segments]
    assert starts == sorted(starts)
    assert all(s >= 0.0 for s in starts)
    assert all(e >= s for s, e in zip(starts, ends, strict=True))
    # The source video is never modified.
    assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime)
    # JSON carries provenance plus the explicit read-only guarantees.
    payload = result.to_dict()
    assert payload["backend"] == "whisper.cpp"
    assert payload["model"] == _MODEL.stem
    assert payload["source_modified"] is False
    assert payload["renamed"] is False
    assert "segments" in payload
