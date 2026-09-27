"""Transcription use case: video audio -> timestamped transcript (read-only).

Composes the existing probe (Stage 0A) with two new adapters: FFmpeg audio
extraction and whisper.cpp transcription. The transcript is *evidence for the
human*; this stage never derives filenames or keywords from it, persists
nothing, and never modifies the source. Temporary audio lives in a
``TemporaryDirectory`` and is removed automatically.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from video_learning.adapters.ffmpeg_audio import FfmpegAudioExtractor
from video_learning.adapters.whisper_cpp import (
    BACKEND_NAME,
    WhisperCpp,
    resolve_model,
)
from video_learning.core.errors import TranscriptionModelMissingError
from video_learning.core.transcript import TranscriptResult
from video_learning.services.inspect_service import InspectService


class TranscribeService:
    """Use case: produce a timestamped transcript of a video's spoken audio."""

    def __init__(
        self,
        inspect_service: InspectService,
        audio_extractor: FfmpegAudioExtractor,
        whisper: WhisperCpp,
    ) -> None:
        self._inspect = inspect_service
        self._audio = audio_extractor
        self._whisper = whisper

    def transcribe(self, path: Path, *, model: Path | None = None) -> TranscriptResult:
        model_path = resolve_model(model)
        if model_path is None:
            if model is not None:
                raise TranscriptionModelMissingError(
                    f"whisper.cpp model not found at '{model}'."
                )
            raise TranscriptionModelMissingError(
                "No whisper.cpp model configured. Pass --model "
                "/path/to/ggml-base.en.bin or set VLS_WHISPER_MODEL. "
                "Models are never downloaded automatically."
            )

        media = self._inspect.inspect(path)  # reuse the Stage 0A probe
        with tempfile.TemporaryDirectory(prefix="vls-transcribe-") as tmp:
            out_dir = Path(tmp)
            wav_path = self._audio.extract(media, out_dir)
            segments, language = self._whisper.transcribe(wav_path, model_path, out_dir)

        return TranscriptResult(
            source=str(path),
            backend=BACKEND_NAME,
            model=model_path.stem,
            language=language or "unknown",
            segments=tuple(segments),
        )
