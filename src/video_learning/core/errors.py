"""Errors shared across the application layers.

Each error carries a message that is understandable to a normal user;
the CLI renders it and exits non-zero without a traceback.
"""

from __future__ import annotations


class VideoLearningError(Exception):
    """Base class for all user-facing errors."""


class FileNotFoundError_(VideoLearningError):
    """The given path does not exist."""


class NotAFileError(VideoLearningError):
    """The given path is a directory, not a media file."""


class MediaProbeError(VideoLearningError):
    """Base class for probe failures."""


class ProbeToolMissingError(MediaProbeError):
    """ffprobe is not installed or not on PATH."""


class UnreadableMediaError(MediaProbeError):
    """ffprobe could not read the file as media (unsupported or corrupt)."""


class MalformedProbeOutputError(MediaProbeError):
    """ffprobe ran but produced output we could not understand."""


class ContentAnalysisError(VideoLearningError):
    """Base class for on-screen-text analysis failures."""


class AnalysisToolMissingError(ContentAnalysisError):
    """ffmpeg or tesseract is not installed / not on PATH."""


class FrameExtractionError(ContentAnalysisError):
    """ffmpeg could not extract frames from the video."""


class OcrError(ContentAnalysisError):
    """tesseract failed to read text from an extracted frame."""


class TranscriptionError(VideoLearningError):
    """Base for local audio-transcription failures (Stage 0E)."""


class AudioExtractionError(TranscriptionError):
    """FFmpeg could not produce the temporary WAV needed for transcription."""


class TranscriptionToolMissingError(TranscriptionError):
    """whisper-cli is not installed, or is a broken shim that cannot run."""


class TranscriptionModelMissingError(TranscriptionError):
    """No usable whisper.cpp ggml model is configured or found."""


class WhisperExecutionError(TranscriptionError):
    """whisper.cpp ran but failed, timed out, or produced unparsable output."""


class OcrArtifactError(VideoLearningError):
    """An OCR-evidence artifact could not be written, read, or parsed (Stage 0F)."""


class ReportArtifactError(VideoLearningError):
    """A combined report artifact could not be written (Stage 0G)."""


class SceneDetectionError(VideoLearningError):
    """FFmpeg scene detection ran but failed, timed out, or was unparsable (Stage 0I).

    A missing FFmpeg binary is reported with the existing
    :class:`AnalysisToolMissingError` instead; this error covers a scene-detection
    pass that could not produce trustworthy visual-change evidence.
    """
