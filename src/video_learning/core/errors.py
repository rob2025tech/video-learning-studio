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
