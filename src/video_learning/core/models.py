"""Core domain models for media inspection.

Pure data: no I/O, no dependencies on adapters or CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

UNKNOWN = "unknown"


@dataclass(frozen=True)
class VideoStreamInfo:
    """Metadata of a single video stream."""

    codec: str | None = None
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None
    pixel_format: str | None = None


@dataclass(frozen=True)
class AudioStreamInfo:
    """Metadata of a single audio stream."""

    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    language: str | None = None


@dataclass(frozen=True)
class MediaInfo:
    """Everything Stage 0 knows about a media file.

    Absent values are ``None`` and rendered as ``unknown`` at the edges;
    they are never invented.
    """

    path: Path
    filename: str
    exists: bool = True
    size_bytes: int | None = None
    modified_at: datetime | None = None
    container_format: str | None = None
    duration_seconds: float | None = None
    video: VideoStreamInfo | None = None
    audio: AudioStreamInfo | None = None

    @property
    def has_video(self) -> bool:
        return self.video is not None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable representation (JSON-safe, nulls for unknown)."""
        return {
            "filename": self.filename,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_human": _format_size(self.size_bytes),
            "modified_at": self.modified_at.astimezone(UTC).isoformat()
            if self.modified_at is not None
            else None,
            "container_format": self.container_format,
            "duration_seconds": self.duration_seconds,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "video": {
                "codec": self.video.codec,
                "width": self.video.width,
                "height": self.video.height,
                "frame_rate": self.video.frame_rate,
                "pixel_format": self.video.pixel_format,
            }
            if self.video is not None
            else None,
            "audio": {
                "codec": self.audio.codec,
                "sample_rate": self.audio.sample_rate,
                "channels": self.audio.channels,
                "language": self.audio.language,
            }
            if self.audio is not None
            else None,
        }


def _format_size(size_bytes: int | None) -> str | None:
    if size_bytes is None:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return None  # pragma: no cover
