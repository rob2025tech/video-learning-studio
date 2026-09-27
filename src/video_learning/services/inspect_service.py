"""Application/service layer.

Orchestrates use cases; contains no CLI concerns and no direct subprocess work.
"""

from __future__ import annotations

from pathlib import Path

from video_learning.core.errors import FileNotFoundError_, NotAFileError
from video_learning.core.models import MediaInfo
from video_learning.providers.media import MediaProbe


class InspectService:
    """Use case: inspect a local media file and return its MediaInfo."""

    def __init__(self, probe: MediaProbe) -> None:
        self._probe = probe

    def inspect(self, path: Path) -> MediaInfo:
        if not path.exists():
            raise FileNotFoundError_(f"File not found: {path}")
        if not path.is_file():
            raise NotAFileError(f"Not a file (expected a media file, got a directory): {path}")
        return self._probe.probe(path)
