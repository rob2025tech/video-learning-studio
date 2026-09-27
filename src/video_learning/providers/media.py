"""The Stage 0 media-probe seam.

``MediaProbe`` is the only provider interface introduced so far: it is the
actual boundary between the application and local media tooling. Future
providers (transcription, OCR, ...) get their own protocols in their own
stages, not before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from video_learning.core.models import MediaInfo


class MediaProbe(Protocol):
    """Probes a local media file and returns its raw container/stream data.

    Implementations must raise a ``MediaProbeError`` subclass on failure;
    they must not return partially invented metadata.
    """

    def probe(self, path: Path) -> MediaInfo: ...
