"""Visual-change evidence domain (Stage 0I).

Pure data with no I/O and no FFmpeg knowledge. A :class:`VisualChange` is
*measurable evidence that one portion of the video differs visually from another*
— it is explicitly NOT a semantic scene, chapter, object, topic, or activity. All
FFmpeg/``scdet`` specifics (stderr, regexes, ``lavfi.scd.*`` keys, command-line
arguments) are isolated in :mod:`video_learning.adapters.ffmpeg_scene`; this
module never sees them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ``scdet`` scores are on a 0–100 scale (NOT 0–1). This is the one authoritative
# threshold: it is never exposed as a CLI option, tuned per video, or made
# adaptive, and no configuration machinery is built around it.
VISUAL_CHANGE_THRESHOLD = 10.0


@dataclass(frozen=True)
class VisualChange:
    """A single visual-change observation: a timestamp and its ``scdet`` score.

    ``change_score`` is on the ``scdet`` 0–100 scale. Value-like and immutable,
    matching the project's other core evidence objects (``OcrEvent``,
    ``TranscriptSegment``).
    """

    timestamp_seconds: float
    change_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "change_score": self.change_score,
        }
