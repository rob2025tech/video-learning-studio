"""Unified evidence timeline models (Stage 0G).

The timeline is *simple chronological evidence*, not semantic interpretation. It
interleaves two DISTINCT kinds of event without forcing them into one common
domain model:

- :class:`OcrEvent` is a point-in-time on-screen-text observation (one timestamp).
- :class:`SpeechEvent` is an interval of transcribed speech (start -> end).

Both are pure data (no I/O). They share only a ``sort_key`` used to merge them
into one deterministic chronological order: OCR (a point) sorts before speech (an
interval) when their start times are equal, and Python's stable sort preserves the
original snapshot/segment order for any remaining ties. No relationship between
OCR and speech is inferred here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True)
class OcrEvent:
    """A point-in-time on-screen-text observation carrying the complete OCR text."""

    timestamp_seconds: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ocr",
            "timestamp_seconds": self.timestamp_seconds,
            "text": self.text,
        }

    @property
    def sort_key(self) -> tuple[float, int, float]:
        """Chronological key ``(time, type rank, tie-break)``; OCR is a point (0)."""
        return (self.timestamp_seconds, 0, self.timestamp_seconds)


@dataclass(frozen=True)
class SpeechEvent:
    """An interval of transcribed speech (start -> end)."""

    start_seconds: float
    end_seconds: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "speech",
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "text": self.text,
        }

    @property
    def sort_key(self) -> tuple[float, int, float]:
        """Chronological key ``(time, type rank, tie-break)``; speech is an interval (1)."""
        return (self.start_seconds, 1, self.end_seconds)


TimelineEvent: TypeAlias = OcrEvent | SpeechEvent


def build_timeline(
    ocr_events: Sequence[OcrEvent], speech_events: Sequence[SpeechEvent]
) -> list[TimelineEvent]:
    """Merge OCR points and speech intervals into one chronological timeline.

    Deterministic: events are sorted by ``sort_key`` (time, then type, then
    tie-break) with Python's stable sort, so equal-key events keep their input
    order. No semantic alignment between OCR and speech is attempted.
    """
    events: list[TimelineEvent] = [*ocr_events, *speech_events]
    events.sort(key=lambda event: event.sort_key)
    return events
