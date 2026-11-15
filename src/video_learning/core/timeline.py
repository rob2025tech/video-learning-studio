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


def _require_seconds(value: Any, key: str) -> None:
    """Reject a non-numeric timestamp rather than let it reach a time comparison.

    Nothing downstream compares these numbers when the segmentation has no segments,
    so a string would otherwise survive rehydration and fail in the renderer. ``bool``
    is excluded explicitly because it is an ``int`` subclass.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number of seconds, got {value!r}")


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OcrEvent:
        """Inverse of :meth:`to_dict` (used to reload a persisted timeline)."""
        timestamp = data["timestamp_seconds"]
        _require_seconds(timestamp, "timestamp_seconds")
        return cls(timestamp_seconds=timestamp, text=data["text"])


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeechEvent:
        """Inverse of :meth:`to_dict` (used to reload a persisted timeline)."""
        start = data["start_seconds"]
        end = data["end_seconds"]
        _require_seconds(start, "start_seconds")
        _require_seconds(end, "end_seconds")
        return cls(start_seconds=start, end_seconds=end, text=data["text"])


TimelineEvent: TypeAlias = OcrEvent | SpeechEvent


def timeline_event_from_dict(data: dict[str, Any]) -> TimelineEvent:
    """Rebuild one serialized timeline event from its ``type`` discriminator.

    The timeline mixes two event kinds in one array, so the caller cannot know which
    :meth:`from_dict` to use. An unrecognised ``type`` is rejected rather than guessed
    at: the artifact came from a model this version does not know.
    """
    kind = data["type"]
    if kind == "ocr":
        return OcrEvent.from_dict(data)
    if kind == "speech":
        return SpeechEvent.from_dict(data)
    raise ValueError(f"unknown timeline event type {kind!r}")


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
