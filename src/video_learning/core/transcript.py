"""Transcription domain models (Stage 0E).

A transcript is *evidence for the human*: timestamped spoken segments produced
locally by whisper.cpp. These are pure data containers with no I/O.

``TranscriptSegment`` is deliberately kept separate from the analyze stage's
``FrameSnapshot``: a snapshot is a *point* in time (one sampled frame), while a
transcript segment is an *interval* (start -> end). They share conventions
(float seconds, ``to_dict``) but not a base class, avoiding a speculative
common abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    """One timestamped span of transcribed speech (an interval, in seconds)."""

    start_seconds: float
    end_seconds: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "text": self.text,
        }


@dataclass(frozen=True)
class TranscriptResult:
    """Outcome of transcribing one video's audio (read-only and auditable).

    ``backend``/``model``/``language`` record what was actually used so the
    result is auditable rather than an anonymous blob of text.
    """

    source: str
    backend: str
    model: str
    language: str
    segments: tuple[TranscriptSegment, ...]

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
            "segment_count": self.segment_count,
            # Explicit guarantees for this stage: transcription is read-only.
            "renamed": False,
            "applied": False,
            "source_modified": False,
        }


@dataclass(frozen=True)
class CapabilityItem:
    """Availability of one transcription prerequisite (a backend or a model).

    ``state`` is one of ``"available"``, ``"missing"``, or ``"unusable"`` (the
    latter covers a shim that exists on PATH but cannot actually run).
    """

    label: str
    state: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "state": self.state, "detail": self.detail}


@dataclass(frozen=True)
class TranscriptionCapability:
    """Deterministic snapshot of this machine's local transcription capability."""

    backend: CapabilityItem
    model: CapabilityItem
    recommended: str
    recommended_detail: str
    using_backend: str
    using_model: str
    using_language: str
    ready: bool

    def blocking_message(self) -> str:
        """Human-readable, actionable reason transcription cannot run yet."""
        issues = [
            item.detail or f"{item.label} not available"
            for item in (self.backend, self.model)
            if item.state != "available"
        ]
        return " ".join(issues) if issues else "Transcription prerequisites are not met."

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.to_dict(),
            "model": self.model.to_dict(),
            "recommended": self.recommended,
            "recommended_detail": self.recommended_detail,
            "using": {
                "backend": self.using_backend,
                "model": self.using_model,
                "language": self.using_language,
            },
            "ready": self.ready,
        }
