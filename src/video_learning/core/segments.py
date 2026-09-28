"""Deterministic segment/boundary assembly (Stage 0I).

Pure logic — no I/O, no subprocesses, no FFmpeg. Two independent evidence streams
are combined into video segments:

* visual-change evidence (:class:`~video_learning.core.scene.VisualChange`), and
* speech-gap evidence (:class:`SpeechBoundary`), derived only from gaps between
  consecutive transcript segments.

The pipeline is deliberately staged so evidence is never lost:

1. :func:`detect_speech_boundaries` turns transcript gaps into speech evidence.
2. :func:`merge_boundaries` (Phase 1) merges nearby visual + speech evidence into
   :class:`BoundaryCandidate` groups using an earliest-timestamp representative and
   an anti-chaining rule.
3. :func:`assemble_segments` (Phase 2) promotes candidates into final
   :class:`Segment` objects under a minimum-duration rule.

A boundary is only "measurable evidence that one portion may reasonably be
separated from another" — never a topic, chapter title, or semantic scene.
Merging and suppression change *promotion* only; the underlying evidence is
always retained.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from video_learning.core.scene import VISUAL_CHANGE_THRESHOLD, VisualChange
from video_learning.core.transcript import TranscriptSegment

# A silence gap of at least this many seconds between consecutive transcript
# segments is speech-gap evidence. An exact-threshold gap qualifies.
SPEECH_GAP_THRESHOLD_SECONDS = 2.0
# Candidates whose timestamps fall within this window describe one boundary.
MERGE_TOLERANCE_SECONDS = 1.0
# A promoted boundary must not create a segment shorter than this.
MIN_SEGMENT_DURATION_SECONDS = 2.0

BoundaryReason: TypeAlias = Literal["visual", "speech_gap", "combined"]


@dataclass(frozen=True)
class SpeechBoundary:
    """Speech-gap evidence: a silence gap centred at ``timestamp_seconds``.

    ``timestamp_seconds`` is the midpoint of the gap and ``gap_seconds`` its
    length. Both are retained so the evidence stays meaningful even after it
    merges with a visual candidate or is suppressed as a final boundary.
    """

    timestamp_seconds: float
    gap_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "gap_seconds": self.gap_seconds,
        }


@dataclass(frozen=True)
class BoundaryCandidate:
    """A merged boundary: a representative timestamp plus the evidence behind it.

    ``timestamp_seconds`` is the EARLIEST candidate timestamp in the merged group
    (the representative). ``reason`` is deterministic — ``combined`` when both
    evidence kinds merged, otherwise ``visual`` or ``speech_gap``. The underlying
    evidence is retained verbatim (original timestamps, scores, and gaps).
    """

    timestamp_seconds: float
    reason: BoundaryReason
    visual_changes: tuple[VisualChange, ...]
    speech_boundaries: tuple[SpeechBoundary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "reason": self.reason,
            "visual_changes": [change.to_dict() for change in self.visual_changes],
            "speech_boundaries": [boundary.to_dict() for boundary in self.speech_boundaries],
        }


@dataclass(frozen=True)
class Segment:
    """A final video segment ``[start, end]`` labelled by its END boundary reason.

    ``boundary_reason`` is the reason of the promoted boundary at ``end_seconds``;
    it is ``None`` for the final segment, which ends at the video duration rather
    than at a detected boundary. No semantic chapter name is attached.
    """

    start_seconds: float
    end_seconds: float
    boundary_reason: BoundaryReason | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "boundary_reason": self.boundary_reason,
        }


@dataclass(frozen=True)
class Segmentation:
    """Complete Stage 0I result: evidence, merged boundaries, and final segments.

    Every evidence item is retained even when its boundary was suppressed, so the
    report stays fully auditable. ``video_duration_seconds`` is the authoritative
    duration the final segment ends at; it is ``None`` only when ffprobe reported
    no duration, in which case ``segments`` is empty (assembly never estimates a
    duration) while all evidence is preserved. ``to_dict`` also records the four
    authoritative constants, so a reader can interpret the numbers without source.
    """

    visual_changes: tuple[VisualChange, ...]
    speech_boundaries: tuple[SpeechBoundary, ...]
    boundaries: tuple[BoundaryCandidate, ...]
    segments: tuple[Segment, ...]
    video_duration_seconds: float | None

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_change_threshold": VISUAL_CHANGE_THRESHOLD,
            "speech_gap_threshold_seconds": SPEECH_GAP_THRESHOLD_SECONDS,
            "merge_tolerance_seconds": MERGE_TOLERANCE_SECONDS,
            "min_segment_duration_seconds": MIN_SEGMENT_DURATION_SECONDS,
            "video_duration_seconds": self.video_duration_seconds,
            "visual_changes": [change.to_dict() for change in self.visual_changes],
            "speech_boundaries": [boundary.to_dict() for boundary in self.speech_boundaries],
            "boundaries": [candidate.to_dict() for candidate in self.boundaries],
            "segments": [segment.to_dict() for segment in self.segments],
            "segment_count": self.segment_count,
        }


def detect_speech_boundaries(
    segments: Sequence[TranscriptSegment],
) -> list[SpeechBoundary]:
    """Speech-gap evidence from gaps between CONSECUTIVE transcript segments.

    For each adjacent pair a boundary exists when
    ``next.start - current.end >= SPEECH_GAP_THRESHOLD_SECONDS`` (an exact
    threshold qualifies). The evidence timestamp is the midpoint of the silence
    gap. Only inter-segment gaps count: leading silence, trailing silence, fewer
    than two segments, and overlapping speech (a negative gap) never create a
    boundary. Input order is preserved, so the output is deterministic.
    """
    boundaries: list[SpeechBoundary] = []
    for current, following in zip(segments, segments[1:], strict=False):
        gap_seconds = following.start_seconds - current.end_seconds
        if gap_seconds >= SPEECH_GAP_THRESHOLD_SECONDS:
            midpoint = (current.end_seconds + following.start_seconds) / 2
            boundaries.append(
                SpeechBoundary(timestamp_seconds=midpoint, gap_seconds=gap_seconds)
            )
    return boundaries


def _reason(
    visual: Sequence[VisualChange], speech: Sequence[SpeechBoundary]
) -> BoundaryReason:
    """Deterministic reason for a merged group (never an arbitrary string)."""
    if visual and speech:
        return "combined"
    if visual:
        return "visual"
    return "speech_gap"


def _combine(existing: BoundaryCandidate, incoming: BoundaryCandidate) -> BoundaryCandidate:
    """Merge ``incoming`` into ``existing``, keeping the earliest timestamp."""
    visual = (*existing.visual_changes, *incoming.visual_changes)
    speech = (*existing.speech_boundaries, *incoming.speech_boundaries)
    return BoundaryCandidate(
        timestamp_seconds=min(existing.timestamp_seconds, incoming.timestamp_seconds),
        reason=_reason(visual, speech),
        visual_changes=visual,
        speech_boundaries=speech,
    )


def merge_boundaries(
    visual_changes: Sequence[VisualChange],
    speech_boundaries: Sequence[SpeechBoundary],
) -> list[BoundaryCandidate]:
    """Merge visual + speech evidence into boundary candidates (Phase 1).

    Candidates are sorted by timestamp (visual before speech at equal times, for
    determinism) and walked chronologically. Each candidate is compared against
    the REPRESENTATIVE timestamp of the last accepted group — the earliest
    timestamp in that group — never against the most recently merged candidate.
    This anti-chaining rule stops a run of small steps from merging across a span
    larger than ``MERGE_TOLERANCE_SECONDS``. Merging unions the evidence and never
    destroys it.
    """
    ranked: list[tuple[float, int, BoundaryCandidate]] = []
    for change in visual_changes:
        ranked.append(
            (
                change.timestamp_seconds,
                0,
                BoundaryCandidate(
                    timestamp_seconds=change.timestamp_seconds,
                    reason="visual",
                    visual_changes=(change,),
                    speech_boundaries=(),
                ),
            )
        )
    for boundary in speech_boundaries:
        ranked.append(
            (
                boundary.timestamp_seconds,
                1,
                BoundaryCandidate(
                    timestamp_seconds=boundary.timestamp_seconds,
                    reason="speech_gap",
                    visual_changes=(),
                    speech_boundaries=(boundary,),
                ),
            )
        )
    ranked.sort(key=lambda entry: (entry[0], entry[1]))

    merged: list[BoundaryCandidate] = []
    for timestamp, _, candidate in ranked:
        if merged and timestamp - merged[-1].timestamp_seconds <= MERGE_TOLERANCE_SECONDS:
            merged[-1] = _combine(merged[-1], candidate)
        else:
            merged.append(candidate)
    return merged


def assemble_segments(
    boundaries: Sequence[BoundaryCandidate],
    video_duration: float,
) -> list[Segment]:
    """Promote merged boundaries into final segments (Phase 2).

    Boundaries are considered chronologically. A candidate is promoted only when
    the segment it would close is at least ``MIN_SEGMENT_DURATION_SECONDS`` long
    AND the tail from it to ``video_duration`` is not left shorter than the
    minimum; otherwise it is suppressed (its evidence is untouched and still
    reported). The first segment always starts at ``0.0`` and the last always ends
    at exactly ``video_duration``; every emitted segment has ``end > start``.
    """
    accepted: list[BoundaryCandidate] = []
    last_edge = 0.0
    for candidate in sorted(boundaries, key=lambda item: item.timestamp_seconds):
        timestamp = candidate.timestamp_seconds
        if timestamp >= video_duration:
            continue  # at/after the end: cannot close a valid internal segment
        if timestamp - last_edge < MIN_SEGMENT_DURATION_SECONDS:
            continue  # would close a segment shorter than the minimum
        if video_duration - timestamp < MIN_SEGMENT_DURATION_SECONDS:
            continue  # would leave a final segment shorter than the minimum
        accepted.append(candidate)
        last_edge = timestamp

    edges = [0.0, *(candidate.timestamp_seconds for candidate in accepted), video_duration]
    segments: list[Segment] = []
    for index in range(len(edges) - 1):
        start = edges[index]
        end = edges[index + 1]
        if end <= start:
            continue  # defensive: never emit a non-positive-duration segment
        reason = accepted[index].reason if index < len(accepted) else None
        segments.append(Segment(start_seconds=start, end_seconds=end, boundary_reason=reason))
    return segments
