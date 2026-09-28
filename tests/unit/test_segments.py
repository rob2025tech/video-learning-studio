"""Unit tests for Stage 0I pure boundary logic (core/segments.py).

Covers the authoritative rules without any I/O: speech-gap detection, the
boundary-reason taxonomy, merge tolerance with the anti-chaining requirement,
minimum-duration suppression, known-duration assembly, and serialization.
"""

from __future__ import annotations

from typing import get_args

import pytest

from video_learning.core.scene import VisualChange
from video_learning.core.segments import (
    MERGE_TOLERANCE_SECONDS,
    MIN_SEGMENT_DURATION_SECONDS,
    SPEECH_GAP_THRESHOLD_SECONDS,
    BoundaryCandidate,
    BoundaryReason,
    Segment,
    Segmentation,
    SpeechBoundary,
    assemble_segments,
    detect_speech_boundaries,
    merge_boundaries,
)
from video_learning.core.transcript import TranscriptSegment


def _seg(start: float, end: float, text: str = "speech") -> TranscriptSegment:
    return TranscriptSegment(start_seconds=start, end_seconds=end, text=text)


def _visual(timestamp: float, score: float = 30.0) -> VisualChange:
    return VisualChange(timestamp_seconds=timestamp, change_score=score)


def _speech(timestamp: float, gap: float = 3.0) -> SpeechBoundary:
    return SpeechBoundary(timestamp_seconds=timestamp, gap_seconds=gap)


def _candidate(timestamp: float, reason: BoundaryReason) -> BoundaryCandidate:
    if reason == "visual":
        return BoundaryCandidate(timestamp, reason, (_visual(timestamp),), ())
    if reason == "speech_gap":
        return BoundaryCandidate(timestamp, reason, (), (_speech(timestamp),))
    return BoundaryCandidate(timestamp, reason, (_visual(timestamp),), (_speech(timestamp),))


# -- authoritative constants -------------------------------------------------


def test_authoritative_constants() -> None:
    assert SPEECH_GAP_THRESHOLD_SECONDS == 2.0
    assert MERGE_TOLERANCE_SECONDS == 1.0
    assert MIN_SEGMENT_DURATION_SECONDS == 2.0


# -- speech-gap boundaries ---------------------------------------------------


def test_gap_exactly_threshold_creates_boundary() -> None:
    # gap = 5.0 - 3.0 = 2.0 qualifies (>= 2.0); midpoint = (3 + 5) / 2 = 4.0
    boundaries = detect_speech_boundaries([_seg(0.0, 3.0), _seg(5.0, 8.0)])
    assert len(boundaries) == 1
    assert boundaries[0].gap_seconds == pytest.approx(2.0)
    assert boundaries[0].timestamp_seconds == pytest.approx(4.0)


def test_gap_greater_than_threshold_creates_boundary() -> None:
    boundaries = detect_speech_boundaries([_seg(0.0, 2.0), _seg(7.5, 9.0)])
    assert len(boundaries) == 1
    assert boundaries[0].gap_seconds == pytest.approx(5.5)
    assert boundaries[0].timestamp_seconds == pytest.approx((2.0 + 7.5) / 2)


def test_gap_below_threshold_creates_no_boundary() -> None:
    # gap = 3.5 - 2.0 = 1.5 < 2.0
    assert detect_speech_boundaries([_seg(0.0, 2.0), _seg(3.5, 5.0)]) == []


def test_midpoint_calculation() -> None:
    boundaries = detect_speech_boundaries([_seg(1.0, 4.0), _seg(10.0, 12.0)])
    assert boundaries[0].timestamp_seconds == pytest.approx(7.0)  # (4 + 10) / 2
    assert boundaries[0].gap_seconds == pytest.approx(6.0)


def test_no_leading_silence_boundary() -> None:
    # Leading silence before the first segment never creates a boundary; only the
    # inter-segment gap (12 -> 15 = 3.0) does.
    boundaries = detect_speech_boundaries([_seg(10.0, 12.0), _seg(15.0, 20.0)])
    assert len(boundaries) == 1
    assert boundaries[0].timestamp_seconds == pytest.approx(13.5)


def test_no_trailing_silence_boundary() -> None:
    # Trailing silence after the last segment never creates a boundary; only the
    # inter-segment gap (5 -> 8 = 3.0) does.
    boundaries = detect_speech_boundaries([_seg(0.0, 5.0), _seg(8.0, 10.0)])
    assert len(boundaries) == 1
    assert boundaries[0].timestamp_seconds == pytest.approx(6.5)


def test_zero_segments_no_boundaries() -> None:
    assert detect_speech_boundaries([]) == []


def test_one_segment_no_boundaries() -> None:
    assert detect_speech_boundaries([_seg(0.0, 5.0)]) == []


def test_overlapping_speech_no_positive_gap() -> None:
    # next.start (3.0) < current.end (5.0) -> negative gap -> no boundary
    assert detect_speech_boundaries([_seg(0.0, 5.0), _seg(3.0, 8.0)]) == []


def test_multiple_gaps_preserve_input_order() -> None:
    segments = [
        _seg(0.0, 1.0),
        _seg(4.0, 5.0),  # gap 3.0 -> boundary at 2.5
        _seg(5.5, 6.0),  # gap 0.5 -> none
        _seg(9.0, 10.0),  # gap 3.0 -> boundary at 7.5
    ]
    boundaries = detect_speech_boundaries(segments)
    assert [b.timestamp_seconds for b in boundaries] == pytest.approx([2.5, 7.5])


# -- boundary reason taxonomy ------------------------------------------------


def test_boundary_reason_taxonomy_is_exactly_three_values() -> None:
    assert set(get_args(BoundaryReason)) == {"visual", "speech_gap", "combined"}


def test_merged_reason_visual_only() -> None:
    merged = merge_boundaries([_visual(10.0)], [])
    assert len(merged) == 1
    assert merged[0].reason == "visual"


def test_merged_reason_speech_only() -> None:
    merged = merge_boundaries([], [_speech(10.0)])
    assert merged[0].reason == "speech_gap"


def test_merged_reason_combined() -> None:
    merged = merge_boundaries([_visual(10.0)], [_speech(10.4)])
    assert len(merged) == 1
    assert merged[0].reason == "combined"


# -- merge tolerance + anti-chaining -----------------------------------------


def test_candidates_within_tolerance_merge() -> None:
    merged = merge_boundaries([_visual(10.0), _visual(10.8)], [])
    assert len(merged) == 1
    assert merged[0].timestamp_seconds == pytest.approx(10.0)  # earliest representative
    assert merged[0].reason == "visual"
    assert len(merged[0].visual_changes) == 2  # evidence retained


def test_candidates_beyond_tolerance_stay_separate() -> None:
    merged = merge_boundaries([_visual(10.0), _visual(11.5)], [])
    assert [c.timestamp_seconds for c in merged] == pytest.approx([10.0, 11.5])


def test_merge_at_exact_tolerance_is_inclusive() -> None:
    # exactly 1.0 apart -> within tolerance -> merged
    merged = merge_boundaries([_visual(10.0), _visual(11.0)], [])
    assert len(merged) == 1
    assert merged[0].timestamp_seconds == pytest.approx(10.0)


def test_visual_plus_speech_within_tolerance_is_combined() -> None:
    merged = merge_boundaries([_visual(10.0)], [_speech(10.5)])
    assert len(merged) == 1
    assert merged[0].reason == "combined"
    assert merged[0].timestamp_seconds == pytest.approx(10.0)
    assert len(merged[0].visual_changes) == 1
    assert len(merged[0].speech_boundaries) == 1


def test_visual_plus_visual_within_tolerance_is_one_visual() -> None:
    merged = merge_boundaries([_visual(10.0), _visual(10.5)], [])
    assert len(merged) == 1
    assert merged[0].reason == "visual"


def test_speech_plus_speech_within_tolerance_is_one_speech_gap() -> None:
    merged = merge_boundaries([], [_speech(10.0), _speech(10.5)])
    assert len(merged) == 1
    assert merged[0].reason == "speech_gap"


def test_representative_timestamp_is_earliest_in_cluster() -> None:
    # speech at 9.8 is earlier than visual at 10.0; both within tolerance
    merged = merge_boundaries([_visual(10.0)], [_speech(9.8)])
    assert len(merged) == 1
    assert merged[0].timestamp_seconds == pytest.approx(9.8)
    assert merged[0].reason == "combined"


def test_anti_chaining_does_not_merge_across_representative() -> None:
    # Authoritative anti-chaining example: visual 10.0, speech_gap 10.5, visual 11.2.
    # 11.2 is within 1.0 of 10.5, but 10.5 already merged into the group whose
    # representative is 10.0; 11.2 - 10.0 = 1.2 > 1.0, so 11.2 stays separate.
    merged = merge_boundaries([_visual(10.0), _visual(11.2)], [_speech(10.5)])
    assert len(merged) == 2
    assert merged[0].timestamp_seconds == pytest.approx(10.0)
    assert merged[0].reason == "combined"
    assert merged[1].timestamp_seconds == pytest.approx(11.2)
    assert merged[1].reason == "visual"


# -- minimum segment duration ------------------------------------------------


def test_min_duration_suppresses_short_leading_boundary() -> None:
    # boundary at 1.0 would create a 1.0s segment (< 2.0) -> suppressed
    segments = assemble_segments([_candidate(1.0, "visual"), _candidate(5.0, "visual")], 10.0)
    assert [s.start_seconds for s in segments] == pytest.approx([0.0, 5.0])
    assert [s.end_seconds for s in segments] == pytest.approx([5.0, 10.0])


def test_exact_min_duration_boundary_is_retained() -> None:
    # boundary at 2.0 -> segment 0 -> 2 is exactly 2.0 (qualifies)
    segments = assemble_segments([_candidate(2.0, "visual")], 10.0)
    assert [s.start_seconds for s in segments] == pytest.approx([0.0, 2.0])
    assert [s.end_seconds for s in segments] == pytest.approx([2.0, 10.0])


def test_boundary_too_close_to_end_is_suppressed() -> None:
    # boundary at 9.0 leaves a 1.0s tail (< 2.0) -> suppressed
    segments = assemble_segments([_candidate(9.0, "visual")], 10.0)
    assert [s.start_seconds for s in segments] == pytest.approx([0.0])
    assert [s.end_seconds for s in segments] == pytest.approx([10.0])


def test_suppression_does_not_mutate_input_evidence() -> None:
    boundaries = [_candidate(1.0, "visual"), _candidate(5.0, "visual")]
    before = list(boundaries)
    assemble_segments(boundaries, 10.0)
    assert boundaries == before  # suppressed boundary still present as evidence
    assert len(boundaries) == 2


def test_segmentation_retains_boundaries_suppressed_from_segments() -> None:
    visual = (_visual(1.0), _visual(5.0))
    boundaries = tuple(merge_boundaries(list(visual), []))
    segments = tuple(assemble_segments(list(boundaries), 10.0))
    segmentation = Segmentation(
        visual_changes=visual,
        speech_boundaries=(),
        boundaries=boundaries,
        segments=segments,
        video_duration_seconds=10.0,
    )
    # 1.0 suppressed from segments, but both boundaries + both changes retained
    assert segmentation.segment_count == 2  # [0,5],[5,10]
    assert len(segmentation.boundaries) == 2
    assert len(segmentation.visual_changes) == 2


# -- known-duration assembly -------------------------------------------------


def test_known_duration_assembles_expected_ranges() -> None:
    boundaries = [_candidate(10.0, "visual"), _candidate(20.0, "speech_gap")]
    segments = assemble_segments(boundaries, 30.0)
    assert [s.start_seconds for s in segments] == pytest.approx([0.0, 10.0, 20.0])
    assert [s.end_seconds for s in segments] == pytest.approx([10.0, 20.0, 30.0])


def test_first_starts_at_zero_and_last_ends_at_duration() -> None:
    segments = assemble_segments([_candidate(10.0, "visual"), _candidate(20.0, "visual")], 30.0)
    assert segments[0].start_seconds == pytest.approx(0.0)
    assert segments[-1].end_seconds == pytest.approx(30.0)


def test_segments_are_ordered_and_contiguous() -> None:
    segments = assemble_segments([_candidate(10.0, "visual"), _candidate(20.0, "visual")], 30.0)
    starts = [s.start_seconds for s in segments]
    ends = [s.end_seconds for s in segments]
    assert starts == sorted(starts)
    assert ends[:-1] == pytest.approx(starts[1:])  # each end == next start


def test_all_segment_durations_positive() -> None:
    segments = assemble_segments([_candidate(10.0, "visual"), _candidate(20.0, "visual")], 30.0)
    assert all(s.end_seconds - s.start_seconds > 0 for s in segments)


def test_no_boundaries_yields_single_full_duration_segment() -> None:
    segments = assemble_segments([], 30.0)
    assert len(segments) == 1
    assert segments[0].start_seconds == pytest.approx(0.0)
    assert segments[0].end_seconds == pytest.approx(30.0)
    assert segments[0].boundary_reason is None


def test_no_segment_dropped_when_all_well_spaced() -> None:
    boundaries = [_candidate(5.0, "visual"), _candidate(10.0, "visual"), _candidate(15.0, "visual")]
    segments = assemble_segments(boundaries, 20.0)
    assert len(segments) == 4
    assert [s.start_seconds for s in segments] == pytest.approx([0.0, 5.0, 10.0, 15.0])
    assert [s.end_seconds for s in segments] == pytest.approx([5.0, 10.0, 15.0, 20.0])


def test_assembly_is_deterministic() -> None:
    boundaries = [_candidate(20.0, "visual"), _candidate(10.0, "speech_gap")]  # unsorted input
    first = assemble_segments(boundaries, 30.0)
    second = assemble_segments(boundaries, 30.0)
    assert first == second
    assert [s.start_seconds for s in first] == pytest.approx([0.0, 10.0, 20.0])


def test_segment_boundary_reason_end_convention_is_implementation_detail() -> None:
    # NOT an authoritative spec field. Documents the current END-boundary
    # convention: each segment carries the reason of the boundary at its END; the
    # final segment (ends at duration) carries None. Authoritative reason
    # assertions live on BoundaryCandidate / boundaries[].
    boundaries = [_candidate(10.0, "visual"), _candidate(20.0, "speech_gap")]
    segments = assemble_segments(boundaries, 30.0)
    assert [s.boundary_reason for s in segments] == ["visual", "speech_gap", None]


# -- serialization (report shape) --------------------------------------------


def test_visual_change_serialization() -> None:
    assert _visual(12.34, 51.953).to_dict() == {
        "timestamp_seconds": 12.34,
        "change_score": 51.953,
    }


def test_speech_boundary_serialization() -> None:
    assert _speech(14.5, 3.0).to_dict() == {"timestamp_seconds": 14.5, "gap_seconds": 3.0}


def test_boundary_candidate_serializes_reason_and_evidence() -> None:
    candidate = BoundaryCandidate(
        timestamp_seconds=10.0,
        reason="combined",
        visual_changes=(_visual(10.0, 30.0),),
        speech_boundaries=(_speech(10.4, 2.5),),
    )
    assert candidate.to_dict() == {
        "timestamp_seconds": 10.0,
        "reason": "combined",
        "visual_changes": [{"timestamp_seconds": 10.0, "change_score": 30.0}],
        "speech_boundaries": [{"timestamp_seconds": 10.4, "gap_seconds": 2.5}],
    }


def test_segment_serialization() -> None:
    assert Segment(0.0, 10.0, "visual").to_dict() == {
        "start_seconds": 0.0,
        "end_seconds": 10.0,
        "boundary_reason": "visual",
    }


def test_segmentation_to_dict_known_duration_full_shape() -> None:
    visual = (_visual(10.0, 30.0),)
    speech = (_speech(20.0, 3.0),)
    boundaries = tuple(merge_boundaries(list(visual), list(speech)))
    segments = tuple(assemble_segments(list(boundaries), 30.0))
    payload = Segmentation(
        visual_changes=visual,
        speech_boundaries=speech,
        boundaries=boundaries,
        segments=segments,
        video_duration_seconds=30.0,
    ).to_dict()

    # thresholds serialized
    assert payload["visual_change_threshold"] == 10.0
    assert payload["speech_gap_threshold_seconds"] == 2.0
    assert payload["merge_tolerance_seconds"] == 1.0
    assert payload["min_segment_duration_seconds"] == 2.0
    # known duration serialized as a NUMBER
    assert payload["video_duration_seconds"] == 30.0
    assert isinstance(payload["video_duration_seconds"], float)
    # evidence serialized
    assert payload["visual_changes"] == [{"timestamp_seconds": 10.0, "change_score": 30.0}]
    assert payload["speech_boundaries"] == [{"timestamp_seconds": 20.0, "gap_seconds": 3.0}]
    # merged boundaries serialized WITH reason (authoritative reason location)
    assert [b["reason"] for b in payload["boundaries"]] == ["visual", "speech_gap"]
    # final segments + count
    assert payload["segment_count"] == len(payload["segments"]) == 3
    # no prohibited field
    assert "sample_interval_seconds" not in payload


def test_segmentation_to_dict_unknown_duration_serializes_null() -> None:
    visual = (_visual(542.0, 30.0),)
    speech = (_speech(550.0, 3.0),)
    payload = Segmentation(
        visual_changes=visual,
        speech_boundaries=speech,
        boundaries=tuple(merge_boundaries(list(visual), list(speech))),
        segments=(),
        video_duration_seconds=None,
    ).to_dict()

    assert payload["video_duration_seconds"] is None
    assert payload["segments"] == []
    assert payload["segment_count"] == 0
    # evidence retained despite unknown duration
    assert len(payload["visual_changes"]) == 1
    assert len(payload["speech_boundaries"]) == 1
    assert len(payload["boundaries"]) == 2
    assert "sample_interval_seconds" not in payload
