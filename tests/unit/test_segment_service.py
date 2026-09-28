"""Unit tests for SegmentService orchestration (Stage 0I).

Uses a local test double for the concrete ``FfmpegSceneDetector`` (no new
production Protocol/abstraction is introduced for testing). Covers the full path:
detector invocation, speech-gap evidence, merge, minimum-duration filtering,
known-duration assembly, and the critical unknown-duration behavior (evidence
retained, zero segments, and NO duration inferred from evidence timestamps).
"""

from __future__ import annotations

from pathlib import Path

from video_learning.core.models import MediaInfo, VideoStreamInfo
from video_learning.core.scene import VisualChange
from video_learning.core.transcript import TranscriptResult, TranscriptSegment
from video_learning.services.segment_service import SegmentService


class FakeSceneDetector:
    """Test double for FfmpegSceneDetector: returns canned visual changes."""

    def __init__(self, changes: list[VisualChange] | None = None) -> None:
        self._changes = changes or []
        self.calls: list[MediaInfo] = []

    def detect(self, media: MediaInfo) -> list[VisualChange]:
        self.calls.append(media)
        return list(self._changes)


def _media(tmp_path: Path, *, duration: float | None) -> MediaInfo:
    return MediaInfo(
        path=tmp_path / "clip.mov",
        filename="clip.mov",
        duration_seconds=duration,
        video=VideoStreamInfo(width=64, height=64),
    )


def _transcript(*segments: TranscriptSegment) -> TranscriptResult:
    return TranscriptResult(
        source="clip.mov",
        backend="whisper.cpp",
        model="ggml-base.en",
        language="en",
        segments=segments,
    )


def _service(
    changes: list[VisualChange] | None = None,
) -> tuple[SegmentService, FakeSceneDetector]:
    detector = FakeSceneDetector(changes)
    service = SegmentService(scene_detector=detector)  # type: ignore[arg-type]
    return service, detector


# -- orchestration -----------------------------------------------------------


def test_invokes_visual_detector_with_the_same_media(tmp_path: Path) -> None:
    service, detector = _service([VisualChange(10.0, 30.0)])
    media = _media(tmp_path, duration=30.0)
    service.segment(media, None)
    # Reuses the media it was given; never re-inspects.
    assert detector.calls == [media]


def test_speech_gap_evidence_computed_from_transcription(tmp_path: Path) -> None:
    service, _ = _service([])
    transcript = _transcript(
        TranscriptSegment(0.0, 2.0, "a"),
        TranscriptSegment(6.0, 8.0, "b"),  # gap 4.0 -> midpoint 4.0
    )
    result = service.segment(_media(tmp_path, duration=30.0), transcript)
    assert len(result.speech_boundaries) == 1
    assert result.speech_boundaries[0].timestamp_seconds == 4.0
    assert result.speech_boundaries[0].gap_seconds == 4.0


def test_merges_visual_and_speech_into_combined_candidate(tmp_path: Path) -> None:
    service, _ = _service([VisualChange(10.0, 30.0)])
    transcript = _transcript(
        TranscriptSegment(0.0, 9.0, "a"),
        TranscriptSegment(11.5, 20.0, "b"),  # gap 2.5 -> midpoint 10.25
    )
    result = service.segment(_media(tmp_path, duration=30.0), transcript)
    # visual@10.0 and speech@10.25 are within 1.0 -> one combined candidate
    assert len(result.boundaries) == 1
    assert result.boundaries[0].reason == "combined"
    assert result.boundaries[0].timestamp_seconds == 10.0


def test_applies_min_duration_and_assembles_known_duration(tmp_path: Path) -> None:
    service, _ = _service([VisualChange(10.0, 30.0), VisualChange(20.0, 40.0)])
    result = service.segment(_media(tmp_path, duration=30.0), None)
    assert result.video_duration_seconds == 30.0
    assert result.segment_count == 3
    assert [s.start_seconds for s in result.segments] == [0.0, 10.0, 20.0]
    assert [s.end_seconds for s in result.segments] == [10.0, 20.0, 30.0]


def test_min_duration_suppression_keeps_evidence(tmp_path: Path) -> None:
    # 1.0 is suppressed from segments but retained as a boundary + visual change.
    service, _ = _service([VisualChange(1.0, 30.0), VisualChange(5.0, 30.0)])
    result = service.segment(_media(tmp_path, duration=10.0), None)
    assert len(result.visual_changes) == 2
    assert len(result.boundaries) == 2
    assert result.segment_count == 2  # [0,5],[5,10]
    assert [s.start_seconds for s in result.segments] == [0.0, 5.0]


def test_no_evidence_yields_single_full_duration_segment(tmp_path: Path) -> None:
    service, _ = _service([])
    result = service.segment(_media(tmp_path, duration=30.0), None)
    assert result.segment_count == 1
    assert result.segments[0].start_seconds == 0.0
    assert result.segments[0].end_seconds == 30.0


def test_transcription_none_is_valid(tmp_path: Path) -> None:
    service, _ = _service([VisualChange(10.0, 30.0)])
    result = service.segment(_media(tmp_path, duration=30.0), None)
    assert result.speech_boundaries == ()
    assert result.segment_count == 2  # [0,10],[10,30]


# -- unknown duration (critical) ---------------------------------------------


def test_unknown_duration_returns_no_segments_but_retains_evidence(tmp_path: Path) -> None:
    service, _ = _service([VisualChange(10.0, 30.0), VisualChange(20.0, 40.0)])
    transcript = _transcript(
        TranscriptSegment(0.0, 2.0, "a"),
        TranscriptSegment(6.0, 8.0, "b"),  # gap 4.0 -> speech boundary at 4.0
    )
    result = service.segment(_media(tmp_path, duration=None), transcript)

    # Required unknown-duration behavior:
    assert result.video_duration_seconds is None
    assert result.segments == ()
    assert result.segment_count == 0

    # ALL evidence retained:
    assert len(result.visual_changes) == 2
    assert len(result.speech_boundaries) == 1
    assert len(result.boundaries) == 3  # 10.0, 20.0, 4.0 (all > 1.0 apart)

    # Serialized form makes the absence explicit:
    payload = result.to_dict()
    assert payload["video_duration_seconds"] is None
    assert payload["segments"] == []
    assert payload["segment_count"] == 0


def test_unknown_duration_never_infers_from_latest_evidence(tmp_path: Path) -> None:
    # The exact regression the design review called out: with duration unknown and
    # latest evidence at 542/550, the result must NOT end at 550 (no truncation,
    # no inference). It must produce zero segments while retaining evidence.
    service, _ = _service([VisualChange(542.0, 30.0)])
    transcript = _transcript(
        TranscriptSegment(0.0, 548.0, "a"),
        TranscriptSegment(550.0, 552.0, "b"),  # gap 2.0 -> speech boundary at 549.0
    )
    result = service.segment(_media(tmp_path, duration=None), transcript)

    assert result.video_duration_seconds is None
    assert result.segments == ()
    assert result.segment_count == 0
    # No segment ends at any evidence timestamp.
    assert all(s.end_seconds not in (549.0, 550.0, 552.0) for s in result.segments)
    # Evidence still retained.
    assert len(result.visual_changes) == 1
    assert len(result.speech_boundaries) == 1


# -- determinism -------------------------------------------------------------


def test_segmentation_is_deterministic(tmp_path: Path) -> None:
    changes = [VisualChange(10.0, 30.0), VisualChange(20.0, 40.0)]
    transcript = _transcript(
        TranscriptSegment(0.0, 2.0, "a"),
        TranscriptSegment(6.0, 8.0, "b"),
    )
    media = _media(tmp_path, duration=30.0)
    service_a, _ = _service(changes)
    service_b, _ = _service(changes)

    first = service_a.segment(media, transcript)
    second = service_b.segment(media, transcript)
    assert first == second
    assert first.to_dict() == second.to_dict()
