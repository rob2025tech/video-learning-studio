"""Round-trip and validation tests for report-artifact rehydration (Multimodal Search v1, Step 3).

``to_dict`` and ``from_dict`` must be exact inverses on the shapes the report really
writes, and a reader must refuse an artifact it cannot honestly interpret instead of
repairing it. Each rejection test names the specific error, because a loader that
blew up on everything would also pass a test that only asked "did it raise".
"""

from __future__ import annotations

import pytest

from video_learning.core.scene import VISUAL_CHANGE_THRESHOLD, VisualChange
from video_learning.core.segments import (
    MERGE_TOLERANCE_SECONDS,
    MIN_SEGMENT_DURATION_SECONDS,
    SPEECH_GAP_THRESHOLD_SECONDS,
    BoundaryCandidate,
    Segment,
    Segmentation,
    SpeechBoundary,
)
from video_learning.core.timeline import OcrEvent, SpeechEvent, timeline_event_from_dict


def _segmentation() -> Segmentation:
    """A segmentation that exercises every serialized field, including suppressed evidence.

    The second visual change produced no segment of its own (only two of the three
    boundaries are promoted), so the reload has to prove evidence arrays survive even
    when they no longer back any final segment.
    """
    merged_visual = VisualChange(timestamp_seconds=10.0, change_score=27.5)
    merged_speech = SpeechBoundary(timestamp_seconds=12.0, gap_seconds=3.0)
    return Segmentation(
        visual_changes=(merged_visual, VisualChange(30.0, 41.25)),
        speech_boundaries=(merged_speech, SpeechBoundary(45.5, 2.0)),
        boundaries=(
            BoundaryCandidate(
                timestamp_seconds=10.0,
                reason="combined",
                visual_changes=(merged_visual,),
                speech_boundaries=(merged_speech,),
            ),
            BoundaryCandidate(30.0, "visual", (VisualChange(30.0, 41.25),), ()),
        ),
        segments=(
            Segment(0.0, 10.0, "visual"),
            Segment(10.0, 30.0, "combined"),
            Segment(30.0, 60.0, None),
        ),
        video_duration_seconds=60.0,
    )


def test_segmentation_round_trips_through_dict() -> None:
    segmentation = _segmentation()
    assert Segmentation.from_dict(segmentation.to_dict()) == segmentation


def test_round_trip_keeps_the_evidence_nested_inside_a_boundary() -> None:
    reloaded = Segmentation.from_dict(_segmentation().to_dict())
    assert reloaded.boundaries[0].visual_changes == (VisualChange(10.0, 27.5),)
    assert reloaded.boundaries[0].speech_boundaries == (SpeechBoundary(12.0, 3.0),)


def test_segmentation_without_duration_or_segments_round_trips() -> None:
    # The documented "ffprobe gave no duration" shape: evidence, but no assembly.
    empty = Segmentation((), (), (), (), None)
    assert Segmentation.from_dict(empty.to_dict()) == empty


def test_writer_still_emits_the_constants_the_reader_demands() -> None:
    """The four output-only constants in the artifact are this code's own values."""
    document = _segmentation().to_dict()
    assert document["visual_change_threshold"] == VISUAL_CHANGE_THRESHOLD
    assert document["speech_gap_threshold_seconds"] == SPEECH_GAP_THRESHOLD_SECONDS
    assert document["merge_tolerance_seconds"] == MERGE_TOLERANCE_SECONDS
    assert document["min_segment_duration_seconds"] == MIN_SEGMENT_DURATION_SECONDS


@pytest.mark.parametrize(
    "key",
    [
        "visual_change_threshold",
        "speech_gap_threshold_seconds",
        "merge_tolerance_seconds",
        "min_segment_duration_seconds",
    ],
)
def test_segmentation_rejects_a_constant_that_disagrees_with_this_code(key: str) -> None:
    document = _segmentation().to_dict()
    document[key] = document[key] + 0.5
    with pytest.raises(ValueError, match=key):
        Segmentation.from_dict(document)


def test_segmentation_rejects_a_segment_count_that_disagrees() -> None:
    document = _segmentation().to_dict()
    document["segment_count"] = len(document["segments"]) + 1
    with pytest.raises(ValueError, match="segment_count"):
        Segmentation.from_dict(document)


def test_boundary_candidate_rejects_an_unknown_reason() -> None:
    document = _segmentation().to_dict()
    document["boundaries"][0]["reason"] = "semantic"
    with pytest.raises(ValueError, match="unknown boundary reason"):
        Segmentation.from_dict(document)


def test_segment_rejects_an_unknown_boundary_reason() -> None:
    document = _segmentation().to_dict()
    document["segments"][0]["boundary_reason"] = "topic"
    with pytest.raises(ValueError, match="unknown boundary reason"):
        Segmentation.from_dict(document)


def test_final_segment_keeps_its_missing_boundary_reason() -> None:
    document = _segmentation().to_dict()
    assert document["segments"][-1]["boundary_reason"] is None
    assert Segmentation.from_dict(document).segments[-1].boundary_reason is None


def test_missing_values_raise_instead_of_being_invented() -> None:
    document = _segmentation().to_dict()
    del document["video_duration_seconds"]
    with pytest.raises(KeyError):
        Segmentation.from_dict(document)

    candidate = document["boundaries"][0]
    del candidate["speech_boundaries"]
    with pytest.raises(KeyError):
        BoundaryCandidate.from_dict(candidate)


def test_timeline_events_round_trip_through_dict() -> None:
    ocr = OcrEvent(timestamp_seconds=15.2, text="pyproject.toml")
    speech = SpeechEvent(start_seconds=17.4, end_seconds=23.1, text="Today we build")
    assert OcrEvent.from_dict(ocr.to_dict()) == ocr
    assert SpeechEvent.from_dict(speech.to_dict()) == speech


def test_timeline_event_from_dict_dispatches_on_the_type_tag() -> None:
    ocr = OcrEvent(15.2, "pyproject.toml")
    speech = SpeechEvent(17.4, 23.1, "Today we build")
    assert timeline_event_from_dict(ocr.to_dict()) == ocr
    assert timeline_event_from_dict(speech.to_dict()) == speech


def test_timeline_event_from_dict_rejects_an_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown timeline event type"):
        timeline_event_from_dict({"type": "caption", "start_seconds": 1.0})


def test_timeline_event_from_dict_requires_a_type() -> None:
    with pytest.raises(KeyError):
        timeline_event_from_dict({"timestamp_seconds": 1.0, "text": "hello"})


@pytest.mark.parametrize("assembled", [True, False])
@pytest.mark.parametrize("value", ["60", True])
def test_segmentation_rejects_a_non_numeric_duration(value: object, assembled: bool) -> None:
    """Checked in both shapes: with no segments, no later comparison would catch it."""
    segmentation = _segmentation() if assembled else Segmentation((), (), (), (), None)
    document = segmentation.to_dict()
    document["video_duration_seconds"] = value
    with pytest.raises(ValueError, match="video_duration_seconds"):
        Segmentation.from_dict(document)


@pytest.mark.parametrize("value", ["15.2", True])
def test_ocr_event_rejects_a_non_numeric_timestamp(value: object) -> None:
    document = OcrEvent(15.2, "pyproject.toml").to_dict()
    document["timestamp_seconds"] = value
    with pytest.raises(ValueError, match="timestamp_seconds"):
        OcrEvent.from_dict(document)


@pytest.mark.parametrize("key", ["start_seconds", "end_seconds"])
@pytest.mark.parametrize("value", ["17.4", True])
def test_speech_event_rejects_a_non_numeric_interval_end(key: str, value: object) -> None:
    document = SpeechEvent(17.4, 23.1, "Today we build").to_dict()
    document[key] = value
    with pytest.raises(ValueError, match=key):
        SpeechEvent.from_dict(document)


def test_integer_seconds_are_accepted_as_numbers() -> None:
    """JSON has one number type, so refusing quoted numbers must not refuse ``15``."""
    ocr = OcrEvent.from_dict({"type": "ocr", "timestamp_seconds": 15, "text": "pyproject.toml"})
    speech = SpeechEvent.from_dict(
        {"type": "speech", "start_seconds": 17, "end_seconds": 23, "text": "Today we build"}
    )
    document = _segmentation().to_dict()
    document["video_duration_seconds"] = 60

    assert ocr.timestamp_seconds == 15.0
    assert (speech.start_seconds, speech.end_seconds) == (17.0, 23.0)
    assert Segmentation.from_dict(document).video_duration_seconds == 60.0
