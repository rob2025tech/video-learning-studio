"""Unit tests for the unified evidence timeline (Stage 0G).

These are pure models: OCR is a *point*, speech is an *interval*. They stay
distinct (no forced common domain model) and are merged into one deterministic
chronological order with stable tie-breaking.
"""

from __future__ import annotations

from video_learning.core.timeline import OcrEvent, SpeechEvent, build_timeline


def test_ocr_event_to_dict_is_a_point() -> None:
    event = OcrEvent(timestamp_seconds=15.2, text="video-learning-studio\npyproject.toml")

    assert event.to_dict() == {
        "type": "ocr",
        "timestamp_seconds": 15.2,
        "text": "video-learning-studio\npyproject.toml",
    }


def test_speech_event_to_dict_is_an_interval() -> None:
    event = SpeechEvent(start_seconds=17.4, end_seconds=23.1, text="Today we build")

    assert event.to_dict() == {
        "type": "speech",
        "start_seconds": 17.4,
        "end_seconds": 23.1,
        "text": "Today we build",
    }


def test_events_remain_distinct_types() -> None:
    # No forced common model: the point keeps a timestamp, the interval keeps a span.
    assert not hasattr(OcrEvent(1.0, "x"), "start_seconds")
    assert not hasattr(SpeechEvent(1.0, 2.0, "x"), "timestamp_seconds")


def test_build_timeline_sorts_chronologically() -> None:
    ocr = [OcrEvent(45.6, "late"), OcrEvent(15.2, "early")]
    speech = [SpeechEvent(17.4, 23.1, "middle")]

    timeline = build_timeline(ocr, speech)

    times = [event.sort_key[0] for event in timeline]
    assert times == sorted(times)
    assert [type(event) for event in timeline] == [OcrEvent, SpeechEvent, OcrEvent]


def test_build_timeline_ocr_before_speech_on_equal_time() -> None:
    # Deterministic tie-break: a point sorts before an interval at the same time.
    timeline = build_timeline([OcrEvent(10.0, "o")], [SpeechEvent(10.0, 12.0, "s")])

    assert isinstance(timeline[0], OcrEvent)
    assert isinstance(timeline[1], SpeechEvent)


def test_build_timeline_speech_tie_break_by_end() -> None:
    # Same start: the earlier-ending interval sorts first.
    longer = SpeechEvent(5.0, 9.0, "longer")
    shorter = SpeechEvent(5.0, 7.0, "shorter")

    assert build_timeline([], [longer, shorter]) == [shorter, longer]


def test_build_timeline_stable_for_identical_ocr_points() -> None:
    first = OcrEvent(3.0, "first")
    second = OcrEvent(3.0, "second")

    # Identical sort keys keep their input order (Python's sort is stable).
    assert build_timeline([first, second], []) == [first, second]


def test_build_timeline_empty_inputs() -> None:
    assert build_timeline([], []) == []


def test_build_timeline_only_ocr() -> None:
    timeline = build_timeline([OcrEvent(1.0, "a")], [])

    assert len(timeline) == 1
    assert isinstance(timeline[0], OcrEvent)


def test_build_timeline_only_speech() -> None:
    timeline = build_timeline([], [SpeechEvent(1.0, 2.0, "a")])

    assert len(timeline) == 1
    assert isinstance(timeline[0], SpeechEvent)
