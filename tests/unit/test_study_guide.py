"""Unit tests for the deterministic study guide (LearnLens Task 2).

No ffmpeg, tesseract, or whisper.cpp is needed: every guide is built by calling
the real Task 1 derivation over hand-authored timelines, except where a Stage 0I
or report fixture is assembled through the actual pipeline.

The suite protects the four properties that make the guide trustworthy:

* **reuse, not reconstruction** — sections carry Task 1's evidence objects;
* **no invention** — every content string in the artifact is evidence-derived,
  with no loss and nothing added;
* **honest limits** — empty and sparse evidence is reported as such, in counts;
* **carry-through** — provenance, unattributed evidence, and Task 1's attribution
  rules reach the guide unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from video_learning.core.models import AudioStreamInfo, MediaInfo
from video_learning.core.scene import VisualChange
from video_learning.core.segments import (
    BoundaryReason,
    Segment,
    Segmentation,
    SpeechBoundary,
    assemble_segments,
    merge_boundaries,
)
from video_learning.core.study import (
    STUDY_EVIDENCE_SCHEMA,
    StudyEvidence,
    build_study_evidence,
)
from video_learning.core.study_guide import (
    STUDY_GUIDE_SCHEMA,
    StudyGuide,
    build_study_guide,
)
from video_learning.core.timeline import (
    OcrEvent,
    SpeechEvent,
    TimelineEvent,
    build_timeline,
)
from video_learning.core.transcript import TranscriptResult, TranscriptSegment
from video_learning.services.analyze_service import AnalyzeResult
from video_learning.services.report_service import ReportResult, ReportService
from video_learning.services.study_guide_service import StudyGuideService
from video_learning.services.study_service import StudyEvidenceService

_SOURCE = Path("/tmp/sample.mp4")
_MODEL = Path("/tmp/ggml-base.en.bin")
_UNAVAILABLE_DETAIL = "whisper-cli not found on PATH — install whisper.cpp"
_SECTION_KEYS = {
    "segment_id",
    "timestamp_range",
    "boundary_reason",
    "evidence_state",
    "key_terms",
    "on_screen",
    "spoken",
    "key_term_count",
    "on_screen_count",
    "spoken_count",
}


def _segment(start: float, end: float, reason: BoundaryReason | None) -> Segment:
    return Segment(start_seconds=start, end_seconds=end, boundary_reason=reason)


def _ocr(seconds: float, text: str) -> OcrEvent:
    return OcrEvent(timestamp_seconds=seconds, text=text)


def _speech(start: float, end: float, text: str) -> SpeechEvent:
    return SpeechEvent(start_seconds=start, end_seconds=end, text=text)


def _segmentation(segments: list[Segment], duration: float | None) -> Segmentation:
    """Wrap segments in a ``Segmentation``, asserting the real Stage 0I invariants.

    A fixture shaped differently from what ``assemble_segments`` can return would
    test a fiction, so an impossible shape fails here rather than quietly.
    """
    if duration is not None and segments:
        assert segments[0].start_seconds == 0.0, "Stage 0I always starts at 0.0"
        assert segments[-1].end_seconds == duration, "Stage 0I always ends at the duration"
        assert segments[-1].boundary_reason is None, "the final segment ends at the duration"
        for previous, following in zip(segments, segments[1:], strict=False):
            assert previous.end_seconds == following.start_seconds, "segments must tile"
        for segment in segments:
            assert segment.end_seconds > segment.start_seconds, "segments have positive length"
    return Segmentation(
        visual_changes=(),
        speech_boundaries=(),
        boundaries=(),
        segments=tuple(segments),
        video_duration_seconds=duration,
    )


def _evidence(
    segments: list[Segment],
    duration: float | None,
    timeline: Sequence[TimelineEvent],
    status: str = "ok",
    detail: str | None = None,
) -> StudyEvidence:
    return build_study_evidence(
        _segmentation(segments, duration), timeline, status, detail, source=_SOURCE
    )


def _guide(
    segments: list[Segment],
    duration: float | None,
    timeline: Sequence[TimelineEvent],
    status: str = "ok",
    detail: str | None = None,
) -> StudyGuide:
    return build_study_guide(_evidence(segments, duration, timeline, status, detail))


def _rich_evidence() -> StudyEvidence:
    """Both evidence kinds, several eligible phrases, and one unattributed event."""
    segments = [
        _segment(0.0, 20.0, "visual"),
        _segment(20.0, 45.0, "combined"),
        _segment(45.0, 60.0, None),
    ]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nmodel usage\nQoder CLI Release Notes"),
        _speech(6.0, 19.0, "the panel shows the remaining quota"),
        _ocr(25.0, "Model Usage\nProjects/Templates/video-learning-studio"),
        _speech(22.0, 44.0, "we open the repository root"),
        _ocr(50.0, "Model Usage\nQoder CLI Release Notes"),
        _speech(48.0, 58.0, "notes ship with every release"),
        _ocr(95.0, "Model Usage"),  # beyond the duration: unattributed
    ]
    return _evidence(segments, 60.0, timeline)


def _key_term_names(guide: StudyGuide, index: int) -> list[str]:
    return [term.phrase for term in guide.sections[index].key_terms]


def _collect_content(node: Any, texts: list[str], phrases: list[str]) -> None:
    """Gather every string under a content key, leaving metadata keys alone.

    Only ``text`` (OCR and speech) and ``phrase`` (key terms) carry captured
    content. Schema ids, source paths, status values, durations and boundary
    labels are metadata, so inspecting them would reject legitimate fields.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "text" and isinstance(value, str):
                texts.append(value)
            elif key == "phrase" and isinstance(value, str):
                phrases.append(value)
            else:
                _collect_content(value, texts, phrases)
    elif isinstance(node, list):
        for item in node:
            _collect_content(item, texts, phrases)


def _payload_content(payload: Any) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    phrases: list[str] = []
    _collect_content(payload, texts, phrases)
    return texts, phrases


def _all_keys(node: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(str(key))
            keys |= _all_keys(value)
    elif isinstance(node, list):
        for item in node:
            keys |= _all_keys(item)
    return keys


class _RecordingReportService(ReportService):
    """A :class:`ReportService` that returns a fixed report and records its calls.

    Subclassed rather than duck-typed so strict mypy checks the signature
    compatibility instead of trusting a structural match.
    """

    def __init__(self, report: ReportResult) -> None:
        self.report = report
        self.calls: list[tuple[Path, Path | None]] = []

    def build(self, path: Path, *, model: Path | None = None) -> ReportResult:
        self.calls.append((path, model))
        return self.report


def _valid_report() -> ReportResult:
    """A ``ReportResult`` obeying every production invariant of the report layer.

    ``transcription_status == "ok"`` carries a real ``TranscriptResult``, the media
    genuinely has audio, ``transcription_detail`` is ``None`` on that path, and the
    timeline's speech events come only from the transcript's segments — exactly as
    ``ReportService.build`` assembles them.
    """
    media = MediaInfo(
        path=_SOURCE,
        filename=_SOURCE.name,
        container_format="mp4",
        duration_seconds=40.0,
        audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2),
    )
    transcript = TranscriptResult(
        source=str(_SOURCE),
        backend="whisper.cpp",
        model="ggml-base.en",
        language="en",
        segments=(TranscriptSegment(12.0, 18.0, "spoken words"),),
    )
    timeline = build_timeline(
        [OcrEvent(timestamp_seconds=10.0, text="Model Usage\nModel Usage")],
        [
            SpeechEvent(segment.start_seconds, segment.end_seconds, segment.text)
            for segment in transcript.segments
        ],
    )
    segments = _segmentation([_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)], 40.0)
    return ReportResult(
        source=_SOURCE,
        media=media,
        analysis=AnalyzeResult(source=_SOURCE),
        transcription=transcript,
        transcription_status="ok",
        transcription_detail=None,
        segmentation=segments,
        timeline=timeline,
    )


# ---------------------------------------------------------------------------
# 1. The guide projects Task 1's objects instead of re-deriving anything.
# ---------------------------------------------------------------------------
def test_sections_are_the_task1_objects_themselves() -> None:
    """Identity, not merely equality: the guide holds the very same tuples.

    If a section carried a reconstruction, a second phrase or attribution algorithm
    would exist by accident and could drift. Passing the tuples through makes
    divergence structurally impossible.
    """
    evidence = _rich_evidence()
    guide = build_study_guide(evidence)

    assert len(guide.sections) == len(evidence.segments)
    for section, segment in zip(guide.sections, evidence.segments, strict=True):
        assert section.key_terms is segment.phrases
        assert section.on_screen is segment.ocr_events
        assert section.spoken is segment.transcript_events
        assert section.segment_id == segment.segment_id
        assert (section.start_seconds, section.end_seconds) == (
            segment.start_seconds,
            segment.end_seconds,
        )
        assert section.boundary_reason == segment.boundary_reason
    assert guide.unattributed is evidence.unattributed


def test_no_content_string_is_invented() -> None:
    """The serialized guide holds exactly the evidence strings: none added, none lost.

    The multiset of ``text`` values must equal every OCR and speech event in the
    evidence (sections plus unattributed), and the multiset of ``phrase`` values
    must equal every key term Task 1 produced. A generated summary, a padded
    section, or a dropped event each fails this.
    """
    evidence = _rich_evidence()
    payload = build_study_guide(evidence).to_dict()
    texts, phrases = _payload_content(payload)

    attributed: list[TimelineEvent] = []
    for segment in evidence.segments:
        attributed.extend(segment.ocr_events)
        attributed.extend(segment.transcript_events)
    attributed.extend(evidence.unattributed.ocr_events)
    attributed.extend(evidence.unattributed.transcript_events)
    expected_texts = [event.text for event in attributed]
    expected_phrases = [term.phrase for segment in evidence.segments for term in segment.phrases]

    assert sorted(texts) == sorted(expected_texts)
    assert sorted(phrases) == sorted(expected_phrases)


def test_guide_uses_segment_vocabulary_and_never_chapter() -> None:
    """Stage 0I boundaries are not semantic chapters, and the artifact must not imply it."""
    payload = build_study_guide(_rich_evidence()).to_dict()

    assert "chapter" not in json.dumps(payload).lower()
    assert not any("chapter" in key.lower() for key in _all_keys(payload))
    assert "section_count" in payload
    assert "segment_id" in payload["sections"][0]


def test_key_terms_come_from_screen_text_and_never_from_speech() -> None:
    """Transcript wording must not be laundered into a key-term list.

    The spoken text repeats a two-word phrase three times, which would qualify on
    frequency if speech ever fed the phrase analysis; the screen text holds one
    one-off title, which correctly does not.
    """
    segments = [_segment(0.0, 30.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "SLIDE TITLE"),
        _speech(4.0, 6.0, "model usage"),
        _speech(8.0, 10.0, "model usage"),
        _speech(12.0, 14.0, "model usage"),
    ]
    guide = _guide(segments, 30.0, timeline)

    assert _key_term_names(guide, 0) == []
    assert guide.sections[0].evidence_state == "ocr_and_transcript"
    assert [event.text for event in guide.sections[0].spoken] == ["model usage"] * 3


# ---------------------------------------------------------------------------
# 2. evidence_state labels two lists; empty and sparse stay honest.
# ---------------------------------------------------------------------------
def test_empty_segment_is_reported_without_filler() -> None:
    """A window with nothing captured gets no content and no generated prose."""
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [_ocr(5.0, "Model Usage\nModel Usage")]
    guide = _guide(segments, 40.0, timeline)

    assert len(guide.sections) == 2, "the segment shape survives even where evidence does not"
    empty = guide.sections[1]
    assert empty.evidence_state == "no_evidence"
    assert (empty.key_terms, empty.on_screen, empty.spoken) == ((), (), ())

    payload: Any = guide.to_dict()["sections"][1]
    assert payload["key_terms"] == []
    assert payload["on_screen"] == []
    assert payload["spoken"] == []
    assert payload["key_term_count"] == 0
    assert payload["on_screen_count"] == 0
    assert payload["spoken_count"] == 0
    assert set(payload) == _SECTION_KEYS, "a section serializes its evidence, never commentary"
    assert set(_all_keys(payload)) == _SECTION_KEYS | {"start_seconds", "end_seconds"}


def test_sparse_ocr_still_yields_a_useful_but_limited_guide() -> None:
    """One occurrence per segment: eligible whole-video, credited once each locally.

    This is Task 1's H1 behaviour reaching the guide intact — the case a per-segment
    frequency gate would erase.
    """
    segments = [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)]
    timeline: list[TimelineEvent] = [_ocr(5.0, "Model Usage"), _ocr(35.0, "model usage")]
    guide = _guide(segments, 60.0, timeline)

    assert [len(section.key_terms) for section in guide.sections] == [1, 1]
    assert [section.evidence_state for section in guide.sections] == ["ocr_only", "ocr_only"]
    assert guide.sections[0].key_terms[0].count_in_segment == 1
    assert guide.sections[0].key_terms[0].count_in_video == 2
    assert guide.coverage.phrase_occurrence_count == 2
    assert guide.coverage.attributed_speech_event_count == 0
    assert guide.transcription_available is True


def test_transcript_only_sections_have_spoken_evidence_and_no_key_terms() -> None:
    """With no screen text captured, the section reports speech and nothing invented."""
    segments = [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)]
    timeline: list[TimelineEvent] = [_speech(5.0, 28.0, "the lecture explains the workflow")]
    guide = _guide(segments, 60.0, timeline)

    assert guide.sections[0].evidence_state == "transcript_only"
    assert guide.sections[0].key_terms == ()
    assert guide.sections[0].on_screen == ()
    assert guide.sections[1].evidence_state == "no_evidence"
    assert guide.coverage.segments_with_speech == 1
    assert guide.coverage.phrase_occurrence_count == 0


def test_all_four_evidence_states_and_coverage_counts() -> None:
    """One guide containing every state, with coverage asserted number by number."""
    segments = [
        _segment(0.0, 20.0, "visual"),
        _segment(20.0, 40.0, "speech_gap"),
        _segment(40.0, 60.0, "visual"),
        _segment(60.0, 80.0, None),
    ]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nModel Usage"),
        _speech(6.0, 18.0, "first window speech"),
        _ocr(25.0, "Model Usage"),
        _speech(45.0, 58.0, "third window speech"),
    ]
    guide = _guide(segments, 80.0, timeline)

    assert [section.evidence_state for section in guide.sections] == [
        "ocr_and_transcript",
        "ocr_only",
        "transcript_only",
        "no_evidence",
    ]
    coverage = guide.coverage.to_dict()
    assert coverage["segment_count"] == 4
    assert coverage["segments_with_ocr"] == 2
    assert coverage["segments_with_speech"] == 2
    assert coverage["segments_without_evidence"] == 1
    assert coverage["attributed_ocr_event_count"] == 2
    assert coverage["attributed_speech_event_count"] == 2
    assert coverage["phrase_occurrence_count"] == 3
    assert coverage["unattributed_ocr_event_count"] == 0
    assert coverage["unattributed_speech_event_count"] == 0
    # The state label and the raw lists can never disagree.
    assert guide.to_dict()["sections"][0]["key_term_count"] == 1
    assert coverage["segments_without_evidence"] == sum(
        1
        for section in guide.sections
        if not section.on_screen and not section.spoken and not section.key_terms
    )


# ---------------------------------------------------------------------------
# 3. Task 1's attribution and provenance reach the guide unchanged.
# ---------------------------------------------------------------------------
def test_speech_spanning_a_boundary_is_attributed_once_and_not_clipped() -> None:
    """Task 1's start-timestamp rule is inherited, not re-decided here."""
    segments = [
        _segment(0.0, 20.0, "visual"),
        _segment(20.0, 45.0, "combined"),
        _segment(45.0, 60.0, None),
    ]
    timeline: list[TimelineEvent] = [
        _speech(10.0, 50.0, "one sentence that runs across two boundaries"),
        _ocr(25.0, "Model Usage\nModel Usage"),
    ]
    guide = _guide(segments, 60.0, timeline)

    assert [len(section.spoken) for section in guide.sections] == [1, 0, 0]
    owning = guide.sections[0].spoken[0]
    assert (owning.start_seconds, owning.end_seconds) == (10.0, 50.0), "never clipped"
    assert owning.text == "one sentence that runs across two boundaries"
    assert guide.sections[1].evidence_state == "ocr_only"
    assert guide.sections[2].evidence_state == "no_evidence"
    assert guide.coverage.attributed_speech_event_count == 1
    assert guide.speech_event_count == 1


def test_unattributed_evidence_is_carried_and_not_segregated_into_a_section() -> None:
    """Evidence outside every window stays where Task 1 put it: in ``unattributed``."""
    segments = [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nModel Usage"),
        _ocr(95.0, "Model Usage"),  # past the end of the video
        _speech(70.0, 75.0, "spoken after the video ended"),
    ]
    guide = _guide(segments, 60.0, timeline)

    assert [event.text for event in guide.unattributed.ocr_events] == ["Model Usage"]
    assert [event.text for event in guide.unattributed.transcript_events] == [
        "spoken after the video ended"
    ]
    sectioned_times = [
        event.timestamp_seconds for section in guide.sections for event in section.on_screen
    ]
    assert sectioned_times == [5.0], "the out-of-window observation reaches no section"
    assert guide.coverage.unattributed_ocr_event_count == 1
    assert guide.coverage.unattributed_speech_event_count == 1
    # Carried totals describe the timeline; attributed totals describe the sections.
    assert guide.speech_event_count == 1
    assert guide.coverage.attributed_speech_event_count == 0


def test_provenance_fields_are_carried_rather_than_recomputed() -> None:
    """Every transcription field on the guide equals Task 1's, field for field."""
    evidence = _rich_evidence()
    guide = build_study_guide(evidence)
    payload: Any = guide.to_dict()

    assert guide.source is evidence.source
    assert guide.study_evidence_schema == STUDY_EVIDENCE_SCHEMA
    assert STUDY_EVIDENCE_SCHEMA == "video-learning.study-evidence/v2"
    for field in (
        "transcription_status",
        "transcription_available",
        "transcription_detail",
        "transcript_evidence_present",
        "speech_event_count",
        "transcription_evidence_conflict",
        "video_duration_seconds",
    ):
        assert getattr(guide, field) == getattr(evidence, field), field
        assert payload[field] == getattr(evidence, field), field

    assert payload["source"] == str(_SOURCE)
    assert payload["transcription_status"] == "ok"
    assert payload["transcription_available"] is True
    assert payload["transcription_detail"] is None
    assert payload["transcription_evidence_conflict"] is False
    assert payload["speech_event_count"] == 3
    assert payload["video_duration_seconds"] == 60.0

    # Carrying a value through only shows up as a bug where it differs from a
    # recomputation, so include the one case the two speech totals cannot agree on.
    spanning: list[TimelineEvent] = [
        _speech(10.0, 20.0, "inside a window"),
        _speech(70.0, 78.0, "past the end of the video"),
    ]
    differing = _evidence(
        [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)], 60.0, spanning
    )
    assert differing.speech_event_count == 2, "Task 1 counts every interval in the timeline"
    carried = build_study_guide(differing)
    assert carried.speech_event_count == differing.speech_event_count == 2
    assert carried.coverage.attributed_speech_event_count == 1, "sections only"


def test_absent_transcription_is_explained_by_carried_status_and_detail() -> None:
    """A guide with no spoken content still says which production case produced it."""
    segments = [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)]
    timeline: list[TimelineEvent] = [_ocr(5.0, "Model Usage\nModel Usage")]
    payload: Any = _guide(
        segments, 60.0, timeline, status="unavailable", detail=_UNAVAILABLE_DETAIL
    ).to_dict()

    assert payload["transcription_status"] == "unavailable"
    assert payload["transcription_available"] is False
    assert payload["transcription_detail"] == _UNAVAILABLE_DETAIL
    assert payload["transcript_evidence_present"] is False
    assert payload["transcription_evidence_conflict"] is False
    assert all(section["spoken"] == [] for section in payload["sections"])
    assert payload["coverage"]["segments_with_ocr"] == 1
    assert payload["coverage"]["segments_without_evidence"] == 1


def test_conflict_flag_reaches_the_guide_verbatim() -> None:
    """The one combination production cannot make is carried, not smoothed over."""
    segments = [_segment(0.0, 30.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "NOTE"),
        _speech(6.0, 9.0, "spoken words"),
    ]
    payload: Any = _guide(
        segments, 30.0, timeline, status="unavailable", detail=_UNAVAILABLE_DETAIL
    ).to_dict()

    assert payload["transcription_available"] is False
    assert payload["transcript_evidence_present"] is True
    assert payload["transcription_evidence_conflict"] is True
    assert payload["sections"][0]["evidence_state"] == "ocr_and_transcript"


def test_zero_segment_evidence_yields_empty_sections_and_honest_coverage() -> None:
    """Unknown duration means no windows at all; the evidence survives as unattributed."""
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nModel Usage"),
        _speech(8.0, 12.0, "spoken without a known duration"),
    ]
    guide = _guide([], None, timeline)

    assert guide.sections == ()
    payload: Any = guide.to_dict()
    assert payload["sections"] == []
    assert payload["section_count"] == 0
    assert payload["video_duration_seconds"] is None
    assert payload["coverage"]["segment_count"] == 0
    assert payload["coverage"]["attributed_ocr_event_count"] == 0
    assert payload["coverage"]["attributed_speech_event_count"] == 0
    assert payload["coverage"]["unattributed_ocr_event_count"] == 1
    assert payload["coverage"]["unattributed_speech_event_count"] == 1
    assert len(payload["unattributed"]["ocr_events"]) == 1
    # Nothing is invented to compensate for the missing windows.
    assert _payload_content(payload)[1] == []


# ---------------------------------------------------------------------------
# 4. Ordering and serialization are inherited and stable.
# ---------------------------------------------------------------------------
def test_section_order_is_chronological_and_matches_the_evidence() -> None:
    guide = build_study_guide(_rich_evidence())

    assert [section.segment_id for section in guide.sections] == [0, 1, 2]
    assert [section.start_seconds for section in guide.sections] == [0.0, 20.0, 45.0]
    assert [section.end_seconds for section in guide.sections] == [20.0, 45.0, 60.0]
    payload: Any = guide.to_dict()
    assert [section["segment_id"] for section in payload["sections"]] == [0, 1, 2]


def test_key_term_order_is_inherited_from_task1() -> None:
    """Local frequency first, whole-video rank as the tie-break — decided upstream."""
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "ECHO FOXTROT\nECHO FOXTROT\nalpha bravo\ngolf hotel"),
        _ocr(25.0, "alpha bravo\nalpha bravo\ngolf hotel"),
    ]
    evidence = _evidence(segments, 40.0, timeline)
    guide = build_study_guide(evidence)

    # "alpha bravo" ranks first globally yet once locally; "ECHO FOXTROT" is the
    # reverse, so a guide that re-sorted by global rank would produce the wrong order.
    assert _key_term_names(guide, 0) == ["ECHO FOXTROT", "alpha bravo", "golf hotel"]
    assert _key_term_names(guide, 0) == [term.phrase for term in evidence.segments[0].phrases]
    assert _key_term_names(guide, 1) == ["alpha bravo", "golf hotel"]


def test_evidence_lists_keep_their_timeline_order() -> None:
    segments = [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)]
    # Already in the order build_timeline would emit; Task 1 preserves, never re-sorts.
    timeline: list[TimelineEvent] = [
        _speech(4.0, 6.0, "first"),
        _ocr(5.0, "Model Usage"),
        _speech(10.0, 12.0, "second"),
        _ocr(25.0, "Model Usage"),
        _ocr(35.0, "Model Usage"),
    ]
    guide = _guide(segments, 60.0, timeline)

    assert [event.timestamp_seconds for event in guide.sections[0].on_screen] == [5.0, 25.0]
    assert [event.start_seconds for event in guide.sections[0].spoken] == [4.0, 10.0]
    assert [event.timestamp_seconds for event in guide.sections[1].on_screen] == [35.0]
    assert guide.sections[1].spoken == ()
    sectioned = [
        event.timestamp_seconds for section in guide.sections for event in section.on_screen
    ]
    assert sectioned == [5.0, 25.0, 35.0], "input order survives both projections untouched"


def test_serialization_is_deterministic_across_runs() -> None:
    first = build_study_guide(_rich_evidence()).to_dict()
    second = build_study_guide(_rich_evidence()).to_dict()

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    # Key insertion order is part of the artifact, so compare unsorted as well.
    assert json.dumps(first) == json.dumps(second)

    payload = json.loads(json.dumps(first))
    assert payload["schema"] == STUDY_GUIDE_SCHEMA == "video-learning.study-guide/v1"
    assert payload["study_evidence_schema"] == STUDY_EVIDENCE_SCHEMA


def test_read_only_projection_reports_no_operation_flags() -> None:
    """Nothing here renames, applies, or modifies anything, so no flags are claimed."""
    payload = build_study_guide(_rich_evidence()).to_dict()
    keys = _all_keys(payload)

    assert "renamed" not in keys
    assert "applied" not in keys
    assert "source_modified" not in keys


# ---------------------------------------------------------------------------
# 5. Service orchestration: report -> evidence -> guide, with no second analysis.
# ---------------------------------------------------------------------------
def test_service_from_report_matches_the_direct_projection() -> None:
    report = _valid_report()
    expected = build_study_guide(StudyEvidenceService().derive(report))

    guide = StudyGuideService(_RecordingReportService(report)).from_report(report)

    assert guide == expected
    assert [section.evidence_state for section in guide.sections] == [
        "ocr_and_transcript",
        "no_evidence",
    ]
    assert guide.source == _SOURCE
    assert guide.video_duration_seconds == 40.0


def test_service_build_calls_the_report_service_once_and_threads_model() -> None:
    report = _valid_report()
    fake = _RecordingReportService(report)
    service = StudyGuideService(fake)

    guide = service.build(_SOURCE, model=_MODEL)

    assert fake.calls == [(_SOURCE, _MODEL)], "one report build per guide, never a second pass"
    assert guide == build_study_guide(StudyEvidenceService().derive(report))

    service.build(_SOURCE)
    assert fake.calls[1] == (_SOURCE, None)
    assert len(fake.calls) == 2


def test_service_does_not_modify_the_report_or_the_evidence() -> None:
    report = _valid_report()
    report_before = json.dumps(report.to_dict(), sort_keys=True)
    evidence = StudyEvidenceService().derive(report)
    evidence_before = json.dumps(evidence.to_dict(), sort_keys=True)

    StudyGuideService(_RecordingReportService(report)).build(_SOURCE)
    build_study_guide(evidence)

    assert json.dumps(report.to_dict(), sort_keys=True) == report_before
    assert json.dumps(evidence.to_dict(), sort_keys=True) == evidence_before


# ---------------------------------------------------------------------------
# 6. Through the real Stage 0I pipeline, not a hand-shaped segmentation.
# ---------------------------------------------------------------------------
def test_guide_built_over_assemble_segments_output() -> None:
    """visual@20.0 merges with speech-gap@20.4 into one ``combined`` boundary."""
    changes = [VisualChange(20.0, 30.0), VisualChange(45.0, 25.0)]
    gaps = [SpeechBoundary(20.4, 2.5)]
    boundaries = merge_boundaries(changes, gaps)
    segments = assemble_segments(boundaries, 60.0)
    segmentation = Segmentation(
        visual_changes=tuple(changes),
        speech_boundaries=tuple(gaps),
        boundaries=tuple(boundaries),
        segments=tuple(segments),
        video_duration_seconds=60.0,
    )
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "ALPHA BRAVO\nALPHA BRAVO"),
        _speech(10.0, 19.0, "first span"),
        _ocr(20.0, "CHARLIE DELTA\nCHARLIE DELTA"),  # exactly on the first boundary
        _speech(22.0, 44.0, "second span"),
        _ocr(45.0, "ECHO FOXTROT\nECHO FOXTROT"),  # exactly on the second boundary
        _speech(50.0, 60.0, "third span"),  # ends exactly at the duration
    ]
    evidence = build_study_evidence(segmentation, timeline, "ok", None, source=_SOURCE)
    guide = build_study_guide(evidence)

    assert [
        (s.segment_id, s.start_seconds, s.end_seconds, s.boundary_reason)
        for s in guide.sections
    ] == [
        (0, 0.0, 20.0, "combined"),
        (1, 20.0, 45.0, "visual"),
        (2, 45.0, 60.0, None),
    ]
    assert [len(s.on_screen) for s in guide.sections] == [1, 1, 1]
    assert [len(s.spoken) for s in guide.sections] == [1, 1, 1]
    assert [s.evidence_state for s in guide.sections] == ["ocr_and_transcript"] * 3
    assert [_key_term_names(guide, index) for index in range(3)] == [
        ["ALPHA BRAVO"],
        ["CHARLIE DELTA"],
        ["ECHO FOXTROT"],
    ]
    assert guide.coverage.segment_count == 3
    assert guide.coverage.unattributed_ocr_event_count == 0
    assert guide.to_dict()["source"] == str(_SOURCE)
