"""Unit tests for per-segment learning evidence (LearnLens Task 1).

Pure, deterministic, and dependency-free: no ffmpeg, tesseract, whisper.cpp, or
LLM is involved. Covers binding existing timeline evidence to Stage 0I segments,
the half-open / closed-final boundary rule, per-segment phrase scoping, missing
transcription, empty OCR, evidence outside any segment (never discarded),
determinism across repeated runs, and the thin ``StudyEvidenceService`` adapter.

Fixture legality (Step 0 / M2)
------------------------------
Hand-shaped ``Segmentation`` fixtures go through :func:`_segmentation`, which
asserts the invariants the real Stage 0I pipeline guarantees, so a fixture can no
longer silently describe a state production cannot produce:

* segments tile ``[0, video_duration_seconds]`` contiguously, each ``end > start``;
* the FINAL segment has ``boundary_reason=None`` — it ends at the duration rather
  than at a detected boundary, so ``assemble_segments`` gives it no reason.

The one exception is the explicitly ``untiled=True`` defensive test, which
exercises the window rule for an input Stage 0I cannot emit. ``_stage0i_segmentation``
builds its segments from ``merge_boundaries`` + ``assemble_segments`` directly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from video_learning.core.keywords import PhraseSuggestion, suggest_phrases
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
from video_learning.core.study import StudyEvidence, build_study_evidence
from video_learning.core.timeline import OcrEvent, SpeechEvent, TimelineEvent, build_timeline
from video_learning.core.transcript import TranscriptResult, TranscriptSegment
from video_learning.services.analyze_service import AnalyzeResult
from video_learning.services.report_service import ReportResult
from video_learning.services.study_service import StudyEvidenceService

_SOURCE = Path("/tmp/sample.mp4")

# A two-word phrase, so it earns eligibility on frequency alone: it is not a
# distinctive compound or path, and therefore cannot bypass the frequency gate.
_RECURRING_PHRASE = "Model Usage"


def _segment(start: float, end: float, reason: BoundaryReason | None) -> Segment:
    return Segment(start_seconds=start, end_seconds=end, boundary_reason=reason)


def _segmentation(
    segments: list[Segment], duration: float | None, *, untiled: bool = False
) -> Segmentation:
    """Build a ``Segmentation`` obeying the real Stage 0I output invariants.

    Raises ``AssertionError`` for a shape ``assemble_segments`` could never return,
    so an illegal fixture fails loudly instead of testing a fiction. ``untiled`` is
    reserved for the single defensive test of the closed-final-end rule.
    """
    if not untiled and duration is not None and segments:
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


def _ocr(t: float, text: str) -> OcrEvent:
    return OcrEvent(timestamp_seconds=t, text=text)


def _speech(start: float, end: float, text: str) -> SpeechEvent:
    return SpeechEvent(start_seconds=start, end_seconds=end, text=text)


def _noise_lines(count: int) -> str:
    """``count`` distinct single-occurrence two-word runs (sparse-OCR noise).

    Each is a valid multi-word candidate that is not distinctive, so the existing
    whole-video frequency gate rejects it: this is exactly the noise that a naive
    per-segment ``min_count=1`` would flood the results with.
    """
    return "\n".join(f"glitchword{index} testword{index}" for index in range(count))


def _whole_video_suggestions(
    timeline: Sequence[TimelineEvent],
) -> list[PhraseSuggestion]:
    """The existing whole-video phrase analysis (Stage 0D) over every OCR event."""
    corpus = "\n".join(event.text for event in timeline if isinstance(event, OcrEvent))
    return suggest_phrases(corpus)


def _whole_video_phrases(timeline: Sequence[TimelineEvent]) -> set[str]:
    """Just the eligible phrase names, in no particular order."""
    return {phrase.phrase for phrase in _whole_video_suggestions(timeline)}


def _segment_phrase_names(result: StudyEvidence, segment_index: int) -> list[str]:
    """Phrase names reported for one segment, read from the serialized artifact."""
    phrases: Any = result.to_dict()["segments"][segment_index]["phrases"]
    return [str(item["phrase"]) for item in phrases]


def _build(
    segments: list[Segment],
    duration: float | None,
    timeline: Sequence[TimelineEvent],
    status: str = "ok",
    detail: str | None = None,
) -> StudyEvidence:
    return build_study_evidence(
        _segmentation(segments, duration), timeline, status, detail, source=_SOURCE
    )


# ---------------------------------------------------------------------------
# Stage 0I fixture legality (M2 infrastructure)
# ---------------------------------------------------------------------------
def test_segmentation_helper_rejects_illegal_stage0i_shape() -> None:
    """The fixture validator itself must reject shapes production cannot emit."""
    # A final segment carrying a boundary reason (Stage 0I never does this).
    with pytest.raises(AssertionError, match="final segment"):
        _segmentation([_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, "speech_gap")], 40.0)
    # Segments that do not tile the duration.
    with pytest.raises(AssertionError, match="ends at the duration"):
        _segmentation([_segment(0.0, 30.0, "visual"), _segment(30.0, 55.0, None)], 60.0)
    # A gap between segments.
    with pytest.raises(AssertionError, match="tile"):
        _segmentation([_segment(0.0, 20.0, "visual"), _segment(25.0, 40.0, None)], 40.0)


# ---------------------------------------------------------------------------
# 1. Multiple segments with OCR and transcript evidence.
# ---------------------------------------------------------------------------
def test_binds_ocr_and_transcript_per_segment() -> None:
    segments = [
        _segment(0.0, 20.0, "combined"),
        _segment(20.0, 40.0, "visual"),
        _segment(40.0, 60.0, None),  # final: ends at the duration, so no reason
    ]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "ALPHA BRAVO\nALPHA BRAVO"),
        _speech(10.0, 15.0, "first span"),
        _ocr(25.0, "CHARLIE DELTA\nCHARLIE DELTA"),
        _speech(22.0, 28.0, "second span"),
        _ocr(45.0, "ECHO FOXTROT\nECHO FOXTROT"),
        _speech(50.0, 55.0, "third span"),
    ]
    result = _build(segments, 60.0, timeline)

    assert [s.start_seconds for s in result.segments] == [0.0, 20.0, 40.0]
    assert [s.boundary_reason for s in result.segments] == ["combined", "visual", None]
    # One OCR event and one speech excerpt per segment.
    assert [len(s.ocr_events) for s in result.segments] == [1, 1, 1]
    assert [len(s.transcript_events) for s in result.segments] == [1, 1, 1]
    assert result.segments[0].transcript_events[0].text == "first span"
    # Phrases are scoped to the segment's own OCR text, not the whole video.
    seg0_phrases = set(_segment_phrase_names(result, 0))
    assert "ALPHA BRAVO" in seg0_phrases
    assert "CHARLIE DELTA" not in seg0_phrases
    assert result.transcription_available is True


# ---------------------------------------------------------------------------
# H1. Eligibility is decided whole-video; occurrences are reported per segment.
# ---------------------------------------------------------------------------
def test_h1_eligible_phrase_recurring_across_segments_appears_in_both() -> None:
    """A phrase seen once per segment qualifies globally and must be reported twice.

    The core Step 0 / H1 regression: the whole-video analysis accepts the phrase on
    two occurrences, yet each segment's own corpus only ever offers one, so gating
    per segment reports it nowhere at all.
    """
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, f"{_RECURRING_PHRASE}\nintro slide"),
        _ocr(25.0, f"{_RECURRING_PHRASE}\nsummary slide"),
    ]
    result = _build(segments, 40.0, timeline)

    # It genuinely qualifies under the existing whole-video extraction.
    assert _RECURRING_PHRASE in _whole_video_phrases(timeline)
    # So it must be reported in both segments it actually occurs in.
    assert _RECURRING_PHRASE in _segment_phrase_names(result, 0)
    assert _RECURRING_PHRASE in _segment_phrase_names(result, 1)


def test_h1_per_segment_counts_are_scoped_and_video_totals_preserved() -> None:
    """Each per-segment phrase record carries both its local and its whole-video count."""
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, f"{_RECURRING_PHRASE}\n{_RECURRING_PHRASE}\nintro slide"),
        _ocr(25.0, _RECURRING_PHRASE),
    ]
    result = _build(segments, 40.0, timeline)

    phrases: Any = result.to_dict()["segments"][0]["phrases"]
    record = next((item for item in phrases if item["phrase"] == _RECURRING_PHRASE), None)
    assert record is not None, f"expected {_RECURRING_PHRASE!r} in segment 0 phrases: {phrases}"
    assert {"count_in_segment", "count_in_video", "global_rank"} <= set(record), record
    assert record["count_in_segment"] == 2, record
    assert record["count_in_video"] == 3, record


def test_h1_sparse_ocr_noise_stays_filtered() -> None:
    """Eligibility must not be bought by simply lowering the frequency gate.

    One segment holds 60 distinct single-occurrence two-word runs plus one
    occurrence of the globally eligible phrase. A per-segment ``min_count=1``
    shortcut would report all 61; whole-video gating must report only one.
    """
    segments = [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, f"{_noise_lines(60)}\n{_RECURRING_PHRASE}"),
        _ocr(35.0, _RECURRING_PHRASE),
    ]
    result = _build(segments, 60.0, timeline)

    reported = set(_segment_phrase_names(result, 0)) | set(_segment_phrase_names(result, 1))
    assert reported == {_RECURRING_PHRASE}, sorted(reported)


def test_h1_eligible_phrase_survives_a_noisy_segment() -> None:
    """Eligibility must be exact, not a by-product of a small per-segment scan cap.

    The eligible phrase ties with the 60 noise candidates on both count and word
    count, and sorts last alphabetically, so a truncated candidate scan can drop it
    purely on presentation order.
    """
    segments = [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, f"{_noise_lines(60)}\n{_RECURRING_PHRASE}"),
        _ocr(35.0, _RECURRING_PHRASE),
    ]
    result = _build(segments, 60.0, timeline)

    assert _RECURRING_PHRASE in _segment_phrase_names(result, 0)


def test_h1_segment_phrases_are_subset_of_whole_video_analysis() -> None:
    """Invariant: per-segment phrases are always a subset of the whole-video set."""
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "ALPHA BRAVO\ncharlie delta echo\nALPHA BRAVO"),
        _ocr(25.0, "foxtrot golf hotel\nALPHA BRAVO\ncharlie delta echo"),
    ]
    result = _build(segments, 40.0, timeline)

    global_phrases = _whole_video_phrases(timeline)
    assert global_phrases, "fixture must yield at least one eligible phrase"
    reported = set(_segment_phrase_names(result, 0)) | set(_segment_phrase_names(result, 1))
    assert reported <= global_phrases, sorted(reported - global_phrases)
    assert "foxtrot golf hotel" not in reported  # seen once overall: still filtered out


def test_h1_segment_phrase_ordering_is_deterministic_and_local_first() -> None:
    """Order is (segment-local count desc, whole-video rank asc) — a total order.

    The fixture makes the two criteria disagree: "alpha bravo" is ranked FIRST
    whole-video (3 occurrences) yet occurs once in segment 0, while "ECHO FOXTROT"
    ranks second globally but twice locally — so segment 0 must lead with ECHO
    FOXTROT. The remaining two tie locally at 1, so whole-video rank breaks it.
    """
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "ECHO FOXTROT\nECHO FOXTROT\nalpha bravo\ngolf hotel"),
        _ocr(25.0, "alpha bravo\nalpha bravo\ngolf hotel"),
    ]
    result = _build(segments, 40.0, timeline)

    assert _segment_phrase_names(result, 0) == ["ECHO FOXTROT", "alpha bravo", "golf hotel"]
    assert _segment_phrase_names(result, 1) == ["alpha bravo", "golf hotel"]
    assert _segment_phrase_names(_build(segments, 40.0, timeline), 0) == [
        "ECHO FOXTROT",
        "alpha bravo",
        "golf hotel",
    ]


def test_h1_segment_occurrences_sum_to_the_whole_video_counts() -> None:
    """Normalization equivalence: per-segment counting must not diverge.

    When every OCR event falls inside a segment, summing ``count_in_segment``
    across all segments must reproduce the whole-video counts. That only holds if
    the segment scan uses the same candidate tokenization, trimming and
    case-folding as ``suggest_phrases`` itself — including merging a lowercase
    occurrence into the dominant surface form, and crediting only maximal runs.
    """
    segments = [
        _segment(0.0, 20.0, "visual"),
        _segment(20.0, 45.0, "combined"),
        _segment(45.0, 60.0, None),
    ]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nmodel usage\nQoder CLI Release Notes"),
        _ocr(25.0, "Projects/Templates/video-learning-studio\nModel Usage"),
        _ocr(50.0, "release notes\nModel Usage\nvideo-learning-studio"),
    ]
    result = _build(segments, 60.0, timeline)
    global_phrases = {
        phrase.phrase: phrase.count for phrase in _whole_video_suggestions(timeline)
    }

    total = sum(p.count_in_segment for seg in result.segments for p in seg.phrases)
    assert total == sum(global_phrases.values()), total
    # Reported phrases cover exactly the whole-video eligible set, no more.
    assert {p.phrase for seg in result.segments for p in seg.phrases} == set(global_phrases)
    # Case-folded occurrences merge into the one dominant surface form.
    seg0 = [(p.phrase, p.count_in_segment, p.count_in_video) for p in result.segments[0].phrases]
    assert seg0 == [("Model Usage", 2, 4)]


def test_h1_unattributed_evidence_is_eligible_but_reports_no_segment_count() -> None:
    """Known boundary: eligibility sees all evidence, occurrence counts do not.

    Unattributed OCR still contributes to whole-video eligibility (evidence is
    never lost), but there is no segment record to attribute its occurrences to, so
    the per-segment counts then fall short of ``count_in_video``.
    """
    segments = [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, _RECURRING_PHRASE),
        _ocr(75.0, _RECURRING_PHRASE),  # beyond the duration: unattributed
    ]
    result = _build(segments, 60.0, timeline)

    assert _RECURRING_PHRASE in _whole_video_phrases(timeline)
    assert [p.count_in_segment for p in result.segments[0].phrases] == [1]
    assert result.segments[0].phrases[0].count_in_video == 2
    assert result.segments[1].phrases == ()
    assert [e.text for e in result.unattributed.ocr_events] == [_RECURRING_PHRASE]


# ---------------------------------------------------------------------------
# 2. OCR-only input (transcription simply skipped for a silent video).
# ---------------------------------------------------------------------------
def test_ocr_only_when_no_audio() -> None:
    segments = [_segment(0.0, 30.0, None)]
    timeline: list[TimelineEvent] = [_ocr(10.0, "SLIDE TITLE")]
    result = _build(segments, 30.0, timeline, status="no-audio")

    assert result.transcription_available is False
    assert result.transcription_status == "no-audio"
    assert result.segments[0].ocr_events[0].text == "SLIDE TITLE"
    assert result.segments[0].transcript_events == ()


# ---------------------------------------------------------------------------
# 3. Missing transcription is represented accurately (backend unavailable).
# ---------------------------------------------------------------------------
def test_missing_transcription_recorded_with_detail() -> None:
    segments = [_segment(0.0, 30.0, None)]
    timeline: list[TimelineEvent] = [_ocr(5.0, "NOTE")]
    result = _build(segments, 30.0, timeline, status="unavailable", detail="whisper model missing")

    assert result.transcription_available is False
    assert result.transcription_status == "unavailable"
    assert result.transcription_detail == "whisper model missing"
    assert all(seg.transcript_events == () for seg in result.segments)


# ---------------------------------------------------------------------------
# 4. Empty OCR yields no events and no phrases but keeps the segment shape.
# ---------------------------------------------------------------------------
def test_empty_ocr_is_graceful() -> None:
    segments = [_segment(0.0, 30.0, None)]
    result = _build(segments, 30.0, [], status="ok")

    assert result.segments[0].ocr_events == ()
    assert result.segments[0].phrases == ()
    assert result.unattributed.is_empty


# ---------------------------------------------------------------------------
# 5. Evidence exactly at segment boundaries lands in exactly one segment.
# ---------------------------------------------------------------------------
def test_boundary_evidence_attributed_once() -> None:
    segments = [
        _segment(0.0, 20.0, "visual"),
        _segment(20.0, 40.0, "combined"),
        _segment(40.0, 60.0, None),  # final, end == duration -> closed at its end
    ]
    timeline: list[TimelineEvent] = [
        _ocr(20.0, "AT-START"),
        _ocr(40.0, "AT-START"),
        _speech(40.0, 45.0, "speech at start of the final segment"),
    ]
    result = _build(segments, 60.0, timeline)

    # A point exactly at a boundary belongs to the segment that STARTS there.
    assert result.segments[0].ocr_events == ()
    assert result.segments[1].ocr_events[0].text == "AT-START"  # t == 20.0
    assert result.segments[2].ocr_events[0].text == "AT-START"  # t == 40.0
    # The final segment closes at the known duration, so its end hosts the point.
    assert result.segments[2].transcript_events[0].text == "speech at start of the final segment"
    assert result.unattributed.is_empty


# ---------------------------------------------------------------------------
# 5b. The final endpoint is closed ONLY when it equals the known duration.
# ---------------------------------------------------------------------------
def test_final_segment_not_closed_when_end_differs_from_duration() -> None:
    """Defensive rule for an input Stage 0I cannot produce (hence ``untiled=True``).

    ``assemble_segments`` always ends the final segment at the duration, so a last
    end of 55.0 with duration 60.0 never happens in production. The half-open rule
    is what makes the closed-final-end case safe, so it stays pinned here.
    """
    segments = [_segment(0.0, 30.0, "visual"), _segment(30.0, 55.0, None)]
    timeline: list[TimelineEvent] = [_ocr(55.0, "AT-LAST-END"), _ocr(50.0, "INSIDE")]
    result = build_study_evidence(
        _segmentation(segments, 60.0, untiled=True), timeline, "ok", None, source=_SOURCE
    )

    assert result.segments[1].ocr_events[0].text == "INSIDE"  # 50 in [30, 55)
    assert all(e.text != "AT-LAST-END" for e in result.segments[1].ocr_events)
    # 55.0 is not < 55.0 and the end is not the duration, so it is unattributed.
    assert [e.text for e in result.unattributed.ocr_events] == ["AT-LAST-END"]


# ---------------------------------------------------------------------------
# 6. Evidence outside every segment is preserved, never discarded.
# ---------------------------------------------------------------------------
def test_out_of_range_evidence_is_unattributed() -> None:
    segments = [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(-1.0, "BEFORE"),
        _ocr(70.0, "AFTER"),
        _speech(120.0, 125.0, "late speech"),
    ]
    result = _build(segments, 60.0, timeline)

    assert all(seg.ocr_events == () for seg in result.segments)
    assert [e.text for e in result.unattributed.ocr_events] == ["BEFORE", "AFTER"]
    assert [e.text for e in result.unattributed.transcript_events] == ["late speech"]


# ---------------------------------------------------------------------------
# 6b. Unknown duration means no segments, but all evidence is still retained.
# ---------------------------------------------------------------------------
def test_unknown_duration_keeps_all_evidence_unattributed() -> None:
    timeline: list[TimelineEvent] = [_ocr(10.0, "OCR"), _speech(5.0, 8.0, "speech")]
    result = _build([], None, timeline)

    assert result.video_duration_seconds is None
    assert result.segments == ()
    assert [e.text for e in result.unattributed.ocr_events] == ["OCR"]
    assert [e.text for e in result.unattributed.transcript_events] == ["speech"]


# ---------------------------------------------------------------------------
# H2. Availability (from status) vs actual evidence, and their disagreement.
# ---------------------------------------------------------------------------
def test_h2_serialized_transcription_fields_exist_with_clear_semantics() -> None:
    """The artifact must distinguish "status says ok" from "speech evidence exists"."""
    segments = [_segment(0.0, 30.0, None)]
    timeline: list[TimelineEvent] = [_ocr(5.0, "NOTE"), _speech(6.0, 9.0, "spoken words")]
    payload: dict[str, Any] = _build(segments, 30.0, timeline).to_dict()

    expected = {
        "transcription_available",
        "speech_event_count",
        "transcript_evidence_present",
        "transcription_evidence_conflict",
    }
    assert expected <= set(payload), sorted(payload)
    assert payload["speech_event_count"] == 1
    assert payload["transcript_evidence_present"] is True
    assert payload["transcription_available"] is True
    assert payload["transcription_evidence_conflict"] is False


def test_h2_empty_transcript_is_available_but_has_no_evidence() -> None:
    """Legal production case: transcription ran ("ok") but produced no speech.

    ``ReportService`` returns ``(TranscriptResult, "ok", None)`` however many
    segments whisper.cpp produced, so an empty transcript must NOT be reported as a
    status/evidence conflict — availability and evidence simply differ here.
    """
    segments = [_segment(0.0, 30.0, None)]
    timeline: list[TimelineEvent] = [_ocr(5.0, "NOTE")]
    result = _build(segments, 30.0, timeline, status="ok")
    payload: dict[str, Any] = result.to_dict()

    assert payload["speech_event_count"] == 0
    assert payload["transcript_evidence_present"] is False
    assert payload["transcription_available"] is True
    assert payload["transcription_evidence_conflict"] is False
    assert result.unattributed.transcript_events == ()


def test_h2_speech_without_available_transcription_is_a_conflict() -> None:
    """Defensive detector: speech evidence cannot exist unless the status is "ok".

    ``build_timeline`` sources speech only from ``transcription.segments``, and
    ``ReportService`` only builds a non-None transcript when the status is "ok", so
    this combination is unreachable in production — which is precisely why the
    disagreement is worth serializing instead of silently over-reporting
    availability.
    """
    segments = [_segment(0.0, 30.0, None)]
    timeline: list[TimelineEvent] = [_ocr(5.0, "NOTE"), _speech(6.0, 9.0, "spoken words")]
    payload: dict[str, Any] = _build(
        segments, 30.0, timeline, status="unavailable", detail="whisper model missing"
    ).to_dict()

    assert payload["transcription_status"] == "unavailable"
    assert payload["transcription_available"] is False
    assert payload["speech_event_count"] == 1
    assert payload["transcript_evidence_present"] is True
    assert payload["transcription_evidence_conflict"] is True


def test_h2_speech_event_count_totals_every_interval_including_unattributed() -> None:
    """The count describes the timeline, so evidence outside segments is still counted."""
    segments = [_segment(0.0, 30.0, None)]
    timeline: list[TimelineEvent] = [_speech(5.0, 8.0, "inside"), _speech(70.0, 75.0, "outside")]
    payload: dict[str, Any] = _build(segments, 30.0, timeline).to_dict()

    assert payload["speech_event_count"] == 2
    assert payload["transcript_evidence_present"] is True


# ---------------------------------------------------------------------------
# M3. Provenance and stable segment identity.
# ---------------------------------------------------------------------------
def test_m3_schema_version_and_source_are_serialized() -> None:
    segments = [_segment(0.0, 30.0, None)]
    timeline: list[TimelineEvent] = [_ocr(5.0, "NOTE")]
    payload: dict[str, Any] = _build(segments, 30.0, timeline).to_dict()

    assert payload["schema"] == "video-learning.study-evidence/v2"
    # Provenance: every other artifact in this project already carries its source.
    assert payload.get("source") == str(_SOURCE), payload.get("source")


def test_m3_segment_ids_are_chronological_and_stable() -> None:
    segments = [
        _segment(0.0, 20.0, "visual"),
        _segment(20.0, 45.0, "combined"),
        _segment(45.0, 60.0, None),
    ]
    timeline: list[TimelineEvent] = [_ocr(5.0, "NOTE")]
    records: Any = _build(segments, 60.0, timeline).to_dict()["segments"]

    assert [item.get("segment_id") for item in records] == [0, 1, 2], records
    assert [item["start_seconds"] for item in records] == [0.0, 20.0, 45.0]
    # Identity follows chronology, so a rebuild gives the same ids.
    rebuilt: Any = _build(segments, 60.0, timeline).to_dict()["segments"]
    assert [item.get("segment_id") for item in rebuilt] == [0, 1, 2]


# ---------------------------------------------------------------------------
# M2. Integration through the real Stage 0I pipeline.
# ---------------------------------------------------------------------------
def _stage0i_segmentation(duration: float) -> tuple[Segmentation, list[TimelineEvent]]:
    """A ``Segmentation`` produced by the actual Stage 0I logic, not hand-shaped.

    visual@20.0 merges with speech-gap@20.4 (inside the 1.0s tolerance) into one
    ``combined`` boundary at 20.0, and visual@45.0 stays ``visual``, so
    ``assemble_segments`` tiles 60.0s as [0,20) [20,45) [45,60].
    """
    changes = [VisualChange(20.0, 30.0), VisualChange(45.0, 25.0)]
    gaps = [SpeechBoundary(20.4, 2.5)]
    boundaries = merge_boundaries(changes, gaps)
    segments = assemble_segments(boundaries, duration)
    segmentation = Segmentation(
        visual_changes=tuple(changes),
        speech_boundaries=tuple(gaps),
        boundaries=tuple(boundaries),
        segments=tuple(segments),
        video_duration_seconds=duration,
    )
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "ALPHA BRAVO\nALPHA BRAVO"),
        _speech(10.0, 19.0, "first span"),
        _ocr(20.0, "CHARLIE DELTA\nCHARLIE DELTA"),  # exactly on the first boundary
        _speech(22.0, 44.0, "second span"),
        _ocr(45.0, "ECHO FOXTROT\nECHO FOXTROT"),  # exactly on the second boundary
        _speech(50.0, 60.0, "third span"),  # ends exactly at the duration
    ]
    return segmentation, timeline


def test_stage0i_pipeline_output_binds_end_to_end() -> None:
    segmentation, timeline = _stage0i_segmentation(60.0)

    # The very Stage 0I invariants the hand-shaped fixtures above rely on.
    assert [(s.start_seconds, s.end_seconds, s.boundary_reason) for s in segmentation.segments] == [
        (0.0, 20.0, "combined"),
        (20.0, 45.0, "visual"),
        (45.0, 60.0, None),
    ]

    result = build_study_evidence(segmentation, timeline, "ok", None, source=_SOURCE)

    assert [s.segment_id for s in result.segments] == [0, 1, 2]
    assert [s.boundary_reason for s in result.segments] == ["combined", "visual", None]
    # Boundary points belong to the segment that STARTS there; the final window is
    # closed at the duration, so the last speech interval is owned by segment 2.
    assert [len(s.ocr_events) for s in result.segments] == [1, 1, 1]
    assert [len(s.transcript_events) for s in result.segments] == [1, 1, 1]
    assert result.segments[2].transcript_events[0].text == "third span"
    assert result.unattributed.is_empty
    # Phrases are scoped per segment, each qualifying on whole-video frequency.
    assert set(_segment_phrase_names(result, 0)) == {"ALPHA BRAVO"}
    assert set(_segment_phrase_names(result, 1)) == {"CHARLIE DELTA"}
    assert set(_segment_phrase_names(result, 2)) == {"ECHO FOXTROT"}
    assert result.to_dict().get("source") == str(_SOURCE)


def test_stage0i_pipeline_keeps_unattributed_evidence_outside_segments() -> None:
    segmentation, timeline = _stage0i_segmentation(60.0)
    extra: list[TimelineEvent] = [*timeline, _ocr(95.0, "BEYOND THE END")]

    result = build_study_evidence(segmentation, extra, "ok", None, source=_SOURCE)

    assert [e.text for e in result.unattributed.ocr_events] == ["BEYOND THE END"]
    assert len(result.segments) == 3


# ---------------------------------------------------------------------------
# 7. Output is deterministic across repeated runs.
# ---------------------------------------------------------------------------
def test_serialization_is_deterministic() -> None:
    segments = [_segment(0.0, 20.0, "combined"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "ALPHA BETA\nALPHA BETA"),
        _speech(3.0, 6.0, "hello"),
        _ocr(25.0, "GAMMA\nGAMMA GAMMA"),
        _speech(21.0, 24.0, "there"),
    ]
    first_payload = _build(segments, 40.0, timeline).to_dict()
    first = json.dumps(first_payload, sort_keys=True)
    second = json.dumps(_build(segments, 40.0, timeline).to_dict(), sort_keys=True)

    assert first == second
    # Key insertion order is part of the artifact, so compare unsorted as well.
    assert json.dumps(first_payload) == json.dumps(_build(segments, 40.0, timeline).to_dict())

    payload = json.loads(first)
    assert payload["schema"] == "video-learning.study-evidence/v2"
    assert payload["renamed"] is False
    assert payload["applied"] is False
    assert payload["source_modified"] is False


# ---------------------------------------------------------------------------
# StudyEvidenceService derives from an existing ReportResult unchanged.
# ---------------------------------------------------------------------------
def _valid_report() -> ReportResult:
    """A ``ReportResult`` obeying every production invariant of the report layer.

    Mirrors what :class:`ReportService` actually emits for a video WITH audio and a
    ready backend: ``transcription_status == "ok"`` implies a non-None
    ``TranscriptResult``; ``transcription_detail`` is ``None`` on that path; the
    media really does carry an audio stream (otherwise transcription is skipped);
    and the timeline's speech events are exactly the transcript's segments, which is
    the only source ``build_timeline`` has for them.
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
        segments=(TranscriptSegment(12.0, 18.0, "spoken"),),
    )
    # The timeline's speech events come ONLY from the transcript's segments, exactly
    # as ReportService.build does, so the fixture cannot contradict itself.
    speech_events = [
        SpeechEvent(segment.start_seconds, segment.end_seconds, segment.text)
        for segment in transcript.segments
    ]
    timeline = build_timeline(
        [OcrEvent(timestamp_seconds=10.0, text="SLIDE")],
        speech_events,
    )
    segmentation = _segmentation([_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)], 40.0)
    return ReportResult(
        source=_SOURCE,
        media=media,
        analysis=AnalyzeResult(source=_SOURCE),
        transcription=transcript,
        transcription_status="ok",
        transcription_detail=None,
        segmentation=segmentation,
        timeline=timeline,
    )


def test_service_derives_from_report_result() -> None:
    evidence = StudyEvidenceService().derive(_valid_report())

    assert evidence.transcription_available is True
    assert [s.start_seconds for s in evidence.segments] == [0.0, 20.0]
    assert evidence.segments[0].ocr_events[0].text == "SLIDE"
    assert evidence.segments[0].transcript_events[0].text == "spoken"
    # Provenance flows from the report, and the evidence agrees with the status.
    assert evidence.source == _SOURCE
    payload = evidence.to_dict()
    assert payload.get("source") == str(_SOURCE)
    assert payload["speech_event_count"] == 1
    assert payload["transcript_evidence_present"] is True
    assert payload["transcription_evidence_conflict"] is False
    assert [s.segment_id for s in evidence.segments] == [0, 1]


def test_service_derives_valid_no_audio_report_without_conflict() -> None:
    """The other legal production combination: no audio, so no transcript at all."""
    report = _valid_report()
    silent_media = MediaInfo(path=_SOURCE, filename=_SOURCE.name, duration_seconds=40.0)
    assert silent_media.has_audio is False, "no-audio requires an audio-less media fixture"
    silent_timeline: list[TimelineEvent] = [OcrEvent(timestamp_seconds=10.0, text="SLIDE")]
    silent_report = ReportResult(
        source=_SOURCE,
        media=silent_media,
        analysis=report.analysis,
        transcription=None,
        transcription_status="no-audio",
        transcription_detail=None,
        segmentation=report.segmentation,
        timeline=silent_timeline,
    )

    evidence = StudyEvidenceService().derive(silent_report)

    assert evidence.transcription_available is False
    assert evidence.transcription_detail is None
    payload = evidence.to_dict()
    assert payload["speech_event_count"] == 0
    assert payload["transcript_evidence_present"] is False
    assert payload["transcription_evidence_conflict"] is False
