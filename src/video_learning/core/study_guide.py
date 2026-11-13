"""Deterministic study guide (LearnLens Task 2): evidence -> per-segment review.

This is a *projection*, not a second analysis. It reorganizes the evidence the
Task 1 layer already bound to the Stage 0I segments into one review section per
segment, so a learner can see what is actually on screen and what is actually
spoken inside each timestamp window, with the original timestamps intact.

Pure data: no I/O, no subprocesses, no LLM/API, and no new dependencies. It
imports no service and reads no file — it takes exactly one
:class:`~video_learning.core.study.StudyEvidence` and returns exactly one
:class:`StudyGuide`.

What this deliberately does NOT do:

* It invents nothing. No summaries, no definitions, no explanations, no causal
  claims, no section headings, and no prose of any kind. Every string in the
  output is copied from evidence (or is a fixed schema/status/boundary label),
  and the empty case stays empty instead of filling in.
* It analyses nothing. Phrase eligibility, phrase counts, per-segment
  attribution, and the ordering of both come from Task 1 unchanged; no second
  phrase-ranking or scanning algorithm exists here, and the section lists are the
  very same :class:`SegmentPhrase` / ``OcrEvent`` / ``SpeechEvent`` objects, not
  reconstructions of them (see ``test_sections_are_the_task1_objects_themselves``).
* It names no chapters. ``Segment`` and ``segment_id`` are the only unit here:
  Stage 0I boundaries are measurable visual-change and speech-gap evidence, not
  semantic chapters, so nothing resembling a chapter title is produced.
* It changes no attribution rule. A speech interval that crosses a boundary stays
  attributed to the segment its ``start_seconds`` falls in, with its original
  ``end_seconds`` unclipped and no duplicate emitted — that is Task 1's rule and
  this layer inherits it verbatim.
* It adds no term-level timestamps. ``SegmentPhrase`` carries counts, not
  positions; locating a key term means the section window and the OCR events
  inside it, which is what :attr:`GuideSection.on_screen` provides.

``evidence_state`` is the only categorisation performed here, and it is a
function of nothing but two emptiness tests on the section's own evidence lists:

===========================  ==========================  =========================
``on_screen`` non-empty      ``spoken`` non-empty         ``evidence_state``
===========================  ==========================  =========================
no                           no                          ``no_evidence``
yes                          no                          ``ocr_only``
no                           yes                         ``transcript_only``
yes                          yes                         ``ocr_and_transcript``
===========================  ==========================  =========================

Two different speech totals appear in the artifact and both are correct, because
they answer different questions. The top-level ``speech_event_count`` is *carried*
from Task 1 and describes the whole timeline, including intervals outside every
segment window (that is what Task 1's provenance semantics are). The
``coverage.attributed_speech_event_count`` is derived *here* and counts only the
speech inside sections; ``coverage.attributed_ocr_event_count`` is the same
distinction for screen text. The ``attributed_`` prefix is what keeps the two
readable apart, and when every event is attributed the totals agree; a test says so.

Ordering is never decided again. Sections follow ``StudyEvidence.segments`` order,
key terms follow Task 1's ``(count_in_segment desc, global_rank asc)`` order, and
``on_screen`` / ``spoken`` keep the timeline order Task 1 bucketed them into. No
sort, no wall-clock, no ``random``, no ``hash()`` — so serialization is stable
across processes.

Schema: ``video-learning.study-guide/v1``, which also records the Task 1 schema
(``study_evidence_schema``) it was projected from. Out of scope for this task:
quiz generation, CLI wiring, persistence, UI integration, and any change to the
report, timeline, segmentation, or study-evidence schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from video_learning.core.segments import BoundaryReason
from video_learning.core.study import (
    STUDY_EVIDENCE_SCHEMA,
    SegmentEvidence,
    SegmentPhrase,
    StudyEvidence,
    UnattributedEvidence,
)
from video_learning.core.timeline import OcrEvent, SpeechEvent

STUDY_GUIDE_SCHEMA = "video-learning.study-guide/v1"

GuideEvidenceState: TypeAlias = Literal[
    "no_evidence", "ocr_only", "transcript_only", "ocr_and_transcript"
]


@dataclass(frozen=True)
class GuideSection:
    """One Stage 0I segment as study material: its window plus the evidence inside it.

    ``segment_id``, ``start_seconds``, ``end_seconds`` and ``boundary_reason`` are
    copied from the Task 1 segment record, and the three evidence fields are that
    record's own tuples. ``key_terms`` are phrases eligible across the whole video
    that also occur here, already in Task 1's presentation order; they come only
    from on-screen text, never from the transcript.

    ``evidence_state`` is a two-emptiness-test label, not an interpretation of the
    content. A ``no_evidence`` section is reported as exactly that: real windows can
    contain captured nothing, and an empty list is the honest answer.
    """

    segment_id: int
    start_seconds: float
    end_seconds: float
    boundary_reason: BoundaryReason | None
    evidence_state: GuideEvidenceState
    key_terms: tuple[SegmentPhrase, ...]
    on_screen: tuple[OcrEvent, ...]
    spoken: tuple[SpeechEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "timestamp_range": {
                "start_seconds": self.start_seconds,
                "end_seconds": self.end_seconds,
            },
            "boundary_reason": self.boundary_reason,
            "evidence_state": self.evidence_state,
            # Each nested item keeps its own native timestamps, so a future UI can
            # seek to an OCR point or play a speech interval without reconstruction.
            "key_terms": [term.to_dict() for term in self.key_terms],
            "on_screen": [event.to_dict() for event in self.on_screen],
            "spoken": [event.to_dict() for event in self.spoken],
            "key_term_count": len(self.key_terms),
            "on_screen_count": len(self.on_screen),
            "spoken_count": len(self.spoken),
        }


@dataclass(frozen=True)
class GuideCoverage:
    """How much of the guide evidence can support, expressed only as counts.

    Everything here is a sum over :attr:`StudyGuide.sections` (or over the carried
    unattributed tuples). It states a limitation numerically rather than with
    generated prose: ``segments_without_evidence`` is how many sections a learner
    genuinely cannot study from, and the ``unattributed_*`` counts say how much
    evidence lies outside every window.

    The ``attributed_*`` prefix means "reaches a section", never "exists in the
    video". The same evidence can be counted by Task 1's carried top-level
    ``speech_event_count`` and not appear here, because it fell into no segment
    window.
    """

    segment_count: int
    segments_with_ocr: int
    segments_with_speech: int
    segments_without_evidence: int
    attributed_ocr_event_count: int
    attributed_speech_event_count: int
    phrase_occurrence_count: int
    unattributed_ocr_event_count: int
    unattributed_speech_event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_count": self.segment_count,
            "segments_with_ocr": self.segments_with_ocr,
            "segments_with_speech": self.segments_with_speech,
            "segments_without_evidence": self.segments_without_evidence,
            "attributed_ocr_event_count": self.attributed_ocr_event_count,
            "attributed_speech_event_count": self.attributed_speech_event_count,
            "phrase_occurrence_count": self.phrase_occurrence_count,
            "unattributed_ocr_event_count": self.unattributed_ocr_event_count,
            "unattributed_speech_event_count": self.unattributed_speech_event_count,
        }


@dataclass(frozen=True)
class StudyGuide:
    """A deterministic, evidence-linked study guide for one video.

    The transcription-provenance fields are carried from :class:`StudyEvidence`
    without recomputation, so a reader of this artifact alone can tell *why* a
    section has no spoken content — ``no-audio`` (nothing to transcribe),
    ``unavailable`` with its ``transcription_detail`` (a local prerequisite is
    missing), or ``ok`` with no speech intervals (a transcript that carried no
    speech). ``transcription_evidence_conflict`` is carried too, including its
    asymmetric Task 1 meaning.

    ``unattributed`` is preserved rather than dropped or redistributed. When the
    duration is unknown there are no segments at all, ``sections`` is empty, and the
    evidence is still present here.

    This artifact reports no operation on the source, so it carries no
    ``renamed`` / ``applied`` / ``source_modified`` flags: it is a pure in-memory
    projection that cannot change anything.
    """

    source: Path
    video_duration_seconds: float | None
    study_evidence_schema: str
    transcription_status: str
    transcription_available: bool
    transcription_detail: str | None
    transcript_evidence_present: bool
    speech_event_count: int
    transcription_evidence_conflict: bool
    sections: tuple[GuideSection, ...]
    coverage: GuideCoverage
    unattributed: UnattributedEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STUDY_GUIDE_SCHEMA,
            "study_evidence_schema": self.study_evidence_schema,
            "source": str(self.source),
            "video_duration_seconds": self.video_duration_seconds,
            "transcription_status": self.transcription_status,
            "transcription_available": self.transcription_available,
            "transcription_detail": self.transcription_detail,
            "transcript_evidence_present": self.transcript_evidence_present,
            "speech_event_count": self.speech_event_count,
            "transcription_evidence_conflict": self.transcription_evidence_conflict,
            "sections": [section.to_dict() for section in self.sections],
            "section_count": len(self.sections),
            "coverage": self.coverage.to_dict(),
            "unattributed": self.unattributed.to_dict(),
        }


def _evidence_state(*, has_ocr: bool, has_speech: bool) -> GuideEvidenceState:
    """Label a section from two emptiness tests on its own evidence lists."""
    if has_ocr and has_speech:
        return "ocr_and_transcript"
    if has_ocr:
        return "ocr_only"
    if has_speech:
        return "transcript_only"
    return "no_evidence"


def _section_for(segment: SegmentEvidence) -> GuideSection:
    """Project one Task 1 segment record onto a guide section.

    The evidence tuples are passed through by reference, never rebuilt, so a
    divergence between the two artifacts is structurally impossible rather than
    merely unlikely.
    """
    return GuideSection(
        segment_id=segment.segment_id,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        boundary_reason=segment.boundary_reason,
        evidence_state=_evidence_state(
            has_ocr=bool(segment.ocr_events),
            has_speech=bool(segment.transcript_events),
        ),
        key_terms=segment.phrases,
        on_screen=segment.ocr_events,
        spoken=segment.transcript_events,
    )


def _coverage_for(
    sections: tuple[GuideSection, ...], unattributed: UnattributedEvidence
) -> GuideCoverage:
    """Count what the sections and the carried unattributed evidence actually hold."""
    return GuideCoverage(
        segment_count=len(sections),
        segments_with_ocr=sum(1 for section in sections if section.on_screen),
        segments_with_speech=sum(1 for section in sections if section.spoken),
        segments_without_evidence=sum(
            1 for section in sections if section.evidence_state == "no_evidence"
        ),
        attributed_ocr_event_count=sum(len(section.on_screen) for section in sections),
        attributed_speech_event_count=sum(len(section.spoken) for section in sections),
        phrase_occurrence_count=sum(
            term.count_in_segment for section in sections for term in section.key_terms
        ),
        unattributed_ocr_event_count=len(unattributed.ocr_events),
        unattributed_speech_event_count=len(unattributed.transcript_events),
    )


def build_study_guide(evidence: StudyEvidence) -> StudyGuide:
    """Project Task 1 evidence into a per-segment study guide (pure, deterministic).

    Nothing is derived from the video here: every field comes from ``evidence``, and
    the only computation performed is the two-emptiness ``evidence_state`` label and
    the ``coverage`` counts. Same input always yields the same artifact, byte for
    byte.
    """
    sections = tuple(_section_for(segment) for segment in evidence.segments)
    return StudyGuide(
        source=evidence.source,
        video_duration_seconds=evidence.video_duration_seconds,
        study_evidence_schema=STUDY_EVIDENCE_SCHEMA,
        transcription_status=evidence.transcription_status,
        transcription_available=evidence.transcription_available,
        transcription_detail=evidence.transcription_detail,
        transcript_evidence_present=evidence.transcript_evidence_present,
        speech_event_count=evidence.speech_event_count,
        transcription_evidence_conflict=evidence.transcription_evidence_conflict,
        sections=sections,
        coverage=_coverage_for(sections, evidence.unattributed),
        unattributed=evidence.unattributed,
    )
