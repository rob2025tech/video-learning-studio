"""Per-segment learning evidence (LearnLens Task 1).

Pure derivation, no I/O, no subprocesses, no LLM/API: it reorganizes the evidence
the Stage 0G report already assembled into one structured record per Stage 0I
segment. Each record binds the timestamps and text that already exist; nothing is
invented — no chapter titles, no topic labels, no semantic interpretation.

Input is taken from existing core models only:

* :class:`~video_learning.core.segments.Segmentation` for the final segments and
  the authoritative ``video_duration_seconds``;
* the unified :mod:`~video_learning.core.timeline` (``OcrEvent`` points and
  ``SpeechEvent`` intervals) for the timestamped OCR and speech evidence;
* :func:`~video_learning.core.keywords.suggest_phrases` (the whole-video Stage 0D
  ranker) to decide which phrases are *eligible* at all, and the same
  normalization/tokenization semantics to count where they occur.

Attribution rules (deterministic, and chosen so no evidence is ever dropped):

* Segments tile ``[0, video_duration_seconds]``. An ``OcrEvent`` belongs to a
  segment when its ``timestamp_seconds`` falls in the segment window; a
  ``SpeechEvent`` is attributed to exactly one segment by its ``start_seconds``.
* A window is half-open ``[start, end)``. The FINAL segment is closed at its end
  ``[start, end]`` only when its ``end_seconds`` equals the known
  ``video_duration_seconds``; otherwise every window (including the last) stays
  half-open. This guarantees a shared boundary point lands in exactly one segment.
* Evidence that matches no window (for example a timestamp at or beyond the end,
  a negative timestamp, or any event when the duration is unknown and so there
  are no segments) is preserved under :class:`UnattributedEvidence`, never lost.

Input contract: ``segmentation.segments`` is expected in ascending chronological
order and tiling the duration contiguously, which is what
:func:`~video_learning.core.segments.assemble_segments` guarantees. ``segment_id``
is the position in that order, and window matching returns the first match, so a
non-chronological or overlapping caller-supplied list would both mis-order the
output and attribute to list order rather than time order.

Phrase scoping (why eligibility is whole-video, counts are per segment)
-----------------------------------------------------------------------
The frequency gate that rejects OCR garble needs the corpus the gate was
calibrated against: the whole video. Applying it to one segment's text would
discard a phrase that recurs across the video but appears only once per segment —
which is the common case for sparse slide OCR — so nothing would be reported
anywhere. Instead:

1. eligibility is decided once, by running ``suggest_phrases`` over every OCR
   event in the timeline (the whole video, including unattributed evidence);
2. each segment then reports only eligible phrases that actually occur inside it,
   counting maximal candidate runs with the very same normalization the ranker
   uses, so the counts sum back to the whole-video count.

Consequently the union of per-segment phrases is always a subset of the
whole-video phrase list, and the noise gate is never weakened: a phrase that is
not eligible whole-video is not reported no matter how often a segment repeats it.

A ``SpeechEvent`` is stored whole, so its original ``start_seconds``/
``end_seconds`` survive even when the interval spans a segment boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_learning.core.keywords import (
    _PHRASE_RUN,
    _TRIM_CHARS,
    PhraseSuggestion,
    suggest_phrases,
)
from video_learning.core.segments import BoundaryReason, Segment, Segmentation
from video_learning.core.timeline import OcrEvent, SpeechEvent, TimelineEvent

STUDY_EVIDENCE_SCHEMA = "video-learning.study-evidence/v2"


@dataclass(frozen=True)
class SegmentPhrase:
    """One whole-video-eligible phrase as it occurs inside a single segment.

    ``count_in_segment`` is how many times the phrase occurs in this segment's own
    OCR text; ``count_in_video`` is the whole-video count that made it eligible in
    the first place; ``global_rank`` is its 1-based position in the whole-video
    phrase list, so a reader can see the presentation order without the source.
    """

    phrase: str
    count_in_segment: int
    count_in_video: int
    global_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "count_in_segment": self.count_in_segment,
            "count_in_video": self.count_in_video,
            "global_rank": self.global_rank,
        }


@dataclass(frozen=True)
class SegmentEvidence:
    """One Stage 0I segment plus the on-screen and speech evidence inside it.

    ``segment_id`` is the segment's chronological position *within the one
    generated StudyEvidence artifact* that carries it. It is deterministic — the
    same input always yields the same ids — which is why it, not ``start_seconds``,
    is the handle later stages should cite, and why citations must name the artifact
    (via its ``source``) that produced them.

    It is **not** stable across re-segmentation: adding a boundary earlier in the
    video shifts every subsequent id. An id is therefore meaningful only against
    the artifact that generated it, never as a durable reference to a point in a
    video.
    """

    segment_id: int
    start_seconds: float
    end_seconds: float
    boundary_reason: BoundaryReason | None
    ocr_events: tuple[OcrEvent, ...]
    phrases: tuple[SegmentPhrase, ...]
    transcript_events: tuple[SpeechEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "boundary_reason": self.boundary_reason,
            "ocr_events": [event.to_dict() for event in self.ocr_events],
            "phrases": [phrase.to_dict() for phrase in self.phrases],
            "transcript_events": [event.to_dict() for event in self.transcript_events],
            "ocr_event_count": len(self.ocr_events),
            "phrase_count": len(self.phrases),
            "transcript_event_count": len(self.transcript_events),
        }


@dataclass(frozen=True)
class UnattributedEvidence:
    """Evidence that falls inside no segment window — retained, never discarded."""

    ocr_events: tuple[OcrEvent, ...]
    transcript_events: tuple[SpeechEvent, ...]

    @property
    def is_empty(self) -> bool:
        return not self.ocr_events and not self.transcript_events

    def to_dict(self) -> dict[str, Any]:
        return {
            "ocr_events": [event.to_dict() for event in self.ocr_events],
            "transcript_events": [event.to_dict() for event in self.transcript_events],
            "ocr_event_count": len(self.ocr_events),
            "transcript_event_count": len(self.transcript_events),
        }


@dataclass(frozen=True)
class StudyEvidence:
    """Structured, per-segment learning evidence derived from one report.

    Four fields describe transcription, and they answer different questions:

    * ``transcription_status`` / ``transcription_detail`` — what the report said,
      verbatim;
    * ``transcription_available`` — ``True`` only when ``transcription_status`` is
      ``"ok"``, i.e. *the run succeeded*;
    * ``speech_event_count`` / ``transcript_evidence_present`` — whether the
      timeline actually carries speech intervals, i.e. *the evidence exists*;
    * ``transcription_evidence_conflict`` — the two disagree in the one direction
      production cannot legitimately produce.

    The conflict flag is deliberately asymmetric. ``build_timeline`` sources speech
    only from ``transcription.segments``, and ``ReportService`` only builds a
    non-``None`` transcript when the status is ``"ok"``, so speech evidence with a
    non-``"ok"`` status is unreachable and worth flagging. The reverse case —
    status ``"ok"`` with zero speech events — is normal (a transcript with no
    segments, or a caller who passed no timeline speech) and must not look like an
    error.

    When the status is not ``"ok"``, every segment's ``transcript_events`` is
    naturally empty and the reported status/detail record why. The complete list of
    ``segments`` is empty when there are no segments (for example when the duration
    is unknown), but the evidence itself is still preserved under ``unattributed``.
    """

    source: Path
    video_duration_seconds: float | None
    transcription_status: str
    transcription_available: bool
    transcription_detail: str | None
    speech_event_count: int
    transcript_evidence_present: bool
    transcription_evidence_conflict: bool
    segments: tuple[SegmentEvidence, ...]
    unattributed: UnattributedEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STUDY_EVIDENCE_SCHEMA,
            "source": str(self.source),
            "video_duration_seconds": self.video_duration_seconds,
            "transcription_status": self.transcription_status,
            "transcription_available": self.transcription_available,
            "transcription_detail": self.transcription_detail,
            "speech_event_count": self.speech_event_count,
            "transcript_evidence_present": self.transcript_evidence_present,
            "transcription_evidence_conflict": self.transcription_evidence_conflict,
            "segments": [segment.to_dict() for segment in self.segments],
            "segment_count": len(self.segments),
            "unattributed": self.unattributed.to_dict(),
            # Explicit non-destructive guarantees (project-wide convention).
            "renamed": False,
            "applied": False,
            "source_modified": False,
        }


def _in_window(event_time: float, segment: Segment, *, closed_end: bool) -> bool:
    """Whether ``event_time`` falls in ``segment``'s window.

    Half-open ``[start, end)`` by default; closed ``[start, end]`` only for the
    final segment whose end equals the authoritative duration.
    """
    if closed_end:
        return segment.start_seconds <= event_time <= segment.end_seconds
    return segment.start_seconds <= event_time < segment.end_seconds


def _segment_index_for(
    event_time: float, segments: Sequence[Segment], closed_last_end: bool
) -> int:
    """Index of the unique segment owning ``event_time``, or ``-1`` if none.

    Windows are contiguous and half-open (except the optional closed final end),
    so at most one segment matches; the first chronological match is returned.
    """
    for index, segment in enumerate(segments):
        is_final = closed_last_end and index == len(segments) - 1
        if _in_window(event_time, segment, closed_end=is_final):
            return index
    return -1


def _whole_video_corpus(ocr_events: Sequence[OcrEvent]) -> str:
    """Every on-screen text observation, joined as the analyze stage joins frames."""
    return "\n".join(event.text for event in ocr_events)


def _normalize_run(run: str) -> str:
    """Reduce one candidate run to its comparison key.

    Mirrors :func:`~video_learning.core.keywords.suggest_phrases` exactly: split on
    whitespace, trim sentence/path separators off each word, drop what is left with
    nothing, then case-fold. Reusing the ranker's own ``_PHRASE_RUN`` and
    ``_TRIM_CHARS`` keeps this consistent with the whole-video counts instead of
    defining a second, subtly different notion of "the same phrase".
    """
    return " ".join(token.strip(_TRIM_CHARS) for token in run.split() if token).lower()


def _eligible_lookup(
    eligible: Sequence[PhraseSuggestion],
) -> Mapping[str, tuple[PhraseSuggestion, int]]:
    """Index eligible phrases by normalized key, with their 1-based whole-video rank.

    The key is the same case-folded, separator-trimmed form the ranker itself uses,
    so an eligible "Model Usage" is found whether the slide wrote it upper or lower.
    """
    return {
        phrase.phrase.lower(): (phrase, rank)
        for rank, phrase in enumerate(eligible, start=1)
    }


def _count_eligible_occurrences(
    segment_text: str, eligible_lookup: Mapping[str, tuple[PhraseSuggestion, int]]
) -> list[SegmentPhrase]:
    """Count eligible phrases inside one segment's OCR text.

    Only maximal candidate runs count, exactly as in the whole-video pass, so a run
    longer than the eligible phrase is a different candidate and is not credited to
    it. A key can only ever be in ``eligible_lookup`` if the whole-video pass
    accepted it, which means the frequency and meaningfulness filters are already
    applied — a phrase that is not eligible whole-video is invisible here however
    often the segment repeats it.
    """
    counts: dict[str, int] = {}
    for line in segment_text.splitlines():
        for run in _PHRASE_RUN.findall(line):
            key = _normalize_run(run)
            if key in eligible_lookup:
                counts[key] = counts.get(key, 0) + 1

    records: list[SegmentPhrase] = []
    for key, count in counts.items():
        phrase, rank = eligible_lookup[key]
        records.append(
            SegmentPhrase(
                phrase=phrase.phrase,
                count_in_segment=count,
                count_in_video=phrase.count,
                global_rank=rank,
            )
        )
    # Segment-local frequency first, then the whole-video rank as the tie-break.
    # global_rank is unique, so this is a total order and needs no further rules.
    records.sort(key=lambda item: (-item.count_in_segment, item.global_rank))
    return records


def build_study_evidence(
    segmentation: Segmentation,
    timeline: Sequence[TimelineEvent],
    transcription_status: str,
    transcription_detail: str | None,
    *,
    source: Path,
) -> StudyEvidence:
    """Bind existing timeline evidence to the Stage 0I segments (pure, deterministic).

    ``source`` is required keyword-only provenance: this artifact describes one
    specific video, and every other artifact in the project already says so.

    ``transcription_available`` is derived solely from ``transcription_status ==
    "ok"`` so the availability signal stays consistent with the report's own
    transcription semantics, while ``speech_event_count`` and
    ``transcript_evidence_present`` independently report the evidence that actually
    arrived.
    """
    segments = segmentation.segments
    duration = segmentation.video_duration_seconds
    # The final window closes at its end only when that end is the known duration.
    closed_last_end = (
        bool(segments) and duration is not None and segments[-1].end_seconds == duration
    )

    ocr_buckets: list[list[OcrEvent]] = [[] for _ in segments]
    speech_buckets: list[list[SpeechEvent]] = [[] for _ in segments]
    unassigned_ocr: list[OcrEvent] = []
    unassigned_speech: list[SpeechEvent] = []
    # Every observation in timeline order: the corpus phrase eligibility is judged
    # against, and the population speech_event_count describes.
    all_ocr: list[OcrEvent] = []
    all_speech: list[SpeechEvent] = []

    # Timeline is already chronologically ordered; filtering preserves that order,
    # so no re-sorting (and no new tie-breaking) is introduced here.
    for event in timeline:
        if isinstance(event, OcrEvent):
            all_ocr.append(event)
            index = _segment_index_for(event.timestamp_seconds, segments, closed_last_end)
            (ocr_buckets[index] if index >= 0 else unassigned_ocr).append(event)
        else:
            all_speech.append(event)
            # A SpeechEvent is attributed by its start only; the whole interval is
            # stored so its original start/end survive even across a boundary.
            index = _segment_index_for(event.start_seconds, segments, closed_last_end)
            (speech_buckets[index] if index >= 0 else unassigned_speech).append(event)

    # Eligibility is decided once against the whole video, never per segment.
    eligible_lookup = _eligible_lookup(suggest_phrases(_whole_video_corpus(all_ocr)))

    segment_records: list[SegmentEvidence] = []
    for index, segment in enumerate(segments):
        ocr_events = tuple(ocr_buckets[index])
        segment_records.append(
            SegmentEvidence(
                segment_id=index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                boundary_reason=segment.boundary_reason,
                ocr_events=ocr_events,
                phrases=tuple(
                    _count_eligible_occurrences(_whole_video_corpus(ocr_events), eligible_lookup)
                ),
                transcript_events=tuple(speech_buckets[index]),
            )
        )

    speech_event_count = len(all_speech)
    transcript_evidence_present = speech_event_count > 0

    return StudyEvidence(
        source=source,
        video_duration_seconds=duration,
        transcription_status=transcription_status,
        transcription_available=transcription_status == "ok",
        transcription_detail=transcription_detail,
        speech_event_count=speech_event_count,
        transcript_evidence_present=transcript_evidence_present,
        # Only the unreachable direction: evidence present while the status says no.
        transcription_evidence_conflict=transcript_evidence_present
        and transcription_status != "ok",
        segments=tuple(segment_records),
        unattributed=UnattributedEvidence(
            ocr_events=tuple(unassigned_ocr),
            transcript_events=tuple(unassigned_speech),
        ),
    )
