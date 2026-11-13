"""Deterministic, evidence-grounded quiz (LearnLens Task 3, Phase A).

Pure derivation, no I/O, no subprocesses, no LLM/API and no new dependencies: it
takes exactly one :class:`~video_learning.core.study.StudyEvidence` and returns
exactly one :class:`StudyQuiz`. Phase A asks only the two questions this evidence
can answer without asserting anything about the video it did not observe.

Why exactly these two kinds
---------------------------
The audit of the evidence graph showed that only two relations are *certifiable*
from what Task 1 holds, and each of them needs no distractors at all:

* a ``SpeechEvent`` interval carries its own verbatim text
  (:class:`~video_learning.core.transcript.TranscriptResult` is the only source of
  speech events, and the adapter strips the utterance — nothing here rewrites it);
* a segment's :class:`~video_learning.core.study.SegmentPhrase` tuple *is* Task 1's
  answer to "which key phrases were identified for this section".

Everything else — definitions, paraphrases, main ideas, causes, true/false on
meaning — would require asserting a fact no extracted observation supports, so it is
out of scope here rather than deferred to a later phase of this module.

No multiple-choice questions exist in Phase A, and that is deliberate rather than
incomplete. A wrong answer must be *demonstrably* wrong, and the only falsity
predicate the evidence supports is ``count_in_segment == 0`` — which means "not
credited in this section", not "not on screen in this section". The ranker credits
only maximal candidate runs of ``2..max_words`` words, so a phrase visible inside a
longer sentence, or repeated adjacently on one line, is present yet credited zero
(see :func:`~video_learning.core.keywords.suggest_phrases`). Turning that gap into
an answer option would invent a claim about the video, so no option pools, no
distractors, and no answer-position ordering are introduced here.

Cloze blank rule (deterministic, and total)
-------------------------------------------
For each ``SpeechEvent`` of each segment, in Task 1's order:

1. the utterance is split into whitespace tokens **with their offsets**, using
   ``\\S+`` matches, so nothing is joined, case-folded, trimmed or re-spaced;
2. an utterance with fewer than :data:`_MIN_UTTERANCE_TOKENS` tokens is skipped:
   after one blank too little context would remain to answer from. This covers the
   empty and whitespace-only texts the adapter can produce;
3. candidates are the tokens of at least :data:`_MIN_BLANK_CHARS` characters that
   are neither in the project's existing stopword list nor letter-free; if none
   qualifies the event is skipped rather than answered with a trivial word or with a
   transcription artifact such as a ``12:34:56`` timer overlay. The letter test is
   ``str.isalpha``, which is Unicode-wide, so accented and non-Latin words are
   ordinary candidates;
4. the blank is the candidate with the most characters, ties broken by the earliest
   offset. Token offsets are unique, so this is a total order and no further rule is
   needed;
5. the blank spans exactly that one token — never a fragment of one, never the
   whitespace around it, never across two events.

The artifact therefore reports the untouched utterance plus ``blank_start`` and
``blank_end``, and ``answer`` is ``source_utterance[blank_start:blank_end]`` by
construction. A test asserts that equality, that the surrounding characters are
whitespace or an edge of the string, and that the stem is the same slice with the
marker substituted. Offsets — not the answer string — locate the blank, so an
utterance that repeats a word is unambiguous: only one occurrence is hidden.

Grounding, ordering and what is never read
------------------------------------------
Unattributed evidence grounds nothing: a question must name the section it came
from, and an event outside every window has none. It is counted in
:attr:`StudyQuiz.coverage` instead of being silently discarded. When Task 1 flags
``transcription_evidence_conflict`` (speech evidence with a status that says there is
no transcript) every cloze is suppressed, because which of the two conflicting
signals to believe is exactly the judgement this module must not make.

Speech attributed across a boundary keeps its whole interval and its whole text: the
blank is located inside the utterance, and the utterance is never clipped to the
window that owns it.

Ordering is never decided again. Cloze questions come first (sections ascending,
then the section's transcript events in Task 1's order), then the recall questions
in section order; answers keep Task 1's ``(count_in_segment desc, global_rank asc)``
order. No sort is applied to questions, no wall-clock, no ``random``, no ``hash()``,
no digest — so the artifact is byte-stable across processes.

Coverage states the limits numerically, and the two skip counts plus the suppression
count sum with the question count back to every attributed speech event. There is no
minimum question count and no failure case: insufficient evidence yields a valid,
self-describing empty quiz.

Schema: ``video-learning.quiz/v1``, recording the Task 1 schema it was derived from.
Out of scope: grading, answer normalization, feedback, persistence, CLI wiring, UI,
and the option-pool and ambiguity-guard work proposed for later phases.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from video_learning.core.keywords import _STOPWORDS
from video_learning.core.study import (
    STUDY_EVIDENCE_SCHEMA,
    SegmentEvidence,
    StudyEvidence,
)
from video_learning.core.timeline import SpeechEvent

QUIZ_SCHEMA = "video-learning.quiz/v1"

#: Substituted for the blank inside :attr:`QuizQuestion.stem`. Public because a
#: renderer shows the same marker a rebuilt stem would contain.
BLANK_MARKER = "____"

QuestionKind: TypeAlias = Literal["transcript_cloze", "section_phrases_recall"]

# Authored scaffolding only: both strings describe the task being asked and assert
# nothing about the video. In particular the recall stem asks which phrases were
# *identified*, never which phrases appeared — Task 1 reports ranked suggestions
# over sampled frames, not an inventory of the screen.
_CLOZE_STEM_TEMPLATE = "Fill in the missing word: {cloze}"
_RECALL_STEM = "Which key phrases were identified for this section?"

_MIN_UTTERANCE_TOKENS = 4
_MIN_BLANK_CHARS = 4

#: Offsets come from the matcher, so token spans index the original utterance.
_TOKEN = re.compile(r"\S+")

# Reusing the ranker's own stopword list keeps one project-wide notion of an
# uninformative word; a second list here would drift from the one that already
# decides what a keyword may be.
_STOPWORD_LOOKUP = frozenset(_STOPWORDS)


@dataclass(frozen=True)
class ClozeBlank:
    """One verbatim blank inside one original utterance, with its position.

    ``source_utterance`` is a ``SpeechEvent.text`` copied whole, and ``answer`` is
    exactly ``source_utterance[blank_start:blank_end]``. The interval fields are the
    event's own unclipped timestamps, so a consumer can play the passage the answer
    came from.
    """

    source_utterance: str
    blank_start: int
    blank_end: int
    answer: str
    speech_start_seconds: float
    speech_end_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_utterance": self.source_utterance,
            "blank_start": self.blank_start,
            "blank_end": self.blank_end,
            "answer": self.answer,
            "speech_start_seconds": self.speech_start_seconds,
            "speech_end_seconds": self.speech_end_seconds,
        }


@dataclass(frozen=True)
class QuizQuestion:
    """One question, grounded entirely in the evidence its fields name.

    ``question_id`` follows the same discipline as ``segment_id``: it is a
    deterministic handle *within the artifact that generated it* — ``cloze:s<segment>
    :e<event>`` or ``recall:s<segment>`` — and it is not durable across
    re-segmentation. ``segment_id`` and ``event_index`` point at Task 1's own lists.

    ``answers`` and ``blank`` are mutually exclusive by kind: a cloze carries its one
    answer inside :class:`ClozeBlank` (so the string is never duplicated away from
    the offsets that justify it) and leaves ``answers`` empty; a recall question
    carries the section's phrase strings in ``answers`` and no blank.
    """

    question_id: str
    kind: QuestionKind
    stem: str
    segment_id: int
    answers: tuple[str, ...]
    event_index: int | None = None
    blank: ClozeBlank | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "kind": self.kind,
            "segment_id": self.segment_id,
            "stem": self.stem,
            "answers": list(self.answers),
            "event_index": self.event_index,
            "blank": self.blank.to_dict() if self.blank is not None else None,
        }


@dataclass(frozen=True)
class QuizCoverage:
    """What the quiz could and could not ground, expressed only as counts.

    The identity a test pins down is
    ``attributed_speech_event_count == cloze_question_count +`` the two skip counts
    ``+ cloze_suppressed_by_conflict_count``: every speech event that reaches a
    section is accounted for as a question or as an explicitly labelled rejection.
    No question is dropped quietly, and no count claims more than the evidence
    supports — ``sections_with_key_phrases`` counts sections the ranker credited a
    phrase in, which is not a claim about how many phrases a slide displayed.
    """

    section_count: int
    sections_with_key_phrases: int
    sections_without_key_phrases: int
    attributed_speech_event_count: int
    cloze_question_count: int
    recall_question_count: int
    question_count: int
    cloze_skipped_too_short_count: int
    cloze_skipped_no_eligible_token_count: int
    cloze_suppressed_by_conflict_count: int
    unattributed_ocr_event_count: int
    unattributed_speech_event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_count": self.section_count,
            "sections_with_key_phrases": self.sections_with_key_phrases,
            "sections_without_key_phrases": self.sections_without_key_phrases,
            "attributed_speech_event_count": self.attributed_speech_event_count,
            "cloze_question_count": self.cloze_question_count,
            "recall_question_count": self.recall_question_count,
            "question_count": self.question_count,
            "cloze_skipped_too_short_count": self.cloze_skipped_too_short_count,
            "cloze_skipped_no_eligible_token_count": (
                self.cloze_skipped_no_eligible_token_count
            ),
            "cloze_suppressed_by_conflict_count": self.cloze_suppressed_by_conflict_count,
            "unattributed_ocr_event_count": self.unattributed_ocr_event_count,
            "unattributed_speech_event_count": self.unattributed_speech_event_count,
        }


@dataclass(frozen=True)
class StudyQuiz:
    """A deterministic quiz for one video, or a valid empty one.

    The transcription-provenance fields are carried from :class:`StudyEvidence`
    without recomputation, so a reader of this artifact alone can tell why there are
    no cloze questions: ``no-audio`` (nothing to transcribe), ``unavailable`` with
    its ``transcription_detail`` (a local prerequisite is missing), or ``ok`` with no
    usable utterance (recorded in the coverage counts).
    ``transcription_evidence_conflict`` is carried too, and drives the cloze
    suppression; ``speech_event_count`` is the whole-timeline total Task 1 reports,
    distinct from ``coverage.attributed_speech_event_count`` by design.

    Like the study guide this reports no operation on the source, so it carries no
    ``renamed`` / ``applied`` / ``source_modified`` flags.
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
    questions: tuple[QuizQuestion, ...]
    coverage: QuizCoverage

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": QUIZ_SCHEMA,
            "study_evidence_schema": self.study_evidence_schema,
            "source": str(self.source),
            "video_duration_seconds": self.video_duration_seconds,
            "transcription_status": self.transcription_status,
            "transcription_available": self.transcription_available,
            "transcription_detail": self.transcription_detail,
            "transcript_evidence_present": self.transcript_evidence_present,
            "speech_event_count": self.speech_event_count,
            "transcription_evidence_conflict": self.transcription_evidence_conflict,
            "questions": [question.to_dict() for question in self.questions],
            "question_count": len(self.questions),
            "coverage": self.coverage.to_dict(),
        }


@dataclass(frozen=True)
class _ClozeOutcome:
    """Cloze questions plus the labelled reasons the other events produced none."""

    questions: tuple[QuizQuestion, ...]
    skipped_too_short: int
    skipped_no_eligible_token: int
    suppressed_by_conflict: int


def _select_blank(text: str) -> tuple[re.Match[str] | None, bool]:
    """Choose the span to blank in ``text``.

    Returns ``(match, too_short)``: on success the match is the token to hide; when
    no token qualifies the boolean says whether the utterance itself was too short,
    which is the distinction the coverage counts report. Offsets come straight from
    the matcher, so the utterance is never re-derived from its tokens.
    """
    tokens = list(_TOKEN.finditer(text))
    if len(tokens) < _MIN_UTTERANCE_TOKENS:
        return None, True
    # Three independent gates, all deterministic: a floor on length, the ranker's own
    # stopword list, and at least one letter. `str.isalpha` is Unicode-wide, so
    # accented and non-Latin words qualify; the letter gate only rejects fragments
    # with no letters at all, mirroring how the ranker refuses a recording timer such
    # as `12:34:56` instead of treating it as meaningful.
    candidates = [
        token
        for token in tokens
        if len(token.group()) >= _MIN_BLANK_CHARS
        and token.group().lower() not in _STOPWORD_LOOKUP
        and any(char.isalpha() for char in token.group())
    ]
    if not candidates:
        return None, False
    # Longest token, then earliest occurrence. Offsets are unique, so this decides
    # every case without a fallback.
    return min(candidates, key=lambda token: (-len(token.group()), token.start())), False


def _cloze_question(
    event: SpeechEvent, selected: re.Match[str], *, segment_id: int, event_index: int
) -> QuizQuestion:
    """Blank the ``selected`` span of ``event``, taking offsets straight from the match."""
    start, end = selected.start(), selected.end()
    stem = _CLOZE_STEM_TEMPLATE.format(
        cloze=event.text[:start] + BLANK_MARKER + event.text[end:]
    )
    return QuizQuestion(
        question_id=f"cloze:s{segment_id}:e{event_index}",
        kind="transcript_cloze",
        stem=stem,
        segment_id=segment_id,
        answers=(),
        event_index=event_index,
        blank=ClozeBlank(
            source_utterance=event.text,
            blank_start=start,
            blank_end=end,
            answer=event.text[start:end],
            speech_start_seconds=event.start_seconds,
            speech_end_seconds=event.end_seconds,
        ),
    )


def _cloze_questions(segments: Sequence[SegmentEvidence], *, suppressed: bool) -> _ClozeOutcome:
    """Walk sections and their transcript events in Task 1's order, never re-sorted."""
    if suppressed:
        return _ClozeOutcome(
            questions=(),
            skipped_too_short=0,
            skipped_no_eligible_token=0,
            suppressed_by_conflict=sum(len(segment.transcript_events) for segment in segments),
        )
    questions: list[QuizQuestion] = []
    skipped_too_short = 0
    skipped_no_eligible_token = 0
    for segment in segments:
        for event_index, event in enumerate(segment.transcript_events):
            selected, too_short = _select_blank(event.text)
            if selected is None:
                if too_short:
                    skipped_too_short += 1
                else:
                    skipped_no_eligible_token += 1
                continue
            questions.append(
                _cloze_question(
                    event, selected, segment_id=segment.segment_id, event_index=event_index
                )
            )
    return _ClozeOutcome(
        questions=tuple(questions),
        skipped_too_short=skipped_too_short,
        skipped_no_eligible_token=skipped_no_eligible_token,
        suppressed_by_conflict=0,
    )


def _recall_questions(segments: Sequence[SegmentEvidence]) -> tuple[QuizQuestion, ...]:
    """One recall question per section the ranker credited at least one phrase in.

    The answers are that section's own ``SegmentPhrase.phrase`` values, in Task 1's
    order and with their exact spelling — no second ranking, no deduplication, and
    nothing added for a section whose list happens to be short.
    """
    return tuple(
        QuizQuestion(
            question_id=f"recall:s{segment.segment_id}",
            kind="section_phrases_recall",
            stem=_RECALL_STEM,
            segment_id=segment.segment_id,
            answers=tuple(phrase.phrase for phrase in segment.phrases),
        )
        for segment in segments
        if segment.phrases
    )


def build_study_quiz(evidence: StudyEvidence) -> StudyQuiz:
    """Derive a quiz from Task 1 evidence (pure, deterministic, never failing).

    Nothing here reads a video, a file, or a model: every content string in the
    result is a ``SpeechEvent.text`` or a ``SegmentPhrase.phrase`` that ``evidence``
    already holds, and every relation it asserts (an utterance's own wording, a
    phrase credited inside a section window) is a field of ``evidence`` itself.
    """
    cloze = _cloze_questions(
        evidence.segments, suppressed=evidence.transcription_evidence_conflict
    )
    recall = _recall_questions(evidence.segments)
    questions = (*cloze.questions, *recall)
    coverage = QuizCoverage(
        section_count=len(evidence.segments),
        sections_with_key_phrases=len(recall),
        sections_without_key_phrases=sum(1 for segment in evidence.segments if not segment.phrases),
        attributed_speech_event_count=sum(
            len(segment.transcript_events) for segment in evidence.segments
        ),
        cloze_question_count=len(cloze.questions),
        recall_question_count=len(recall),
        question_count=len(questions),
        cloze_skipped_too_short_count=cloze.skipped_too_short,
        cloze_skipped_no_eligible_token_count=cloze.skipped_no_eligible_token,
        cloze_suppressed_by_conflict_count=cloze.suppressed_by_conflict,
        unattributed_ocr_event_count=len(evidence.unattributed.ocr_events),
        unattributed_speech_event_count=len(evidence.unattributed.transcript_events),
    )
    return StudyQuiz(
        source=evidence.source,
        video_duration_seconds=evidence.video_duration_seconds,
        study_evidence_schema=STUDY_EVIDENCE_SCHEMA,
        transcription_status=evidence.transcription_status,
        transcription_available=evidence.transcription_available,
        transcription_detail=evidence.transcription_detail,
        transcript_evidence_present=evidence.transcript_evidence_present,
        speech_event_count=evidence.speech_event_count,
        transcription_evidence_conflict=evidence.transcription_evidence_conflict,
        questions=questions,
        coverage=coverage,
    )
