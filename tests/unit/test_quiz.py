"""Unit tests for the deterministic evidence-grounded quiz (LearnLens Task 3, Phase A).

No ffmpeg, tesseract, or whisper.cpp is needed: every quiz is built by calling the
real Task 1 derivation over hand-authored timelines, except where a report fixture
is assembled through the actual pipeline.

Phase A has exactly two question kinds and neither can be wrong about the video:

* ``transcript_cloze`` blanks a contiguous, whole-token span of ONE original
  ``SpeechEvent.text`` and reports the untouched utterance plus its offsets, so the
  answer is reconstructible byte for byte (``test_cloze_blank_is_reconstructible``);
* ``section_phrases_recall`` asks which key phrases were *identified* for a section
  and its answers are that section's own ``SegmentPhrase.phrase`` values, in Task 1's
  order.

The properties protected here are therefore: verbatim reconstruction, no invention
(no string in the artifact that is not carried provenance, a fixed label, or
evidence content), honest empty/sparse inputs reported as counts, skip rules that
always fire (never a guess), and carry-through of Task 1's provenance.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from video_learning.core.models import AudioStreamInfo, MediaInfo
from video_learning.core.quiz import (
    BLANK_MARKER,
    QUIZ_SCHEMA,
    QuizQuestion,
    StudyQuiz,
    build_study_quiz,
)
from video_learning.core.segments import BoundaryReason, Segment, Segmentation
from video_learning.core.study import (
    STUDY_EVIDENCE_SCHEMA,
    StudyEvidence,
    build_study_evidence,
)
from video_learning.core.timeline import (
    OcrEvent,
    SpeechEvent,
    TimelineEvent,
    build_timeline,
)
from video_learning.core.transcript import TranscriptResult, TranscriptSegment
from video_learning.services.analyze_service import AnalyzeResult
from video_learning.services.quiz_service import StudyQuizService
from video_learning.services.report_service import ReportResult, ReportService
from video_learning.services.study_service import StudyEvidenceService

_SOURCE = Path("/tmp/sample.mp4")
_MODEL = Path("/tmp/ggml-base.en.bin")
_UNAVAILABLE_DETAIL = "whisper-cli not found on PATH — install whisper.cpp"

# The wording is a specification, not an implementation detail: it must describe the
# data available (Task 1's *identified* phrases) and must never claim to enumerate
# every phrase visible on screen.
_RECALL_STEM = "Which key phrases were identified for this section?"
_CLOZE_PREFIX = "Fill in the missing word: "


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


def _quiz(
    segments: list[Segment],
    duration: float | None,
    timeline: Sequence[TimelineEvent],
    status: str = "ok",
    detail: str | None = None,
) -> StudyQuiz:
    return build_study_quiz(_evidence(segments, duration, timeline, status, detail))


def _cloze_quiz(utterance: str) -> StudyQuiz:
    """One section holding exactly one OCR event (so a phrase exists) and one utterance.

    The OCR line is repeated so it clears the whole-video frequency gate; without it
    the section would carry no key phrases and the recall question would crowd the
    assertions about the cloze.
    """
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nmodel usage"),
        _speech(6.0, 18.0, utterance),
    ]
    return _quiz([_segment(0.0, 20.0, "visual"), _segment(20.0, 30.0, None)], 30.0, timeline)


def _mixed_evidence() -> StudyEvidence:
    """Three sections: two usable utterances + one too short, and phrases throughout.

    The timeline is ascending by event start, as ``build_timeline`` guarantees,
    because Task 1 buckets in input order rather than sorting (its documented input
    contract). Two events deliberately fall past the 60 s duration to exercise the
    unattributed path.
    """
    segments = [
        _segment(0.0, 20.0, "visual"),
        _segment(20.0, 45.0, "combined"),
        _segment(45.0, 60.0, None),
    ]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nmodel usage\nQoder CLI Release Notes"),
        _speech(6.0, 19.0, "the panel shows the remaining quota"),
        _speech(22.0, 44.0, "we open the repository root"),
        _ocr(25.0, "Model Usage"),
        _speech(48.0, 58.0, "notes"),  # a single token: too short to blank
        _ocr(50.0, "Qoder CLI Release Notes"),
        _speech(70.0, 75.0, "nothing owns this interval"),  # beyond the duration
        _ocr(95.0, "Model Usage"),  # beyond the duration
    ]
    return _evidence(segments, 60.0, timeline)


def _mixed_quiz() -> StudyQuiz:
    return build_study_quiz(_mixed_evidence())


def _cloze_questions(quiz: StudyQuiz) -> list[QuizQuestion]:
    return [question for question in quiz.questions if question.kind == "transcript_cloze"]


def _recall_questions(quiz: StudyQuiz) -> list[QuizQuestion]:
    return [question for question in quiz.questions if question.kind == "section_phrases_recall"]


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


def _evidence_content(evidence: StudyEvidence) -> set[str]:
    """Every content string Task 1 actually holds: speech text and phrase text.

    OCR ``text`` is deliberately absent: Phase A grounds nothing on raw screen text,
    only on the phrases the ranker credited for a section.
    """
    content: set[str] = set()
    for segment in evidence.segments:
        for phrase in segment.phrases:
            content.add(phrase.phrase)
        for event in segment.transcript_events:
            content.add(event.text)
    for event in evidence.unattributed.transcript_events:
        content.add(event.text)
    return content


def _all_strings(node: Any, *, skip_keys: frozenset[str]) -> set[str]:
    """Every string value in a serialized payload, minus values under ``skip_keys``.

    ``stem`` is skipped because it is a template slice rather than an atomic value,
    and ``answer`` because the cloze answer is a *substring* of evidence text; both
    have their own dedicated reconstruction tests.
    """
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in skip_keys:
                continue
            if isinstance(value, str):
                found.add(value)
            else:
                found |= _all_strings(value, skip_keys=skip_keys)
    elif isinstance(node, list):
        for item in node:
            found |= _all_strings(item, skip_keys=skip_keys)
    elif isinstance(node, str):
        found.add(node)
    return found


# ---------------------------------------------------------------------------
# 1. Cloze: the blank is an exact, verbatim, whole-token window on one utterance.
# ---------------------------------------------------------------------------
def test_cloze_blank_is_reconstructible_from_the_original_utterance() -> None:
    """The core Phase A guarantee: offsets, not interpretation, define the answer."""
    quiz = _mixed_quiz()
    questions = _cloze_questions(quiz)
    assert len(questions) == 2

    for question in questions:
        blank = question.blank
        assert blank is not None
        utterance = blank.source_utterance
        assert 0 <= blank.blank_start < blank.blank_end <= len(utterance)
        answer = utterance[blank.blank_start : blank.blank_end]
        assert answer == blank.answer
        assert answer
        # The stem is the same utterance with exactly that slice replaced, so the
        # learner sees the original text unaltered apart from the blank.
        rebuilt = (
            utterance[: blank.blank_start] + BLANK_MARKER + utterance[blank.blank_end :]
        )
        assert question.stem == _CLOZE_PREFIX + rebuilt
        # Whole-token boundaries: the blank never cuts a token or swallows its
        # punctuation, because the characters around it are whitespace or an edge.
        before = utterance[: blank.blank_start]
        after = utterance[blank.blank_end :]
        assert before == "" or before[-1].isspace()
        assert after == "" or after[0].isspace()
        assert question.answers == ()


def test_cloze_selects_the_longest_eligible_token() -> None:
    quiz = _cloze_quiz("the panel shows the remaining quota")
    questions = _cloze_questions(quiz)
    assert len(questions) == 1
    blank = questions[0].blank
    assert blank is not None
    assert blank.answer == "remaining"
    assert blank.source_utterance == "the panel shows the remaining quota"
    assert blank.speech_start_seconds == 6.0
    assert blank.speech_end_seconds == 18.0


def test_cloze_ties_break_to_the_earliest_occurrence() -> None:
    """Same-length candidates must not depend on iteration order.

    The repeated ``quota`` also proves the blank is positional: the second one is
    still visible in the stem, so the artifact does not replace every match.
    """
    quiz = _cloze_quiz("quota and quota share")
    questions = _cloze_questions(quiz)
    assert len(questions) == 1
    blank = questions[0].blank
    assert blank is not None
    assert blank.answer == "quota"
    assert blank.blank_start == 0
    assert questions[0].stem == _CLOZE_PREFIX + f"{BLANK_MARKER} and quota share"


def test_cloze_offsets_survive_irregular_internal_whitespace() -> None:
    """A double space inside the utterance must be preserved, not collapsed."""
    quiz = _cloze_quiz("model  USAGE metrics daily")
    questions = _cloze_questions(quiz)
    assert len(questions) == 1
    blank = questions[0].blank
    assert blank is not None
    assert blank.source_utterance == "model  USAGE metrics daily"
    assert blank.answer == "metrics"
    assert blank.source_utterance[blank.blank_start : blank.blank_end] == "metrics"
    # The blank starts right after the double-spaced pair, so offsets index the
    # original characters rather than a re-joined token list.
    assert blank.source_utterance[: blank.blank_start] == "model  USAGE "
    assert questions[0].stem == _CLOZE_PREFIX + f"model  USAGE {BLANK_MARKER} daily"


def test_cloze_never_blanks_a_stopword_or_a_short_token() -> None:
    """Length and the ranker's own stopword list decide eligibility."""
    # ``code`` is the only token of >= 4 characters that is not a stopword.
    quiz = _cloze_quiz("code is ok now")
    blank = _cloze_questions(quiz)[0].blank
    assert blank is not None
    assert blank.answer == "code"

    # Every token of this utterance is a stopword, so nothing may be blanked.
    stopword_quiz = _cloze_quiz("the and of with")
    assert _cloze_questions(stopword_quiz) == []
    assert stopword_quiz.coverage.cloze_skipped_no_eligible_token_count == 1
    assert stopword_quiz.coverage.cloze_question_count == 0


def test_cloze_skips_short_or_empty_utterances_rather_than_guessing() -> None:
    # Fewer than four whitespace tokens leaves no context once blanked.
    for utterance in ("", "   ", "one two", "one two three"):
        quiz = _cloze_quiz(utterance)
        assert _cloze_questions(quiz) == [], utterance
        assert quiz.coverage.cloze_skipped_too_short_count == 1, utterance
        assert quiz.coverage.cloze_question_count == 0, utterance

    # Four tokens is enough, and the only eligible token is then the answer.
    quiz = _cloze_quiz("run the tests now")
    blank = _cloze_questions(quiz)[0].blank
    assert blank is not None
    assert blank.answer == "tests"


def test_cloze_excludes_letter_free_numeric_tokens() -> None:
    """A timer overlay is verbatim evidence but not an answerable blank."""
    quiz = _cloze_quiz("26.09.26 12:34:56 42 7")
    assert _cloze_questions(quiz) == []
    assert quiz.coverage.cloze_skipped_no_eligible_token_count == 1
    assert quiz.coverage.cloze_skipped_too_short_count == 0
    assert quiz.coverage.cloze_question_count == 0


def test_cloze_never_prefers_a_longer_numeric_token_over_a_word() -> None:
    """The letter gate runs before the length ranking, not after it.

    ``42424242`` is the longest token in the utterance, so an implementation that
    filtered after selecting would blank it.
    """
    quiz = _cloze_quiz("42424242 records daily notes")
    blank = _cloze_questions(quiz)[0].blank
    assert blank is not None
    assert blank.answer == "records"
    assert blank.source_utterance[blank.blank_start : blank.blank_end] == "records"


def test_cloze_allows_digits_inside_the_selected_token() -> None:
    """Only a letter-free *token* is excluded, not a token containing digits.

    ``v2release`` and ``yesterday`` tie on length, so this also re-checks the
    earliest-offset tie-break under the new gate.
    """
    quiz = _cloze_quiz("we shipped v2release yesterday here")
    blank = _cloze_questions(quiz)[0].blank
    assert blank is not None
    assert blank.answer == "v2release"
    assert blank.blank_start == 11


def test_cloze_letter_gate_is_unicode_wide() -> None:
    """An ASCII-only letter test would silently drop non-Latin vocabulary."""
    quiz = _cloze_quiz("日本語 の テキスト です")
    blank = _cloze_questions(quiz)[0].blank
    assert blank is not None
    assert blank.answer == "テキスト"


def test_cloze_tie_break_remains_the_earliest_equal_length_word() -> None:
    """Four equal-length eligible words: the rule must still be offset order."""
    quiz = _cloze_quiz("alpha beta gamma delta")
    blank = _cloze_questions(quiz)[0].blank
    assert blank is not None
    assert blank.answer == "alpha"
    assert blank.blank_start == 0
    assert quiz.questions[0].stem == _CLOZE_PREFIX + f"{BLANK_MARKER} beta gamma delta"


def test_cloze_emits_at_most_one_question_per_speech_event() -> None:
    segments = [_segment(0.0, 30.0, "combined"), _segment(30.0, 60.0, None)]
    # Chronological, as ``build_timeline`` guarantees: Task 1 buckets in input order.
    timeline: list[TimelineEvent] = [
        _speech(1.0, 9.0, "the panel shows the remaining quota"),
        _ocr(2.0, "Model Usage\nmodel usage"),
        _speech(10.0, 19.0, "we open the repository root"),
        _speech(20.0, 29.0, "usage statistics refresh every morning"),
        _speech(31.0, 44.0, "usage statistics refresh every morning"),
        _ocr(35.0, "Model Usage"),
    ]
    quiz = _quiz(segments, 60.0, timeline)
    questions = _cloze_questions(quiz)
    utterances = [
        question.blank.source_utterance
        for question in questions
        if question.blank is not None
    ]
    # Two events carry identical text but belong to different sections, so each
    # contributes exactly one distinct question.
    assert utterances.count("usage statistics refresh every morning") == 2
    assert len({question.question_id for question in questions}) == len(questions)
    assert [question.question_id for question in questions] == [
        "cloze:s0:e0",
        "cloze:s0:e1",
        "cloze:s0:e2",
        "cloze:s1:e0",
    ]


# ---------------------------------------------------------------------------
# 2. Recall: the answers ARE Task 1's credited phrases, same values, same order.
# ---------------------------------------------------------------------------
def test_recall_answers_are_the_exact_phrase_values_in_task1_order() -> None:
    evidence = _mixed_evidence()
    quiz = build_study_quiz(evidence)

    questions = _recall_questions(quiz)
    assert len(questions) == 3
    assert [question.segment_id for question in questions] == [0, 1, 2]
    for question, segment in zip(questions, evidence.segments, strict=True):
        assert question.answers == tuple(phrase.phrase for phrase in segment.phrases)
        assert question.blank is None
        assert question.event_index is None
    # Section 0 holds both phrases, ordered by segment-local count then global rank.
    assert questions[0].answers == ("Model Usage", "Qoder CLI Release Notes")
    assert questions[1].answers == ("Model Usage",)
    assert questions[2].answers == ("Qoder CLI Release Notes",)


def test_recall_stem_describes_identification_and_never_completeness() -> None:
    """Requirement: never imply the list is everything visible on screen."""
    quiz = _mixed_quiz()
    questions = _recall_questions(quiz)
    assert questions
    for question in questions:
        assert question.stem == _RECALL_STEM
        lowered = question.stem.lower()
        for overreach in ("all", "every", "only", "complete", "entire"):
            assert overreach not in lowered, question.stem


def test_section_without_phrases_emits_no_recall_question_and_says_so() -> None:
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nmodel usage"),
        _speech(6.0, 18.0, "the panel shows the remaining quota"),
        # Section 1 has a speech event but no OCR at all, so it can hold no phrase.
        _speech(22.0, 38.0, "we open the repository root"),
    ]
    quiz = _quiz(segments, 40.0, timeline)
    assert [question.segment_id for question in _recall_questions(quiz)] == [0]
    assert quiz.coverage.sections_with_key_phrases == 1
    assert quiz.coverage.sections_without_key_phrases == 1
    assert quiz.coverage.section_count == 2


# ---------------------------------------------------------------------------
# 3. Nothing is invented: every string is provenance, a fixed label, or evidence.
# ---------------------------------------------------------------------------
def test_no_string_in_the_quiz_is_invented() -> None:
    evidence = _mixed_evidence()
    quiz = build_study_quiz(evidence)
    payload: Any = quiz.to_dict()
    strings = _all_strings(payload, skip_keys=frozenset({"stem", "answer"}))

    allowed = _evidence_content(evidence) | {
        QUIZ_SCHEMA,
        STUDY_EVIDENCE_SCHEMA,
        str(_SOURCE),
        quiz.transcription_status,
        "transcript_cloze",
        "section_phrases_recall",
    }
    allowed |= {question.question_id for question in quiz.questions}
    assert strings <= allowed, strings - allowed


def test_question_ids_are_derived_from_evidence_handles() -> None:
    quiz = _mixed_quiz()
    assert [question.question_id for question in quiz.questions] == [
        "cloze:s0:e0",
        "cloze:s1:e0",
        "recall:s0",
        "recall:s1",
        "recall:s2",
    ]


# ---------------------------------------------------------------------------
# 4. Grounding: only evidence that reaches a section may seed a question.
# ---------------------------------------------------------------------------
def test_unattributed_evidence_never_grounds_a_question() -> None:
    quiz = _mixed_quiz()
    # The 70-75s utterance is a valid blank candidate ("interval") but owns no
    # section, so it must not appear anywhere in the artifact.
    assert all(
        "interval" not in (question.blank.source_utterance if question.blank else "")
        for question in quiz.questions
    )
    assert quiz.coverage.unattributed_speech_event_count == 1
    assert quiz.coverage.unattributed_ocr_event_count == 1
    assert quiz.coverage.attributed_speech_event_count == 3
    assert quiz.speech_event_count == 4  # carried from Task 1: the timeline total


def test_conflicting_transcription_suppresses_cloze_and_records_why() -> None:
    """Evidence present with a non-``ok`` status is Task 1's conflict case."""
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nmodel usage"),
        _speech(6.0, 18.0, "the panel shows the remaining quota"),
    ]
    quiz = _quiz(segments, 40.0, timeline, status="unavailable", detail=_UNAVAILABLE_DETAIL)

    assert quiz.transcription_evidence_conflict is True
    assert _cloze_questions(quiz) == []
    assert quiz.coverage.cloze_suppressed_by_conflict_count == 1
    assert quiz.coverage.attributed_speech_event_count == 1
    # The phrase question is unaffected: it never depended on the transcript.
    assert [question.stem for question in _recall_questions(quiz)] == [_RECALL_STEM]


def test_speech_spanning_a_boundary_is_blanked_whole_and_attributed_once() -> None:
    segments = [_segment(0.0, 15.0, "combined"), _segment(15.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nmodel usage"),
        _speech(10.0, 30.0, "the panel shows the remaining quota"),
    ]
    quiz = _quiz(segments, 40.0, timeline)
    questions = _cloze_questions(quiz)
    assert len(questions) == 1
    blank = questions[0].blank
    assert blank is not None
    assert questions[0].segment_id == 0
    # The interval is NOT clipped to the window it was attributed to.
    assert (blank.speech_start_seconds, blank.speech_end_seconds) == (10.0, 30.0)
    assert blank.source_utterance == "the panel shows the remaining quota"


# ---------------------------------------------------------------------------
# 5. Honest limits: empty and sparse inputs produce a valid, self-describing quiz.
# ---------------------------------------------------------------------------
def test_empty_evidence_yields_a_valid_empty_quiz() -> None:
    quiz = _quiz([], None, [])
    assert quiz.questions == ()
    payload: Any = quiz.to_dict()
    assert payload["schema"] == QUIZ_SCHEMA
    assert payload["questions"] == []
    assert quiz.coverage.question_count == 0
    assert quiz.coverage.section_count == 0
    assert quiz.coverage.attributed_speech_event_count == 0


def test_no_usable_evidence_yields_empty_questions_not_invented_ones() -> None:
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    # OCR text with no phrase that clears the frequency gate, and only short speech.
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "xyzzy"),
        _speech(22.0, 30.0, "ok"),
    ]
    quiz = _quiz(segments, 40.0, timeline)
    assert quiz.questions == ()
    assert quiz.coverage.sections_without_key_phrases == 2
    assert quiz.coverage.cloze_skipped_too_short_count == 1
    assert quiz.coverage.question_count == 0
    # An empty quiz is still a complete artifact, not a partial failure.
    assert quiz.study_evidence_schema == STUDY_EVIDENCE_SCHEMA


def test_absent_transcription_leaves_recall_questions_intact() -> None:
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [_ocr(5.0, "Model Usage\nmodel usage")]
    quiz = _quiz(segments, 40.0, timeline, status="no-audio")

    assert quiz.transcription_available is False
    assert _cloze_questions(quiz) == []
    assert quiz.coverage.attributed_speech_event_count == 0
    assert [question.answers for question in _recall_questions(quiz)] == [("Model Usage",)]


def test_coverage_counts_add_up_to_every_event_considered() -> None:
    quiz = _mixed_quiz()
    coverage = quiz.coverage
    assert coverage.question_count == (
        coverage.cloze_question_count + coverage.recall_question_count
    )
    assert (
        coverage.attributed_speech_event_count
        == coverage.cloze_question_count
        + coverage.cloze_skipped_too_short_count
        + coverage.cloze_skipped_no_eligible_token_count
    )
    assert coverage.section_count == (
        coverage.sections_with_key_phrases + coverage.sections_without_key_phrases
    )
    assert coverage.sections_with_key_phrases == len(_recall_questions(quiz))


# ---------------------------------------------------------------------------
# 6. Provenance is carried, never recomputed.
# ---------------------------------------------------------------------------
def test_provenance_fields_are_carried_rather_than_recomputed() -> None:
    """Field-for-field equality with Task 1, on a fixture where they cannot coincide."""
    evidence = _mixed_evidence()
    quiz = build_study_quiz(evidence)
    payload: Any = quiz.to_dict()

    assert quiz.source is evidence.source
    assert quiz.study_evidence_schema == STUDY_EVIDENCE_SCHEMA
    for field in (
        "transcription_status",
        "transcription_available",
        "transcription_detail",
        "transcript_evidence_present",
        "speech_event_count",
        "transcription_evidence_conflict",
        "video_duration_seconds",
    ):
        assert getattr(quiz, field) == getattr(evidence, field), field
        assert payload[field] == getattr(evidence, field), field

    # The carried timeline total and the sectioned count genuinely differ here, so a
    # recomputation of ``speech_event_count`` would be caught rather than masked.
    assert quiz.speech_event_count == 4
    assert quiz.coverage.attributed_speech_event_count == 3
    assert payload["source"] == str(_SOURCE)
    assert payload["video_duration_seconds"] == 60.0


def test_absent_transcription_is_explained_by_carried_status_and_detail() -> None:
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nmodel usage"),
        _speech(6.0, 18.0, "the panel shows the remaining quota"),
    ]
    quiz = _quiz(segments, 40.0, timeline, status="unavailable", detail=_UNAVAILABLE_DETAIL)

    assert quiz.transcription_status == "unavailable"
    assert quiz.transcription_available is False
    assert quiz.transcription_detail == _UNAVAILABLE_DETAIL
    assert _cloze_questions(quiz) == []
    assert quiz.coverage.cloze_skipped_too_short_count == 0
    assert quiz.coverage.attributed_speech_event_count == 1


def test_read_only_derivation_reports_no_operation_flags() -> None:
    payload: Any = _mixed_quiz().to_dict()
    keys = _all_keys(payload)
    for flag in ("renamed", "applied", "source_modified"):
        assert flag not in keys
        assert flag not in payload


# ---------------------------------------------------------------------------
# 7. Determinism.
# ---------------------------------------------------------------------------
def test_serialization_is_deterministic() -> None:
    evidence = _mixed_evidence()

    def payload() -> dict[str, Any]:
        result: dict[str, Any] = build_study_quiz(evidence).to_dict()
        return result

    first = payload()
    assert json.dumps(first, sort_keys=True) == json.dumps(payload(), sort_keys=True)
    # Key insertion order is part of the artifact, so compare unsorted as well.
    assert json.dumps(first) == json.dumps(payload())
    # The blank offsets — the discriminating part of a cloze — are in there twice.
    assert "blank_start" in json.dumps(first)


def test_question_order_is_kind_then_evidence_order() -> None:
    quiz = _mixed_quiz()
    kinds = [question.kind for question in quiz.questions]
    assert kinds == [
        "transcript_cloze",
        "transcript_cloze",
        "section_phrases_recall",
        "section_phrases_recall",
        "section_phrases_recall",
    ]
    assert [question.segment_id for question in quiz.questions] == [0, 1, 0, 1, 2]


def test_answers_keep_task1_ranking_and_are_never_resorted() -> None:
    """Answers keep Task 1's ranking; a second sort would contradict it.

    ``Qoder CLI Release Notes`` recurs more often in this section than ``Model
    Usage``, so Task 1 puts it first even though it sorts later alphabetically.
    """
    segments = [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)]
    timeline: list[TimelineEvent] = [
        _ocr(
            5.0,
            "Qoder CLI Release Notes\nqoder cli release notes\nQoder CLI Release Notes\n"
            "Model Usage\nmodel usage",
        ),
        _speech(6.0, 18.0, "the panel shows the remaining quota"),
    ]
    quiz = _quiz(segments, 40.0, timeline)
    answers = _recall_questions(quiz)[0].answers
    assert answers == ("Qoder CLI Release Notes", "Model Usage")
    assert list(answers) != sorted(answers)


# ---------------------------------------------------------------------------
# 8. StudyQuizService derives from an existing ReportResult unchanged.
# ---------------------------------------------------------------------------
def _valid_report() -> ReportResult:
    """A ``ReportResult`` obeying every production invariant of the report layer.

    Mirrors what :class:`ReportService` emits for a video WITH audio and a ready
    backend: ``transcription_status == "ok"`` implies a non-None ``TranscriptResult``,
    ``transcription_detail`` is ``None`` on that path, the media really does carry an
    audio stream, and the timeline's speech events come only from the transcript's
    segments — which is the single source ``build_timeline`` has for them.
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
        segments=(TranscriptSegment(12.0, 18.0, "the panel shows the remaining quota"),),
    )
    speech_events = [
        SpeechEvent(segment.start_seconds, segment.end_seconds, segment.text)
        for segment in transcript.segments
    ]
    timeline = build_timeline(
        [OcrEvent(timestamp_seconds=10.0, text="Model Usage\nmodel usage")],
        speech_events,
    )
    segmentation = _segmentation(
        [_segment(0.0, 20.0, "visual"), _segment(20.0, 40.0, None)], 40.0
    )
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


def test_service_derives_from_study_evidence() -> None:
    evidence = StudyEvidenceService().derive(_valid_report())
    quiz = StudyQuizService.from_evidence(evidence)

    assert quiz == build_study_quiz(evidence)
    assert [q.question_id for q in quiz.questions] == ["cloze:s0:e0", "recall:s0"]
    blank = quiz.questions[0].blank
    assert blank is not None
    assert blank.answer == "remaining"


def test_service_derives_from_report_result() -> None:
    report = _valid_report()
    quiz = StudyQuizService(_RecordingReportService(report)).from_report(report)

    assert quiz == build_study_quiz(StudyEvidenceService().derive(report))
    assert quiz.source == _SOURCE
    assert quiz.transcription_status == "ok"
    assert quiz.coverage.cloze_question_count == 1
    assert _recall_questions(quiz)[0].answers == ("Model Usage",)


def test_service_builds_from_a_path_through_the_existing_pipeline() -> None:
    report = _valid_report()
    recording = _RecordingReportService(report)
    quiz = StudyQuizService(recording).build(_SOURCE, model=_MODEL)

    assert recording.calls == [(_SOURCE, _MODEL)], "the report is built exactly once"
    assert quiz == StudyQuizService(recording).from_report(report)


def test_service_reports_a_valid_empty_quiz_for_a_silent_sparse_video() -> None:
    report = _valid_report()
    silent_media = MediaInfo(path=_SOURCE, filename=_SOURCE.name, duration_seconds=40.0)
    assert silent_media.has_audio is False, "no-audio requires an audio-less media fixture"
    silent_report = ReportResult(
        source=_SOURCE,
        media=silent_media,
        analysis=report.analysis,
        transcription=None,
        transcription_status="no-audio",
        transcription_detail=None,
        segmentation=report.segmentation,
        timeline=[OcrEvent(timestamp_seconds=10.0, text="Model Usage\nmodel usage")],
    )

    quiz = StudyQuizService(_RecordingReportService(silent_report)).from_report(silent_report)

    assert _cloze_questions(quiz) == []
    assert quiz.coverage.attributed_speech_event_count == 0
    assert quiz.coverage.cloze_suppressed_by_conflict_count == 0
    assert [q.question_id for q in quiz.questions] == ["recall:s0"]
    assert quiz.transcription_status == "no-audio"
