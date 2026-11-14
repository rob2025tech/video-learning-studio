"""Unit tests for the pure lexical search core (Multimodal Evidence Search v1, Step 1).

Nothing here touches ffmpeg, tesseract, whisper.cpp or the video pipeline: every
index is built by calling the real Stage 0G/0I derivation over hand-authored
timelines, exactly as ``tests/unit/test_quiz.py`` does.

The properties protected:

* **Projection is lossless and invents nothing** — one record per timeline event,
  verbatim text, that event's own timestamps, ``None`` for an OCR end that was
  never measured, and unattributed evidence still searchable.
* **Tokenizer keeps the punctuation that carries meaning** — ``whisper.cpp``,
  ``ggmlmedium.bin``, paths and command-line flags survive intact, because losing
  them would make a software tutorial unsearchable.
* **A fragment cannot carry intent, but a negation can** — a clitic tail (``'s``,
  ``'re``, ``'ll``) and any lone letter behind an apostrophe are indexed on neither
  side, so ``what's`` stops looking like a search and ``Qoder's editor`` is still
  about Qoder; ``n't`` is the one tail kept, indexed whole so ``won`` cannot borrow
  ``won't``'s hits, with a typographic quote folding to a straight one in the term
  only and every span selecting exactly its own surface. A negation the evidence wrote
  badly (``won''t``, ``won' t``, ``won-'t``, ``won'ts``), whether or not a separator
  character follows it, is suppressed rather than repaired. A quoted ``grep 'whisper'``
  keeps its ``whisper``.
* **BM25 is recomputable** — a score here is re-derived from the textbook formula
  with the constants pinned in the module docstring; if those change, this test
  fails on purpose. And it is recomputable *from the artifact*: the records with no
  meaningful term (blank frames, timers, UI chrome) are excluded from ``N``, from
  the average length and from the document frequencies, because IDF grows with
  ``N`` and dead screen text must not be able to inflate a score.
* **States are structural, never calibrated** — ``no_evidence``,
  ``no_confident_match`` and ``ok`` are decided by emptiness and by real term
  overlap. There is no score threshold to cross, so a weak-but-real match must
  still be ``ok``.
* **Determinism** — repeated calls, and a rebuilt index, produce identical JSON, and
  ties resolve by the lexicographic ``evidence_id`` compare (so ``speech:10``
  precedes ``speech:2``), which is tested at a scale where that is visible.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from video_learning.core.search import (
    DEFAULT_RESULT_LIMIT,
    SEARCH_SCHEMA,
    SearchEvidence,
    SearchIndex,
    SearchResults,
    build_search_index,
    project_evidence,
    search,
    tokenize,
)
from video_learning.core.segments import BoundaryReason, Segment, Segmentation
from video_learning.core.study import STUDY_EVIDENCE_SCHEMA, StudyEvidence, build_study_evidence
from video_learning.core.timeline import OcrEvent, SpeechEvent, TimelineEvent

_SOURCE = Path("/tmp/sample.mp4")

# BM25 constants, restated here on purpose: they are part of the documented
# contract of `core/search.py`, and a silent change to them must break a test.
_K1 = 1.5
_B = 0.75


def _ocr(seconds: float, text: str) -> OcrEvent:
    return OcrEvent(timestamp_seconds=seconds, text=text)


def _speech(start: float, end: float, text: str) -> SpeechEvent:
    return SpeechEvent(start_seconds=start, end_seconds=end, text=text)


def _segment(start: float, end: float, reason: BoundaryReason | None) -> Segment:
    return Segment(start_seconds=start, end_seconds=end, boundary_reason=reason)


def _evidence(
    timeline: Sequence[TimelineEvent],
    *,
    segments: list[Segment] | None = None,
    duration: float | None = 60.0,
    status: str = "ok",
    detail: str | None = None,
) -> StudyEvidence:
    """Derive real ``StudyEvidence`` from a hand-authored timeline.

    The default windows tile ``[0, 60)`` so every fixture event lands inside a
    segment. Pass ``segments=[]`` with ``duration=None`` for the shape the pipeline
    produces when the duration is unknown and nothing is attributable.
    """
    chosen = [_segment(0.0, 30.0, "visual"), _segment(30.0, 60.0, None)]
    if segments is not None:
        chosen = segments
    if chosen and duration is not None:
        # Stage 0I invariants: a fixture shaped differently would test a fiction.
        assert chosen[0].start_seconds == 0.0, "Stage 0I always starts at 0.0"
        assert chosen[-1].end_seconds == duration, "the last segment ends at the duration"
        for previous, following in zip(chosen, chosen[1:], strict=False):
            assert previous.end_seconds == following.start_seconds, "segments must tile"
    segmentation = Segmentation(
        visual_changes=(),
        speech_boundaries=(),
        boundaries=(),
        segments=tuple(chosen),
        video_duration_seconds=duration,
    )
    return build_study_evidence(segmentation, timeline, status, detail, source=_SOURCE)


def _index(*events: TimelineEvent) -> SearchIndex:
    return build_search_index(_evidence(list(events)))


def _results(index: SearchIndex, query: str, **kwargs: Any) -> SearchResults:
    return search(index, query, **kwargs)


def _texts(result: SearchResults) -> list[str]:
    return [match.evidence.text for match in result.results]


def _idf(document_frequency: int, total_documents: int) -> float:
    return math.log(1.0 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))


def _bm25(
    term_frequency: int,
    document_length: int,
    average_length: float,
    document_frequency: int,
    total_documents: int,
) -> float:
    """The formula from the module docstring, written out independently here."""
    norm = _K1 * (1.0 - _B + _B * (document_length / average_length))
    saturation = term_frequency * (_K1 + 1.0) / (term_frequency + norm)
    return _idf(document_frequency, total_documents) * saturation


# ---------------------------------------------------------------------------
# Projection: coverage, verbatim text, timestamps, attribution
# ---------------------------------------------------------------------------


def test_projection_covers_every_speech_and_ocr_event_exactly_once() -> None:
    """One record per event, and no event represented twice.

    This is why the projection reads ``StudyEvidence`` rather than a report: a
    report carries the same text both in its ``timeline`` and in its source
    sections, which would double-count every hit.
    """
    evidence = _evidence(
        [
            _ocr(5.0, "Model Usage"),
            _speech(6.0, 12.0, "we open the settings panel"),
            _ocr(35.0, "Release Notes"),
            _speech(40.0, 50.0, "here are the release notes"),
        ]
    )
    records = project_evidence(evidence)
    assert [record.text for record in records] == [
        "Model Usage",
        "we open the settings panel",
        "Release Notes",
        "here are the release notes",
    ]
    # Cross-checked against the artifact's own counts, not a hand-typed number.
    declared = sum(
        len(segment.ocr_events) + len(segment.transcript_events)
        for segment in evidence.segments
    )
    assert len(records) == declared == 4


def test_projection_preserves_verbatim_text() -> None:
    """Whitespace, casing, punctuation and newlines survive byte for byte.

    The surrounding spaces matter: OCR of screen chrome and whisper.cpp's own
    output both carry padding, and trimming it would make the record text stop
    matching the evidence the caller already has.
    """
    raw_ocr = "  Qoder Model Usage\nmodel usage\t(v0.1) "
    raw_speech = "  Today we're going to build a small command line tool. "
    records = project_evidence(_evidence([_ocr(5.0, raw_ocr), _speech(6.0, 12.0, raw_speech)]))
    assert [record.text for record in records] == [raw_ocr, raw_speech]


def test_ocr_records_are_points_with_no_end_seconds() -> None:
    """A sampled frame says *when*, never for how long.

    An end for OCR would be an invention, so the field is ``None`` even though the
    record still carries the observation's own timestamp.
    """
    records = project_evidence(_evidence([_ocr(12.5, "Release Notes")]))
    assert [(r.source_type, r.start_seconds, r.end_seconds) for r in records] == [
        ("ocr", 12.5, None)
    ]


def test_speech_records_keep_their_whole_interval() -> None:
    """Attribution uses the start; the interval itself is retained untouched."""
    records = project_evidence(_evidence([_speech(28.0, 45.0, "across the boundary")]))
    assert [(r.source_type, r.start_seconds, r.end_seconds) for r in records] == [
        ("speech", 28.0, 45.0)
    ]


def test_no_record_field_is_invented_and_none_can_be() -> None:
    """Every record is one source event, and the type has nowhere to put a lie.

    There is no frame reference because extraction leaves none behind, no
    confidence because neither backend reports one, and no duration for OCR.
    """
    timeline = [_ocr(5.0, "Model Usage"), _speech(6.0, 12.0, "we open settings")]
    sources: list[tuple[str, float, float | None, str]] = []
    for event in timeline:
        if isinstance(event, OcrEvent):
            sources.append(("ocr", event.timestamp_seconds, None, event.text))
        else:
            sources.append(("speech", event.start_seconds, event.end_seconds, event.text))
    projected = [
        (r.source_type, r.start_seconds, r.end_seconds, r.text)
        for r in project_evidence(_evidence(timeline))
    ]
    assert sorted(projected) == sorted(sources)
    assert set(SearchEvidence.__dataclass_fields__) == {
        "evidence_id",
        "source_type",
        "text",
        "start_seconds",
        "end_seconds",
        "segment_id",
    }


def test_projection_emits_ocr_before_speech_within_each_segment() -> None:
    """The documented intra-segment order, independent of timeline interleaving."""
    records = project_evidence(
        _evidence(
            [
                _speech(3.0, 4.0, "first segment speech"),
                _ocr(2.0, "first segment ocr"),
                _ocr(31.0, "second segment ocr"),
                _speech(32.0, 33.0, "second segment speech"),
            ]
        )
    )
    assert [(r.segment_id, r.source_type, r.start_seconds) for r in records] == [
        (0, "ocr", 2.0),
        (0, "speech", 3.0),
        (1, "ocr", 31.0),
        (1, "speech", 32.0),
    ]


def test_evidence_ids_are_per_source_ordinals_in_projection_order() -> None:
    """``ocr:N`` and ``speech:N`` count separately, so ids never renumber across sources."""
    records = project_evidence(
        _evidence(
            [
                _ocr(2.0, "alpha screen"),
                _speech(3.0, 4.0, "alpha speech"),
                _ocr(31.0, "beta screen"),
                _speech(32.0, 33.0, "beta speech"),
            ]
        )
    )
    assert [r.evidence_id for r in records] == ["ocr:0", "speech:0", "ocr:1", "speech:1"]


def test_duplicate_timestamps_are_a_locator_not_an_identity() -> None:
    """The module docstring refuses to call ``(source_type, start_seconds)`` a key.

    Two OCR points can legitimately share one second — frame sampling clamps every
    midpoint of a sub-second video onto the same instant — so the pair addresses a
    place, not a record. ``evidence_id`` is what separates them, in the index and in
    the tie-break; nothing here invents a durable identity to do that job instead.
    """
    index = _index(_ocr(5.0, "whisper.cpp"), _ocr(5.0, "Release Notes"))
    locators = [(record.source_type, record.start_seconds) for record in index.records]
    assert locators == [("ocr", 5.0), ("ocr", 5.0)]
    assert len(set(locators)) == 1
    assert [record.evidence_id for record in index.records] == ["ocr:0", "ocr:1"]
    assert [m.evidence.evidence_id for m in _results(index, "whisper").results] == ["ocr:0"]


def test_unattributed_evidence_keeps_no_segment_and_stays_searchable() -> None:
    """The unknown-duration shape drops nothing: it just has nowhere to attribute to."""
    evidence = _evidence(
        [_ocr(5.0, "whisper.cpp"), _speech(6.0, 12.0, "we run whisper on a model file")],
        segments=[],
        duration=None,
    )
    records = project_evidence(evidence)
    assert [r.segment_id for r in records] == [None, None]
    assert len(evidence.unattributed.ocr_events) == 1
    assert len(evidence.unattributed.transcript_events) == 1

    result = _results(build_search_index(evidence), "whisper")
    assert result.status == "ok"
    assert {match.evidence.segment_id for match in result.results} == {None}
    assert result.coverage.unattributed_record_count == 2


def test_records_are_attributed_to_the_segment_that_contains_them() -> None:
    result = _results(
        _index(_ocr(5.0, "transcribe locally"), _speech(32.0, 40.0, "transcribe locally")),
        "transcribe",
    )
    assert [(m.evidence.segment_id, m.evidence.evidence_id) for m in result.results] == [
        (0, "ocr:0"),
        (1, "speech:0"),
    ]


def test_empty_evidence_projects_no_records() -> None:
    evidence = _evidence([], segments=[], duration=None)
    assert project_evidence(evidence) == ()
    assert build_search_index(evidence).records == ()


def test_projection_is_deterministic_and_does_not_mutate_its_input() -> None:
    events: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nmodel usage"),
        _speech(6.0, 18.0, "it reads a video and prints what it sees."),
    ]
    evidence = _evidence(events)
    before: dict[str, Any] = evidence.to_dict()
    first = [record.to_dict() for record in project_evidence(evidence)]
    second = [record.to_dict() for record in project_evidence(_evidence(events))]
    assert first == second
    assert evidence.to_dict() == before


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_dotted_filenames_survive_as_one_term_and_also_index_their_parts() -> None:
    """The decision to use ``_WORD`` rather than ``_TOKEN``, pinned.

    ``_TOKEN`` must start with a letter and forbids a dot, so it would shred these
    into unrelated fragments — and they are the highest-value terms in a software
    tutorial. Each compound is also indexed per part, so a query for the plain word
    still reaches the dotted name.
    """
    for text, expected in [
        ("whisper.cpp", ("whisper.cpp", "whisper", "cpp")),
        ("ggmlmedium.bin", ("ggmlmedium.bin", "ggmlmedium", "bin")),
        ("FFmpeg.exit", ("ffmpeg.exit", "ffmpeg", "exit")),
    ]:
        assert tuple(token.term for token in tokenize(text)) == expected, text


def test_paths_stay_whole_and_expose_every_component() -> None:
    path = "Projects/Templates/video-learning-studio"
    terms = [token.term for token in tokenize(path)]
    assert terms[0] == path.lower()
    assert set(terms[1:]) == {"projects", "templates", "video", "learning", "studio"}


def test_absolute_path_is_found_by_its_project_folder() -> None:
    """The single biggest recall win of splitting compounds.

    The leading ``/`` is not part of a word, so the whole term is the path *from*
    ``Users`` onward; the query ``video-learning-studio`` can only match through
    the components — and it must.
    """
    index = _index(_ocr(5.0, "cwd /Users/r/Projects/Templates/video-learning-studio"))
    result = _results(index, "video-learning-studio")
    assert result.status == "ok"
    assert result.results[0].matched_terms == ("video", "learning", "studio")


def test_command_line_flags_are_tokenized_as_written() -> None:
    command = "whisper-cli -m ggml-medium.bin --output-json"
    terms = tuple(token.term for token in tokenize(command))
    assert terms == (
        "whisper-cli",
        "whisper",
        "cli",
        "m",
        "ggml-medium.bin",
        "ggml",
        "medium",
        "bin",
        "output-json",
        "output",
        "json",
    )


def test_a_query_term_finds_the_command_line_it_was_written_in() -> None:
    index = _index(_ocr(5.0, "whisper-cli -m ggml-medium.bin --output-json"))
    for query in ("ggml-medium.bin", "output-json", "medium", "whisper"):
        assert _results(index, query).status == "ok", query


def test_trailing_punctuation_is_not_part_of_a_term() -> None:
    tokens = tokenize("Install ffmpeg.")
    assert [token.term for token in tokens] == ["install", "ffmpeg"]
    assert tokens[1].surface == "ffmpeg"
    assert tokens[1].end == len("Install ffmpeg")


def test_separators_alone_produce_no_terms() -> None:
    for noise in ("...", "---", "///", "", "   \n  "):
        assert tokenize(noise) == (), repr(noise)


def test_offsets_always_point_at_their_own_surface_in_the_untouched_text() -> None:
    text = "Model Usage\nwhisper.cpp and ggml-medium.bin are local."
    for token in tokenize(text):
        assert text[token.start : token.end] == token.surface
        assert token.term == token.surface.lower()


def test_case_is_folded_for_matching_and_preserved_for_display() -> None:
    tokens = tokenize("Qoder CLI\nqoder cli")
    assert {token.term for token in tokens} == {"qoder", "cli"}
    assert {token.surface for token in tokens} == {"Qoder", "CLI", "qoder", "cli"}


def test_searching_a_stem_does_not_match_a_different_inflection() -> None:
    """No stemming exists, and this test says so out loud rather than hoping.

    Recall against inflected speech is the documented gap a later embedding step is
    there to close; pretending otherwise would overstate the capability.
    """
    index = _index(_speech(6.0, 12.0, "we transcribed the audio locally"))
    assert _results(index, "transcribe").status == "no_confident_match"


# ---------------------------------------------------------------------------
# Contractions
# ---------------------------------------------------------------------------


def test_clitic_tails_are_not_terms() -> None:
    """``'s``, ``'re``, ``'ll`` are fragments, and no fragment carries intent.

    The rule is structural — a run that continues a preceding alphanumeric across an
    apostrophe is a tail — not a clitic vocabulary, so it cannot drift as English (or
    a typo) does. A straight and a typographic apostrophe behave identically.
    """
    for text, expected in [
        ("today's meeting", ("today", "meeting")),
        ("we're going", ("we", "going")),
        ("what's new", ("what", "new")),
        ("she'll do it", ("she", "do", "it")),
        ("it\u2019s ready", ("it", "ready")),
    ]:
        assert tuple(token.term for token in tokenize(text)) == expected, text


def test_negative_contractions_are_indexed_as_one_term() -> None:
    """``n't`` is the tail whose head is a real word, so the pair stays together.

    Every form arrives through the same shape test rather than this list, which is
    why ``aren't``, ``wasn't``, ``haven't`` and ``needn't`` work too; casing and a
    sentence-final period are the ranker's ordinary trimming, and a tail longer than
    ``t`` (``don'tknow``) is not a contraction at all.
    """
    for text, expected in [
        ("don't", ("don't",)),
        ("won't", ("won't",)),
        ("can't", ("can't",)),
        ("isn't", ("isn't",)),
        ("doesn't", ("doesn't",)),
        ("couldn't", ("couldn't",)),
        ("shouldn't", ("shouldn't",)),
        ("mustn't", ("mustn't",)),
        ("ain't", ("ain't",)),
        ("aren't", ("aren't",)),
        ("DON'T", ("don't",)),
        ("won't.", ("won't",)),
        ("it won\u2019t", ("it", "won't")),
        ("don'tknow", ("don",)),
    ]:
        assert tuple(token.term for token in tokenize(text)) == expected, text


def test_won_does_not_match_won_t_and_won_t_does_not_match_won() -> None:
    """The precision defect this rule exists to remove, in both directions.

    Before it, ``won't`` tokenized as ``won``, so one term covered both senses and
    the *negated* clip — being shorter — ranked first for the affirmative query.
    Each query must now find only its own sentence, and the shared word must still
    find both, which is the proof that the negated clip was not deleted from the
    index rather than merely disambiguated.
    """
    index = _index(
        _speech(1.0, 2.0, "we won the award locally"),
        _speech(3.0, 4.0, "it won't work locally"),
    )
    affirmative = _results(index, "won")
    assert _texts(affirmative) == ["we won the award locally"]
    assert affirmative.results[0].matched_terms == ("won",)
    assert affirmative.coverage.matched_record_count == 1

    negated = _results(index, "won't")
    assert _texts(negated) == ["it won't work locally"]
    assert negated.results[0].matched_terms == ("won't",)
    assert negated.coverage.matched_record_count == 1

    shared = _results(index, "locally")
    assert set(_texts(shared)) == {"we won the award locally", "it won't work locally"}
    assert shared.coverage.matched_record_count == 2


@pytest.mark.parametrize(
    ("contraction", "affirmative", "negated"),
    [
        ("don't", "we do transcribe locally", "we don't transcribe locally"),
        ("can't", "you can transcribe locally", "you can't transcribe locally"),
        ("isn't", "it is transcribed locally", "it isn't transcribed locally"),
        ("doesn't", "she does transcribe locally", "she doesn't transcribe locally"),
        ("couldn't", "they could transcribe locally", "they couldn't transcribe locally"),
        ("shouldn't", "you should transcribe locally", "you shouldn't transcribe locally"),
    ],
)
def test_a_negative_contraction_finds_only_the_negated_clip(
    contraction: str, affirmative: str, negated: str
) -> None:
    """Six forms, one shape test — and the affirmative partner never answers.

    The partner record deliberately carries the bare head (``can``, ``do``, ``is``)
    so a collapse would show up as a second hit. ``locally`` finds both, so no clip
    is lost; only the negation is kept distinct. (The shared word has to be one both
    rows spell identically — there is no stemmer here either.)
    """
    index = _index(_speech(1.0, 2.0, affirmative), _speech(3.0, 4.0, negated))
    result = _results(index, contraction)
    assert _texts(result) == [negated], contraction
    assert result.results[0].matched_terms == (contraction,), contraction
    assert set(_texts(_results(index, "locally"))) == {affirmative, negated}


def test_typo_and_spaced_negations_are_not_guessed_at() -> None:
    """No stemmer, no repair: a missing quote or a stray space stays a different string.

    ``dont``/``cant`` reach only text that really spells them, and ``ca n't`` is two
    runs whose lone ``n`` is dropped as a one-letter head — so it becomes neither
    ``can`` nor ``can't``. Predicting otherwise would be the synonym table this step
    must not add.
    """
    index = _index(
        _speech(1.0, 2.0, "we don't transcribe locally"),
        _speech(3.0, 4.0, "open the file ca n't you see"),
    )
    for query in ("dont", "cant", "do", "can", "n"):
        assert _results(index, query).coverage.matched_record_count == 0, query
    assert _results(index, "don't").status == "ok"
    assert _results(index, "transcribe").status == "ok"


@pytest.mark.parametrize(
    "text",
    [
        "it won-'t work locally",
        "London.'t is the caption",
        "run-on.'t breaks the line",
        "it won''t work locally",
        "it won' t work locally",
        "it won\u2019 t work locally",
        "it won'ts work locally",
        "it won''t. opened it",
        "so it won-'t/ opened it",
        "it won't work locally",
        "it won\u2019t work locally",
        "v0.1-don't is the filename",
    ],
)
def test_every_contraction_span_selects_its_own_surface(text: str) -> None:
    """The invariant, checked on the search path instead of assumed from the rule.

    ``surface`` is what a caller prints and ``start``/``end`` are what it highlights,
    so they have to agree even when a separator or a second quote sits between the
    head and its ``t``. They disagreed for ``won-'t`` — the span was measured against
    the trimmed core while the surface kept the untrimmed run, which selected
    ``won-'`` for a surface of ``won-'t``. A malformed negation is now suppressed
    rather than spanned, so no returned match can be wrong about its own text.
    """
    for token in tokenize(text):
        assert text[token.start : token.end] == token.surface, text
    index = _index(_speech(6.0, 12.0, text))
    for query in ("won", "won't", "london", "work", "locally", "filename", "t"):
        for match in _results(index, query, limit=10).results:
            assert match.evidence.text == text, (text, query)
            for span in match.matched_spans:
                assert text[span.start : span.end] == span.surface, (text, query, span.term)


def test_a_malformed_negation_never_answers_an_affirmative_query() -> None:
    """Five broken spellings of ``won't``, and none of them is ``won``.

    Before this rule each one indexed the bare head, so an affirmative search found
    the negated clip — the same inversion ``won't``/``won`` was separated for, reached
    through a doubled quote, a stray space, a curly quote or a plural-looking tail.
    Suppression is the answer for all five: the malformed records stop answering
    ``won``, stop answering ``won't`` (guessing the correction would be repair), and
    keep answering their own real words, so nothing is deleted from the corpus.
    """
    affirmative = "we won the award locally"
    malformed = [
        "it won''t work locally",
        "it won' t work locally",
        "it won\u2019 t work locally",
        "it won'ts work locally",
        "it won-'t work locally",
    ]
    well_formed = "it won\u2019t fail locally"
    corpus = [affirmative, *malformed, well_formed]
    index = _index(
        *(_speech(float(i), float(i) + 1.0, text) for i, text in enumerate(corpus))
    )
    assert _texts(_results(index, "won", limit=10)) == [affirmative]
    assert _texts(_results(index, "won't", limit=10)) == [well_formed]
    assert _results(index, "won't").results[0].matched_terms == ("won't",)
    # The fragments the malformed forms used to leak are not intent either.
    for query in ("t", "ts"):
        assert _results(index, query).coverage.matched_record_count == 0, query
    # And the records themselves are still evidence, reachable by what they really say.
    assert set(_texts(_results(index, "work", limit=10))) == set(malformed)
    assert _texts(_results(index, "fail")) == [well_formed]


@pytest.mark.parametrize("separator", [".", "-", "/", "#", "+"])
def test_a_broken_negation_running_into_a_separator_is_still_dropped(separator: str) -> None:
    """Each of the ranker's own trim characters used to reopen the affirmative.

    The swallowed fragment is located with ``_WORD_RE``, which counts ``. - / # +`` as
    part of the run, so ``won''t.`` reported it as ``t.``, compared that to ``t``, found
    no match and concluded this was not a broken negation at all. The head then went in
    as ``won`` — the one outcome the rule exists to prevent — and a sentence-ending
    period is how text normally ends, which made the common case the broken one. The
    comparison now strips the same characters the ranker strips; the boundary it returns
    still covers the whole run, so nothing after the fragment is left to index twice.
    """
    for shape in ("won''t", "won' t", "won\u2019 t", "won-'t", "won'ts", "won ''t"):
        text = f"we {shape}{separator} opened it"
        assert tuple(token.term for token in tokenize(text)) == ("we", "opened", "it"), text


def test_a_malformed_negation_ending_a_sentence_is_not_an_affirmative() -> None:
    """The same class asked as a search, because the tokenizer is not the user's view.

    Five captions whose ``won't`` the transcription wrote badly and a period or slash
    closed off, one genuine affirmative and both well-formed spellings of the
    contraction. Asking for ``won`` must answer the affirmative and nothing else: a
    negated caption ranking for the word its whole point denies is the inversion this
    step was written against. Asking for ``t`` must answer nothing, since the leftover
    fragment is the other half of the same leak. And asking for ``won't`` still reaches
    only the two captions that really wrote it, straight or curly — the correction
    removes the artifact, it does not guess a replacement for it.
    """
    affirmative = "we won the award locally"
    malformed = [
        "so it won''t. transcribe locally",
        "so it won\u2019 t. transcribe locally",
        "so it won'ts. transcribe locally",
        "so it won-'t/ transcribe locally",
        "so it won ''t+ transcribe locally",
    ]
    straight = "so it won't fail locally"
    curly = "so it won\u2019t fail locally"
    corpus = [affirmative, *malformed, straight, curly]
    index = _index(
        *(_speech(float(i), float(i) + 1.0, text) for i, text in enumerate(corpus))
    )
    assert _texts(_results(index, "won", limit=10)) == [affirmative]
    assert set(_texts(_results(index, "won't", limit=10))) == {straight, curly}
    for match in _results(index, "won't", limit=10).results:
        assert match.matched_terms == ("won't",), match.evidence.text
    for query in ("t", "ts"):
        assert _results(index, query).coverage.matched_record_count == 0, query
    # Suppressed terms are not deleted captions: each malformed record still answers
    # the words it really contains.
    assert set(_texts(_results(index, "transcribe", limit=10))) == set(malformed)


def test_a_broken_negation_after_a_line_break_suppresses_the_word_before_it() -> None:
    """The whitespace the rule skips is whitespace, newline included — pinned as a limit.

    A head at the end of one caption line and a quoted ``t`` opening the next read as
    one broken negation, so the head goes. Repairing that would mean deciding whether
    the ``t`` belongs to the word or to the reader, which is exactly the guess this step
    refuses to make; the honest options were to reach no further than a space or to
    accept the collateral, and the shape rule as written accepts it. Asserted here so
    the documented limitation stays the actual limitation.
    """
    text = "Section\n’t was the label"
    assert tuple(token.term for token in tokenize(text)) == ("was", "the", "label")
    index = _index(_ocr(5.0, text))
    assert _results(index, "section").coverage.matched_record_count == 0
    assert _results(index, "label").coverage.matched_record_count == 1


def test_only_the_two_ordinary_apostrophe_spellings_are_read_as_apostrophes() -> None:
    """Which quote characters count is a set, and the set is ``'`` and ``’``.

    A left single quote, a modifier letter apostrophe or a backtick where the
    contraction's apostrophe should be is a separator, so the head stands as the
    affirmative and the leftover ``t`` is indexed. That is stated rather than fixed:
    folding more code points would be a guess about what the evidence meant to write,
    and the two spellings recognized are the ones whisper.cpp and a title card
    actually produce. The head of ``whisper'cli`` is kept while its tail is dropped for
    the opposite reason — a tail joined without a space is a clitic, a head never is.
    """
    for quote in ("\u2018", "\u02bc", "`", "\u00b4"):
        text = f"we won{quote}t work"
        assert tuple(token.term for token in tokenize(text)) == ("we", "won", "t", "work"), text
    assert tuple(token.term for token in tokenize("whisper'cli")) == ("whisper",)
    leaked = "we won\u2018t work"
    well_formed = "we won't work"
    index = _index(_ocr(5.0, leaked), _ocr(6.0, well_formed))
    assert set(_texts(_results(index, "won", limit=10))) == {leaked}
    assert set(_texts(_results(index, "won't", limit=10))) == {well_formed}


def test_a_one_letter_fragment_behind_an_apostrophe_is_not_intent() -> None:
    """The tail rule was supposed to remove these; it only did the fused ones.

    ``abc-'t``, ``the '-t flag`` and ``Qoder-'s editor`` each left a searchable ``t``
    or ``s`` standing alone, because the old test required a letter *immediately*
    before the quote and a separator is not one. A lone letter behind an apostrophe is
    a fragment however it got there, so it contributes nothing — while ``-m``, which
    has a space behind its hyphen and no apostrophe at all, keeps its ``m``.
    """
    for text, expected in [
        ("abc-'t", ("abc",)),
        ("the '-t flag", ("the", "flag")),
        ("Qoder-'s editor", ("qoder", "editor")),
        ("5't", ()),
        ("'t", ()),
    ]:
        assert tuple(token.term for token in tokenize(text)) == expected, text
    index = _index(
        _ocr(5.0, "the '-t flag"),
        _ocr(6.0, "Qoder-'s editor"),
        _ocr(7.0, "whisper-cli -m ggml-medium.bin"),
    )
    for query in ("t", "s"):
        assert _results(index, query).status == "no_confident_match", query
    assert _texts(_results(index, "flag")) == ["the '-t flag"]
    assert _texts(_results(index, "qoder")) == ["Qoder-'s editor"]
    assert _results(index, "m").status == "ok"


def test_suppression_does_not_reach_for_a_word_that_is_not_a_negation() -> None:
    """The two shapes one condition away from the artifact, and neither is one.

    ``Lincoln's`` is an ``n``-terminated head followed by an apostrophe whose tail is
    a possessive, and ``Eisen 'twas`` is an apostrophe followed by a ``t``. Both used
    to be handled, both must stay handled: a name that vanishes because someone typed
    ``'twas`` is a worse failure than the fragment the rule removes.
    """
    for text, expected in [
        ("Lincoln's speech", ("lincoln", "speech")),
        ("Eisen 'twas written", ("eisen", "twas", "written")),
        ("Boston 'to the bay", ("boston", "to", "the", "bay")),
    ]:
        assert tuple(token.term for token in tokenize(text)) == expected, text
    index = _index(
        _ocr(5.0, "Lincoln's speech"),
        _speech(6.0, 12.0, "Eisen 'twas written"),
    )
    for query in ("lincoln", "speech", "twas", "written"):
        assert _results(index, query).status == "ok", query


def test_a_negation_inside_a_compound_stops_its_components_from_splitting() -> None:
    """The documented cost of the rule, asserted as behavior rather than as prose.

    ``v0.1-don't`` indexes whole only: splitting it would index ``don`` and ``1``,
    which is the affirmative-fragment error the negation term exists to remove, in a
    different costume. So the version prefix does not reach it and the contraction
    does — and asking for ``don't`` alone is a different search, not a narrower one.
    """
    text = "v0.1-don't is the filename"
    assert tuple(token.term for token in tokenize(text)) == (
        "v0.1-don't",
        "is",
        "the",
        "filename",
    )
    index = _index(_ocr(5.0, text))
    assert _texts(_results(index, "v0.1-don't")) == [text]
    assert _results(index, "v0.1-don't").results[0].matched_terms == ("v0.1-don't",)
    for query in ("v0.1", "don", "1", "don't"):
        assert _results(index, query).coverage.matched_record_count == 0, query
    assert _texts(_results(index, "filename")) == [text]


def test_straight_and_curly_apostrophes_find_the_same_evidence() -> None:
    """One matching key for two spellings, in whichever order they meet.

    The fold is in the term only: every score below is identical across quote styles,
    which is what lets a transcript that mixes whisper.cpp's straight quotes with a
    title card's typographic one behave as one corpus.
    """
    straight = _index(_speech(1.0, 2.0, "it won't work locally"))
    curly = _index(_speech(1.0, 2.0, "it won\u2019t work locally"))
    for query in ("won't", "won\u2019t"):
        assert _texts(_results(straight, query)) == ["it won't work locally"]
        assert _texts(_results(curly, query)) == ["it won\u2019t work locally"]
    assert _results(straight, "won't").results[0].score == pytest.approx(
        _results(curly, "won\u2019t").results[0].score
    )
    # And the affirmative head stays out of both, whichever quote the evidence used.
    for index in (straight, curly):
        assert _results(index, "won").results == ()


def test_normalization_never_rewrites_the_evidence_a_reader_is_shown() -> None:
    """The whole point of a quote-folding rule: the key folds, the evidence does not.

    ``surface`` and the offsets stay a true substring of the untouched text, so a
    highlight lands on ``won\u2019t`` exactly as transcribed while ``matched_terms``
    carries the canonical ``won't`` key. The serialized artifact keeps the curly
    apostrophe byte for byte.
    """
    text = "  it won\u2019t work locally "
    result = _results(_index(_speech(6.0, 12.0, text)), "won't")
    match = result.results[0]
    assert match.evidence.text == text
    span = next(s for s in match.matched_spans if s.term == "won't")
    assert span.surface == "won\u2019t"
    assert text[span.start : span.end] == span.surface
    assert match.matched_terms == ("won't",)
    assert json.loads(json.dumps(result.to_dict()))["results"][0]["text"] == text


def test_one_letter_clitic_heads_leave_no_term_behind() -> None:
    """``o'clock`` and ``y'all`` are unsearchable rather than misleading.

    The tail is a clitic and a bare ``o`` is the artifact the rule exists to remove,
    so the head goes with it. The cost is stated here rather than discovered later:
    ``clock`` finds nothing. The enclosing records must still answer their real words.
    """
    for text in ("o'clock", "y'all"):
        assert tokenize(text) == (), text
    index = _index(_speech(1.0, 2.0, "it's five o'clock"), _speech(3.0, 4.0, "y'all got it"))
    for query in ("o", "y", "clock", "all", "o'clock", "y'all"):
        assert _results(index, query).coverage.matched_record_count == 0, query
    assert _texts(_results(index, "five")) == ["it's five o'clock"]
    assert _texts(_results(index, "got")) == ["y'all got it"]


def test_a_corpus_of_only_apostrophe_artifacts_is_no_evidence() -> None:
    """The searchable-record rule and the clitic rule agree instead of arguing.

    Frames whose every run is a one-letter head are projected and counted, and
    contribute no document to the statistics at all.
    """
    index = _index(_ocr(5.0, "o'clock"), _ocr(6.0, "y'all"))
    assert len(index.records) == 2
    assert index.searchable_positions == ()
    result = _results(index, "o'clock")
    assert result.status == "no_evidence"
    assert result.coverage.record_count == 2
    assert result.coverage.searchable_record_count == 0


@pytest.mark.parametrize("query", ["what's", "it's", "we're", "that's", "there's"])
def test_a_conventional_question_is_not_a_search(query: str) -> None:
    """Before the fix these were ``ok`` with an unrelated match, on a one-letter term.

    Every head left behind (``what``, ``it``, ``we``, ``that``, ``there``) is a
    stopword, so the honest answer to the question is that it asked for nothing.
    """
    index = _index(
        _speech(6.0, 12.0, "we open the settings panel"),
        _ocr(13.0, "Release Notes"),
    )
    result = _results(index, query)
    assert result.coverage.meaningful_query_term_count == 0, query
    assert result.status == "no_confident_match", query


def test_a_quoted_term_after_a_space_is_still_indexed() -> None:
    """Only a tail joined to a preceding run is dropped, so shell text survives.

    ``grep 'whisper' file`` is exactly the kind of command a tutorial shows; its
    apostrophe follows a space, so there is nothing for it to fuse with.
    """
    index = _index(_ocr(5.0, "grep 'whisper' file"))
    assert tuple(token.term for token in tokenize("grep 'whisper' file")) == (
        "grep",
        "whisper",
        "file",
    )
    for query in ("whisper", "grep"):
        assert _results(index, query).status == "ok", query
    # "file" is one of the project's UI-chrome stopwords, so it is not intent. That
    # is the pre-existing stopword rule, not the apostrophe rule: the term is indexed.
    assert _results(index, "file").status == "no_confident_match"


def test_a_possessive_and_its_bare_name_reach_the_same_evidence() -> None:
    """One rule applied to query and evidence is what makes them interchangeable."""
    index = _index(_speech(6.0, 12.0, "today's session is about transcription"))
    for query in ("today's", "today", "session"):
        assert _results(index, query).status == "ok", query
    assert _results(index, "today").results[0].matched_terms == ("today",)


def test_possessives_of_technical_names_stay_searchable_by_the_name() -> None:
    """The win the tail rule is for: ``Qoder's editor`` is about Qoder and an editor.

    The name keeps its own term, the tail leaves nothing, and the query does not need
    the apostrophe — which is exactly why possessives must not be treated like ``n't``
    and indexed whole: a term ``qoder's`` would never be typed by a searcher.
    """
    index = _index(
        _ocr(5.0, "Qoder's editor"),
        _speech(6.0, 12.0, "whisper's output is a text file"),
    )
    for query, expected in (
        ("qoder", "Qoder's editor"),
        ("editor", "Qoder's editor"),
        ("whisper", "whisper's output is a text file"),
        ("output", "whisper's output is a text file"),
    ):
        assert _texts(_results(index, query)) == [expected], query
    assert _results(index, "qoder").results[0].matched_terms == ("qoder",)
    # Asking with the apostrophe is the same search, not a second one.
    assert _texts(_results(index, "Qoder's")) == ["Qoder's editor"]


def test_short_technical_terms_survive_the_contraction_fix() -> None:
    """The easy fix was a two-character minimum; it would have deleted ``bin``.

    ``bin`` is the real intent inside ``ggml-medium.bin``, and the ``-m`` flag is one
    letter, so the fix had to be about apostrophes rather than about length: a lone
    letter stays intent as long as no apostrophe follows it, and stops being one only
    when the clitic rule would have left it behind as an artifact.
    """
    index = _index(
        _ocr(5.0, "whisper-cli -m ggml-medium.bin"),
        _speech(6.0, 12.0, "use bin local"),
    )
    for text, expected in [
        ("bin", ("bin",)),
        ("-m", ("m",)),
        ("whisper.cpp", ("whisper.cpp", "whisper", "cpp")),
        ("ggmlmedium.bin", ("ggmlmedium.bin", "ggmlmedium", "bin")),
        ("FFmpeg.exit", ("ffmpeg.exit", "ffmpeg", "exit")),
    ]:
        assert tuple(token.term for token in tokenize(text)) == expected, text
    for query in ("bin", "m", "whisper", "ggml-medium.bin"):
        assert _results(index, query).status == "ok", query


# ---------------------------------------------------------------------------
# Result states
# ---------------------------------------------------------------------------


def test_no_searchable_evidence_is_reported_as_no_evidence() -> None:
    index = build_search_index(_evidence([], segments=[], duration=None))
    result = _results(index, "whisper")
    assert result.status == "no_evidence"
    assert result.results == ()
    assert result.coverage.record_count == 0
    assert result.coverage.searchable_record_count == 0
    assert result.coverage.query_token_count == 1  # the query was fine; the corpus was not


def test_no_lexical_overlap_is_no_confident_match_not_an_error() -> None:
    index = _index(_speech(6.0, 12.0, "we open the settings panel"))
    result = _results(index, "pineapple")
    assert result.status == "no_confident_match"
    assert result.results == ()
    assert result.coverage.matched_record_count == 0
    assert result.coverage.meaningful_query_term_count == 1


@pytest.mark.parametrize(
    "query",
    ["the and", "what is the", "how do I", "how is that in the", "", "   "],
)
def test_stopword_only_or_blank_queries_have_no_meaningful_terms(query: str) -> None:
    index = _index(_speech(6.0, 12.0, "we open the settings panel and read them"))
    result = _results(index, query)
    assert result.status == "no_confident_match"
    assert result.coverage.meaningful_query_term_count == 0
    assert result.coverage.matched_record_count == 0


def test_a_query_of_pure_numbers_matches_nothing() -> None:
    """A ``12:34:56`` timer or a frame count carries no search intent.

    The same rule the keyword ranker applies to letter-free fragments, and the
    reason the corpus below really does contain those digits.
    """
    index = _index(
        _ocr(5.0, "00:12:34:56 elapsed"),
        _speech(6.0, 9.0, "the model is 1500 tokens"),
    )
    assert [token.term for token in tokenize("12:34:56")] == ["12", "34", "56"]
    result = _results(index, "12:34:56")
    assert result.status == "no_confident_match"
    assert result.coverage.query_token_count == 3
    assert result.coverage.meaningful_query_term_count == 0


def test_a_weak_but_real_match_is_still_ok() -> None:
    """There is deliberately no score threshold, so nothing can be filtered by it.

    The long document below earns the lowest positive score this corpus can
    produce for a term that appears in both documents; it is still a real overlap
    and must be returned, not silently dropped.
    """
    index = _index(
        _speech(1.0, 2.0, "whisper"),
        _speech(
            3.0,
            4.0,
            "whisper is the tool that we open from the terminal window to transcribe audio",
        ),
    )
    result = _results(index, "whisper")
    assert result.status == "ok"
    assert result.coverage.matched_record_count == 2
    first, second = (match.score for match in result.results)
    assert first > second > 0.0


def test_status_is_always_one_of_the_documented_three() -> None:
    cases = [
        (build_search_index(_evidence([], segments=[], duration=None)), "whisper"),
        (_index(_ocr(5.0, "alpha screen")), "beta"),
        (_index(_ocr(5.0, "alpha screen")), "alpha"),
        (_index(_ocr(5.0, "alpha screen")), "the"),
    ]
    for index, query in cases:
        assert _results(index, query).status in {
            "no_evidence",
            "no_confident_match",
            "ok",
        }, query


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_bm25_score_matches_the_textbook_formula_recomputed_here() -> None:
    """A reader can reproduce any score in this artifact with pen and paper.

    Two equal-length documents, one mentioning the term twice, in a two-document
    corpus: IDF is ``ln(1 + 0.5/2.5)`` and the length normalization collapses to
    exactly ``k1``, so both expected scores are pure arithmetic.
    """
    index = _index(
        _speech(1.0, 6.0, "install whisper"),
        _speech(7.0, 12.0, "whisper whisper"),
    )
    result = _results(index, "whisper")
    assert _texts(result) == ["whisper whisper", "install whisper"]
    assert result.results[0].score == pytest.approx(_bm25(2, 2, 2.0, 2, 2), rel=1e-12)
    assert result.results[1].score == pytest.approx(_bm25(1, 2, 2.0, 2, 2), rel=1e-12)
    # The same number, written out by hand: IDF * (2 * 2.5 / (2 + 1.5)).
    assert result.results[0].score == pytest.approx(math.log(1.2) * 5.0 / 3.5, rel=1e-12)


def test_repeating_a_term_raises_the_score_without_multiplying_it() -> None:
    """Term-frequency saturation — the whole point of ``k1`` — visible in the numbers."""
    index = _index(
        _speech(1.0, 2.0, "whisper"),
        _speech(3.0, 4.0, "whisper whisper whisper"),
    )
    scores = {match.evidence.text: match.score for match in _results(index, "whisper").results}
    assert scores["whisper whisper whisper"] > scores["whisper"]
    assert scores["whisper whisper whisper"] < 3 * scores["whisper"]


def test_rare_term_outranks_common_term_when_frequency_and_length_agree() -> None:
    """IDF is doing its job: identical term frequency and length, different distribution."""
    index = _index(
        _speech(1.0, 2.0, "whisper cli"),
        _speech(3.0, 4.0, "ffmpeg cli"),
        _speech(5.0, 6.0, "ffmpeg cli"),
        _speech(7.0, 8.0, "ffmpeg cli"),
    )
    result = _results(index, "whisper ffmpeg")
    assert _texts(result)[0] == "whisper cli"
    rare, common = result.results[0].score, result.results[1].score
    assert rare == pytest.approx(_bm25(1, 2, 2.0, 1, 4), rel=1e-12)
    assert common == pytest.approx(_bm25(1, 2, 2.0, 3, 4), rel=1e-12)
    assert rare > common


def test_a_longer_document_is_penalized_for_the_same_single_hit() -> None:
    """Length normalization, ``b``, in one assertion."""
    long_text = "whisper is the tool that we open from the terminal window"
    index = _index(_speech(1.0, 2.0, "whisper"), _speech(3.0, 4.0, long_text))
    result = _results(index, "whisper")
    assert _texts(result) == ["whisper", long_text]
    assert result.results[0].score > result.results[1].score


def test_covering_more_query_terms_outranks_covering_one() -> None:
    index = _index(
        _speech(1.0, 2.0, "transcribe the audio"),
        _speech(3.0, 4.0, "transcribe the audio locally with a model"),
    )
    result = _results(index, "transcribe locally")
    assert _texts(result)[0] == "transcribe the audio locally with a model"


def test_non_matching_records_are_never_padded_into_the_results() -> None:
    index = _index(
        _speech(1.0, 2.0, "whisper"),
        _speech(3.0, 4.0, "nothing related at all here"),
    )
    result = _results(index, "whisper", limit=10)
    assert _texts(result) == ["whisper"]
    assert result.coverage.record_count == 2
    assert result.coverage.matched_record_count == 1
    assert all(match.score > 0.0 for match in result.results)


def test_equal_scores_at_twelve_records_break_by_lexicographic_id() -> None:
    """The id tie-break is a *string* compare, and twelve records make it visible.

    All twelve carry the same term at the same second with the same length, so every
    score and every timestamp ties and order is decided by ``evidence_id`` alone.
    Ordinal order would put ``speech:2`` third; the documented rule puts it fifth,
    behind ``speech:10`` and ``speech:11``, because ``"1" < "2"``. A reader
    recomputing this list has to get the same answer, which is why the expectation is
    spelled out rather than sorted in Python.
    """
    events = [_speech(5.0, 6.0, "whisper") for _ in range(12)]
    expected = [f"speech:{i}" for i in (0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9)]
    index = _index(*events)
    result = _results(index, "whisper", limit=20)
    assert len(result.results) == 12
    assert {match.score for match in result.results} == {result.results[0].score}
    assert [match.evidence.evidence_id for match in result.results] == expected
    assert len(set(expected)) == 12  # unique inside one artifact, so the order is total

    # Same records, rebuilt: the order is a property of the rule, not of this run.
    assert [
        match.evidence.evidence_id for match in _results(_index(*events), "whisper").results
    ] == expected[:5]


def test_ties_break_chronologically_then_by_evidence_id() -> None:
    """Equal scores must still produce one stable order.

    ``ocr:0`` and ``speech:0`` are the same term at the same second with the same
    length, so only the deterministic id tie-break can separate them; ``ocr:1`` is
    later in the video and so comes last regardless of its id.
    """
    index = _index(_ocr(5.0, "whisper"), _speech(5.0, 9.0, "whisper"), _ocr(10.0, "whisper"))
    result = _results(index, "whisper")
    assert {match.score for match in result.results} == {result.results[0].score}
    assert [match.evidence.evidence_id for match in result.results] == [
        "ocr:0",
        "speech:0",
        "ocr:1",
    ]


def test_ranking_is_reproducible_as_identical_json() -> None:
    events: list[TimelineEvent] = [
        _ocr(5.0, "Model Usage\nwhisper.cpp"),
        _speech(6.0, 18.0, "we run whisper on a local model file"),
        _speech(20.0, 30.0, "the model is ggml-medium.bin"),
        _ocr(25.0, "Release Notes\nrelease notes"),
    ]
    query = "whisper model"
    payload_a = json.dumps(
        search(build_search_index(_evidence(events)), query).to_dict(), sort_keys=True
    )
    payload_b = json.dumps(
        search(build_search_index(_evidence(events)), query).to_dict(), sort_keys=True
    )
    assert payload_a == payload_b
    reused = build_search_index(_evidence(events))
    assert json.dumps(search(reused, query).to_dict(), sort_keys=True) == payload_a


def test_searching_does_not_mutate_the_index() -> None:
    index = _index(_speech(6.0, 12.0, "we open settings"))
    snapshot = [record.to_dict() for record in index.records]
    lengths = index.doc_lengths
    search(index, "settings")
    search(index, "the")
    assert [record.to_dict() for record in index.records] == snapshot
    assert index.doc_lengths == lengths


# ---------------------------------------------------------------------------
# Matched terms and spans
# ---------------------------------------------------------------------------


def test_matched_terms_follow_document_order_not_query_order() -> None:
    index = _index(_speech(6.0, 12.0, "cli whisper cli"))
    result = _results(index, "whisper cli")
    assert result.results[0].matched_terms == ("cli", "whisper")


def test_matched_terms_are_distinct_and_exclude_stopwords() -> None:
    index = _index(_speech(6.0, 12.0, "the model model is local"))
    matched = _results(index, "model the").results[0]
    assert matched.matched_terms == ("model",)
    assert matched.score > 0.0


def test_matched_spans_reproduce_the_evidence_text_and_are_sorted() -> None:
    text = "we run whisper on ggml-medium.bin locally"
    index = _index(_speech(6.0, 12.0, text))
    matched = _results(index, "whisper ggml-medium.bin").results[0]
    starts = [span.start for span in matched.matched_spans]
    assert starts == sorted(starts)
    for span in matched.matched_spans:
        assert text[span.start : span.end] == span.surface
    # The query's compound arrives as three scored terms, so its parts match too;
    # listing them here is what makes the overlap visible instead of surprising.
    assert {span.surface for span in matched.matched_spans} == {
        "whisper",
        "ggml-medium.bin",
        "ggml",
        "medium",
        "bin",
    }
    whole = [span for span in matched.matched_spans if span.surface == "ggml-medium.bin"]
    assert text[whole[0].start : whole[0].end] == "ggml-medium.bin"


def test_a_compound_and_its_parts_produce_nested_spans() -> None:
    """Overlapping spans are the honest result of indexing both forms.

    Ordered by offset, so the shorter part that starts at the same place is
    reported first — a caller highlighting these should paint longest-first.
    """
    text = "run whisper.cpp now"
    index = _index(_speech(6.0, 12.0, text))
    spans = _results(index, "whisper.cpp").results[0].matched_spans
    assert [(span.surface, span.start, span.end) for span in spans] == [
        ("whisper", 4, 11),
        ("whisper.cpp", 4, 15),
        ("cpp", 12, 15),
    ]


def test_matched_spans_never_widen_or_narrow_the_record_text() -> None:
    index = _index(_ocr(5.0, "Release Notes"))
    matched = _results(index, "release").results[0]
    assert matched.evidence.text == "Release Notes"
    assert matched.matched_spans[0].start == 0
    assert matched.matched_spans[0].surface == "Release"


# ---------------------------------------------------------------------------
# Limits, coverage and the artifact shape
# ---------------------------------------------------------------------------


def _seven_whisper_records() -> list[TimelineEvent]:
    return [_speech(float(i), float(i) + 1.0, f"whisper step {i}") for i in range(7)]


def test_limit_caps_results_but_never_what_is_considered() -> None:
    assert DEFAULT_RESULT_LIMIT == 5
    result = _results(_index(*_seven_whisper_records()), "whisper", limit=3)
    assert len(result.results) == 3
    assert result.coverage.matched_record_count == 7
    assert result.coverage.returned_record_count == 3
    assert result.limit == 3
    assert result.status == "ok"


def test_default_limit_applies_when_not_given() -> None:
    result = _results(_index(*_seven_whisper_records()), "whisper")
    assert len(result.results) == DEFAULT_RESULT_LIMIT
    assert result.coverage.matched_record_count == 7
    assert result.coverage.returned_record_count == DEFAULT_RESULT_LIMIT


def test_zero_limit_reports_counts_without_results() -> None:
    """An "explain my empty answer" surface needs the state and the numbers anyway."""
    result = _results(_index(_speech(1.0, 2.0, "whisper")), "whisper", limit=0)
    assert result.status == "ok"
    assert result.results == ()
    assert result.coverage.matched_record_count == 1
    assert result.coverage.returned_record_count == 0


def test_coverage_describes_the_index_independently_of_the_query() -> None:
    evidence = _evidence(
        [_ocr(5.0, "Model Usage\nmodel usage"), _speech(6.0, 12.0, "we open the panel")],
        segments=[],
        duration=None,
    )
    coverage = _results(build_search_index(evidence), "panel").coverage
    assert (
        coverage.record_count,
        coverage.speech_record_count,
        coverage.ocr_record_count,
    ) == (2, 1, 1)
    assert (coverage.attributed_record_count, coverage.unattributed_record_count) == (0, 2)


def test_coverage_separates_query_tokens_from_scored_terms() -> None:
    """A compound query is two things at once: several tokens, three scored terms."""
    result = _results(_index(_ocr(5.0, "whisper.cpp")), "whisper.cpp the")
    assert result.coverage.query_token_count == 4  # compound, its two parts, plus "the"
    assert result.coverage.meaningful_query_term_count == 3
    assert result.coverage.matched_record_count == 1


def test_results_carry_the_evidence_provenance_verbatim() -> None:
    """A reader can tell why a result set is OCR-only without another artifact.

    The fixture is the shape the pipeline actually produces when transcription
    failed: OCR points and no speech events at all. Pairing a failed status with
    transcript evidence would set ``transcription_evidence_conflict`` and test
    search against a contradiction that :func:`build_study_evidence` reports itself.
    """
    events = [
        _ocr(5.0, "Model Usage\nmodel usage"),
        _ocr(35.0, "Release Notes\nrelease notes"),
    ]
    evidence = _evidence(events, status="failed", detail="whisper-cli not found on PATH")
    assert evidence.transcription_evidence_conflict is False
    assert evidence.speech_event_count == 0
    result = _results(build_search_index(evidence), "model")
    assert result.source == _SOURCE
    assert result.study_evidence_schema == STUDY_EVIDENCE_SCHEMA
    assert result.transcription_status == "failed"
    assert result.transcription_available is False
    assert result.video_duration_seconds == 60.0
    assert result.query == "model"
    assert {match.evidence.source_type for match in result.results} == {"ocr"}


def test_results_dict_is_json_safe_and_self_describing() -> None:
    result = _results(_index(_speech(6.0, 12.0, "we open settings")), "settings")
    payload = result.to_dict()
    assert payload["schema"] == SEARCH_SCHEMA
    assert payload["status"] == "ok"
    assert payload["result_count"] == len(payload["results"])
    assert payload["results"][0]["source_type"] == "speech"
    assert json.loads(json.dumps(payload)) == payload


def test_only_ocr_results_serialize_a_null_end_seconds() -> None:
    result = _results(_index(_ocr(5.0, "whisper"), _speech(10.0, 14.0, "whisper")), "whisper")
    rows = {row["source_type"]: row for row in result.to_dict()["results"]}
    assert rows["ocr"]["end_seconds"] is None
    assert rows["speech"]["end_seconds"] == 14.0


def test_screen_garble_alone_cannot_produce_a_confident_match() -> None:
    """OCR noise and timers must not manufacture an answer.

    One real record sits among three pieces of screen noise. A digit-only,
    stopword-only or one-letter-fragment query has to find nothing, and the one
    query with real intent has to find exactly the record that carries the term —
    never a near-miss scored against the garbage.
    """
    index = _index(
        _ocr(5.0, "00:12:34 45%"),
        _ocr(6.0, "--- /// ..."),
        _ocr(7.0, "the a an"),
        _speech(8.0, 9.0, "whisper transcribes locally"),
    )
    for query in ("12:34", "45", "the", "s", "t"):
        assert _results(index, query).status == "no_confident_match", query
    result = _results(index, "whisper")
    assert _texts(result) == ["whisper transcribes locally"]
    assert result.coverage.matched_record_count == 1
    assert result.coverage.searchable_record_count == 1


# ---------------------------------------------------------------------------
# Unsearchable records: projected, counted, and kept out of the statistics
# ---------------------------------------------------------------------------


def test_unsearchable_records_are_still_projected_and_counted() -> None:
    """Nothing is dropped: a blank frame is evidence, it just cannot be matched.

    ``doc_lengths`` stays parallel to ``records`` so the projection is auditable,
    while ``searchable_positions`` — the only list BM25 reads — names the one record
    that carries a meaningful term.
    """
    index = _index(_ocr(5.0, "   "), _speech(6.0, 8.0, "whisper"), _ocr(9.0, "the a an"))
    assert [record.text for record in index.records] == ["   ", "the a an", "whisper"]
    assert [record.evidence_id for record in index.records] == ["ocr:0", "ocr:1", "speech:0"]
    assert index.doc_lengths == (0, 3, 1)
    assert index.searchable_positions == (2,)
    result = _results(index, "whisper")
    assert (result.coverage.record_count, result.coverage.searchable_record_count) == (3, 1)


def test_blank_records_cannot_inflate_a_score() -> None:
    """IDF rises with ``N``, so dead screen text used to raise every real score.

    The same two speech records alone, and padded with the kinds of OCR a real video
    frame produces, must score byte-identically. Measured against the pre-fix
    statistics, the padding lifted them from 0.235 and 0.149 to 1.294 and 0.800 — a
    score that depended on how much garbage the sampling happened to capture.
    """
    core = [_speech(1.0, 2.0, "whisper"), _speech(3.0, 4.0, "install whisper now")]
    padded = [
        _ocr(5.0, "00:12:34 45%"),
        _ocr(6.0, "   "),
        _ocr(7.0, "--- /// ..."),
        _ocr(8.0, "the a an"),
        *core,
    ]
    bare = _results(_index(*core), "whisper", limit=10)
    wide = _results(_index(*padded), "whisper", limit=10)
    assert _texts(bare) == _texts(wide)
    assert [match.score for match in bare.results] == [match.score for match in wide.results]
    assert bare.coverage.record_count == 2
    assert wide.coverage.record_count == 6
    assert wide.coverage.searchable_record_count == 2


def test_idf_and_average_length_use_only_searchable_records() -> None:
    """The formula is recomputable from the artifact's own counts, blanks included.

    Four records, two of them unsearchable, so ``N`` is ``2`` and the mean length is
    ``(1 + 2) / 2`` — neither of which is true of ``records``. Writing the numbers
    out here is the point: a reader can check them against
    ``coverage.searchable_record_count`` without reading the implementation.
    """
    index = _index(
        _speech(1.0, 2.0, "whisper"),
        _speech(3.0, 4.0, "whisper cli"),
        _ocr(5.0, "00:12:34 45%"),
        _ocr(6.0, "   "),
    )
    result = _results(index, "whisper")
    assert _texts(result) == ["whisper", "whisper cli"]
    assert result.results[0].score == pytest.approx(_bm25(1, 1, 1.5, 2, 2), rel=1e-12)
    assert result.results[1].score == pytest.approx(_bm25(1, 2, 1.5, 2, 2), rel=1e-12)
    assert result.coverage.searchable_record_count == 2


def test_a_corpus_of_only_unsearchable_records_is_no_evidence() -> None:
    """The honest state for "transcription failed and the screen was a timer".

    Reporting this as ``no_confident_match`` would blame the query; the corpus is
    what has nothing matchable. The records are still counted, and the check runs
    before the query is even looked at.
    """
    index = _index(
        _ocr(5.0, "00:12:34 45%"),
        _ocr(6.0, "--- /// ..."),
        _ocr(7.0, "the a an"),
    )
    assert len(index.records) == 3
    assert index.searchable_positions == ()
    assert index.average_searchable_length == 0.0
    for query in ("whisper", "the", "45", ""):
        result = _results(index, query)
        assert result.status == "no_evidence", query
        assert result.coverage.record_count == 3
        assert result.coverage.searchable_record_count == 0
