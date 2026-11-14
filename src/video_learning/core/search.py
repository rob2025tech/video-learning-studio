"""Deterministic lexical search over existing study evidence (Search v1, Step 1).

Pure derivation, no I/O, no subprocesses, no database, no model and no new
dependency: it projects one :class:`~video_learning.core.study.StudyEvidence` into
immutable searchable records and ranks them for a natural-language query with a
small, auditable BM25. It searches the evidence the pipeline already produced; it
never re-runs inspection, analysis, transcription or segmentation, and never
mutates an input.

Why a projection and not a second store
---------------------------------------
:class:`~video_learning.core.study.StudyEvidence` is already the canonical,
de-duplicated view of the two searchable sources: every OCR point and speech
interval appears exactly once, either inside a segment or under ``unattributed``
(see :mod:`~video_learning.core.study`). A report, by contrast, carries the same
text twice — once as ``ocr.snapshots``/``transcription.segments`` and again in the
unified ``timeline`` — so indexing a report would double-count every hit. This
module therefore reads the evidence and nothing else.

Projection rules
----------------
* one record per timeline event: every ``SpeechEvent`` and every ``OcrEvent`` of
  every segment, then the unattributed events;
* ``text`` is the event's own string, copied whole — never trimmed, folded,
  summarized or re-spaced;
* ``source_type`` is ``"speech"`` for an interval and ``"ocr"`` for a point;
* ``end_seconds`` is the speech interval's end, and ``None`` for OCR: a sampled
  frame says *when* text was seen, never for how long, so an OCR end would be an
  invention;
* ``segment_id`` is that segment's id, or ``None`` for unattributed evidence,
  which stays searchable rather than being silently dropped.

``evidence_id`` is a per-source ordinal (``speech:0``, ``ocr:2`, …) assigned in
projection order, so ids never renumber across sources. Like ``segment_id`` it is
**artifact-scoped, not durable**: re-transcribing or re-segmenting a video changes
them. Within one artifact, though, it is unique, which is what the sort relies on.

The timestamps are *not* an identity. ``(source, source_type, start_seconds)`` is a
descriptive locator — it names the video, the source and the second — and it is not
unique: frame sampling clamps every midpoint of a sub-second video onto the same
instant (``adapters/ffmpeg_frames.py`` clamps to ``duration - 0.05``), and an OCR
point can share its second with the speech interval that starts there. No key
in this module is claimed to be unique except ``evidence_id``, and no new identity
mechanism (durable id, content hash) is introduced here; that would be a change to
the evidence artifact, not to search.

Tokenization
------------
Words use the ranker's own ``_WORD`` shape (:data:`video_learning.core.keywords._WORD`),
*not* its single-word ``_TOKEN``: ``_TOKEN`` must start with a letter and forbids a
dot, so it would shred ``ggmlmedium.bin``, ``FFmpeg.exit`` and ``whisper.cpp`` —
precisely the highest-value terms in a software tutorial. ``_WORD`` keeps internal
``. / - + #`` intact, and each match is trimmed with the ranker's ``_TRIM_CHARS`` so
a sentence-final period is not part of the term. Trimming shifts the span, so the
recorded offsets point at the trimmed core inside the untouched text.

Each trimmed compound is indexed *twice*: once whole, so ``whisper.cpp`` can be
found exactly, and once per part split on those same separator characters, so the
far more common query ``whisper`` also reaches the on-screen ``whisper.cpp``. This
is word splitting, the same job FTS5's ``simple`` tokenizer does, not stemming:
there is **no stemmer and no synonym table** (both would need a model or a resource
this step must not add), so ``transcribe`` still does not match ``transcribed``.
That is a known recall limit and the measurable reason a later embedding step
exists.

Splitting means a record's token list can hold overlapping spans — ``whisper.cpp``
(0,11), ``whisper`` (0,7), ``cpp`` (8,11) — each of which is a true substring of the
verbatim text. Every span, whole or part, satisfies ``text[start:end] == surface``.

The whole-then-parts rule has one exception, and it is a negation: ``v0.1-don't`` is
indexed whole and its components (``v0``, ``1``, ``don``) are not emitted, because the
head of a contraction is not an ordinary compound. Splitting one would index ``don``
and ``1`` — the same affirmative-fragment error the ``n't`` rule was added for, in a
different costume. Asking for ``v0.1`` therefore does not reach ``v0.1-don't``, which
is the cost of keeping the negation honest rather than a gap in the split.

Apostrophes are not word characters, so ``_WORD`` already cuts ``we're`` into ``we``
and ``re``. Cutting there is wrong twice over: the fragment ``re`` (or ``s``, ``t``,
``ll``) is not a word, and it made the query ``what's`` — ``what`` is a stopword —
look like it carried intent. So a run that continues a preceding letter across an
apostrophe is a clitic tail and is not indexed at all: ``today's`` contributes
``today``, ``grep 'whisper' file`` keeps both ``whisper`` (its apostrophe follows a
space, not a letter) and ``file``, and a head that would be left as a single letter
goes with its tail — ``o'clock`` and ``y'all`` contribute nothing, because a bare
``o`` is exactly the artifact this rule exists to remove. A one-letter run behind a
quote is dropped wherever the quote stands, fused or not, since that is the same
fragment: the ``t`` of ``abc-'t``, the ``s`` of ``Qoder-'s``, and nothing else about
those strings is searchable.

``n't`` is the exception, because stripping it made the opposite mistake. Dropping
the tail left ``won`` standing for ``won't``, and ``won`` is a real, meaningful word
whose sense is inverted: a search for ``won`` ranked the negated clip first. English
has exactly one such suffix, so the test stays structural instead of a word list — a
run ending in ``n``, immediately followed by ``'t`` and then a non-word character, is
indexed as one term, the whole contraction. "Immediately" is load-bearing: the quote
has to touch the trimmed core, because only then do the head and its tail select one
contiguous span. ``won't`` and ``won`` are therefore different terms and cannot borrow
each other's hits, and ``don't``, ``can't``, ``isn't``, ``doesn't``, ``couldn't``,
``shouldn't``, ``mustn't``, ``ain't`` and every other ``n't`` form arrive with them,
since none of those heads are words either. The matching key folds a typographic
apostrophe to the straight one so ``won't`` finds ``won’t``; ``surface`` and the
offsets stay verbatim, because the evidence a reader is shown must be what the
evidence said.

A negation the evidence failed to write — ``won''t``, ``won' t``, ``won ’ t``,
``won'ts``, ``won-'t``, and any of those with one of the ranker's own separator
characters (``. - / # +``) directly after the fragment — is
suppressed rather than repaired: the head would become the affirmative and the
leftover ``t`` would become intent, so neither is indexed and the fragment is
swallowed with the head. Suppression is by shape, not by exhaustive list, and two
boundaries come with it: the quotes it recognizes are ``'`` and the typographic
``’``, so an apostrophe written any other way is a separator and the head stands as
``won``, and the whitespace it skips includes line breaks, so a broken negation at the
end of a caption line can suppress the word before it. See
:func:`_malformed_negation` for both.

Ranking
-------
Textbook Okapi BM25 with the constants below. The candidate set is the records
containing at least one *meaningful* query term, so a non-matching record is never
scored 0.0 and shown. A meaningful query term contains a letter and is not one of
the project's stopwords; a query made only of stopwords, or of digits such as a
``12:34:56`` timer, therefore yields no confident match — the same reason the
keyword ranker refuses letter-free fragments.

BM25 scores rank documents; they are **not** calibrated confidence, so there is no
score threshold and none should be added. The honest states are structural.

The same "meaningful" test also decides which records count as *documents* at all. A
record whose every token is a stopword, a digit run or punctuation — a blank frame, a
timer, a strip of UI chrome — contributes nothing to ``N``, to the average document
length or to the document-frequency tally. IDF grows with ``N``, so leaving
unsearchable records in the denominator makes every real score depend on how much
dead screen text the video happened to contain. Those records are still projected,
still listed in ``record_count``, and simply never reachable; ``searchable_positions``
is the one list all three statistics, and the candidate scan, are derived from.

Result states
-------------
``no_evidence`` — no projected record carries a searchable term: no events at all,
or transcription unavailable *and* OCR that is only blanks, timers and chrome.
``no_confident_match`` — searchable evidence exists but shares no meaningful term
with the query. ``ok`` — at least one record overlaps. The counts stay populated in
every state, so an empty answer can say what it was empty *of*.

Known limits, stated rather than discovered later
--------------------------------------------------
* No stemming and no synonyms, so ``transcribe`` will not reach ``transcribed`` and
  "privacy" will not reach "nothing leaves the machine". Recall is lexical, full
  stop.
* A compound query contributes every one of its parts, so asking for
  ``video-learning-studio`` scores a short hit far higher than asking for
  ``studio`` does. Scores are comparable *within* one query, never across queries,
  which is the second reason there is no threshold.
* ``evidence_id`` and ``segment_id`` identify rows inside one artifact only; they
  move when the video is re-transcribed or re-segmented. ``(source, source_type,
  start_seconds)`` does not identify anything: duplicate timestamps are reachable
  and it is a locator, not a key.
* A negative contraction is a term to itself, so ``won't`` will not reach ``will
  not``, and ``dont`` typed without the apostrophe reaches only text that really
  spells ``dont``. Either collapse would be stemming or synonym expansion, both out
  of scope here.
* A negation the evidence wrote badly — ``won''t``, ``won' t``, ``won ’ t``,
  ``won'ts``, ``won-'t``, and any of those with ``.``, ``-``, ``/``, ``#`` or ``+``
  directly after the fragment — is dropped entirely, head included, rather than guessed
  at. The safe reading of ``won ’ t`` is "not won", and the unsafe one is "won";
  indexing neither is the only answer this step can give without a repair rule. What is
  recognized is the shape, not a list of spellings: a quote in ``'`` or ``’``, then
  nothing but a ``t`` or ``ts`` fragment. An apostrophe written any other way is a
  separator, so ``won‘t`` still indexes the affirmative ``won`` and a stray ``t``; so does
  a tail with a letter glued to it (``won'tx``, ``don'tknow``), which is a different word
  rather than a fragment this rule can judge. The
  collateral is narrow but real: an ``n``-terminated word quoted around a lone ``t``
  disappears with it, so neither ``Charlton 't Heston`` nor ``token 't' here`` is
  searchable as written, and the whitespace skipped on the way to the quote includes
  line breaks, so the last word of a caption line can be dropped by a quoted ``t`` that
  opens the next one. ``'twas``, ``'to`` and every other longer quoted run are
  untouched.
* ``o'clock`` and ``y'all`` are unsearchable rather than wrong: the tail is a clitic
  and the one-letter head goes with it, so ``clock`` finds nothing in ``five
  o'clock``. A tail an apostrophe joins without a space is dropped the same way on its
  own — ``whisper'cli`` keeps ``whisper`` and loses ``cli`` — and so is a one-letter run
  sitting behind a quote, as in ``Qoder-'s`` or ``the '-t flag``. A spaced
  ``ca n't`` is two runs, so it becomes neither ``can`` nor ``can't``: the lone ``n``
  is dropped as a one-letter head.
* The negation rule ends compound expansion, so ``v0.1-don't`` indexes no ``v0.1`` and
  no ``don``. See Tokenization above for why splitting it would reintroduce the very
  affirmative term the rule removes.
* One video per index. Cross-video retrieval is a different problem and is not
  approximated here.

Determinism
-----------
No ``random``, no wall-clock, no ``hash()``-derived order, no sort without a total
order, and no set whose iteration order can reach an output: the one set
comprehension here only tallies per-document term frequencies, whose values do not
depend on order. Query terms are summed in a fixed first-occurrence order because
floating-point addition is not associative, so two runs of the same query cannot
differ in the last bit. Dictionaries are keyed by term and only ever looked up,
never iterated to produce output.

The result order is ``(score descending, start_seconds ascending, evidence_id
ascending)``, and the last component is a **lexicographic string compare**, not a
numeric one: ``ocr:0`` sorts before ``speech:0``, and ``speech:10`` before
``speech:2``. That is the rule, not an accident to be tidied — a reader recomputing a
ranking must expect id order, not ordinal order. It is reached only between records
that already tie on score *and* start (which duplicate timestamps do allow), and
because ``evidence_id`` is unique inside one artifact the order is total.

Out of scope here, and not introduced: persistence, ``from_dict``, SQLite/FTS5,
embeddings or any model, reciprocal-rank fusion, query routing, object detection,
CLI and UI, and any change to the report, study-evidence, guide, quiz or
segmentation behavior.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from video_learning.core.keywords import _STOPWORDS, _TRIM_CHARS, _WORD
from video_learning.core.study import STUDY_EVIDENCE_SCHEMA, StudyEvidence
from video_learning.core.timeline import OcrEvent, SpeechEvent

SEARCH_SCHEMA = "video-learning.search/v1"

SearchSourceType: TypeAlias = Literal["speech", "ocr"]
SearchStatus: TypeAlias = Literal["ok", "no_confident_match", "no_evidence"]

#: BM25 term-frequency saturation and length normalization. Fixed, documented
#: constants rather than tuned parameters: any value here is a convention, and a
#: reader must be able to recompute a score from the artifact alone.
_K1 = 1.5
_B = 0.75

#: How many ranked results a query returns by default. A presentation cap only;
#: ``matched_record_count`` in the coverage reports how many were available.
DEFAULT_RESULT_LIMIT = 5

# The ranker's word shape, compiled once. Keeps internal separators so paths,
# dotted filenames and version-like tokens stay whole.
_WORD_RE = re.compile(_WORD)
# The complement of the same set: the pieces a compound splits into. Reusing
# ``_TRIM_CHARS`` (rather than inventing a second separator list) is what makes the
# whole term and its parts add up to the same string.
_PART_RE = re.compile(f"[^{re.escape(_TRIM_CHARS)}]+")
# Apostrophes are not word characters, so ``_WORD`` splits ``we're`` at the quote and
# leaves the pieces after it — ``re``, ``s``, ``t``, ``ll``, ``d``, ``ve`` — to be
# rejected as what they are: contraction and possessive tails. Straight and typographic
# quotes both count, because a transcript can carry either. A tuple, not a string, so
# ``startswith`` sees two alternatives rather than one two-character sequence.
_CLITIC_QUOTES = ("'", "’")
# The only English negation suffix, and the one tail whose head is a real word: ``n``
# plus ``'t``. It is indexed whole so ``won`` and ``won't`` stay separate terms.
_NEGATION_TAIL = "t"
# Every negation term is keyed with this, whichever quote the evidence used, so a
# straight query finds curly evidence and back. Surfaces and spans stay verbatim.
_CANONICAL_NEGATION = "'t"
# The fragments a broken ``n't`` leaves behind: the ``t`` on its own, and that ``t``
# with a stray ``s`` glued to it (``won'ts``). Two spellings of one artifact, both
# derived from :data:`_NEGATION_TAIL` rather than collected as a word list, and narrow
# enough that ``'twas`` and ``'to`` keep their meaning however they were quoted.
_NEGATION_FRAGMENTS = (_NEGATION_TAIL, f"{_NEGATION_TAIL}s")


@dataclass(frozen=True)
class Token:
    """One indexed term with the span it occupies in the untouched evidence text.

    ``term`` is the case-folded, separator-trimmed core used for matching;
    ``surface`` is what the evidence actually wrote, kept so a caller can display
    a match without re-deriving it from the offsets. ``text[start:end] == surface``
    holds for every token, including a compound's parts, whose spans nest inside the
    whole compound's span, and including a negative contraction, whose span covers
    the apostrophe too. ``term`` is ``surface`` lower-cased for every token except a
    negative contraction, which also folds a typographic apostrophe to the straight
    one so the two quote styles match; its surface still quotes the evidence exactly.
    """

    term: str
    surface: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {"term": self.term, "surface": self.surface, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class SearchEvidence:
    """One searchable unit of evidence: an OCR point or a speech interval.

    ``text`` is verbatim. Nothing in this record is inferred: there is no frame
    reference because none survives frame extraction, no confidence because neither
    backend reports one, and no duration for OCR because it was never measured.
    """

    evidence_id: str
    source_type: SearchSourceType
    text: str
    start_seconds: float
    end_seconds: float | None
    segment_id: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "text": self.text,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "segment_id": self.segment_id,
        }


@dataclass(frozen=True)
class SearchMatch:
    """A ranked record with the terms that earned it its place.

    ``matched_terms`` are in first-occurrence order inside the evidence text and
    ``matched_spans`` are sorted by offset, so both can be rendered without
    touching ``evidence.text``. The spans index the verbatim text, which is why the
    record itself stays immutable. A compound and one of its parts can both match,
    so spans may nest; a caller highlighting them should paint longest-first.
    """

    evidence: SearchEvidence
    score: float
    matched_terms: tuple[str, ...]
    matched_spans: tuple[Token, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.evidence.to_dict(),
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "matched_spans": [span.to_dict() for span in self.matched_spans],
        }


@dataclass(frozen=True)
class SearchCoverage:
    """What was indexed and what the query reached, expressed only as counts.

    Each count names exactly what it counts, because the three query numbers mean
    different things on purpose: ``query_token_count`` is every token the tokenizer
    produced from the query, including the parts a compound split into, while
    ``meaningful_query_term_count`` is the distinct tokens with actual search intent
    behind them — the set BM25 scored. The gap between them is how a caller can see
    that a query was all UI chrome and stopwords without reading the results.
    ``record_count`` and ``searchable_record_count`` are the same pair seen from the
    corpus side: projected records, and the subset of them that carries a meaningful
    term and therefore took part in the statistics. A ``no_evidence`` result with a
    non-zero ``record_count`` says precisely what went wrong. ``matched_record_count``
    is how many records passed the candidate test and ``returned_record_count`` how
    many the limit allowed back, so a capped answer never looks like a total one.
    """

    record_count: int
    searchable_record_count: int
    speech_record_count: int
    ocr_record_count: int
    attributed_record_count: int
    unattributed_record_count: int
    query_token_count: int
    meaningful_query_term_count: int
    matched_record_count: int
    returned_record_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "searchable_record_count": self.searchable_record_count,
            "speech_record_count": self.speech_record_count,
            "ocr_record_count": self.ocr_record_count,
            "attributed_record_count": self.attributed_record_count,
            "unattributed_record_count": self.unattributed_record_count,
            "query_token_count": self.query_token_count,
            "meaningful_query_term_count": self.meaningful_query_term_count,
            "matched_record_count": self.matched_record_count,
            "returned_record_count": self.returned_record_count,
        }


@dataclass(frozen=True)
class SearchResults:
    """One query's outcome against one index, self-describing and self-limiting.

    The provenance fields are carried verbatim from the indexed
    :class:`StudyEvidence`, so a reader of this artifact alone can tell that a
    result set is OCR-heavy because transcription was unavailable, and can name the
    video every timestamp belongs to.
    """

    query: str
    status: SearchStatus
    results: tuple[SearchMatch, ...]
    coverage: SearchCoverage
    source: Path
    study_evidence_schema: str
    transcription_status: str
    transcription_available: bool
    video_duration_seconds: float | None
    limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SEARCH_SCHEMA,
            "query": self.query,
            "status": self.status,
            "source": str(self.source),
            "study_evidence_schema": self.study_evidence_schema,
            "transcription_status": self.transcription_status,
            "transcription_available": self.transcription_available,
            "video_duration_seconds": self.video_duration_seconds,
            "limit": self.limit,
            "results": [match.to_dict() for match in self.results],
            "result_count": len(self.results),
            "coverage": self.coverage.to_dict(),
        }


@dataclass(frozen=True)
class SearchIndex:
    """A whole-video lexical index built once from :class:`StudyEvidence`.

    ``records``, ``record_tokens`` and ``doc_lengths`` are parallel and cover *every*
    projected record, including ones nothing can match. ``searchable_positions`` is
    the subset carrying at least one meaningful term, and is the only source of
    candidates: ``doc_freq`` and ``average_searchable_length`` are computed over it,
    never over ``records``, so unsearchable screen text cannot move a score. Holding a
    linear token list rather than an inverted index is deliberate: one video carries
    tens to a few thousand records, a scan is trivially fast at that size, and it keeps
    the scoring path short enough to audit line by line.
    """

    records: tuple[SearchEvidence, ...]
    record_tokens: tuple[tuple[Token, ...], ...]
    searchable_positions: tuple[int, ...]
    doc_freq: Mapping[str, int]
    doc_lengths: tuple[int, ...]
    average_searchable_length: float
    source: Path
    study_evidence_schema: str
    transcription_status: str
    transcription_available: bool
    speech_event_count: int
    video_duration_seconds: float | None


def tokenize(text: str) -> tuple[Token, ...]:
    """Split ``text`` into terms with spans into the original string (pure).

    Order is left-to-right by match position, and within one match the whole
    compound comes before its parts, so ``matched_terms`` and the term-frequency
    scan are reproducible without sorting. Parts whose span equals the whole
    trimmed core are not emitted twice. An apostrophe tail contributes nothing, a
    ``n't`` pair contributes exactly one term, and a one-letter clitic head
    contributes nothing — see the predicates below. A malformed ``n't`` contributes
    nothing at all and absorbs the fragment it left behind, which is a shape test on
    the quote and the fragment's width rather than a list of spellings.
    """
    tokens: list[Token] = []
    # Start position of the last character a broken negation swallowed. Only ever
    # moves forward, because the scan is left-to-right.
    absorbed_until = 0
    for match in _WORD_RE.finditer(text):
        if match.start() < absorbed_until:
            continue
        raw = match.group()
        core = raw.strip(_TRIM_CHARS)
        if not core:
            continue
        if _is_clitic_tail(text, match.start(), len(core)):
            continue
        # Trimming moves the start and shortens the run; the span must still land
        # on the untouched text, so it is shifted by the leading separators removed.
        start = match.start() + len(raw) - len(raw.lstrip(_TRIM_CHARS))
        following = text[match.end() :]
        if _is_single_letter_fragment(core, following):
            continue
        if core.lower().endswith("n"):
            # The quote has to touch the trimmed core. A separator in between
            # (``won-'t``) means the two characters are no longer adjacent text, so
            # no span can name both and this is a malformed negation instead.
            negation = _negation_tail(core, following) if raw == core else None
            if negation is not None:
                # One term for the whole contraction, which is what keeps ``won`` from
                # borrowing ``won't``'s hits. Only the key canonicalizes the quote: the
                # surface is the text this span selects, verbatim and contiguous.
                surface = f"{core}{negation}"
                tokens.append(
                    Token(
                        term=f"{core.lower()}{_CANONICAL_NEGATION}",
                        surface=surface,
                        start=start,
                        end=start + len(surface),
                    )
                )
                continue
            absorbed = _malformed_negation(following)
            if absorbed is not None:
                # Neither the head nor its fragment can be trusted: index ``won`` and
                # a negated clip answers an affirmative query, index ``t`` and a
                # one-letter artifact is search intent. Suppressing both is the honest
                # answer; repairing it to ``won't`` would be guessing.
                absorbed_until = match.end() + absorbed
                continue
        tokens.append(Token(term=core.lower(), surface=core, start=start, end=start + len(core)))
        for part in _PART_RE.finditer(core):
            if part.group() == core:
                continue  # an ordinary word is already indexed as itself
            offset = start + part.start()
            tokens.append(
                Token(
                    term=part.group().lower(),
                    surface=part.group(),
                    start=offset,
                    end=offset + len(part.group()),
                )
            )
    return tuple(tokens)


def _is_clitic_tail(text: str, start: int, length: int) -> bool:
    """Is the run at ``start`` an apostrophe's remainder rather than a word?

    One structural test, applied identically to a query and to evidence, which is why
    ``today's`` matches both ``today's`` and ``today``. An alphanumeric immediately
    before the quote means the quote fused this run to the word before it, so the
    ``'re`` of ``we're`` and the ``'s`` of ``Qoder's`` never become terms. Anything
    else before the quote means quoting — ``grep 'whisper' file`` keeps ``whisper``,
    whose apostrophe follows a space. A run at the very start of the text, or after a
    single leading quote (``'twas``), has no preceding word to fuse with and is
    indexed normally.

    A one-letter run is the exception, and is dropped wherever an apostrophe stands in
    front of it, separator gaps included. ``length`` is the trimmed core, so the ``t``
    of ``'-t`` counts as one letter even though ``_WORD`` swept the hyphen into the
    run. ``t``, ``s`` and ``re`` are not words and never carry intent; ``a`` and ``i``,
    the only single letters English searches for, are stopwords anyway. So the
    fragment left by ``abc-'t`` or ``the '-t flag`` contributes nothing, while ``-m``
    keeps its ``m`` — the only thing behind it is a space. The cost is a lone flag
    letter shown quoted on screen (``grep 'm'``), which is not how the flag is usually
    written down.

    The ``t`` of an ``n't`` contraction arrives here as well, and is meant to: the
    head has already absorbed it into one term, so indexing it again would
    double-count it.
    """
    if length == 1:
        return _behind_a_quote(text, start)
    return start >= 2 and text[start - 1] in _CLITIC_QUOTES and text[start - 2].isalnum()


def _behind_a_quote(text: str, start: int) -> bool:
    """Does an apostrophe open onto the run at ``start``, separators in between aside?

    Walking back over the ranker's own trim characters is what makes ``'-t`` and ``'t``
    the same case. Stopping at anything else — a space, a letter, the start of the
    text — is what keeps ``-m`` and ``whisper.cpp`` from being read as fragments.
    """
    index = start - 1
    while index >= 0 and text[index] in _TRIM_CHARS:
        index -= 1
    return index >= 0 and text[index] in _CLITIC_QUOTES


def _is_single_letter_fragment(core: str, following: str) -> bool:
    """Is this a one-letter run that only ever appears joined to an apostrophe?

    The ``n`` of a spaced ``ca n't`` would otherwise stand alone as a term, and a bare
    letter is precisely the artifact the clitic rule exists to remove, so the head goes
    with the tail it was torn from. The test is the apostrophe itself, not a word list,
    so ``bin``, the ``-m`` flag and every dotted filename pass through untouched.
    """
    return len(core) == 1 and following.startswith(_CLITIC_QUOTES)


def _negation_tail(core: str, following: str) -> str | None:
    """Return the verbatim ``'t`` that turns this head into a negative contraction.

    The caller has already established the two preconditions that make the answer
    trustworthy: ``core`` ends in ``n``, and the quote touches the trimmed core. That
    second one is what lets the span be honest — the two characters returned here are
    the text immediately after the head, so head and tail select one contiguous run of
    the evidence with nothing trimmed out of the middle.

    ``n`` is the only letter English joins to an apostrophe-``t`` for negation, so a
    head ending in it plus an adjacent ``'t`` and then a non-word character is a shape
    test rather than a vocabulary: ``don't``, ``won't``, ``can't``, ``isn't``,
    ``doesn't``, ``couldn't``, ``shouldn't``, ``mustn't``, ``ain't``, ``aren't`` and
    ``wasn't`` all arrive on the same three conditions, while ``Lincoln's`` (quote
    plus ``s``) and ``don'tknow`` (a tail longer than ``t``) do not. The two verbatim
    characters come back so the caller can span them honestly.
    """
    if not core.lower().endswith("n") or len(following) < 2:
        return None
    if following[0] not in _CLITIC_QUOTES or following[1].lower() != _NEGATION_TAIL:
        return None
    if len(following) > 2 and _WORD_RE.match(following, 2):
        return None  # the quote opens a longer word; it does not close a contraction
    return following[:2]


def _malformed_negation(following: str) -> int | None:
    """How far into ``following`` a broken ``n't`` reaches, or ``None``.

    Three shapes, all of them a negation that the evidence failed to write: a second
    quote (``won''t``), whitespace where the letter should follow its apostrophe
    (``won' t``, ``won ’ t``), or a stray letter on the tail (``won'ts``). None of them
    can be indexed as the head, because ``won`` is the affirmative this rule exists to
    keep apart, and none can be indexed as the fragment, because a lone ``t`` is not
    intent. Suppressing the whole region is the conservative answer; rewriting it to
    ``won't`` would be repair, which this step does not do.

    The list is a sample of the shapes rather than a guarantee about all of them: what
    this recognizes is one or more quotes from :data:`_CLITIC_QUOTES` and then a
    fragment the width of ``t`` or ``ts``, the ranker's own separator punctuation aside.
    An apostrophe spelled with any other code point, or a fragment glued to a following
    word across its punctuation, is not a negation this function can see, and the head
    is indexed as the affirmative.

    Only an ``n``-terminated head reaches here, and only a fragment the width of ``t``
    or ``ts`` counts, so a quoted word survives untouched: ``Eisen 'twas`` keeps both
    runs and ``Lincoln's`` keeps its name. The same width test is what takes a quoted
    one-letter identifier with its head, so ``token 't' here`` loses ``token``, and the
    whitespace skipped on the way there includes newlines — a head at the end of one
    caption line and a quoted ``t`` at the start of the next read as one broken
    negation. Returns the length of the swallowed region, which is the span the caller
    must skip so the fragment cannot be indexed twice.
    """
    index = 0
    while index < len(following) and following[index].isspace():
        index += 1
    quotes = 0
    while index < len(following) and following[index] in _CLITIC_QUOTES:
        quotes += 1
        index += 1
        while index < len(following) and following[index].isspace():
            index += 1
    if not quotes:
        return None
    run = _WORD_RE.match(following, index)
    if run is None:
        return None
    # ``_WORD_RE`` is greedy about the ranker's own trim characters, so a sentence-final
    # period arrives as ``t.`` and would fail an exact comparison: the broken negation
    # then indexed its head as the affirmative, which is the one outcome this rule
    # exists to prevent. The boundary stays the whole run, which is only ever the
    # fragment plus separator characters — a word glued on past the punctuation makes
    # the comparison fail instead of reaching further.
    if run.group().strip(_TRIM_CHARS).lower() not in _NEGATION_FRAGMENTS:
        return None
    return run.end()


def _is_meaningful(token: Token) -> bool:
    """Whether this token can carry search intent — asked of queries *and* records.

    One predicate serves both sides on purpose: a record is reachable exactly when a
    query asks for one of its meaningful terms, so the index statistics and the query
    terms can never drift apart and disagree about what is searchable.

    The length floor the phrase ranker applies is deliberately absent: it exists to
    suppress OCR garble in candidate phrases, whereas a short query word such as
    ``bin`` is real intent. Letter-free runs (a ``12:34:56`` timer, a frame count)
    are rejected for the same reason the keyword ranker rejects them as meaningless.
    """
    return any(char.isalpha() for char in token.term) and token.term not in _STOPWORDS


def project_evidence(evidence: StudyEvidence) -> tuple[SearchEvidence, ...]:
    """Project study evidence into searchable records, in deterministic order.

    Segments are walked in their ascending chronological order (the input contract
    :func:`~video_learning.core.study.build_study_evidence` documents), each segment
    contributing its OCR points then its speech intervals, and the unattributed
    events come last so they remain retrievable with ``segment_id=None``.
    """
    records: list[SearchEvidence] = []
    ordinals: dict[str, int] = {"ocr": 0, "speech": 0}

    for segment in evidence.segments:
        for on_screen in segment.ocr_events:
            records.append(_record("ocr", on_screen, segment.segment_id, ordinals))
        for spoken in segment.transcript_events:
            records.append(_record("speech", spoken, segment.segment_id, ordinals))
    for unassigned_screen in evidence.unattributed.ocr_events:
        records.append(_record("ocr", unassigned_screen, None, ordinals))
    for unassigned_speech in evidence.unattributed.transcript_events:
        records.append(_record("speech", unassigned_speech, None, ordinals))

    return tuple(records)


def _record(
    source_type: SearchSourceType,
    event: OcrEvent | SpeechEvent,
    segment_id: int | None,
    ordinals: dict[str, int],
) -> SearchEvidence:
    """One record for one timeline event, with that event's own timestamps."""
    ordinal = ordinals[source_type]
    ordinals[source_type] = ordinal + 1
    if isinstance(event, OcrEvent):
        # A point observation: no end is known, so none is asserted.
        return SearchEvidence(
            evidence_id=f"ocr:{ordinal}",
            source_type="ocr",
            text=event.text,
            start_seconds=event.timestamp_seconds,
            end_seconds=None,
            segment_id=segment_id,
        )
    return SearchEvidence(
        evidence_id=f"speech:{ordinal}",
        source_type="speech",
        text=event.text,
        start_seconds=event.start_seconds,
        end_seconds=event.end_seconds,
        segment_id=segment_id,
    )


def build_search_index(evidence: StudyEvidence) -> SearchIndex:
    """Tokenize every projected record once and gather the corpus statistics.

    Every record keeps its tokens and its length — a blank frame is still evidence a
    caller can point at — but only the records with a meaningful term feed the
    numbers BM25 reads.
    """
    records = project_evidence(evidence)
    record_tokens = tuple(tokenize(record.text) for record in records)
    doc_lengths = tuple(len(tokens) for tokens in record_tokens)
    searchable_positions = tuple(
        position
        for position, tokens in enumerate(record_tokens)
        if any(_is_meaningful(token) for token in tokens)
    )

    # Keyed by term, read back by term only. Iterating it would be nondeterministic
    # across processes, so no output order is ever taken from this mapping.
    # Counted over searchable records only, which is the df that pairs with the N in
    # _inverse_doc_freq; a score has to be recomputable from these two numbers alone.
    doc_freq: dict[str, int] = {}
    for position in searchable_positions:
        for term in {token.term for token in record_tokens[position]}:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    total_tokens = sum(doc_lengths[position] for position in searchable_positions)
    average = total_tokens / len(searchable_positions) if searchable_positions else 0.0
    return SearchIndex(
        records=records,
        record_tokens=record_tokens,
        searchable_positions=searchable_positions,
        doc_freq=doc_freq,
        doc_lengths=doc_lengths,
        average_searchable_length=average,
        source=evidence.source,
        study_evidence_schema=STUDY_EVIDENCE_SCHEMA,
        transcription_status=evidence.transcription_status,
        transcription_available=evidence.transcription_available,
        speech_event_count=evidence.speech_event_count,
        video_duration_seconds=evidence.video_duration_seconds,
    )


def _inverse_doc_freq(index: SearchIndex, term: str) -> float:
    """The Robertson-Sparck Jones IDF used by Okapi BM25, always positive.

    ``N`` is the searchable-record count, the same denominator ``doc_freq`` was
    tallied over, so a score is recomputable from the two numbers in this artifact.
    """
    total = len(index.searchable_positions)
    frequency = index.doc_freq.get(term, 0)
    return math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))


def _score_document(index: SearchIndex, position: int, query_terms: tuple[str, ...]) -> float:
    """BM25 score of one searchable record against the meaningful query terms.

    ``query_terms`` is an ordered tuple, deliberately not a set: the summation
    order below is part of the score's last bits, and a set's iteration order is
    not reproducible across processes.

    ``position`` comes from ``searchable_positions``, so its length is non-zero and
    the average it normalizes against is at least 1.0 by construction. There is
    therefore no empty-document fallback: one would silently absorb a caller that
    passed the wrong list, and no caller can score an unsearchable record.
    """
    tokens = index.record_tokens[position]
    length = index.doc_lengths[position]

    norm = _K1 * (1.0 - _B + _B * (length / index.average_searchable_length))
    score = 0.0
    for term in query_terms:
        term_frequency = sum(1 for token in tokens if token.term == term)
        if not term_frequency:
            continue
        weighted = term_frequency * (_K1 + 1.0) / (term_frequency + norm)
        score += _inverse_doc_freq(index, term) * weighted
    return score


def search(
    index: SearchIndex, query: str, *, limit: int = DEFAULT_RESULT_LIMIT
) -> SearchResults:
    """Rank ``index``'s records for ``query`` (pure, deterministic, read-only).

    Only searchable records are scanned, so ``no_evidence`` here means "nothing in
    this corpus is matchable", not "nothing was projected" — ``record_count`` still
    reports the blank and timer-only frames that were.

    ``limit`` caps what is returned, never what is considered: the coverage reports
    the full matched count. A non-positive ``limit`` yields no results but still
    reports the status and counts, which is what an "explain your empty answer"
    surface needs.
    """
    query_tokens = tokenize(query)
    # Distinct meaningful terms, in first-occurrence order: this ordering is what
    # the scorer sums in, and what the coverage reports, so nothing downstream has
    # to guess whether a set was involved.
    terms: list[str] = []
    for token in query_tokens:
        if _is_meaningful(token) and token.term not in terms:
            terms.append(token.term)
    query_terms = tuple(terms)

    def results(
        status: SearchStatus, matches: tuple[SearchMatch, ...], matched: int
    ) -> SearchResults:
        return SearchResults(
            query=query,
            status=status,
            results=matches,
            coverage=SearchCoverage(
                record_count=len(index.records),
                searchable_record_count=len(index.searchable_positions),
                speech_record_count=sum(1 for r in index.records if r.source_type == "speech"),
                ocr_record_count=sum(1 for r in index.records if r.source_type == "ocr"),
                attributed_record_count=sum(1 for r in index.records if r.segment_id is not None),
                unattributed_record_count=sum(
                    1 for r in index.records if r.segment_id is None
                ),
                query_token_count=len(query_tokens),
                meaningful_query_term_count=len(query_terms),
                matched_record_count=matched,
                returned_record_count=len(matches),
            ),
            source=index.source,
            study_evidence_schema=index.study_evidence_schema,
            transcription_status=index.transcription_status,
            transcription_available=index.transcription_available,
            video_duration_seconds=index.video_duration_seconds,
            limit=limit,
        )

    if not index.searchable_positions:
        # Nothing in the corpus can be matched, whether or not records were projected:
        # blank and timer-only OCR is evidence, but it is not *searchable* evidence.
        return results("no_evidence", (), 0)
    if not query_terms:
        return results("no_confident_match", (), 0)

    scored: list[tuple[float, SearchEvidence, tuple[Token, ...]]] = []
    for position in index.searchable_positions:
        hits = tuple(token for token in index.record_tokens[position] if token.term in query_terms)
        if not hits:
            continue  # no meaningful overlap at all: never scored, never shown
        score = _score_document(index, position, query_terms)
        scored.append((score, index.records[position], hits))
    if not scored:
        return results("no_confident_match", (), 0)

    # Total order: score desc, then chronologically, then the id as a *string*, so
    # speech:10 precedes speech:2. Ids are unique inside one artifact, so nothing is
    # left to Python's sort stability to decide.
    scored.sort(key=lambda item: (-item[0], item[1].start_seconds, item[1].evidence_id))
    matches = tuple(
        SearchMatch(
            evidence=record,
            score=score,
            matched_terms=_unique_terms(hits),
            matched_spans=tuple(sorted(hits, key=lambda token: (token.start, token.end))),
        )
        for score, record, hits in scored[: max(limit, 0)]
    )
    return results("ok", matches, len(scored))


def _unique_terms(tokens: tuple[Token, ...]) -> tuple[str, ...]:
    """Distinct matched terms in first-occurrence order inside the evidence."""
    terms: list[str] = []
    for token in tokens:
        if token.term not in terms:
            terms.append(token.term)
    return tuple(terms)
