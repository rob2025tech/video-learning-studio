"""Service-boundary tests for Step 2's ``SearchService`` (Multimodal Evidence Search v1).

Nothing here touches ffmpeg, tesseract, whisper.cpp or the video pipeline, and nothing
here re-tests the ranker: the tokenizer, BM25 scores, ordering and status rules are
pinned by ``tests/unit/test_search.py`` against the core. What these tests own is the
seam — that the service builds one index, delegates every query to it unchanged, keeps
the evidence's provenance intact, cannot reach a pipeline, and never mutates what it
retains.

The evidence is real, not a stand-in: every fixture is a hand-authored timeline run
through :func:`~video_learning.core.study.build_study_evidence`, exactly as the Task 1,
Task 2 and Step 1 suites do, so the service is exercised on the object shape production
produces.

The properties protected:

* **One index, many queries** — ``build_search_index`` runs once per service no matter
  how many queries follow, and the queries are answered by that same object.
* **Delegation, not a second implementation** — every result equals a direct core
  ``search`` call on the same index, down to the serialized document.
* **Pass-through, not reinterpretation** — all three statuses, the coverage counts and
  the ``limit`` semantics arrive exactly as the core decided them.
* **Provenance survives the trip** — source, transcription fields, duration, stamped
  schema, and per-match timestamps and attribution, including OCR's absent end time and
  an unattributed event's ``segment_id=None``.
* **Read-only** — the retained index is byte-for-byte the same object after any number
  of queries, two artifacts stay in two separate indexes, and no name that could run
  media processing is even reachable from the service module.
* **The call shape is pinned** — the signature the Step 3 CLI is written against,
  including ``limit`` staying keyword-only and defaulting to the core's constant.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import video_learning.core.study as study_core
from video_learning.core.search import (
    DEFAULT_RESULT_LIMIT,
    SearchIndex,
    build_search_index,
    search,
)
from video_learning.core.segments import BoundaryReason, Segment, Segmentation
from video_learning.core.study import STUDY_EVIDENCE_SCHEMA, StudyEvidence, build_study_evidence
from video_learning.core.timeline import OcrEvent, SpeechEvent, TimelineEvent
from video_learning.services import search_service
from video_learning.services.search_service import SearchService

_SOURCE = Path("/tmp/sample.mp4")
_OTHER_SOURCE = Path("/tmp/other-sample.mp4")


def _ocr(seconds: float, text: str) -> OcrEvent:
    return OcrEvent(timestamp_seconds=seconds, text=text)


def _speech(start: float, end: float, text: str) -> SpeechEvent:
    return SpeechEvent(start_seconds=start, end_seconds=end, text=text)


def _segment(start: float, end: float, reason: BoundaryReason | None) -> Segment:
    return Segment(start_seconds=start, end_seconds=end, boundary_reason=reason)


def _evidence(
    timeline: Sequence[TimelineEvent],
    *,
    source: Path = _SOURCE,
    segments: list[Segment] | None = None,
    duration: float | None = 60.0,
    status: str = "ok",
    detail: str | None = None,
) -> StudyEvidence:
    """Derive real ``StudyEvidence`` from a hand-authored timeline.

    The default windows tile ``[0, 60)`` so every fixture event lands inside a segment;
    an event at or beyond the end falls to ``unattributed``, which is the shape the
    pipeline produces when a timestamp cannot be attributed. ``source`` varies only to
    give the two-artifact test two genuinely different artifacts.
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
    return build_study_evidence(segmentation, timeline, status, detail, source=source)


def _mixed_evidence() -> StudyEvidence:
    """One artifact carrying both modalities, an unattributed event, and dead frames."""
    return _evidence(
        [
            _ocr(5.0, "whisper.cpp model usage"),
            _speech(6.0, 12.0, "we transcribe audio locally with whisper"),
            _ocr(20.0, "12:34:56"),
            _ocr(90.0, "local-first offline transcription notes"),
        ]
    )


def _digest(index: SearchIndex) -> dict[str, Any]:
    """Everything a query could silently change about a built index, as plain data.

    Compared before and after a batch of queries. ``doc_freq`` is copied, not iterated,
    so this reads the same way the scorer does — by key.
    """
    return {
        "record_ids": [record.evidence_id for record in index.records],
        "record_texts": [record.text for record in index.records],
        "terms": [[token.term for token in row] for row in index.record_tokens],
        "spans": [
            [(token.start, token.end, token.surface) for token in row]
            for row in index.record_tokens
        ],
        "doc_lengths": list(index.doc_lengths),
        "searchable_positions": list(index.searchable_positions),
        "doc_freq": dict(index.doc_freq),
        "average_searchable_length": index.average_searchable_length,
    }


# ---------------------------------------------------------------------------
# 1. The index is built once, and every query is answered from that one object.
# ---------------------------------------------------------------------------


def test_the_index_is_built_once_for_many_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four queries, one tokenization pass — the whole reason this service exists.

    Counting the constructor's call is the direct check on "multiple queries without
    re-running processing": the expensive work in this feature is projection plus
    tokenization, and it must not be reachable from a query at all.
    """
    built: list[SearchIndex] = []
    real = search_service.build_search_index

    def counting(evidence: StudyEvidence) -> SearchIndex:
        index = real(evidence)
        built.append(index)
        return index

    monkeypatch.setattr(search_service, "build_search_index", counting)
    service = SearchService(_mixed_evidence())
    results = [service.query(term) for term in ("whisper", "transcribe", "model", "notes")]

    assert len(built) == 1
    assert service.index is built[0], "the retained index is the one that was built"
    assert service.index is service.index, "the property hands back one object, not a copy"
    assert [result.status for result in results] == ["ok", "ok", "ok", "ok"]


def test_every_query_matches_a_direct_core_call() -> None:
    """Delegation is exact: the service adds nothing to what the core returned.

    Compared as documents, not just as statuses, so a service that quietly re-sorted,
    re-scored, restamped or rebuilt a result could not pass this.
    """
    evidence = _mixed_evidence()
    service = SearchService(evidence)
    cases: list[tuple[str, dict[str, Any]]] = [
        ("whisper", {}),
        ("transcribe", {"limit": 1}),
        ("local-first", {}),
        ("model usage", {"limit": 0}),
        ("the model", {"limit": 3}),
        ("elephant", {}),
        ("", {}),
    ]
    for query, kwargs in cases:
        through_service = service.query(query, **kwargs)
        through_core = search(service.index, query, **kwargs)
        assert through_service.to_dict() == through_core.to_dict(), (query, kwargs)


def test_identical_queries_are_deterministic() -> None:
    """Same artifact, same words, same bytes — twice in a row and across two services.

    A rebuilt service over an equal timeline must agree with the first one as well: that
    is the determinism Step 1 documented for the core, asserted at the point where a
    service could otherwise smuggle in per-instance state.
    """
    service = SearchService(_mixed_evidence())
    first = json.dumps(service.query("whisper").to_dict(), sort_keys=True)
    second = json.dumps(service.query("whisper").to_dict(), sort_keys=True)
    other = SearchService(_mixed_evidence())
    third = json.dumps(other.query("whisper").to_dict(), sort_keys=True)
    assert first == second == third


def test_default_limit_is_the_core_constant_and_survives_the_call() -> None:
    """The service does not pick its own page size."""
    service = SearchService(_mixed_evidence())
    assert service.query("transcribe").limit == DEFAULT_RESULT_LIMIT
    assert service.query("transcribe", limit=1).limit == 1


# ---------------------------------------------------------------------------
# 2. Statuses and coverage arrive exactly as the core decided them.
# ---------------------------------------------------------------------------


def test_ok_is_passed_through_with_the_core_coverage_counts() -> None:
    """A real match is reported as ``ok`` with the same counts the core computed."""
    evidence = _mixed_evidence()
    service = SearchService(evidence)
    result = service.query("whisper")
    direct = search(build_search_index(evidence), "whisper")

    assert result.status == "ok"
    assert result.coverage.to_dict() == direct.coverage.to_dict()
    assert result.coverage.speech_record_count == 1
    assert result.coverage.ocr_record_count == 3
    assert result.coverage.matched_record_count >= result.coverage.returned_record_count


def test_a_query_without_a_meaningful_term_is_no_confident_match() -> None:
    """Queries with nothing to rank are answered, not raised and not padded with results.

    The blank and stopword-only queries contribute no terms at all; the timer frame's
    digits tokenize but are rejected by the core's meaning filter, which is why an
    on-screen clock is projected evidence that can never be a match.
    """
    service = SearchService(_mixed_evidence())
    for query in ("", "the", "the on", "a an the on", "12:34:56"):
        result = service.query(query)
        assert result.status == "no_confident_match", query
        assert result.results == (), query
        assert result.coverage.meaningful_query_term_count == 0, query
    # One meaningful term inside the same stopwords is enough to answer: the service
    # delegates the core's meaning filter instead of re-implementing a stricter one.
    assert service.query("the model on").status == "ok"


def test_a_meaningful_term_that_no_record_has_is_no_confident_match() -> None:
    """The other ``no_confident_match``: a real term, no overlap — and counts that say so."""
    service = SearchService(_mixed_evidence())
    result = service.query("telepathy")
    assert result.status == "no_confident_match"
    assert result.coverage.meaningful_query_term_count == 1
    assert result.coverage.matched_record_count == 0
    assert result.coverage.record_count == 4


def test_an_artifact_with_nothing_projected_is_no_evidence() -> None:
    """No segments and no events: the core's emptiest answer, passed straight through."""
    service = SearchService(_evidence([], segments=[], duration=None))
    result = service.query("whisper")
    assert result.status == "no_evidence"
    assert result.coverage.record_count == 0
    assert result.coverage.searchable_record_count == 0


def test_only_unsearchable_frames_is_no_evidence_and_says_why() -> None:
    """``no_evidence`` with records present — the distinction the service must not blur.

    A timer-only corpus was projected, tokenized and understood to be unmatchable. The
    service reports the status without quietly upgrading it to a match or downgrading it
    to an error, and the counts that explain the difference go with it.
    """
    service = SearchService(_evidence([_ocr(5.0, "12:34:56"), _ocr(6.0, "00:01:23")]))
    result = service.query("whisper")
    assert result.status == "no_evidence"
    assert result.coverage.record_count == 2
    assert result.coverage.searchable_record_count == 0


def test_a_non_positive_limit_returns_nothing_but_still_reports_the_answer() -> None:
    """The "explain your empty answer" path is the core's, uncopied and uncapped here."""
    service = SearchService(_mixed_evidence())
    for limit in (0, -5):
        result = service.query("whisper", limit=limit)
        assert result.results == (), limit
        assert result.status == "ok", limit
        assert result.coverage.returned_record_count == 0, limit
        assert result.coverage.matched_record_count == 2, limit
        assert result.limit == limit, limit


# ---------------------------------------------------------------------------
# 3. Provenance: the artifact and every record inside it.
# ---------------------------------------------------------------------------


def test_artifact_provenance_is_carried_verbatim() -> None:
    """Everything naming the video comes back untouched, including a failed transcription.

    The service is the first place a caller could plausibly restamp these, so the
    no-audio shape is checked too: ``transcription_available`` is False there, and the
    duration is genuinely unknown rather than zero.
    """
    evidence = _mixed_evidence()
    result = SearchService(evidence).query("whisper")
    assert result.source == _SOURCE
    assert result.transcription_status == "ok"
    assert result.transcription_available is True
    assert result.video_duration_seconds == 60.0
    assert result.study_evidence_schema == STUDY_EVIDENCE_SCHEMA

    silent = _evidence(
        [_ocr(5.0, "whisper.cpp model usage")],
        status="failed",
        detail="whisper-cli not found on PATH",
        duration=None,
        segments=[],
    )
    silent_result = SearchService(silent).query("whisper")
    assert silent_result.transcription_status == "failed"
    assert silent_result.transcription_available is False
    assert silent_result.video_duration_seconds is None


def test_the_stamped_schema_is_a_label_not_a_validation() -> None:
    """``study_evidence_schema`` names what the search layer wrote, nothing more.

    The core writes its own constant unconditionally, so this field could not detect a
    mismatched artifact even if one existed. Asserted so the service can never be quoted
    as evidence that an input was checked.
    """
    result = SearchService(_mixed_evidence()).query("whisper")
    assert result.study_evidence_schema == STUDY_EVIDENCE_SCHEMA
    assert result.to_dict()["study_evidence_schema"] == STUDY_EVIDENCE_SCHEMA


def test_each_match_keeps_its_modality_timestamp_and_attribution() -> None:
    """Per-record provenance is the core's, and a caller must not have to re-derive it.

    Speech keeps both ends of its interval; OCR has no end because none was ever
    measured — an absent value rendered honestly rather than filled in.
    """
    service = SearchService(_mixed_evidence())
    matches = service.query("whisper", limit=10).results
    by_id = {match.evidence.evidence_id: match.evidence for match in matches}
    assert by_id, "the fixture query must reach records"
    speech = [item for item in by_id.values() if item.source_type == "speech"]
    assert speech, "the fixture query must reach the speech record"
    for item in speech:
        # Speech carries both ends of its interval, copied from the transcript segment.
        assert (item.start_seconds, item.end_seconds) == (6.0, 12.0)
        assert item.segment_id == 0
    ocr = [item for item in by_id.values() if item.source_type == "ocr"]
    assert ocr, "the fixture query must reach an OCR record"
    for item in ocr:
        assert item.end_seconds is None, item.evidence_id


def test_unattributed_evidence_stays_searchable_with_no_segment() -> None:
    """An event outside every window is retained and reachable, attributed to nothing.

    The past-the-end timestamp is the production shape, not a fiction: Stage 0G evidence
    older than the last boundary, or anything at all when the duration is unknown.
    """
    service = SearchService(_mixed_evidence())
    matches = [
        match.evidence
        for match in service.query("notes", limit=10).results
        if match.evidence.segment_id is None
    ]
    assert [item.text for item in matches] == ["local-first offline transcription notes"]
    assert matches[0].start_seconds == 90.0
    assert matches[0].end_seconds is None


# ---------------------------------------------------------------------------
# 4. The service reads, and never writes, what it retains.
# ---------------------------------------------------------------------------


def test_queries_do_not_mutate_the_retained_index() -> None:
    """Twenty awkward queries leave the index exactly as the constructor built it.

    Every field a score is computed from is compared, including the per-record token
    spans: a query that trimmed, re-sorted or re-tokenized in place would show up as a
    changed span, not just a changed count.
    """
    service = SearchService(_mixed_evidence())
    before = _digest(service.index)
    queries = (
        "whisper", "Whisper.CPP", "transcribe", "local-first", "notes", "", " ",
        "the the the", "won't", "don\u2019t", "12:34:56", "ggml-medium.bin", "/usr/local/bin",
    )
    for query in queries:
        for limit in (0, 3):
            service.query(query, limit=limit)
    assert _digest(service.index) == before


def test_two_artifacts_do_not_share_an_index() -> None:
    """A second video is a second service — no registry, no bleed between corpora.

    Checked both ways: the query that only the first artifact can answer, and the source
    path each result names, so a shared index would fail on counts or on provenance.
    """
    first = SearchService(_evidence([_ocr(5.0, "whisper.cpp model usage")], source=_SOURCE))
    second = SearchService(
        _evidence([_ocr(5.0, "tesseract reads the slide text")], source=_OTHER_SOURCE)
    )

    assert first.query("whisper").status == "ok"
    assert second.query("whisper").status == "no_confident_match"
    assert first.query("tesseract").status == "no_confident_match"
    assert second.query("tesseract").status == "ok"
    assert first.query("whisper").source == _SOURCE
    assert second.query("tesseract").source == _OTHER_SOURCE
    assert first.query("whisper").coverage.record_count == 1
    assert second.query("tesseract").coverage.record_count == 1


# ---------------------------------------------------------------------------
# 5. The serialized contract, and the absence of any path to a video.
# ---------------------------------------------------------------------------


def test_results_keep_the_core_to_dict_contract() -> None:
    """The service invents no keys and drops none, so one schema describes both layers.

    ``SearchResults.to_dict`` is the documented artifact shape; a service that wrapped
    the result would silently fork the schema and split every consumer of it.
    """
    result = SearchService(_mixed_evidence()).query("whisper")
    document = json.loads(json.dumps(result.to_dict(), sort_keys=True))
    assert set(document) == {
        "schema",
        "query",
        "status",
        "source",
        "study_evidence_schema",
        "transcription_status",
        "transcription_available",
        "video_duration_seconds",
        "limit",
        "results",
        "result_count",
        "coverage",
    }
    assert document["schema"] == "video-learning.search/v1"
    assert document["result_count"] == len(result.results)
    assert set(document["coverage"]) == set(result.coverage.to_dict())
    for row in document["results"]:
        assert {
            "evidence_id",
            "source_type",
            "text",
            "start_seconds",
            "segment_id",
            "score",
        } <= set(row)


def test_no_media_or_derivation_dependency_is_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """There is no name in the service module that could inspect a video, and no need for one.

    Two checks, because one is easy to fake: the module must not import anything that
    runs the pipeline or re-derives evidence, and building a service over existing
    evidence must work even while that derivation raises. Between them they cover both
    ways a refactor could reach for the pipeline — by importing it, and by calling it.
    """
    for name in (
        "ReportService",
        "StudyEvidenceService",
        "SegmentService",
        "InspectService",
        "AnalyzeService",
        "TranscribeService",
        "build_study_evidence",
    ):
        assert not hasattr(search_service, name), name

    evidence = _mixed_evidence()

    def refuse(*args: Any, **kwargs: Any) -> StudyEvidence:
        raise AssertionError("the search service must not derive evidence")

    monkeypatch.setattr(study_core, "build_study_evidence", refuse)
    service = SearchService(evidence)
    assert service.query("whisper").status == "ok"
    assert service.query("notes").status == "ok"


def test_signature_keeps_evidence_positional_and_limit_keyword_only() -> None:
    """Pin the call shape the Step 3 CLI is written against.

    ``SearchService(evidence).query(text, limit=n)`` is the whole public seam, so a
    rename or a re-ordering is a breaking change for every caller. ``limit`` staying
    keyword-only is what stops a second positional string from silently becoming a
    result cap, and its default staying the core's constant is what keeps the CLI's
    ``--limit`` default and the service's own default from drifting apart.
    """
    init = inspect.signature(SearchService.__init__)
    assert list(init.parameters) == ["self", "evidence"]
    assert init.parameters["evidence"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    query = inspect.signature(SearchService.query)
    assert list(query.parameters) == ["self", "query", "limit"]
    limit = query.parameters["limit"]
    assert limit.kind is inspect.Parameter.KEYWORD_ONLY
    assert limit.default is DEFAULT_RESULT_LIMIT

    service = SearchService(_mixed_evidence())
    with pytest.raises(TypeError):
        service.query("whisper", 1)
