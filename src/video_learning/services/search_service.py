"""Evidence-search use case (Multimodal Evidence Search v1, Step 2): evidence -> index -> queries.

A thin orchestration adapter, mirroring the project's "one service per use case"
pattern (``StudyEvidenceService``, ``StudyGuideService``, ``StudyQuizService``). It holds
no search logic of its own and performs no I/O: it builds the lexical index once with the
pure :func:`~video_learning.core.search.build_search_index` and answers each query by
delegating to the pure :func:`~video_learning.core.search.search`.

    StudyEvidence            — already derived (Stage 0G/0I report + Task 1 binding)
    SearchService(evidence)  — one SearchIndex, built once, in the constructor
    service.query(text)      — the core's SearchResults, returned unchanged

Accepting an already-derived :class:`~video_learning.core.study.StudyEvidence` instead of
a video path is what makes the absence of media processing structural rather than a
promise: this module imports no service that inspects, extracts frames, runs OCR,
transcribes audio or segments anything, so there is no call path from :meth:`query` to a
subprocess. There is deliberately no ``build(path)`` or ``from_report(report)`` entry
point either — a caller holding a report derives the evidence with
:meth:`StudyEvidenceService.derive` and passes the result here.

The index is retained on the instance and never rebuilt, which is how several queries
over one video cost exactly one tokenization pass. :class:`~video_learning.core.search.SearchIndex`
is a frozen dataclass of tuples plus a mapping that :func:`~video_learning.core.search.search`
only ever reads by key, so one index is safe to query repeatedly, and a second video is
simply a second ``SearchService``. No module-level cache keyed by source path is
introduced: that would give an artifact an unbounded lifetime and let a stale index
outlive a re-derived ``StudyEvidence``.

The three statuses are the core's and are not reinterpreted here:

* ``ok`` — at least one record shared a meaningful query term. A weak-but-real match is
  ``ok``, because there is no score threshold to cross.
* ``no_confident_match`` — the query carried no meaningful term, or no record matched
  one. That is an answer, not an error, and it raises nothing.
* ``no_evidence`` — nothing in the corpus is matchable at all. ``record_count`` still
  reports the blank and timer-only frames that were projected, so this status does not
  mean "the video produced no evidence".

One stamped field is a label rather than a check: ``study_evidence_schema`` on the
results is the constant of the module that built the index, which
:func:`~video_learning.core.search.build_search_index` writes unconditionally — it is
never read from, or compared against, the incoming artifact. So a result says which
schema the search layer wrote, and cannot be cited as proof that the evidence handed in
was produced by the matching builder.

Introduced outside this module, and still absent from it: CLI commands and any
rehydration of a serialized artifact. Step 3 wired both up around this service —
:meth:`~video_learning.services.report_service.ReportStore.load_evidence` returns the
:class:`StudyEvidence` this constructor takes, and the ``search`` command in
``cli/main.py`` calls this service with it — so nothing about this seam changed shape.
Still out of scope for the feature: persisting a result set, SQLite/FTS5, embeddings,
semantic search, reciprocal-rank fusion, multi-query orchestration, and any change to
the core tokenizer, ranking or status rules. Excerpting is not among them because it
belongs one layer up: :func:`~video_learning.cli.render.render_search_results` windows
an OCR record around its first match for the readable listing, while :meth:`query`
still hands back the core's results with the complete text and exact spans untouched.
The evidence object is only ever read; nothing is written.
"""

from __future__ import annotations

from video_learning.core.search import (
    DEFAULT_RESULT_LIMIT,
    SearchIndex,
    SearchResults,
    build_search_index,
    search,
)
from video_learning.core.study import StudyEvidence


class SearchService:
    """Use case: answer many lexical queries against one already-derived evidence artifact."""

    def __init__(self, evidence: StudyEvidence) -> None:
        """Build the whole-video lexical index for ``evidence``, once and only once.

        The projection, tokenization and corpus statistics all happen here; every
        :meth:`query` after this reads the index instead of re-deriving it, so N queries
        cost one index. Nothing is validated about ``evidence`` beyond what typing gives:
        it is expected to be the output of
        :func:`~video_learning.core.study.build_study_evidence`, whose input contract
        (chronological, tiling segments) :func:`~video_learning.core.search.project_evidence`
        relies on for attribution.

        The service keeps no reference to the caller's object — the records, source path
        and transcription fields it needs were copied into the index — so it cannot be
        surprised by a caller rebuilding an artifact under it.
        """
        self._index = build_search_index(evidence)

    @property
    def index(self) -> SearchIndex:
        """The index built in the constructor, for callers that need it directly.

        Exposed because a later step may want the records or the statistics without
        asking a question (an evidence panel, a multi-query fusion pass), and reading
        this object is exactly what :meth:`query` does. It is frozen and :meth:`query`
        never mutates it, so the reference stays as trustworthy as the service.
        """
        return self._index

    def query(self, query: str, *, limit: int = DEFAULT_RESULT_LIMIT) -> SearchResults:
        """Return the core's ranked results for ``query`` against the retained index.

        Pure delegation: no wrapping, reranking, re-scoring, truncation, excerpting or
        serialization. Everything the core returns survives untouched — the verbatim
        record text, ``matched_terms`` in first-occurrence order, ``matched_spans`` whose
        offsets index that text (and nest, so highlighting should paint longest-first),
        the ``coverage`` counts including ``matched_record_count``, which is the full
        matched total even when ``limit`` returned fewer, and the provenance copied from
        the evidence. ``limit`` caps what is returned, never what is considered, and a
        non-positive ``limit`` still reports the status and the counts — which is what an
        "explain your empty answer" surface needs.
        """
        return search(self._index, query, limit=limit)
