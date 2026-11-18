"""Request models only. Responses are the core's own dicts, never a second schema.

Re-modelling ``SearchResults`` in pydantic would duplicate the wire shape that
``core/search.py`` already owns and let the two drift, so the API validates what comes
in and passes through what the core produced.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from video_learning.core.search import DEFAULT_RESULT_LIMIT

#: An input bound on query length, not a search rule. The CLI accepts any length.
MAX_QUERY_LENGTH = 512


class SearchRequest(BaseModel):
    """One query against one already-loaded report session."""

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(min_length=1, max_length=128)
    query: str = Field(default="", max_length=MAX_QUERY_LENGTH)
    # Deliberately UNCONSTRAINED — no ge/gt/le. core.search treats a non-positive limit
    # as "return no records but still report status and coverage", which the CLI exposes
    # as `--limit 0` and which an "explain your empty answer" surface needs. A positive
    # lower bound here would silently remove that behavior from the API only.
    limit: int = DEFAULT_RESULT_LIMIT
