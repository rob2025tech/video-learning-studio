"""What the API is allowed to show: provenance reduced to a filename, evidence verbatim.

The core's ``SearchResults.to_dict()`` is the single owner of the search wire shape.
This module changes exactly one provenance field and adds one flag; it does not
reshape, trim, re-sort or re-score anything.
"""

from __future__ import annotations

import re
from typing import Any

from video_learning.core.search import SearchResults

#: Control characters have no business in a name we render, and a NUL in particular
#: truncates strings in some downstream C APIs.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
MAX_DISPLAY_NAME_LENGTH = 255
UNKNOWN_VIDEO_NAME = "(unknown video)"


def safe_display_name(raw: str | None, *, fallback: str) -> str:
    """Reduce an untrusted string to a display-only basename.

    Both separators are handled because a Windows browser sends ``C:\\dir\\report.json``
    where a macOS one sends ``report.json``, and a hostile client may send either — or a
    traversal string such as ``../../.ssh/id_rsa``. Only the last component survives, so
    the result cannot name a directory.

    The return value is **display metadata only**. No caller may open, join, resolve or
    stat it: nothing in this package turns a client-supplied name into a filesystem path.
    """
    if not raw:
        return fallback
    candidate = raw.replace("\\", "/").rsplit("/", 1)[-1]
    candidate = _CONTROL_CHARS.sub("", candidate).strip()
    if candidate in {"", ".", ".."}:
        return fallback
    return candidate[:MAX_DISPLAY_NAME_LENGTH]


def search_payload(results: SearchResults) -> dict[str, Any]:
    """``SearchResults.to_dict()`` with the absolute source path replaced by a filename.

    The readable CLI listing already shows only ``source.name``; the JSON does not, and
    an HTTP response is read by a browser and recorded in demos, so the same reduction
    is applied here. Every other key — including ``result_count``, the full ``coverage``
    block and each record's verbatim ``text`` and ``matched_spans`` — passes through
    untouched, because the spans index that text (``text[start:end] == surface``).

    Evidence text is deliberately *not* scanned for path-like content: a slide or a
    spoken sentence may legitimately contain one, and rewriting evidence would break
    the span invariant and misreport what the artifact actually recorded.
    """
    payload = results.to_dict()
    payload["source"] = safe_display_name(str(results.source), fallback=UNKNOWN_VIDEO_NAME)
    payload["source_is_redacted"] = True
    return payload
