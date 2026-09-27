"""Pure keyword suggestion from OCR text.

No I/O, no dependencies beyond the standard library, fully deterministic:
the same input text always yields the same ordered suggestions.

This module deliberately does NOT invent metadata. Suggestions are only ever
derived from words that actually appeared in the provided (OCR'd) text.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

# A token must start with a letter and may contain letters/digits/+/#.
# This drops pure numbers, version strings like "62.12.102", and stray symbols,
# while keeping meaningful terms such as "Qoder", "iPh13m", "nodejs", "C++".
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#]*")

# Small, boring-word filter. Kept intentionally modest: OCR of software screens
# produces many UI/code tokens, and over-filtering would hide real subjects.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by can could did do does for from had has have how i if in into is
    it its just may my no not of on or our out so than that the their them then there these they
    this to too was we were what when where which who will with you your
    file edit view window help search select all copy paste undo redo new open save close
    """.split()
)


@dataclass(frozen=True)
class KeywordSuggestion:
    """A single suggested keyword and how often it was observed."""

    term: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"term": self.term, "count": self.count}


def suggest_keywords(
    text: str,
    *,
    max_keywords: int = 12,
    min_length: int = 3,
    min_count: int = 1,
) -> list[KeywordSuggestion]:
    """Rank words in ``text`` into suggested keywords.

    Ordering is deterministic: descending by count, then alphabetically by the
    lower-cased term. The displayed casing is the most common surface form seen
    in the text (so "Qoder" stays capitalised, "AI" stays upper-case).
    """
    surface_forms: dict[str, Counter[str]] = {}
    counts: Counter[str] = Counter()

    for raw in _TOKEN.findall(text):
        if len(raw) < min_length:
            continue
        key = raw.lower()
        if key in _STOPWORDS:
            continue
        counts[key] += 1
        surface_forms.setdefault(key, Counter())[raw] += 1

    ranked: list[KeywordSuggestion] = []
    for key, count in counts.items():
        if count < min_count:
            continue
        # Most frequent original casing wins; ties broken alphabetically.
        term = sorted(surface_forms[key].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        ranked.append(KeywordSuggestion(term=term, count=count))

    ranked.sort(key=lambda kw: (-kw.count, kw.term.lower()))
    return ranked[:max_keywords]
