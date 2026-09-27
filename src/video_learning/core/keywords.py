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

# --- Phrase extraction (Stage 0D) -------------------------------------------
# A "word" for phrase purposes keeps internal separators (/, -, ., +, #) so that
# paths and hyphenated/dotted names stay intact, e.g.
# "Projects/Templates/video-learning-studio" or "node.js".
_WORD = r"[A-Za-z0-9][A-Za-z0-9+#./\-]*"
# A phrase candidate is a run of such words joined by single spaces/tabs. It stops
# at UI-noise characters and at double spaces, so merged screen chrome is broken
# up rather than swallowed whole. Order within a line is kept.
_PHRASE_RUN = re.compile(rf"{_WORD}(?:[ \t]{_WORD})*")
# Leading/trailing separators trimmed off each word (e.g. a sentence-final period).
_TRIM_CHARS = ".-/#+"
# A one-off single-word compound must be at least this long to count as a
# distinctive path/name worth keeping at count 1 (rejects short OCR garble).
_DISTINCTIVE_MIN_LEN = 6

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


def _is_distinctive_compound(tokens: list[str]) -> bool:
    """True for a path/name-like single token worth keeping even at count 1.

    Conservative by design. A single token qualifies when it is reasonably long,
    contains a letter, and either looks like a path (contains ``/``) or is a
    hyphen/dot compound whose parts each contain a letter (e.g.
    ``video-learning-studio``, ``node.js``). Numeric-only fragments such as a
    ``26.09.26`` recording timer are rejected, so they fall back to the ordinary
    frequency gate instead of being preserved as distinctive.
    """
    if len(tokens) != 1 or len(tokens[0]) < _DISTINCTIVE_MIN_LEN:
        return False
    token = tokens[0]
    if not any(char.isalpha() for char in token):
        return False
    if "/" in token:  # a path separator is strong structural evidence
        return True
    parts = [part for part in re.split(r"[-.]", token) if part]
    return len(parts) >= 2 and all(any(c.isalpha() for c in part) for part in parts)


def _has_meaningful_word(tokens: list[str]) -> bool:
    """At least one token is a real word (>=3 chars, has a letter, not a stopword)."""
    return any(
        len(tok) >= 3 and re.search(r"[A-Za-z]", tok) and tok.lower() not in _STOPWORDS
        for tok in tokens
    )


@dataclass(frozen=True)
class PhraseSuggestion:
    """A suggested phrase (or distinctive compound) and how often it was observed."""

    phrase: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"phrase": self.phrase, "count": self.count}


def _phrase_rank_key(suggestion: PhraseSuggestion) -> tuple[int, int, int, str]:
    """Deterministic presentation order (this never decides what may exist).

    Order by: occurrence count desc, distinctive compound/path before ordinary
    phrase on ties, word count desc, then alphabetical (case-folded).
    """
    tokens = suggestion.phrase.split()
    distinctive = 0 if _is_distinctive_compound(tokens) else 1
    return (-suggestion.count, distinctive, -len(tokens), suggestion.phrase.lower())


def suggest_phrases(
    text: str,
    *,
    max_phrases: int = 8,
    min_count: int = 2,
    max_words: int = 5,
) -> list[PhraseSuggestion]:
    """Rank meaningful phrases in OCR ``text`` (pure, deterministic).

    Unlike :func:`suggest_keywords` (a bag of single words), this preserves the
    OCR's line/word order and internal separators, so phrases such as
    ``Qoder Model Usage``, ``Release Notes`` and paths like
    ``Projects/Templates/video-learning-studio`` survive intact.

    Deterministic rules:

    * Candidates are maximal word runs on each line. A run qualifies when it is
      either a multi-word phrase (``2..max_words`` words containing at least one
      non-stopword) or a single distinctive compound/path (see
      :func:`_is_distinctive_compound`). Ordinary single words are left to
      :func:`suggest_keywords`; random adjacent pairs are suppressed by frequency.
    * Frequency: a candidate must occur at least ``min_count`` times across all
      lines/frames, **except** distinctive compounds/paths, which are kept even at
      count 1 so a one-off ``Projects/Templates/video-learning-studio`` survives.
    * Dedup is conservative: candidates that normalize to the same phrase (case
      and whitespace folded) are merged into one entry with their counts combined
      and the dominant surface form kept. A phrase is NEVER dropped merely for
      being a substring of another, so ``Release Notes`` and
      ``Qoder CLI Release Notes`` can both survive as independently useful tags.
    * Ranking (presentation order only): count desc, distinctive compound/path
      before ordinary phrase on ties, word count desc, then alphabetical.
    """
    surface: dict[str, Counter[str]] = {}
    counts: Counter[str] = Counter()

    for line in text.splitlines():
        for run in _PHRASE_RUN.findall(line):
            tokens = [tok.strip(_TRIM_CHARS) for tok in run.split()]
            tokens = [tok for tok in tokens if tok]
            if not tokens or len(tokens) > max_words:
                continue
            distinctive = _is_distinctive_compound(tokens)
            if len(tokens) == 1 and not distinctive:
                continue  # ordinary single words belong to the keyword list
            if len(tokens) >= 2 and not _has_meaningful_word(tokens):
                continue  # e.g. "the and" carries no real content
            phrase = " ".join(tokens)
            key = phrase.lower()
            counts[key] += 1
            surface.setdefault(key, Counter())[phrase] += 1

    ranked: list[PhraseSuggestion] = []
    for key, count in counts.items():
        # One-off distinctive compounds/paths bypass the frequency gate.
        if count < min_count and not _is_distinctive_compound(key.split()):
            continue
        # Most frequent original casing wins; ties broken alphabetically.
        phrase = sorted(surface[key].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        ranked.append(PhraseSuggestion(phrase=phrase, count=count))

    # Conservative dedup already happened above via the normalized `counts` and
    # `surface` maps (exact case/whitespace-folded duplicates merged, counts
    # summed). No substring elimination: nested phrases are kept as independent
    # tags. Ranking only orders presentation; it never removes candidates.
    ranked.sort(key=_phrase_rank_key)
    return ranked[:max_phrases]
