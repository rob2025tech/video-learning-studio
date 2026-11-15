"""Unit tests for CLI render helpers (timestamps, OCR preview, matched excerpts)."""

from __future__ import annotations

from video_learning.cli.render import _fmt_timestamp, _matched_excerpt, _ocr_preview
from video_learning.core.search import Token, tokenize


def test_timestamp_basic() -> None:
    assert _fmt_timestamp(12.4) == "00:00:12.4"
    assert _fmt_timestamp(227.1) == "00:03:47.1"
    assert _fmt_timestamp(0.0) == "00:00:00.0"


def test_timestamp_hours() -> None:
    assert _fmt_timestamp(3600.0) == "01:00:00.0"
    assert _fmt_timestamp(3723.5) == "01:02:03.5"


def test_timestamp_fraction_carries_into_seconds() -> None:
    # .96 must round up to the next second, never print an invalid ".10".
    assert _fmt_timestamp(115.96) == "00:01:56.0"
    assert _fmt_timestamp(59.99) == "00:01:00.0"


def test_timestamp_negative_clamped() -> None:
    assert _fmt_timestamp(-1.0) == "00:00:00.0"


def test_preview_collapses_whitespace() -> None:
    assert _ocr_preview("Qoder   Model\n\nUsage\tRelease") == "Qoder Model Usage Release"


def test_preview_truncates_long_text() -> None:
    text = "word " * 100
    preview = _ocr_preview(text, limit=20)
    assert len(preview) <= 20
    assert preview.endswith("…")


def test_preview_empty_text() -> None:
    assert _ocr_preview("   \n\t ") == "(no text detected)"


# --- Matched excerpts (OCR search results only) ------------------------------

# 461 characters, with the only match at offset 281: far past a 90-character preview
# of the record's opening words, which is the defect the excerpt window exists to fix.
_LONG_OCR = (
    "Preferences\n\n"
    + "General   Appearance   Advanced\n" * 8
    + "\nAdjust the histogram until the exposure looks right.\n"
    + "Apply   Cancel   Reset\n" * 6
)


def _hits(text: str, *terms: str) -> tuple[Token, ...]:
    """The spans the ranker would hand back for these terms, from the real tokenizer."""
    return tuple(token for token in tokenize(text) if token.term in terms)


def test_excerpt_centers_on_a_late_match() -> None:
    excerpt = _matched_excerpt(_LONG_OCR, _hits(_LONG_OCR, "histogram"))
    assert "histogram" in excerpt
    assert "Adjust" in excerpt
    assert "exposure" in excerpt
    assert "Preferences" not in excerpt
    assert len(excerpt) <= 92


def test_excerpt_keeps_short_text_whole() -> None:
    text = "build a search index locally"
    excerpt = _matched_excerpt(text, _hits(text, "index"))
    assert excerpt == _ocr_preview(text)
    assert "…" not in excerpt


def test_excerpt_marks_both_truncated_ends() -> None:
    excerpt = _matched_excerpt(_LONG_OCR, _hits(_LONG_OCR, "histogram"))
    assert excerpt.startswith("…")
    assert excerpt.endswith("…")


def test_excerpt_omits_ellipsis_at_the_text_edges() -> None:
    head = "histogram " + "filler text " * 20
    assert not _matched_excerpt(head, _hits(head, "histogram")).startswith("…")
    tail = "filler text " * 20 + " histogram"
    assert not _matched_excerpt(tail, _hits(tail, "histogram")).endswith("…")


def test_excerpt_preserves_a_whitespace_bearing_surface_verbatim() -> None:
    """The tokenizer cannot produce this, but the helper must still not corrupt it."""
    surface = "Si,\tlo\nhago."
    text = "alpha\n\n" + "pad " * 40 + surface + " tail" * 40
    start = text.index(surface)
    span = Token(term="si", surface=surface, start=start, end=start + len(surface))
    assert surface in _matched_excerpt(text, (span,))


def test_excerpt_fuses_a_nested_compound_and_its_part() -> None:
    text = "pad " * 30 + "open node.js now " + "pad " * 30
    spans = _hits(text, "node.js", "node", "js")
    assert len(spans) == 3
    assert "node.js" in _matched_excerpt(text, spans)


def test_excerpt_fuses_partially_overlapping_spans() -> None:
    text = "pad " * 30 + "abcdefghij" + " pad" * 30
    start = text.index("abcdefghij")
    spans = (
        Token(term="abcde", surface="abcde", start=start, end=start + 5),
        Token(term="defgh", surface="defgh", start=start + 3, end=start + 8),
    )
    assert "abcdefgh" in _matched_excerpt(text, spans)


def test_excerpt_widens_rather_than_cutting_an_edge_span() -> None:
    text = "x" * 100 + "alpha" + "y" * 41 + "omega" + "z" * 100
    spans = (
        Token(term="alpha", surface="alpha", start=100, end=105),
        Token(term="omega", surface="omega", start=146, end=151),
    )
    excerpt = _matched_excerpt(text, spans)
    assert "alpha" in excerpt
    assert "omega" in excerpt
    # The second surface straddles the 90-character edge, so integrity beats the budget.
    assert len(excerpt) > 92


def test_excerpt_omits_a_disjoint_match_outside_the_window() -> None:
    text = "pad " * 50 + "alpha" + " pad " * 7 + "beta" + " pad " * 20 + "gamma" + " pad " * 50
    spans = _hits(text, "alpha", "beta", "gamma")
    assert len(spans) == 3
    excerpt = _matched_excerpt(text, spans)
    assert "alpha" in excerpt
    assert "beta" in excerpt
    assert "gamma" not in excerpt
    assert "gam" not in excerpt
    assert len(excerpt) <= 92


def test_excerpt_collapses_context_but_keeps_one_space_against_the_match() -> None:
    text = "pad\n\n\thistogram\n\t  tail"
    assert _matched_excerpt(text, _hits(text, "histogram")) == "pad histogram tail"


def test_excerpt_falls_back_without_spans() -> None:
    assert _matched_excerpt(_LONG_OCR, ()) == _ocr_preview(_LONG_OCR)
    assert _matched_excerpt("   \n\t ", ()) == "(no text detected)"


def test_excerpt_omits_ellipsis_when_only_whitespace_was_skipped() -> None:
    head = "\n\n\thistogram " + "tail" * 30
    assert not _matched_excerpt(head, _hits(head, "histogram")).startswith("…")
    tail = "head " * 30 + "histogram\n\n\t"
    assert not _matched_excerpt(tail, _hits(tail, "histogram")).endswith("…")


def test_excerpt_falls_back_on_an_out_of_range_span() -> None:
    start = _LONG_OCR.index("histogram")
    end = start + len("histogram")
    bad = (
        Token(term="histogram", surface="histogram", start=-3, end=end),
        Token(term="histogram", surface="histogram", start=start, end=len(_LONG_OCR) + 1),
        Token(term="x", surface="", start=start, end=start),
        Token(term="histogram", surface="histogram", start=end, end=start),
    )
    for span in bad:
        excerpt = _matched_excerpt(_LONG_OCR, (span,))
        assert excerpt == _ocr_preview(_LONG_OCR)
        assert len(excerpt) <= 92
    good = Token(term="histogram", surface="histogram", start=start, end=end)
    beyond = Token(
        term="exposure",
        surface="exposure",
        start=_LONG_OCR.index("exposure"),
        end=len(_LONG_OCR) + 1,
    )
    assert _matched_excerpt(_LONG_OCR, (good, beyond)) == _ocr_preview(_LONG_OCR)


def test_excerpt_falls_back_when_a_surface_contradicts_the_text() -> None:
    start = _LONG_OCR.index("histogram")
    end = start + len("histogram")
    bad = (
        Token(term="histogram", surface="Histogram", start=start, end=end),
        Token(term="histogram", surface="histograx", start=start, end=end),
        Token(term="until", surface="until", start=start, end=start + 5),
    )
    for span in bad:
        excerpt = _matched_excerpt(_LONG_OCR, (span,))
        assert excerpt == _ocr_preview(_LONG_OCR)
        assert "histogram" not in excerpt


def test_excerpt_keeps_a_surface_longer_than_the_limit() -> None:
    surface = "a" * 120
    text = "pad " * 10 + surface + " pad" * 10
    start = text.index(surface)
    span = Token(term=surface, surface=surface, start=start, end=start + len(surface))
    excerpt = _matched_excerpt(text, (span,), limit=90)
    assert surface in excerpt
    assert len(excerpt) >= len(surface)
