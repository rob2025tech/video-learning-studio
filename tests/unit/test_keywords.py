"""Unit tests for the pure keyword suggester (no I/O)."""

from __future__ import annotations

from video_learning.core.keywords import suggest_keywords


def _terms(text: str, **kwargs: object) -> list[str]:
    return [kw.term for kw in suggest_keywords(text, **kwargs)]  # type: ignore[arg-type]


def test_deterministic_same_input_same_output() -> None:
    text = "Qoder terminal coding Qoder AI coding terminal Qoder"
    assert _terms(text) == _terms(text)


def test_ranks_by_frequency_then_alphabetical() -> None:
    text = "terminal Qoder Qoder Qoder coding coding python"
    result = suggest_keywords(text)
    # Qoder(3) > coding(2) > python(1)/terminal(1); single-count ties alphabetical.
    assert [kw.term for kw in result] == ["Qoder", "coding", "python", "terminal"]
    assert [kw.count for kw in result] == [3, 2, 1, 1]


def test_filters_stopwords() -> None:
    text = "the file edit and window help Qoder"
    terms = _terms(text)
    assert "Qoder" in terms
    for stop in ("the", "file", "edit", "and", "window", "help"):
        assert stop not in terms


def test_filters_short_tokens_and_pure_numbers() -> None:
    text = "AI is ok 42 7 Qoder 62.12.102"
    terms = _terms(text)
    assert "Qoder" in terms
    assert "42" not in terms and "7" not in terms
    # "AI" is length 2 < min_length 3, so dropped by default.
    assert "AI" not in terms


def test_preserves_dominant_casing() -> None:
    text = "QODER qoder Qoder Qoder terminal"
    terms = _terms(text)
    assert terms[0] == "Qoder"  # most frequent surface form wins


def test_min_length_override_keeps_short_terms() -> None:
    text = "AI ML coding"
    assert "AI" in _terms(text, min_length=2)


def test_min_count_filters_rare_tokens() -> None:
    text = "Qoder Qoder onceishword"
    terms = _terms(text, min_count=2)
    assert terms == ["Qoder"]


def test_max_keywords_caps_output() -> None:
    text = "alpha beta gamma delta epsilon zeta"
    assert len(_terms(text, max_keywords=3)) == 3


def test_empty_text_yields_no_suggestions() -> None:
    assert suggest_keywords("") == []
    assert suggest_keywords("   \n\t 123 !!! ") == []


def test_keeps_technical_tokens() -> None:
    text = "nodejs C++ python3 import numpy"
    terms = _terms(text)
    assert "C++" in terms
    assert "nodejs" in terms
    assert "python3" in terms
