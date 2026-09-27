"""Unit tests for the pure keyword and phrase suggesters (no I/O)."""

from __future__ import annotations

from video_learning.core.keywords import suggest_keywords, suggest_phrases


def _terms(text: str, **kwargs: object) -> list[str]:
    return [kw.term for kw in suggest_keywords(text, **kwargs)]  # type: ignore[arg-type]


def _phrases(text: str, **kwargs: object) -> list[str]:
    return [p.phrase for p in suggest_phrases(text, **kwargs)]  # type: ignore[arg-type]


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


# --- Phrase suggestions (Stage 0D) ------------------------------------------


def test_phrase_release_notes_recognized() -> None:
    # Repeated on two frames/lines -> passes the frequency gate and stays intact.
    assert _phrases("Release Notes\nRelease Notes") == ["Release Notes"]


def test_phrase_qoder_model_usage_recognized() -> None:
    text = "Qoder Model Usage\nQoder Model Usage"
    assert _phrases(text) == ["Qoder Model Usage"]


def test_phrase_qoder_cli_release_notes_recognized() -> None:
    # Inolated (no competing shorter form) the full specific phrase is kept.
    text = "Qoder CLI Release Notes\nQoder CLI Release Notes"
    assert _phrases(text) == ["Qoder CLI Release Notes"]


def test_phrase_path_preserved_intact_as_oneoff() -> None:
    # A distinctive path seen on a single frame is NOT discarded, and separators
    # (/, -) are preserved rather than split into Projects/Templates/video/...
    text = "~/Projects/Templates/video-learning-studio %"
    assert _phrases(text) == ["Projects/Templates/video-learning-studio"]


def test_phrase_hyphenated_name_preserved_intact() -> None:
    assert _phrases("video-learning-studio") == ["video-learning-studio"]


def test_phrase_repeated_occurrences_get_stronger_count() -> None:
    result = suggest_phrases("Release Notes\nRelease Notes\nRelease Notes")
    assert result[0].phrase == "Release Notes"
    assert result[0].count == 3


def test_phrase_oneoff_compound_kept_but_oneoff_words_dropped() -> None:
    # Distinctive compound survives at count 1; an ordinary one-off multi-word
    # run (random adjacency) does not.
    text = "Projects/Templates/video-learning-studio\nSome Random Words"
    phrases = _phrases(text)
    assert "Projects/Templates/video-learning-studio" in phrases
    assert "Some Random Words" not in phrases


def test_phrase_generic_garbage_filtered() -> None:
    # Stopword-only runs and one-off adjacent word pairs are suppressed, so we do
    # not emit things like "Qoder RobTech" merely because they sat side by side.
    assert _phrases("the and\nQoder RobTech") == []


def test_phrase_multiword_requires_min_count() -> None:
    # A single sighting of an ordinary multi-word phrase is not enough evidence.
    assert _phrases("Qoder Model Usage") == []


def test_phrase_exact_duplicates_merge_counts() -> None:
    # Case/whitespace-folded duplicates combine into one entry with a summed
    # count and the dominant surface form (conservative dedup).
    result = suggest_phrases("Release Notes\nrelease notes\nRelease Notes")
    assert len(result) == 1
    assert result[0].phrase == "Release Notes"
    assert result[0].count == 3


def test_phrase_nested_phrases_are_both_retained() -> None:
    # A phrase is NOT dropped merely because another contains it; both are
    # independently useful tags (no frequency-based substring elimination).
    text = (
        "Release Notes\nRelease Notes\n"
        "Qoder CLI Release Notes\nQoder CLI Release Notes"
    )
    phrases = _phrases(text)
    assert "Release Notes" in phrases
    assert "Qoder CLI Release Notes" in phrases


def test_phrase_numeric_fragment_is_not_distinctive() -> None:
    # A date/version-like fragment is not a distinctive compound, so as a lone
    # one-off token it is dropped rather than preserved.
    assert _phrases("26.09.26c") == []


def test_phrase_ranking_count_desc_then_distinctive_then_words_then_alpha() -> None:
    text = (
        "video-learning-studio\nvideo-learning-studio\n"  # count2, distinctive, 1 word
        "Beta Gamma\nBeta Gamma\n"                        # count2, ordinary, 2 words
        "Alpha Delta Epsilon\nAlpha Delta Epsilon\n"      # count2, ordinary, 3 words
    )
    # Count tie -> distinctive first, then word count desc, then alphabetical.
    assert _phrases(text) == [
        "video-learning-studio",
        "Alpha Delta Epsilon",
        "Beta Gamma",
    ]


def test_phrase_ranking_prefers_higher_count_first() -> None:
    text = (
        "Alpha Beta\nAlpha Beta\nAlpha Beta\n"      # count3
        "Gamma Delta Epsilon\nGamma Delta Epsilon\n"  # count2
        "video-learning-studio\n"                     # count1, distinctive one-off
    )
    assert _phrases(text) == [
        "Alpha Beta",
        "Gamma Delta Epsilon",
        "video-learning-studio",
    ]


def test_phrase_max_phrases_caps_output() -> None:
    text = (
        "Alpha One\nAlpha One\nBeta Two\nBeta Two\n"
        "Gamma Three\nGamma Three\nDelta Four\nDelta Four"
    )
    assert len(_phrases(text, max_phrases=2)) == 2


def test_phrase_empty_text_yields_no_suggestions() -> None:
    assert suggest_phrases("") == []
    assert suggest_phrases("   \n\t 123 !!! ") == []


def test_phrase_deterministic_same_input_same_output() -> None:
    text = "Qoder Model Usage\nRelease Notes\nProjects/Templates/video-learning-studio"
    assert _phrases(text) == _phrases(text)
