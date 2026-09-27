"""Unit tests for CLI render helpers (timestamp formatting + OCR preview)."""

from __future__ import annotations

from video_learning.cli.render import _fmt_timestamp, _ocr_preview


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
