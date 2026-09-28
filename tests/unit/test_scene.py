"""Unit tests for the Stage 0I visual-change domain (core/scene.py).

Pure data plus the single authoritative threshold constant. No I/O, no FFmpeg.
"""

from __future__ import annotations

import dataclasses

import pytest

from video_learning.core.scene import VISUAL_CHANGE_THRESHOLD, VisualChange


def test_visual_change_threshold_is_authoritative_value() -> None:
    # scdet scores are on a 0-100 scale; the frozen Stage 0I threshold is 10.0.
    assert VISUAL_CHANGE_THRESHOLD == 10.0


def test_visual_change_stores_timestamp_and_score() -> None:
    change = VisualChange(timestamp_seconds=3.0, change_score=15.625)
    assert change.timestamp_seconds == 3.0
    assert change.change_score == 15.625


def test_visual_change_to_dict_is_json_safe() -> None:
    change = VisualChange(timestamp_seconds=12.34, change_score=51.953)
    assert change.to_dict() == {"timestamp_seconds": 12.34, "change_score": 51.953}


def test_visual_change_is_immutable_and_value_equal() -> None:
    a = VisualChange(timestamp_seconds=1.0, change_score=20.0)
    b = VisualChange(timestamp_seconds=1.0, change_score=20.0)
    assert a == b
    assert hash(a) == hash(b)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.change_score = 99.0  # type: ignore[misc]
