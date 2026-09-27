"""Unit tests for OcrSnapshotStore: deterministic OCR-evidence persistence.

Pure file I/O over ``tmp_path``; no binaries, no network, no wall-clock input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_learning.core.errors import OcrArtifactError
from video_learning.core.keywords import KeywordSuggestion, PhraseSuggestion
from video_learning.services.analyze_service import AnalyzeResult, FrameSnapshot
from video_learning.services.ocr_snapshot_store import OCR_ARTIFACT_SCHEMA, OcrSnapshotStore

_TOP_LEVEL_KEYS = {
    "schema",
    "source",
    "snapshots",
    "keywords",
    "phrases",
    "frames_analyzed",
    "ocr_text_chars",
    "renamed",
    "applied",
    "source_modified",
}


def _result(source: Path) -> AnalyzeResult:
    return AnalyzeResult(
        source=source,
        keywords=[KeywordSuggestion("Qoder", 4), KeywordSuggestion("terminal", 2)],
        phrases=[PhraseSuggestion("Qoder Model Usage", 3), PhraseSuggestion("Release Notes", 2)],
        snapshots=[
            FrameSnapshot(12.4, "Projects Templates video-learning-studio"),
            FrameSnapshot(227.1, "Qoder Model Usage\nRelease Notes\nLine three"),
        ],
        frames_analyzed=2,
        ocr_text_chars=128,
    )


def test_save_writes_schema_tagged_artifact(tmp_path: Path) -> None:
    store = OcrSnapshotStore()
    out = tmp_path / "ocr.json"

    returned = store.save(_result(tmp_path / "clip.mov"), out)

    assert returned == out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == OCR_ARTIFACT_SCHEMA
    assert set(data) == _TOP_LEVEL_KEYS


def test_round_trip_preserves_full_result(tmp_path: Path) -> None:
    store = OcrSnapshotStore()
    original = _result(tmp_path / "clip.mov")
    out = tmp_path / "ocr.json"

    store.save(original, out)
    loaded = store.load(out)

    assert loaded == original


def test_complete_ocr_text_preserved_not_replaced_by_suggestions(tmp_path: Path) -> None:
    store = OcrSnapshotStore()
    out = tmp_path / "ocr.json"

    store.save(_result(tmp_path / "clip.mov"), out)
    data = json.loads(out.read_text(encoding="utf-8"))

    # The full multi-line OCR text survives verbatim (never swapped for phrases).
    assert data["snapshots"][1]["ocr_text"] == "Qoder Model Usage\nRelease Notes\nLine three"
    # Phrases/keywords remain separate derived fields.
    assert {"phrase": "Qoder Model Usage", "count": 3} in data["phrases"]
    assert {"term": "Qoder", "count": 4} in data["keywords"]


def test_traceability_video_to_timestamp_to_text(tmp_path: Path) -> None:
    store = OcrSnapshotStore()
    out = tmp_path / "ocr.json"
    video = tmp_path / "clip.mov"

    store.save(_result(video), out)
    data = json.loads(out.read_text(encoding="utf-8"))

    # A human can trace: source video -> snapshot timestamp -> original OCR text,
    # and a derived phrase is findable inside that original text.
    assert data["source"] == str(video)
    assert [s["timestamp_seconds"] for s in data["snapshots"]] == [12.4, 227.1]
    blob = "\n".join(s["ocr_text"] for s in data["snapshots"])
    assert "Qoder Model Usage" in blob


def test_output_is_deterministic_across_saves(tmp_path: Path) -> None:
    store = OcrSnapshotStore()
    result = _result(tmp_path / "clip.mov")
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"

    store.save(result, first)
    store.save(result, second)

    assert first.read_bytes() == second.read_bytes()
    # No wall-clock metadata anywhere in the artifact.
    text = first.read_text(encoding="utf-8").lower()
    for banned in ("generated_at", "created_at", "modified_at"):
        assert banned not in text


def test_multiple_snapshots_order_preserved(tmp_path: Path) -> None:
    store = OcrSnapshotStore()
    result = AnalyzeResult(
        source=tmp_path / "clip.mov",
        snapshots=[
            FrameSnapshot(0.0, "zero"),
            FrameSnapshot(1.5, "one"),
            FrameSnapshot(3.25, "two"),
        ],
        frames_analyzed=3,
        ocr_text_chars=11,
    )
    out = tmp_path / "ocr.json"

    store.save(result, out)
    loaded = store.load(out)

    assert [s.timestamp_seconds for s in loaded.snapshots] == [0.0, 1.5, 3.25]
    assert [s.ocr_text for s in loaded.snapshots] == ["zero", "one", "two"]


def test_empty_ocr_text_round_trips(tmp_path: Path) -> None:
    store = OcrSnapshotStore()
    result = AnalyzeResult(
        source=tmp_path / "clip.mov",
        snapshots=[FrameSnapshot(0.0, ""), FrameSnapshot(1.0, "   ")],
        frames_analyzed=2,
        ocr_text_chars=0,
    )
    out = tmp_path / "ocr.json"

    store.save(result, out)
    loaded = store.load(out)

    assert loaded == result
    assert [s.ocr_text for s in loaded.snapshots] == ["", "   "]


def test_no_snapshots_round_trips(tmp_path: Path) -> None:
    store = OcrSnapshotStore()
    result = AnalyzeResult(source=tmp_path / "clip.mov")
    out = tmp_path / "ocr.json"

    store.save(result, out)
    loaded = store.load(out)

    assert loaded == result
    assert loaded.snapshots == []


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    store = OcrSnapshotStore()
    out = tmp_path / "nested" / "dir" / "ocr.json"

    store.save(_result(tmp_path / "clip.mov"), out)

    assert out.exists()


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    out = tmp_path / "ocr.json"
    out.write_text(json.dumps({"schema": "bogus/v9", "source": "x"}), encoding="utf-8")

    with pytest.raises(OcrArtifactError, match="schema"):
        OcrSnapshotStore().load(out)


def test_load_rejects_malformed_json(tmp_path: Path) -> None:
    out = tmp_path / "ocr.json"
    out.write_text("{not json", encoding="utf-8")

    with pytest.raises(OcrArtifactError, match="not valid JSON"):
        OcrSnapshotStore().load(out)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OcrArtifactError, match="Could not read"):
        OcrSnapshotStore().load(tmp_path / "absent.json")
