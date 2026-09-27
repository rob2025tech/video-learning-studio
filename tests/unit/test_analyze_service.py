"""Unit tests for AnalyzeService: orchestration with fakes, no binaries invoked."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_learning.adapters.ffmpeg_frames import ExtractedFrame
from video_learning.core.errors import FileNotFoundError_
from video_learning.core.models import MediaInfo, VideoStreamInfo
from video_learning.services.analyze_service import AnalyzeService
from video_learning.services.inspect_service import InspectService


class FakeProbe:
    def __init__(self, media: MediaInfo) -> None:
        self._media = media
        self.calls: list[Path] = []

    def probe(self, path: Path) -> MediaInfo:
        self.calls.append(path)
        return self._media


class FakeFrameExtractor:
    def __init__(self, frames: list[ExtractedFrame]) -> None:
        self._frames = frames
        self.calls: list[tuple[MediaInfo, Path]] = []

    def extract(self, media: MediaInfo, out_dir: Path) -> list[ExtractedFrame]:
        self.calls.append((media, out_dir))
        return self._frames


class FakeOcr:
    def __init__(self, text_by_frame: dict[str, str]) -> None:
        self._text = text_by_frame
        self.seen: list[Path] = []

    def extract_text(self, image: Path) -> str:
        self.seen.append(image)
        return self._text.get(image.name, "")


def _service(
    tmp_path: Path, ocr_text: dict[str, str], frame_names: list[str]
) -> tuple[AnalyzeService, FakeProbe, FakeFrameExtractor, FakeOcr]:
    media = MediaInfo(
        path=tmp_path / "clip.mov",
        filename="clip.mov",
        duration_seconds=10.0,
        video=VideoStreamInfo(width=1280, height=720),
    )
    frames = [
        ExtractedFrame(timestamp_seconds=float(index), image_path=tmp_path / name)
        for index, name in enumerate(frame_names)
    ]
    probe = FakeProbe(media)
    extractor = FakeFrameExtractor(frames)
    ocr = FakeOcr(ocr_text)
    service = AnalyzeService(
        inspect_service=InspectService(probe=probe),
        frame_extractor=extractor,  # type: ignore[arg-type]
        ocr=ocr,  # type: ignore[arg-type]
    )
    return service, probe, extractor, ocr


def test_reuses_inspection_pipeline(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    service, probe, extractor, _ = _service(
        tmp_path, {"frame_000.png": "Qoder terminal"}, ["frame_000.png"]
    )

    service.analyze(video)

    assert probe.calls == [video]  # went through InspectService -> MediaProbe once
    assert len(extractor.calls) == 1


def test_aggregates_keywords_across_frames(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    service, _, _, _ = _service(
        tmp_path,
        {
            "frame_000.png": "Qoder terminal coding",
            "frame_001.png": "Qoder coding python",
        },
        ["frame_000.png", "frame_001.png"],
    )

    result = service.analyze(video)

    terms = [kw.term for kw in result.keywords]
    assert "Qoder" in terms and "coding" in terms
    assert result.frames_analyzed == 2
    assert result.ocr_text_chars > 0
    # Non-destructive guarantees are surfaced explicitly.
    assert result.to_dict()["renamed"] is False
    assert result.to_dict()["source_modified"] is False


def test_snapshots_bind_timestamp_to_ocr_text(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    service, _, _, _ = _service(
        tmp_path,
        {
            "frame_000.png": "Projects Templates studio",
            "frame_001.png": "Release Notes",
        },
        ["frame_000.png", "frame_001.png"],
    )

    result = service.analyze(video)

    # One snapshot per frame, in order, each keeping its own timestamp + text.
    assert [s.timestamp_seconds for s in result.snapshots] == [0.0, 1.0]
    assert result.snapshots[0].ocr_text == "Projects Templates studio"
    assert result.snapshots[1].ocr_text == "Release Notes"


def test_result_dict_exposes_snapshots(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    service, _, _, _ = _service(
        tmp_path, {"frame_000.png": "Qoder Model Usage"}, ["frame_000.png"]
    )

    payload = service.analyze(video).to_dict()

    assert payload["snapshots"] == [
        {"timestamp_seconds": 0.0, "ocr_text": "Qoder Model Usage"}
    ]
    assert payload["frames_analyzed"] == 1
    # Existing fields preserved.
    for key in ("source", "keywords", "ocr_text_chars", "renamed", "applied"):
        assert key in payload


def test_no_text_yields_empty_keywords(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    service, _, _, _ = _service(tmp_path, {"frame_000.png": "  \n "}, ["frame_000.png"])

    result = service.analyze(video)

    assert result.keywords == []
    assert result.frames_analyzed == 1


def test_phrases_populated_from_same_ocr_text(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    service, _, _, _ = _service(
        tmp_path,
        {
            "frame_000.png": "Release Notes\nQoder Model Usage",
            "frame_001.png": "Release Notes\nQoder Model Usage",
        },
        ["frame_000.png", "frame_001.png"],
    )

    result = service.analyze(video)

    phrases = [p.phrase for p in result.phrases]
    assert "Release Notes" in phrases
    assert "Qoder Model Usage" in phrases
    # Existing keyword behaviour is unchanged and drawn from the same text.
    assert "Qoder" in [kw.term for kw in result.keywords]
    # Snapshots remain intact (no second OCR pass, no behaviour change).
    assert [s.timestamp_seconds for s in result.snapshots] == [0.0, 1.0]


def test_result_dict_exposes_phrases_and_preserves_fields(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    service, _, _, _ = _service(
        tmp_path,
        {"frame_000.png": "Release Notes", "frame_001.png": "Release Notes"},
        ["frame_000.png", "frame_001.png"],
    )

    payload = service.analyze(video).to_dict()

    assert {"phrase": "Release Notes", "count": 2} in payload["phrases"]
    for key in (
        "source",
        "snapshots",
        "keywords",
        "frames_analyzed",
        "ocr_text_chars",
        "renamed",
        "applied",
        "source_modified",
    ):
        assert key in payload


def test_missing_file_raises_before_any_work(tmp_path: Path) -> None:
    service, probe, extractor, ocr = _service(tmp_path, {}, ["frame_000.png"])

    with pytest.raises(FileNotFoundError_):
        service.analyze(tmp_path / "does-not-exist.mov")

    assert probe.calls == []
    assert extractor.calls == []
    assert ocr.seen == []
