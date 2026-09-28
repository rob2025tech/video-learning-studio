"""Integration test: real report over a generated text + audio video.

Skips unless ffmpeg/ffprobe/tesseract are present (the same guard as the analyze
integration test). whisper.cpp is NOT required: transcription is expected to be
reported as ``unavailable`` here, which is exactly the deterministic path under
test. No binary is downloaded and no network call is made; the source is read-only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from video_learning.adapters.ffmpeg_audio import FfmpegAudioExtractor
from video_learning.adapters.ffmpeg_frames import FfmpegFrameExtractor
from video_learning.adapters.ffmpeg_scene import FfmpegSceneDetector
from video_learning.adapters.ffprobe_media import FfprobeMediaProbe
from video_learning.adapters.tesseract_ocr import TesseractOcr
from video_learning.adapters.whisper_cpp import WhisperCpp
from video_learning.core.timeline import OcrEvent
from video_learning.services.analyze_service import AnalyzeService
from video_learning.services.inspect_service import InspectService
from video_learning.services.report_service import REPORT_SCHEMA, ReportService, ReportStore
from video_learning.services.segment_service import SegmentService
from video_learning.services.transcribe_service import TranscribeService

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]

_HAS_BINS = all(shutil.which(b) is not None for b in ("ffmpeg", "ffprobe", "tesseract"))

pytestmark = pytest.mark.skipif(
    not _HAS_BINS, reason="requires ffmpeg, ffprobe and tesseract installed on PATH"
)


def _first_font() -> str | None:
    return next((font for font in _FONTS if Path(font).exists()), None)


def _service() -> ReportService:
    whisper = WhisperCpp()
    return ReportService(
        inspect_service=InspectService(probe=FfprobeMediaProbe()),
        analyze_service=AnalyzeService(
            inspect_service=InspectService(probe=FfprobeMediaProbe()),
            frame_extractor=FfmpegFrameExtractor(max_frames=2),
            ocr=TesseractOcr(),
        ),
        transcribe_service=TranscribeService(
            inspect_service=InspectService(probe=FfprobeMediaProbe()),
            audio_extractor=FfmpegAudioExtractor(),
            whisper=whisper,
        ),
        segment_service=SegmentService(scene_detector=FfmpegSceneDetector()),
        whisper=whisper,
    )


@pytest.fixture(scope="module")
def text_audio_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    font = _first_font()
    if font is None:
        pytest.skip("no usable system font found for drawtext")
    out = tmp_path_factory.mktemp("media") / "words.mov"
    drawtext = (
        f"drawtext=fontfile={font}:text='QODER TERMINAL CODING':"
        "fontcolor=white:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2"
    )
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=2:r=5",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                "-vf", drawtext,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        pytest.skip(f"could not generate drawtext+audio fixture: {exc}")
    return out


def test_real_report_is_deterministic_and_read_only(
    text_audio_video: Path, tmp_path: Path
) -> None:
    service = _service()
    store = ReportStore()

    before = text_audio_video.stat()
    report = service.build(text_audio_video)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    store.save(report, first)
    store.save(service.build(text_audio_video), second)
    after = text_audio_video.stat()

    # Metadata and OCR are real and present.
    assert report.media.duration_seconds is not None
    assert report.analysis.frames_analyzed >= 1
    # The unified timeline is chronologically ordered and keeps OCR as a point.
    times = [event.sort_key[0] for event in report.timeline]
    assert times == sorted(times)
    assert any(isinstance(event, OcrEvent) for event in report.timeline)
    # Transcription is reported explicitly (unavailable here) — never invented.
    assert report.transcription_status in {"no-audio", "unavailable", "ok"}
    if report.transcription_status != "ok":
        assert report.transcription is None
    # Stage 0I segments are present; a static clip may have no visual changes but
    # always yields at least the full-duration segment starting at 0.0.
    segments_payload = report.to_dict()["segments"]
    assert segments_payload["segment_count"] >= 1
    assert segments_payload["segments"][0]["start_seconds"] == 0.0
    # The source video is never modified.
    assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime)
    # Two independent builds serialize byte-identically (determinism).
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8"))["schema"] == REPORT_SCHEMA
