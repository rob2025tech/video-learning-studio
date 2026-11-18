"""Shared fixtures: canned ffprobe and whisper.cpp output for mocked tests.

Also holds the synthetic report artifact the search CLI and API tests share. It is
built from core models and written by the production ``ReportStore``, so both surfaces
are exercised against the real on-disk shape. Nothing here imports the optional API
dependencies: the CLI test suite must stay runnable without the ``api`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from video_learning.core.keywords import KeywordSuggestion, PhraseSuggestion
from video_learning.core.models import AudioStreamInfo, MediaInfo, VideoStreamInfo
from video_learning.core.segments import Segment, Segmentation
from video_learning.core.timeline import OcrEvent, SpeechEvent, build_timeline
from video_learning.core.transcript import TranscriptResult, TranscriptSegment
from video_learning.services.analyze_service import AnalyzeResult, FrameSnapshot
from video_learning.services.report_service import ReportResult, ReportStore

FFPROBE_JSON_MOV: dict[str, Any] = {
    "streams": [
        {
            "index": 0,
            "codec_name": "h264",
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "pix_fmt": "yuv420p",
            "avg_frame_rate": "30000/1001",
            "r_frame_rate": "30000/1001",
            "duration": "62.500000",
        },
        {
            "index": 1,
            "codec_name": "aac",
            "codec_type": "audio",
            "sample_rate": "48000",
            "channels": 2,
            "tags": {"language": "eng"},
            "duration": "62.500000",
        },
    ],
    "format": {
        "filename": "/tmp/sample.mov",
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration": "62.500000",
        "size": "12345678",
    },
}

FFPROBE_JSON_VIDEO_ONLY: dict[str, Any] = {
    "streams": [
        {
            "index": 0,
            "codec_name": "hevc",
            "codec_type": "video",
            "width": 3840,
            "height": 2160,
            "pix_fmt": "yuv420p10le",
            "avg_frame_rate": "0/0",
            "r_frame_rate": "60/1",
        },
    ],
    "format": {
        "format_name": "matroska,webm",
        "size": "999",
    },
}

FFPROBE_JSON_AUDIO_ONLY: dict[str, Any] = {
    "streams": [
        {
            "index": 0,
            "codec_name": "mp3",
            "codec_type": "audio",
            "sample_rate": "44100",
            "channels": 1,
            "duration": "180.0",
        },
    ],
    "format": {
        "format_name": "mp3",
        "duration": "180.0",
        "size": "2880000",
    },
}

# Representative whisper.cpp `--output-json` (`-oj`) structure. Segment offsets
# are integer MILLISECONDS; text carries a leading space as whisper.cpp emits.
# NOTE: whisper.cpp is not installed in this environment, so this schema follows
# the documented output format and must be verified on first real use.
WHISPER_CPP_JSON: dict[str, Any] = {
    "systeminfo": "avx = 1 | avx2 = 0 | neon = 0",
    "model": {"type": "base", "multilingual": False, "vocab": 51865},
    "params": {
        "model": "models/ggml-base.en.bin",
        "language": "en",
        "translate": False,
    },
    "result": {"language": "en"},
    "transcription": [
        {
            "timestamps": {"from": "00:00:00,000", "to": "00:00:03,840"},
            "offsets": {"from": 0, "to": 3840},
            "text": " Today we're going to build a small command line tool.",
        },
        {
            "timestamps": {"from": "00:00:03,840", "to": "00:00:07,200"},
            "offsets": {"from": 3840, "to": 7200},
            "text": " It reads a video and prints what it sees.",
        },
    ],
}


@pytest.fixture
def ffprobe_json_mov() -> str:
    return json.dumps(FFPROBE_JSON_MOV)


@pytest.fixture
def ffprobe_json_video_only() -> str:
    return json.dumps(FFPROBE_JSON_VIDEO_ONLY)


@pytest.fixture
def ffprobe_json_audio_only() -> str:
    return json.dumps(FFPROBE_JSON_AUDIO_ONLY)


@pytest.fixture
def whisper_cpp_json() -> str:
    return json.dumps(WHISPER_CPP_JSON)


SYNTHETIC_OCR_TEXT = "build a search index locally"


def synthetic_report(
    video: Path, *, speech_records: int = 6, ocr_text: str = SYNTHETIC_OCR_TEXT
) -> ReportResult:
    """The search CLI's synthetic report, with a tunable record count for size tests.

    ``video`` is deliberately never created: reading a report must not need the video it
    names. Defaults reproduce ``tests/unit/test_search_cli.py``'s fixture exactly, so the
    API and the CLI are compared against one artifact rather than two lookalikes.
    """
    speech = tuple(
        TranscriptSegment(index * 4.0, index * 4.0 + 3.0, f"the index ranks record {index}")
        for index in range(speech_records)
    )
    duration = max(24.0, speech_records * 4.0)
    return ReportResult(
        source=video,
        media=MediaInfo(
            path=video,
            filename=video.name,
            container_format="mov",
            duration_seconds=duration,
            video=VideoStreamInfo(
                codec="h264", width=1920, height=1080, frame_rate=30.0, pixel_format="yuv420p"
            ),
            audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2, language="und"),
        ),
        analysis=AnalyzeResult(
            source=video,
            keywords=[KeywordSuggestion("index", speech_records + 1)],
            phrases=[PhraseSuggestion("search index", 2)],
            snapshots=[FrameSnapshot(0.5, ocr_text)],
            frames_analyzed=1,
            ocr_text_chars=len(ocr_text),
        ),
        transcription=TranscriptResult(
            source=str(video),
            backend="whisper.cpp",
            model="ggml-base.en",
            language="en",
            segments=speech,
        ),
        transcription_status="ok",
        transcription_detail=None,
        segmentation=Segmentation(
            visual_changes=(),
            speech_boundaries=(),
            boundaries=(),
            segments=(Segment(start_seconds=0.0, end_seconds=duration, boundary_reason=None),),
            video_duration_seconds=duration,
        ),
        timeline=build_timeline(
            [OcrEvent(0.5, ocr_text)],
            [SpeechEvent(s.start_seconds, s.end_seconds, s.text) for s in speech],
        ),
    )


@pytest.fixture
def report_artifact_bytes(tmp_path: Path) -> bytes:
    """A real ``video-learning.report/v1`` artifact, written by the production store."""
    out = tmp_path / "report.json"
    ReportStore().save(synthetic_report(tmp_path / "clip.mov"), out)
    return out.read_bytes()


@pytest.fixture
def report_artifact_text(report_artifact_bytes: bytes) -> str:
    return report_artifact_bytes.decode("utf-8")
