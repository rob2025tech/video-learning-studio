"""Shared fixtures: canned ffprobe output for mocked unit tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

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


@pytest.fixture
def ffprobe_json_mov() -> str:
    return json.dumps(FFPROBE_JSON_MOV)


@pytest.fixture
def ffprobe_json_video_only() -> str:
    return json.dumps(FFPROBE_JSON_VIDEO_ONLY)


@pytest.fixture
def ffprobe_json_audio_only() -> str:
    return json.dumps(FFPROBE_JSON_AUDIO_ONLY)
