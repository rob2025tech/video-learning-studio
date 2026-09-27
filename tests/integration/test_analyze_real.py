"""Integration test: real ffmpeg + tesseract over a generated text video.

No binary is committed — the fixture is a short clip with large on-screen words
rendered via ffmpeg's drawtext filter. Skipped when any dependency is missing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from video_learning.adapters.ffmpeg_frames import FfmpegFrameExtractor
from video_learning.adapters.ffprobe_media import FfprobeMediaProbe
from video_learning.adapters.tesseract_ocr import TesseractOcr
from video_learning.services.analyze_service import AnalyzeService
from video_learning.services.inspect_service import InspectService

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
    return next((f for f in _FONTS if Path(f).exists()), None)


@pytest.fixture(scope="module")
def text_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    font = _first_font()
    if font is None:
        pytest.skip("no usable system font found for drawtext")
    out = tmp_path_factory.mktemp("media") / "words.mp4"
    drawtext = (
        f"drawtext=fontfile={font}:text='QODER TERMINAL CODING':"
        "fontcolor=white:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2"
    )
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=2:r=5",
                "-vf", drawtext,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        pytest.skip(f"could not generate drawtext fixture (ffmpeg may lack libfreetype): {exc}")
    return out


def test_real_pipeline_reads_on_screen_text(text_video: Path) -> None:
    service = AnalyzeService(
        inspect_service=InspectService(probe=FfprobeMediaProbe()),
        frame_extractor=FfmpegFrameExtractor(max_frames=3),
        ocr=TesseractOcr(),
    )

    result = service.analyze(text_video)

    lowered = {kw.term.lower() for kw in result.keywords}
    assert result.frames_analyzed >= 1
    # At least one of the large rendered words should be recovered by OCR.
    assert lowered & {"qoder", "terminal", "coding"}, result.keywords
