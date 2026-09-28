"""Integration test: real FFmpeg scdet over a synthetic red->blue hard cut.

Proves the actual FFmpeg path emits the authoritative
``lavfi.scd.score: ... lavfi.scd.time: ...`` format and that the adapter extracts
a numeric visual change near the known cut. The existing static real-video report
fixture produces NO visual changes, so it never exercises the numeric scdet
parser; this tiny synthetic fixture does. Skips unless ffmpeg/ffprobe are present.
The clip is generated in a temp dir and is only ever read.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from video_learning.adapters.ffmpeg_scene import FfmpegSceneDetector
from video_learning.adapters.ffprobe_media import FfprobeMediaProbe
from video_learning.core.models import MediaInfo
from video_learning.core.scene import VISUAL_CHANGE_THRESHOLD
from video_learning.services.inspect_service import InspectService
from video_learning.services.segment_service import SegmentService

_HAS_BINS = all(shutil.which(b) is not None for b in ("ffmpeg", "ffprobe"))

pytestmark = pytest.mark.skipif(
    not _HAS_BINS, reason="requires ffmpeg and ffprobe installed on PATH"
)

_CUT_SECONDS = 3.0
_TOLERANCE_SECONDS = 1.0


@pytest.fixture(scope="module")
def red_blue_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny 6s clip: 3s solid red then 3s solid blue -> one hard cut at ~3.0s."""
    out = tmp_path_factory.mktemp("media") / "red_blue.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=red:s=64x64:d=3:r=5",
                "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=3:r=5",
                "-filter_complex",
                "[0:v]format=yuv420p[r];[1:v]format=yuv420p[b];[r][b]concat=n=2:v=1:a=0[v]",
                "-map", "[v]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        pytest.skip(f"could not generate red->blue fixture: {exc}")
    return out


@pytest.fixture(scope="module")
def clip_media(red_blue_clip: Path) -> MediaInfo:
    return InspectService(probe=FfprobeMediaProbe()).inspect(red_blue_clip)


def test_real_ffmpeg_scdet_detects_the_hard_cut(
    clip_media: MediaInfo,
) -> None:
    assert clip_media.has_video
    changes = FfmpegSceneDetector().detect(clip_media)

    # The real FFmpeg path emitted the authoritative scdet format and our parser
    # extracted at least one numeric (timestamp, score) change.
    assert len(changes) >= 1
    near_cut = [
        c for c in changes if abs(c.timestamp_seconds - _CUT_SECONDS) <= _TOLERANCE_SECONDS
    ]
    assert near_cut, f"no detected change near the {_CUT_SECONDS}s cut: {changes}"
    # scdet only emits changes at/above the configured threshold; a solid red->blue
    # cut is a strong change above it.
    assert all(c.change_score >= VISUAL_CHANGE_THRESHOLD for c in changes)
    assert any(c.change_score > VISUAL_CHANGE_THRESHOLD for c in near_cut)


def test_real_scdet_is_deterministic_and_read_only(
    red_blue_clip: Path, clip_media: MediaInfo
) -> None:
    detector = FfmpegSceneDetector()
    before = red_blue_clip.stat()
    first = detector.detect(clip_media)
    second = detector.detect(clip_media)
    after = red_blue_clip.stat()

    first_pairs = [(c.timestamp_seconds, c.change_score) for c in first]
    second_pairs = [(c.timestamp_seconds, c.change_score) for c in second]
    assert first_pairs == second_pairs
    # Segmentation only ever reads the source.
    assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime)


def test_real_segmentation_ends_at_authoritative_duration(clip_media: MediaInfo) -> None:
    service = SegmentService(scene_detector=FfmpegSceneDetector())
    result = service.segment(clip_media, None)

    # Real ffprobe duration is authoritative and known for this fixture.
    assert clip_media.duration_seconds is not None
    assert result.video_duration_seconds == clip_media.duration_seconds
    assert result.segment_count >= 1
    assert result.segments[0].start_seconds == 0.0
    # The final segment ends EXACTLY at the authoritative duration (never at the
    # latest evidence timestamp).
    assert result.segments[-1].end_seconds == clip_media.duration_seconds
    # The real cut became visual evidence.
    assert len(result.visual_changes) >= 1
