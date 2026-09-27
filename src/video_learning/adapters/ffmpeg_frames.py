"""Frame extraction via the system ``ffmpeg`` binary (already required by Stage 0).

Samples a handful of evenly-spaced frames into a caller-provided directory and
returns them together with the timestamp each was taken at. The source video is
only ever read.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from video_learning.core.errors import AnalysisToolMissingError, FrameExtractionError
from video_learning.core.models import MediaInfo

_FRAME_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class ExtractedFrame:
    """A single sampled frame and the media timestamp it was captured at.

    ``image_path`` points into temporary storage owned by the caller; it is only
    valid for the lifetime of that directory.
    """

    timestamp_seconds: float
    image_path: Path


class FfmpegFrameExtractor:
    """Extracts representative still frames from a video for OCR."""

    def __init__(self, max_frames: int = 5, max_width: int = 1600) -> None:
        self._max_frames = max_frames
        self._max_width = max_width

    def extract(self, media: MediaInfo, out_dir: Path) -> list[ExtractedFrame]:
        if not media.has_video:
            raise FrameExtractionError(
                f"'{media.filename}' has no video stream to analyze "
                "(this feature reads on-screen text)."
            )
        if shutil.which("ffmpeg") is None:
            raise AnalysisToolMissingError(
                "ffmpeg was not found on PATH. Install FFmpeg first, e.g. `brew install ffmpeg`."
            )

        timestamps = self._timestamps(media.duration_seconds)
        scale_filter = self._scale_filter(media)

        frames: list[ExtractedFrame] = []
        for index, seconds in enumerate(timestamps):
            out_path = out_dir / f"frame_{index:03d}.png"
            command = [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{seconds:.3f}",
                "-i", str(media.path),
                "-frames:v", "1",
            ]
            if scale_filter is not None:
                command += ["-vf", scale_filter]
            command.append(str(out_path))

            self._run(command, media)
            if out_path.exists():
                frames.append(ExtractedFrame(timestamp_seconds=seconds, image_path=out_path))

        if not frames:
            raise FrameExtractionError(
                f"ffmpeg produced no frames for '{media.filename}'."
            )
        return frames

    def _run(self, command: list[str], media: MediaInfo) -> None:
        try:
            result = subprocess.run(  # noqa: S603
                command, capture_output=True, text=True, timeout=_FRAME_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as exc:
            raise FrameExtractionError(
                f"ffmpeg timed out extracting a frame from '{media.filename}'."
            ) from exc
        except OSError as exc:
            raise AnalysisToolMissingError(f"Could not execute ffmpeg: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            hint = detail[-1] if detail else "no details reported by ffmpeg"
            raise FrameExtractionError(
                f"ffmpeg failed to extract a frame from '{media.filename}': {hint}"
            )

    def _scale_filter(self, media: MediaInfo) -> str | None:
        """Downscale wide frames (helps OCR speed); never upscale."""
        width = media.video.width if media.video is not None else None
        if width is not None and width > self._max_width:
            return f"scale={self._max_width}:-2"
        return None

    def _timestamps(self, duration: float | None) -> list[float]:
        """Midpoints of N equal segments; avoids the very first/last frames."""
        if duration is None or duration <= 0:
            return [0.0]
        count = max(1, self._max_frames)
        step = duration / count
        return [min(duration - 0.05, (i + 0.5) * step) for i in range(count)]
