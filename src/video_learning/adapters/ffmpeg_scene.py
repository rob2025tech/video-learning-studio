"""Adapter: visual-change evidence via FFmpeg's ``scdet`` filter (Stage 0I).

Runs a dedicated full-video decode pass and reads ``scdet`` scores from FFmpeg's
stderr. The verified FFmpeg 8.0 line has the atomic form::

    [scdet @ 0x7fe4c2e0f880] lavfi.scd.score: 15.625, lavfi.scd.time: 3

Scores are on a 0–100 scale. All FFmpeg-specific parsing is isolated here; the
core domain (:class:`~video_learning.core.scene.VisualChange`) never sees stderr,
regexes, ``lavfi.scd.*`` keys, or command-line arguments.

``metadata=print`` is deliberately NOT used: although it can emit
``lavfi.scd.score=``/``lavfi.scd.time=`` on separate lines, the atomic stderr form
keeps score and timestamp together, bounds the parse, needs no temporary
metadata-output workflow, and keeps FFmpeg scraping in one adapter. This stderr
form is an implementation detail, not a general machine-readable API. The source
video is only ever read (``-f null -``); no artifacts are written.
"""

from __future__ import annotations

import re
import shutil
import subprocess

from video_learning.core.errors import AnalysisToolMissingError, SceneDetectionError
from video_learning.core.models import MediaInfo
from video_learning.core.scene import VISUAL_CHANGE_THRESHOLD, VisualChange

_SCENE_TIMEOUT_SECONDS = 600

# Matches the atomic scdet line while tolerating/ignoring the leading
# "[scdet @ 0x...]" prefix. Score and time are captured loosely so a malformed
# value is surfaced as a parse failure rather than silently skipped.
_SCDET_RE = re.compile(
    r"lavfi\.scd\.score:\s*(?P<score>[^,]+),\s*lavfi\.scd\.time:\s*(?P<time>\S+)"
)


class FfmpegSceneDetector:
    """Detects candidate visual changes in a video using FFmpeg's scdet filter."""

    def detect(self, media: MediaInfo) -> list[VisualChange]:
        """Return ordered visual-change evidence for ``media``.

        No video stream is a valid, non-error condition that yields ``[]`` without
        invoking FFmpeg. A missing FFmpeg binary reuses the project's existing
        :class:`AnalysisToolMissingError`; timeouts, launch failures, nonzero exit
        codes, and malformed scdet output raise :class:`SceneDetectionError`.
        """
        if not media.has_video:
            return []
        if shutil.which("ffmpeg") is None:
            raise AnalysisToolMissingError(
                "ffmpeg was not found on PATH. Install FFmpeg first, e.g. `brew install ffmpeg`."
            )
        command = [
            "ffmpeg", "-nostdin",
            "-i", str(media.path),
            "-an", "-sn",
            "-vf", f"scdet=threshold={VISUAL_CHANGE_THRESHOLD}",
            "-f", "null", "-",
        ]
        return self._parse(self._run(command, media.filename), media.filename)

    def _run(self, command: list[str], filename: str) -> str:
        """Run the decode pass and return its stderr (where scdet reports)."""
        try:
            result = subprocess.run(  # noqa: S603
                command, capture_output=True, text=True, timeout=_SCENE_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as exc:
            raise SceneDetectionError(
                f"ffmpeg timed out after {_SCENE_TIMEOUT_SECONDS}s detecting visual "
                f"changes in '{filename}'."
            ) from exc
        except OSError as exc:
            raise SceneDetectionError(f"Could not execute ffmpeg: {exc}") from exc
        if result.returncode != 0:
            lines = (result.stderr or "").strip().splitlines()
            detail = lines[-1] if lines else f"exit code {result.returncode}"
            raise SceneDetectionError(
                f"ffmpeg could not detect visual changes in '{filename}': {detail}"
            )
        return result.stderr or ""

    def _parse(self, stderr: str, filename: str) -> list[VisualChange]:
        """Extract every scdet line into a :class:`VisualChange`, in order."""
        changes: list[VisualChange] = []
        for match in _SCDET_RE.finditer(stderr):
            try:
                change_score = float(match.group("score"))
                timestamp_seconds = float(match.group("time"))
            except ValueError as exc:
                raise SceneDetectionError(
                    f"ffmpeg produced a malformed scdet line for '{filename}': "
                    f"{match.group(0)!r}"
                ) from exc
            changes.append(
                VisualChange(timestamp_seconds=timestamp_seconds, change_score=change_score)
            )
        return changes
