"""Concrete MediaProbe backed by the system ``ffprobe`` binary."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from video_learning.core.errors import (
    MalformedProbeOutputError,
    ProbeToolMissingError,
    UnreadableMediaError,
)
from video_learning.core.models import AudioStreamInfo, MediaInfo, VideoStreamInfo

FFPROBE_COMMAND = [
    "ffprobe",
    "-v", "quiet",
    "-print_format", "json",
    "-show_format",
    "-show_streams",
]

_PROBE_TIMEOUT_SECONDS = 60


class FfprobeMediaProbe:
    """Runs ``ffprobe`` as a subprocess and maps its JSON onto core models."""

    def __init__(self, command: list[str] | None = None) -> None:
        self._command = command if command is not None else list(FFPROBE_COMMAND)

    def probe(self, path: Path) -> MediaInfo:
        data = self._run_ffprobe(path)
        return self.build_media_info(path, data)

    # -- subprocess boundary ------------------------------------------------

    def _run_ffprobe(self, path: Path) -> dict[str, Any]:
        if shutil.which("ffprobe") is None:
            raise ProbeToolMissingError(
                "ffprobe was not found on PATH. Install FFmpeg first, e.g. `brew install ffmpeg`."
            )
        command = [*self._command, str(path)]
        try:
            result = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise UnreadableMediaError(
                f"ffprobe timed out after {_PROBE_TIMEOUT_SECONDS}s while reading '{path.name}'."
            ) from exc
        except OSError as exc:
            raise ProbeToolMissingError(f"Could not execute ffprobe: {exc}") from exc

        if result.returncode != 0:
            stderr_hint = (result.stderr or "").strip().splitlines()
            detail = stderr_hint[-1] if stderr_hint else "no details reported by ffprobe"
            raise UnreadableMediaError(
                f"ffprobe could not read '{path.name}' as media "
                f"(exit code {result.returncode}): {detail}"
            )

        return self._parse_json(result.stdout, path)

    def _parse_json(self, stdout: str, path: Path) -> dict[str, Any]:
        if not stdout.strip():
            raise MalformedProbeOutputError(
                f"ffprobe returned no output for '{path.name}'."
            )
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise MalformedProbeOutputError(
                f"ffprobe output for '{path.name}' was not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise MalformedProbeOutputError(
                f"ffprobe output for '{path.name}' was not a JSON object."
            )
        return data

    # -- mapping ffprobe JSON -> core models --------------------------------

    def build_media_info(self, path: Path, data: dict[str, Any]) -> MediaInfo:
        """Pure mapping: ffprobe JSON plus filesystem facts -> MediaInfo."""
        fmt = _as_dict(data.get("format"))
        streams = _as_list(data.get("streams"))

        size_bytes = _to_int(fmt.get("size"))
        modified_at = _file_mtime(path)

        return MediaInfo(
            path=path,
            filename=path.name,
            size_bytes=size_bytes,
            modified_at=modified_at,
            container_format=_clean_container(fmt.get("format_name")),
            duration_seconds=_to_float(fmt.get("duration"))
            or _duration_from_streams(streams),
            video=_first_video_stream(streams),
            audio=_first_audio_stream(streams),
        )


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _clean_container(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    # ffprobe reports e.g. "mov,mp4,m4a,3gp,3g2,mj2"; keep the first label.
    return raw.split(",")[0]


def _first_stream_of_type(streams: list[Any], kind: str) -> dict[str, Any]:
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == kind:
            return stream
    return {}


def _first_video_stream(streams: list[Any]) -> VideoStreamInfo | None:
    stream = _first_stream_of_type(streams, "video")
    if not stream:
        return None
    # Attached pictures (album art) are not real video streams.
    if isinstance(stream.get("disposition"), dict) and stream["disposition"].get(
        "attached_pic"
    ) == 1:
        return None
    # "0/0" is a non-empty string, so parse each candidate separately
    # instead of relying on `or`.
    frame_rate = _parse_frame_rate(stream.get("avg_frame_rate"))
    if frame_rate is None:
        frame_rate = _parse_frame_rate(stream.get("r_frame_rate"))
    return VideoStreamInfo(
        codec=stream.get("codec_name") if isinstance(stream.get("codec_name"), str) else None,
        width=_to_int(stream.get("width")),
        height=_to_int(stream.get("height")),
        frame_rate=frame_rate,
        pixel_format=stream.get("pix_fmt") if isinstance(stream.get("pix_fmt"), str) else None,
    )


def _first_audio_stream(streams: list[Any]) -> AudioStreamInfo | None:
    stream = _first_stream_of_type(streams, "audio")
    if not stream:
        return None
    tags = _as_dict(stream.get("tags"))
    language = tags.get("language")
    return AudioStreamInfo(
        codec=stream.get("codec_name") if isinstance(stream.get("codec_name"), str) else None,
        sample_rate=_to_int(stream.get("sample_rate")),
        channels=_to_int(stream.get("channels")),
        language=language if isinstance(language, str) else None,
    )


def _parse_frame_rate(raw: Any) -> float | None:
    """Parse ffprobe rational frame rates like ``30000/1001``; ``0/0`` -> None."""
    if not isinstance(raw, str) or "/" not in raw:
        return _to_float(raw)
    num_str, _, den_str = raw.partition("/")
    num = _to_float(num_str)
    den = _to_float(den_str)
    if num is None or den is None or den == 0:
        return None
    return round(num / den, 3)


def _duration_from_streams(streams: list[Any]) -> float | None:
    """Fallback for containers (e.g. raw streams) without format-level duration."""
    for stream in streams:
        if isinstance(stream, dict):
            duration = _to_float(stream.get("duration"))
            if duration is not None:
                return duration
    return None


def _to_float(raw: Any) -> float | None:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _to_int(raw: Any) -> int | None:
    value = _to_float(raw)
    if value is None:
        return None
    return int(value)


def _file_mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None
