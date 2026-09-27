"""Typer application entry point.

Command handlers stay thin: parse args -> call service -> render output.
All rendering helpers live in ``video_learning.cli.render``.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from video_learning import __version__
from video_learning.adapters.ffmpeg_audio import FfmpegAudioExtractor
from video_learning.adapters.ffmpeg_frames import FfmpegFrameExtractor
from video_learning.adapters.ffprobe_media import FfprobeMediaProbe
from video_learning.adapters.tesseract_ocr import TesseractOcr
from video_learning.adapters.whisper_cpp import WhisperCpp
from video_learning.cli.render import (
    render_analysis,
    render_media_info,
    render_transcript,
    render_transcription_capability,
)
from video_learning.core.errors import VideoLearningError
from video_learning.services.analyze_service import AnalyzeService
from video_learning.services.inspect_service import InspectService
from video_learning.services.transcribe_service import TranscribeService

app = typer.Typer(
    name="video-learning",
    help="Local-first toolkit for turning videos on your Mac into a personal learning system.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()
error_console = Console(stderr=True, style="bold red")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"video-learning {__version__}")
        raise typer.Exit(0)


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Top-level callback (handles --version)."""


@app.command()
def inspect(
    video: Path = typer.Argument(
        ...,
        exists=False,  # validated by the service so errors stay user-facing
        help="Path to a local video file.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a rich table."
    ),
) -> None:
    """Inspect a local video file and report its media metadata."""
    service = InspectService(probe=FfprobeMediaProbe())
    try:
        info = service.inspect(video)
    except VideoLearningError as exc:
        error_console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        # Plain print keeps stdout strictly valid JSON (rich may wrap/colorize).
        print(json.dumps(info.to_dict(), indent=2))
    else:
        render_media_info(console, info)


@app.command()
def analyze(
    video: Path = typer.Argument(
        ...,
        exists=False,  # validated by the service so errors stay user-facing
        help="Path to a local video file.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of the readable summary."
    ),
) -> None:
    """Analyze on-screen text and suggest descriptive keywords (proposal-only).

    Reads a few sampled frames with ffmpeg and OCRs them with tesseract. It never
    renames, moves, copies, or modifies the source video.
    """
    service = AnalyzeService(
        inspect_service=InspectService(probe=FfprobeMediaProbe()),
        frame_extractor=FfmpegFrameExtractor(),
        ocr=TesseractOcr(),
    )
    try:
        result = service.analyze(video)
    except VideoLearningError as exc:
        error_console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        # Plain print keeps stdout strictly valid JSON (rich may wrap/colorize).
        print(json.dumps(result.to_dict(), indent=2))
    else:
        render_analysis(console, result)


@app.command(name="transcribe-audio")
def transcribe_audio(
    video: Path = typer.Argument(
        ...,
        exists=False,  # validated by the service so errors stay user-facing
        help="Path to a local video file.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of the readable summary."
    ),
    model: Path | None = typer.Option(
        None,
        "--model",
        help=(
            "Path to a whisper.cpp ggml model file (e.g. ggml-base.en.bin). "
            "Defaults to $VLS_WHISPER_MODEL. Never downloaded automatically."
        ),
    ),
) -> None:
    """Transcribe a video's audio to timestamped text using local whisper.cpp.

    Reads the source audio into a temporary WAV and transcribes it locally. It
    never modifies the source, never downloads a model, and never falls back to
    a cloud API. The transcript is evidence for the human; no filenames or
    keywords are derived from it.
    """
    whisper = WhisperCpp()
    capability = whisper.detect_capability(model)
    service = TranscribeService(
        inspect_service=InspectService(probe=FfprobeMediaProbe()),
        audio_extractor=FfmpegAudioExtractor(),
        whisper=whisper,
    )

    if not capability.ready:
        # Show capability plus a clear, actionable error; never attempt to run.
        if json_output:
            print(
                json.dumps(
                    {
                        "capability": capability.to_dict(),
                        "transcription": None,
                        "error": capability.blocking_message(),
                    },
                    indent=2,
                )
            )
        else:
            render_transcription_capability(console, capability)
            error_console.print(f"Error: {capability.blocking_message()}")
        raise typer.Exit(code=1)

    try:
        result = service.transcribe(video, model=model)
    except VideoLearningError as exc:
        error_console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        # Plain print keeps stdout strictly valid JSON (rich may wrap/colorize).
        print(
            json.dumps(
                {"capability": capability.to_dict(), "transcription": result.to_dict()},
                indent=2,
            )
        )
    else:
        render_transcription_capability(console, capability)
        render_transcript(console, result)
