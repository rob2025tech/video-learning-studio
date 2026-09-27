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
from video_learning.adapters.ffmpeg_frames import FfmpegFrameExtractor
from video_learning.adapters.ffprobe_media import FfprobeMediaProbe
from video_learning.adapters.tesseract_ocr import TesseractOcr
from video_learning.cli.render import render_analysis, render_media_info
from video_learning.core.errors import VideoLearningError
from video_learning.services.analyze_service import AnalyzeService
from video_learning.services.inspect_service import InspectService

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
