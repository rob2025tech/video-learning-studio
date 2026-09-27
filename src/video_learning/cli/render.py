"""Rich rendering for CLI output. Presentation only — no business logic."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from video_learning.core.models import UNKNOWN, MediaInfo
from video_learning.services.analyze_service import AnalyzeResult


def _fmt(value: object, pattern: str = "{}") -> str:
    return pattern.format(value) if value is not None else UNKNOWN


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return UNKNOWN
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    clock = f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"
    return f"{clock} ({seconds:.3f}s)"


def _fmt_modified(info: MediaInfo) -> str:
    if info.modified_at is None:
        return UNKNOWN
    return info.modified_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _fmt_timestamp(seconds: float) -> str:
    """Format a media offset as HH:MM:SS.s for quick QuickTime comparison.

    Rounding is done on total tenths so a fractional part like .96 carries into
    the seconds field instead of printing an invalid ".10".
    """
    total_tenths = int(round(max(0.0, seconds) * 10))
    hours, remainder = divmod(total_tenths, 36_000)
    minutes, remainder = divmod(remainder, 600)
    secs, tenths = divmod(remainder, 10)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{tenths}"


def _ocr_preview(text: str, limit: int = 90) -> str:
    """Collapse OCR text to a single concise line for the readable summary."""
    collapsed = " ".join(text.split())
    if not collapsed:
        return "(no text detected)"
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 1].rstrip()}…"


def render_media_info(console: Console, info: MediaInfo) -> None:
    table = Table(title=f"Media info: {info.filename}", show_lines=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Filename", info.filename)
    table.add_row("Path", str(info.path))
    table.add_row("File size", _fmt(info.size_bytes, "{} bytes"))
    table.add_row("Modified", _fmt_modified(info))
    table.add_row("Container", _fmt(info.container_format))
    table.add_row("Duration", _fmt_duration(info.duration_seconds))

    table.add_section()
    table.add_row("Video stream", "yes" if info.has_video else "no")
    if info.video is not None:
        table.add_row("Video codec", _fmt(info.video.codec))
        dimensions = (
            f"{info.video.width}x{info.video.height}"
            if info.video.width is not None and info.video.height is not None
            else UNKNOWN
        )
        table.add_row("Dimensions", dimensions)
        table.add_row("Frame rate", _fmt(info.video.frame_rate, "{:.3f} fps"))
        table.add_row("Pixel format", _fmt(info.video.pixel_format))

    table.add_section()
    table.add_row("Audio stream", "yes" if info.has_audio else "no")
    if info.audio is not None:
        table.add_row("Audio codec", _fmt(info.audio.codec))
        table.add_row("Sample rate", _fmt(info.audio.sample_rate, "{} Hz"))
        table.add_row("Channels", _fmt(info.audio.channels))
        table.add_row("Language", _fmt(info.audio.language))

    console.print(table)


def render_analysis(console: Console, result: AnalyzeResult) -> None:
    """Render auditable snapshots plus suggested keywords (proposal only).

    Each snapshot shows the sampled timestamp and a concise OCR preview so the
    user can line it up against the source video; the complete OCR text is
    retained in the ``--json`` output.
    """
    console.print(f"Video: {result.source.name}")
    console.print()
    console.print("OCR snapshots:")
    if not result.snapshots:
        console.print("[dim](none)[/dim]")
    else:
        for snapshot in result.snapshots:
            stamp = _fmt_timestamp(snapshot.timestamp_seconds)
            preview = _ocr_preview(snapshot.ocr_text)
            console.print(f"- [cyan]{stamp}[/cyan]  {preview}")
    console.print()
    console.print("Suggested keywords:")
    if not result.keywords:
        console.print(
            "[dim](none — no readable on-screen text was detected in the sampled frames)[/dim]"
        )
        return
    for keyword in result.keywords:
        console.print(f"- {keyword.term}")
    console.print()
    console.print(
        f"[dim](from on-screen text in {result.frames_analyzed} sampled frame"
        f"{'s' if result.frames_analyzed != 1 else ''}; proposal only, source unchanged)[/dim]"
    )
