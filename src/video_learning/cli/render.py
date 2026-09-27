"""Rich rendering for CLI output. Presentation only — no business logic."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from video_learning.core.models import UNKNOWN, MediaInfo
from video_learning.core.timeline import OcrEvent
from video_learning.core.transcript import (
    CapabilityItem,
    TranscriptionCapability,
    TranscriptResult,
)
from video_learning.services.analyze_service import AnalyzeResult
from video_learning.services.report_service import ReportResult


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
    """Render auditable snapshots plus suggested phrases and keywords.

    Each snapshot shows the sampled timestamp and a concise OCR preview so the
    user can line it up against the source video; the complete OCR text is
    retained in the ``--json`` output. Proposal only.
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
    console.print("Suggested phrases:")
    if not result.phrases:
        console.print("[dim](none)[/dim]")
    else:
        for phrase in result.phrases:
            console.print(f"- {phrase.phrase}")
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


def _section(console: Console, title: str) -> None:
    console.print(f"[bold]{title}[/bold]")
    console.print("[dim]" + "─" * 32 + "[/dim]")


def _capability_line(console: Console, item: CapabilityItem, *, kind: str) -> None:
    if item.state == "available":
        mark, word = "[green]✓[/green]", "available"
    elif item.state == "unusable":
        mark, word = "[red]✗[/red]", "not usable"
    elif kind == "backend":
        mark, word = "[red]✗[/red]", "not available"
    else:
        mark, word = "[red]✗[/red]", "model not found"
    console.print(f"{mark} [bold]{item.label:<16}[/bold]{word}")
    if item.state != "available" and item.detail:
        console.print(f"  [dim]{item.detail}[/dim]")


def render_transcription_capability(
    console: Console, capability: TranscriptionCapability
) -> None:
    """Show this machine's local transcription capability (deterministic)."""
    _section(console, "Transcription backends detected")
    _capability_line(console, capability.backend, kind="backend")
    console.print()
    _section(console, "Model")
    _capability_line(console, capability.model, kind="model")
    console.print()
    _section(console, "Recommended")
    console.print(capability.recommended)
    console.print(f"[dim]{capability.recommended_detail}[/dim]")
    console.print()
    _section(console, "Using")
    console.print(capability.using_backend)
    console.print(capability.using_model)
    console.print(capability.using_language)


def render_transcript(console: Console, result: TranscriptResult) -> None:
    """Render a timestamped transcript timeline (evidence for the human)."""
    console.print()
    _section(console, "Transcription")
    console.print(f"[bold]Backend:[/bold] {result.backend}")
    console.print(f"[bold]Model:[/bold]   {result.model}")
    console.print(f"[bold]Language:[/bold] {result.language}")
    console.print()
    if not result.segments:
        console.print("[dim](no speech detected)[/dim]")
    for segment in result.segments:
        start = _fmt_timestamp(segment.start_seconds)
        end = _fmt_timestamp(segment.end_seconds)
        console.print(f"[cyan]{start}[/cyan] → [cyan]{end}[/cyan]  {segment.text}")
    console.print()
    console.print(
        "[dim]Read-only — source unchanged; the transcript is evidence for the "
        "human (no files renamed, nothing persisted).[/dim]"
    )


def render_ocr_artifact_saved(console: Console, path: Path, result: AnalyzeResult) -> None:
    """Confirm OCR snapshot evidence was persisted to a local artifact."""
    _section(console, "OCR evidence saved")
    console.print(f"[bold]Artifact:[/bold]  {path}")
    console.print(f"[bold]Source:[/bold]    {result.source.name}")
    console.print(
        f"[bold]Snapshots:[/bold] {result.frames_analyzed} "
        f"(complete OCR text preserved; {result.ocr_text_chars} chars)"
    )
    console.print(
        f"[bold]Derived:[/bold]   {len(result.phrases)} phrases, "
        f"{len(result.keywords)} keywords"
    )
    console.print()
    console.print(
        "[dim]Read-only — the source video is unchanged; the full OCR text and its "
        "derived suggestions are persisted for audit (no renaming, no keywords "
        "invented).[/dim]"
    )


def render_report(
    console: Console, result: ReportResult, artifact_path: Path | None = None
) -> None:
    """Render a concise, human-readable unified report (metadata + evidence).

    The timeline shows each OCR point as a short preview and each speech interval
    in full; the complete OCR text is retained in the ``--json``/``--out`` report,
    never dumped here. No relationship between OCR and speech is implied.
    """
    media = result.media
    analysis = result.analysis

    _section(console, "VIDEO REPORT")
    console.print(f"[bold]File:[/bold] {media.filename}")
    console.print(f"[dim]{media.path}[/dim]")
    console.print()

    _section(console, "Metadata")
    console.print(f"[bold]Duration:[/bold] {_fmt_duration(media.duration_seconds)}")
    if media.video is not None:
        dimensions = (
            f"{media.video.width}x{media.video.height}"
            if media.video.width is not None and media.video.height is not None
            else UNKNOWN
        )
        console.print(
            f"[bold]Video:[/bold]    {_fmt(media.video.codec)} · {dimensions} · "
            f"{_fmt(media.video.frame_rate, '{:.3f} fps')}"
        )
    else:
        console.print("[bold]Video:[/bold]    none")
    if media.audio is not None:
        console.print(
            f"[bold]Audio:[/bold]    {_fmt(media.audio.codec)} · "
            f"{_fmt(media.audio.sample_rate, '{} Hz')} · {_fmt(media.audio.channels)} ch"
        )
    else:
        console.print("[bold]Audio:[/bold]    none")
    console.print()

    _section(console, "OCR")
    console.print(f"[bold]Snapshots:[/bold]      {analysis.frames_analyzed}")
    console.print(f"[bold]OCR characters:[/bold] {analysis.ocr_text_chars}")
    phrases = ", ".join(phrase.phrase for phrase in analysis.phrases[:6]) or "(none)"
    keywords = ", ".join(keyword.term for keyword in analysis.keywords[:10]) or "(none)"
    console.print(f"[bold]Phrases:[/bold]       {phrases}")
    console.print(f"[bold]Keywords:[/bold]      {keywords}")
    console.print()

    _section(console, "Transcription")
    console.print(f"[bold]Status:[/bold] {result.transcription_status}")
    transcript = result.transcription
    if transcript is not None:
        console.print(f"[bold]Backend:[/bold]  {transcript.backend}")
        console.print(f"[bold]Model:[/bold]    {transcript.model}")
        console.print(f"[bold]Language:[/bold] {transcript.language}")
        console.print(f"[bold]Segments:[/bold] {transcript.segment_count}")
    elif result.transcription_detail:
        console.print(f"[dim]{result.transcription_detail}[/dim]")
    console.print()

    _section(console, "UNIFIED TIMELINE")
    if not result.timeline:
        console.print("[dim](no evidence)[/dim]")
    for event in result.timeline:
        if isinstance(event, OcrEvent):
            stamp = _fmt_timestamp(event.timestamp_seconds)
            preview = _ocr_preview(event.text)
            console.print(f"[cyan]{stamp}[/cyan]  [magenta]OCR[/magenta]     {preview}")
        else:
            start = _fmt_timestamp(event.start_seconds)
            end = _fmt_timestamp(event.end_seconds)
            console.print(
                f"[cyan]{start}[/cyan] → [cyan]{end}[/cyan]  "
                f"[green]SPEECH[/green]  {event.text}"
            )
    console.print()

    if artifact_path is not None:
        console.print(f"[dim]Artifact written: {artifact_path}[/dim]")
    console.print(
        "[dim]Read-only — source unchanged; metadata, OCR, and transcription evidence "
        "combined into one chronological timeline (no renaming, no semantic "
        "alignment).[/dim]"
    )
