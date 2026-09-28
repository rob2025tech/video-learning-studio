"""Reporting/orchestration use case (Stage 0G): combine evidence into a report.

This is a thin orchestration + serialization layer, NOT a second implementation
of OCR or transcription. It reuses the existing services and result models:

    InspectService    -> MediaInfo         (Stage 0A metadata)
    AnalyzeService    -> AnalyzeResult      (Stage 0D/0F OCR evidence)
    TranscribeService -> TranscriptResult   (Stage 0E audio evidence)
    SegmentService    -> Segmentation       (Stage 0I segments)

and merges the OCR snapshots (points) and transcript segments (intervals) into a
single deterministic *unified timeline*, plus a deterministic Stage 0I ``segments``
section (visual-change + speech-gap evidence -> merged boundaries -> segments). The
source video is only ever read; the report is written to a caller-chosen artifact
path and nowhere else.

Transcription follows Stage 0E capability/error semantics: if there is no audio,
transcription is skipped (``no-audio``); if audio is present but the local
backend/model is not ready, the report records an explicit, deterministic
``unavailable`` status (reusing the Stage 0E blocking message) rather than failing
the whole report or inventing content. Models are never downloaded and there is
no cloud fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_learning.adapters.whisper_cpp import WhisperCpp
from video_learning.core.errors import ReportArtifactError, TranscriptionError
from video_learning.core.models import MediaInfo
from video_learning.core.segments import Segmentation
from video_learning.core.timeline import OcrEvent, SpeechEvent, build_timeline
from video_learning.core.transcript import TranscriptResult
from video_learning.services.analyze_service import AnalyzeResult, AnalyzeService
from video_learning.services.inspect_service import InspectService
from video_learning.services.segment_service import SegmentService
from video_learning.services.transcribe_service import TranscribeService

REPORT_SCHEMA = "video-learning.report/v1"

# Fixed, deterministic detail for the rare case where capability was ready but
# transcription still failed at runtime. We deliberately do NOT interpolate the
# exception string, which is not guaranteed to be deterministic or stable.
_RUNTIME_FAILURE_DETAIL = "transcription was attempted but failed at runtime"


def _metadata_dict(media: MediaInfo) -> dict[str, Any]:
    """Stable, intrinsic media metadata for the report.

    Reuses the Stage 0A ``MediaInfo`` fields verbatim. Volatile filesystem values
    (``modified_at``/mtime and size) are intentionally omitted so the artifact is
    deterministic for the same input and the same available backends/models.
    """
    return {
        "filename": media.filename,
        "path": str(media.path),
        "container_format": media.container_format,
        "duration_seconds": media.duration_seconds,
        "has_video": media.has_video,
        "video": {
            "codec": media.video.codec,
            "width": media.video.width,
            "height": media.video.height,
            "frame_rate": media.video.frame_rate,
            "pixel_format": media.video.pixel_format,
        }
        if media.video is not None
        else None,
        "has_audio": media.has_audio,
        "audio": {
            "codec": media.audio.codec,
            "sample_rate": media.audio.sample_rate,
            "channels": media.audio.channels,
            "language": media.audio.language,
        }
        if media.audio is not None
        else None,
    }


@dataclass(frozen=True)
class ReportResult:
    """A deterministic, auditable combination of metadata + OCR + transcription.

    ``transcription`` is ``None`` unless ``transcription_status == "ok"``. The
    status is one of ``"ok"``, ``"no-audio"``, or ``"unavailable"``;
    ``transcription_detail`` carries the Stage 0E blocking message only for
    ``"unavailable"`` and is otherwise ``None``. ``segmentation`` is the Stage 0I
    result (visual/speech evidence, merged boundaries, and final segments).
    """

    source: Path
    media: MediaInfo
    analysis: AnalyzeResult
    transcription: TranscriptResult | None
    transcription_status: str
    transcription_detail: str | None
    segmentation: Segmentation
    timeline: list[OcrEvent | SpeechEvent] = field(default_factory=list)

    def _transcription_dict(self) -> dict[str, Any]:
        """Structured transcription section with explicit, deterministic status."""
        transcript = self.transcription
        if transcript is not None:
            return {
                "status": "ok",
                "detail": None,
                "backend": transcript.backend,
                "model": transcript.model,
                "language": transcript.language,
                "segments": [segment.to_dict() for segment in transcript.segments],
                "segment_count": transcript.segment_count,
            }
        # no-audio / unavailable: nothing ran, so no provenance is invented.
        return {
            "status": self.transcription_status,
            "detail": self.transcription_detail,
            "backend": None,
            "model": None,
            "language": None,
            "segments": [],
            "segment_count": 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "source": str(self.source),
            "metadata": _metadata_dict(self.media),
            # OCR evidence reuses AnalyzeResult's own serialization (no second
            # representation); the complete OCR text is preserved per snapshot.
            "ocr": {
                "snapshots": [snapshot.to_dict() for snapshot in self.analysis.snapshots],
                "phrases": [phrase.to_dict() for phrase in self.analysis.phrases],
                "keywords": [keyword.to_dict() for keyword in self.analysis.keywords],
                "frames_analyzed": self.analysis.frames_analyzed,
                "ocr_text_chars": self.analysis.ocr_text_chars,
            },
            "transcription": self._transcription_dict(),
            "timeline": [event.to_dict() for event in self.timeline],
            # Stage 0I: deterministic segments plus the visual/speech evidence and
            # merged boundaries they were assembled from (evidence is never lost).
            "segments": self.segmentation.to_dict(),
            # Explicit non-destructive guarantees (project-wide convention).
            "renamed": False,
            "applied": False,
            "source_modified": False,
        }


class ReportService:
    """Use case: build a unified report by reusing the existing services."""

    def __init__(
        self,
        inspect_service: InspectService,
        analyze_service: AnalyzeService,
        transcribe_service: TranscribeService,
        segment_service: SegmentService,
        whisper: WhisperCpp,
    ) -> None:
        self._inspect = inspect_service
        self._analyze = analyze_service
        self._transcribe = transcribe_service
        self._segment = segment_service
        self._whisper = whisper

    def build(self, path: Path, *, model: Path | None = None) -> ReportResult:
        media = self._inspect.inspect(path)  # reuse Stage 0A
        analysis = self._analyze.analyze(path)  # reuse Stage 0D/0F

        transcription, status, detail = self._transcription_evidence(path, media, model)

        # Stage 0I reuses the MediaInfo and transcript already gathered above; it
        # never re-runs inspection, analysis, or transcription.
        segmentation = self._segment.segment(media, transcription)

        timeline = build_timeline(
            [
                OcrEvent(timestamp_seconds=snapshot.timestamp_seconds, text=snapshot.ocr_text)
                for snapshot in analysis.snapshots
            ],
            [
                SpeechEvent(
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    text=segment.text,
                )
                for segment in (transcription.segments if transcription is not None else ())
            ],
        )
        return ReportResult(
            source=path,
            media=media,
            analysis=analysis,
            transcription=transcription,
            transcription_status=status,
            transcription_detail=detail,
            segmentation=segmentation,
            timeline=timeline,
        )

    def _transcription_evidence(
        self, path: Path, media: MediaInfo, model: Path | None
    ) -> tuple[TranscriptResult | None, str, str | None]:
        """Return ``(transcription, status, detail)`` using Stage 0E semantics."""
        if not media.has_audio:
            # No audio: skip transcription entirely (never probe the backend).
            return None, "no-audio", None

        capability = self._whisper.detect_capability(model)
        if not capability.ready:
            # Audio present but the local backend/model is missing or unusable:
            # record an explicit, deterministic reason (the Stage 0E message).
            return None, "unavailable", capability.blocking_message()

        try:
            return self._transcribe.transcribe(path, model=model), "ok", None
        except TranscriptionError:
            # Capability was ready yet the run failed; stay explicit/deterministic
            # rather than interpolating a possibly-nondeterministic error string.
            return None, "unavailable", _RUNTIME_FAILURE_DETAIL


class ReportStore:
    """Persist a :class:`ReportResult` as a deterministic local JSON artifact."""

    schema = REPORT_SCHEMA

    def save(self, report: ReportResult, path: Path) -> Path:
        """Write ``report`` to ``path`` deterministically; return ``path``.

        Creates missing parent directories for the caller-chosen artifact and
        overwrites any prior artifact there. The source video is never written.
        """
        document = json.dumps(report.to_dict(), indent=2) + "\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(document, encoding="utf-8")
        except OSError as exc:
            raise ReportArtifactError(
                f"Could not write report artifact to '{path}': {exc}"
            ) from exc
        return path
