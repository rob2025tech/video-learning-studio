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

import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from video_learning.adapters.whisper_cpp import WhisperCpp
from video_learning.core.errors import ReportArtifactError, TranscriptionError
from video_learning.core.models import MediaInfo
from video_learning.core.segments import Segmentation
from video_learning.core.study import StudyEvidence, build_study_evidence
from video_learning.core.timeline import (
    OcrEvent,
    SpeechEvent,
    build_timeline,
    timeline_event_from_dict,
)
from video_learning.core.transcript import TranscriptResult
from video_learning.services.analyze_service import AnalyzeResult, AnalyzeService
from video_learning.services.inspect_service import InspectService
from video_learning.services.segment_service import SegmentService
from video_learning.services.transcribe_service import TranscribeService

REPORT_SCHEMA = "video-learning.report/v1"

# The ordered progress stages emitted by :meth:`ReportService.build`. Each name
# marks the point *immediately before* that stage's work starts (``done`` is
# emitted only once the report is fully assembled), so a consumer can show the
# stage actually in flight rather than the last one that finished.
ReportStage = Literal["inspect", "ocr", "transcribe", "segment", "timeline", "done"]

# A synchronous progress hook. Callback exceptions are swallowed by
# :meth:`ReportService._notify` (see its docstring for the policy), so a raising
# hook can never corrupt or abort an otherwise-good report build.
StageCallback = Callable[[ReportStage], None]

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

    def build(
        self,
        path: Path,
        *,
        model: Path | None = None,
        on_stage: StageCallback | None = None,
    ) -> ReportResult:
        """Assemble the report, optionally notifying ``on_stage`` before each stage.

        ``on_stage`` is a pure progress hook: when it is ``None`` (the default)
        the build behaves exactly as before — same stages, same order, same
        deterministic report. Each name is emitted immediately *before* the work
        it names starts, in pipeline order (``inspect``, ``ocr``, ``transcribe``,
        ``segment``, ``timeline``), and ``done`` only after the report has been
        built successfully. A stage that raises therefore leaves its own
        notification as the last one emitted, which identifies it as the stage
        that was running.
        """
        self._notify(on_stage, "inspect")
        media = self._inspect.inspect(path)  # reuse Stage 0A

        self._notify(on_stage, "ocr")
        analysis = self._analyze.analyze(path)  # reuse Stage 0D/0F

        self._notify(on_stage, "transcribe")
        transcription, status, detail = self._transcription_evidence(path, media, model)

        # Stage 0I reuses the MediaInfo and transcript already gathered above; it
        # never re-runs inspection, analysis, or transcription.
        self._notify(on_stage, "segment")
        segmentation = self._segment.segment(media, transcription)

        self._notify(on_stage, "timeline")
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
        report = ReportResult(
            source=path,
            media=media,
            analysis=analysis,
            transcription=transcription,
            transcription_status=status,
            transcription_detail=detail,
            segmentation=segmentation,
            timeline=timeline,
        )
        self._notify(on_stage, "done")
        return report

    @staticmethod
    def _notify(on_stage: StageCallback | None, stage: ReportStage) -> None:
        """Emit ``stage`` to ``on_stage`` if supplied, swallowing callback errors.

        The agreed safety policy: progress observation must never change the
        outcome of a report build. A callback that raises is treated as a broken
        observer, not a pipeline failure — the exception is caught and ignored
        (only :class:`Exception`; ``KeyboardInterrupt``/``SystemExit`` still
        propagate) and the remaining stages continue normally.
        """
        if on_stage is None:
            return
        # Deliberately silent: the report, not the hook, is the product.
        with contextlib.suppress(Exception):
            on_stage(stage)

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
    """Persist a :class:`ReportResult` as a deterministic local JSON artifact.

    Reading is deliberately partial: :meth:`load_evidence` rebuilds only what the
    downstream evidence derivations consume, rather than the whole report.
    """

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

    def load_evidence(self, path: Path) -> StudyEvidence:
        """Reload the study evidence a saved report artifact describes.

        Only the sections :func:`~video_learning.core.study.build_study_evidence`
        consumes are rebuilt — the segmentation, the timeline, and the transcription
        status/detail — because that is all a downstream pure derivation needs. This is
        deliberately *not* full :class:`ReportResult` rehydration: ``metadata`` is
        written by :func:`_metadata_dict` in a flatter shape than ``MediaInfo`` (and
        drops volatile filesystem values on purpose), ``ocr`` is evidence no learning
        artifact reads back here, ``TranscriptResult`` has no ``from_dict``, and
        rebuilding a transcript object would break the report's own invariant that a
        non-``"ok"`` status pairs with ``None``.

        The artifact is read only; the video it describes is never opened, and no
        analysis, transcription, or segmentation work is re-run.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        # A decode failure is a ValueError, not an OSError, so it must be named here.
        except (OSError, UnicodeDecodeError) as exc:
            raise ReportArtifactError(f"Could not read report artifact '{path}': {exc}") from exc
        return self.load_evidence_from_text(raw, label=str(path))

    def load_evidence_from_text(self, raw: str, *, label: str) -> StudyEvidence:
        """Same derivation as :meth:`load_evidence`, from text the caller already holds.

        Split out so a caller that received the artifact as bytes (an HTTP upload, for
        example) never has to write it to disk first. ``label`` names the artifact in
        every error message and is display-only: it is never opened, resolved or joined
        to a directory. Reading from a path passes ``str(path)``, so the messages a CLI
        user sees are unchanged.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReportArtifactError(
                f"Report artifact '{label}' is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ReportArtifactError(f"Report artifact '{label}' is not a JSON object.")
        found = data.get("schema")
        if found != self.schema:
            raise ReportArtifactError(
                f"Unsupported report artifact schema in '{label}': {found!r} "
                f"(expected {self.schema!r})."
            )
        try:
            return build_study_evidence(
                segmentation=Segmentation.from_dict(data["segments"]),
                timeline=[timeline_event_from_dict(event) for event in data["timeline"]],
                transcription_status=data["transcription"]["status"],
                transcription_detail=data["transcription"]["detail"],
                source=Path(data["source"]),
            )
        except KeyError as exc:
            raise ReportArtifactError(
                f"Report artifact '{label}' is missing the {exc.args[0]!r} section it needs."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ReportArtifactError(
                f"Report artifact '{label}' holds a value this version cannot interpret: {exc}"
            ) from exc
