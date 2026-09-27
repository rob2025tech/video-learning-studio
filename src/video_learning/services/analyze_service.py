"""Content-analysis use case: video -> suggested descriptive keywords.

Proposal-only and non-destructive. It reads a few sampled frames into a
temporary directory, OCRs them, ranks the observed words, and returns
suggestions. It never renames, moves, copies, or writes to the source media,
and it never invents keywords that did not appear on screen.

This layer deliberately has no provider abstractions yet: it wires the concrete
Stage 0B adapters directly. A seam is introduced only when a second real
implementation (e.g. Whisper for spoken audio) is added.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_learning.adapters.ffmpeg_frames import FfmpegFrameExtractor
from video_learning.adapters.tesseract_ocr import TesseractOcr
from video_learning.core.keywords import KeywordSuggestion, suggest_keywords
from video_learning.services.inspect_service import InspectService


@dataclass(frozen=True)
class FrameSnapshot:
    """One sampled frame's timestamp paired with the OCR text read from it.

    This is the audit unit: it lets a user line up
    ``timestamp -> video frame -> OCR text -> keywords`` against the source.
    """

    timestamp_seconds: float
    ocr_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "ocr_text": self.ocr_text,
        }


@dataclass(frozen=True)
class AnalyzeResult:
    """Outcome of analyzing a single video (proposal only)."""

    source: Path
    keywords: list[KeywordSuggestion] = field(default_factory=list)
    snapshots: list[FrameSnapshot] = field(default_factory=list)
    frames_analyzed: int = 0
    ocr_text_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "keywords": [kw.to_dict() for kw in self.keywords],
            "frames_analyzed": self.frames_analyzed,
            "ocr_text_chars": self.ocr_text_chars,
            # Explicit guarantees for this stage.
            "renamed": False,
            "applied": False,
            "source_modified": False,
        }


class AnalyzeService:
    """Use case: suggest descriptive keywords from a video's on-screen text."""

    def __init__(
        self,
        inspect_service: InspectService,
        frame_extractor: FfmpegFrameExtractor,
        ocr: TesseractOcr,
    ) -> None:
        self._inspect = inspect_service
        self._frames = frame_extractor
        self._ocr = ocr

    def analyze(self, path: Path) -> AnalyzeResult:
        media = self._inspect.inspect(path)  # reuse Stage 0A pipeline

        with tempfile.TemporaryDirectory(prefix="vls-frames-") as tmp:
            frames = self._frames.extract(media, Path(tmp))
            # Keep each frame's timestamp bound to the text OCR'd from it.
            snapshots = [
                FrameSnapshot(
                    timestamp_seconds=frame.timestamp_seconds,
                    ocr_text=self._ocr.extract_text(frame.image_path),
                )
                for frame in frames
            ]

        combined = "\n".join(snapshot.ocr_text for snapshot in snapshots)
        keywords = suggest_keywords(combined)
        return AnalyzeResult(
            source=path,
            keywords=keywords,
            snapshots=snapshots,
            frames_analyzed=len(snapshots),
            ocr_text_chars=len(combined.strip()),
        )
