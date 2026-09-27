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
class AnalyzeResult:
    """Outcome of analyzing a single video (proposal only)."""

    source: Path
    keywords: list[KeywordSuggestion] = field(default_factory=list)
    frames_analyzed: int = 0
    ocr_text_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
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
            texts = [self._ocr.extract_text(frame) for frame in frames]
            frames_analyzed = len(frames)

        combined = "\n".join(texts)
        keywords = suggest_keywords(combined)
        return AnalyzeResult(
            source=path,
            keywords=keywords,
            frames_analyzed=frames_analyzed,
            ocr_text_chars=len(combined.strip()),
        )
