"""Persistence for OCR snapshot evidence (Stage 0F).

Writes an :class:`~video_learning.services.analyze_service.AnalyzeResult` to a
single deterministic local JSON artifact — no database, no cloud, no framework.
The artifact simply reuses the result's own serialization (source + the complete
OCR text of every timestamped snapshot + the phrases/keywords already derived
from that same text), so a human can trace any suggestion back through
``video -> snapshot timestamp -> original OCR text``.

Output is deterministic by construction: stable key and list ordering, a fixed
schema tag, no wall-clock timestamps, and no random identifiers. The store only
ever writes the caller-chosen artifact path; it never touches the source video.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_learning.core.errors import OcrArtifactError
from video_learning.services.analyze_service import AnalyzeResult

OCR_ARTIFACT_SCHEMA = "video-learning.ocr-snapshots/v1"


class OcrSnapshotStore:
    """Save and reload OCR snapshot evidence as a local JSON artifact."""

    schema = OCR_ARTIFACT_SCHEMA

    def to_artifact(self, result: AnalyzeResult) -> dict[str, Any]:
        """Return the JSON-safe artifact document for ``result`` (pure)."""
        return {"schema": self.schema, **result.to_dict()}

    def save(self, result: AnalyzeResult, path: Path) -> Path:
        """Write ``result`` to ``path`` deterministically; return ``path``.

        Creates missing parent directories for the caller-chosen artifact and
        overwrites any prior artifact at that path. The source video is never
        read from or written to here.
        """
        document = json.dumps(self.to_artifact(result), indent=2) + "\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(document, encoding="utf-8")
        except OSError as exc:
            raise OcrArtifactError(f"Could not write OCR artifact to '{path}': {exc}") from exc
        return path

    def load(self, path: Path) -> AnalyzeResult:
        """Reload a previously saved artifact, verifying its schema (inverse)."""
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OcrArtifactError(f"Could not read OCR artifact '{path}': {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OcrArtifactError(f"OCR artifact '{path}' is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise OcrArtifactError(f"OCR artifact '{path}' is not a JSON object.")
        found = data.get("schema")
        if found != self.schema:
            raise OcrArtifactError(
                f"Unsupported OCR artifact schema in '{path}': {found!r} "
                f"(expected {self.schema!r})."
            )
        try:
            return AnalyzeResult.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise OcrArtifactError(f"OCR artifact '{path}' is malformed: {exc}") from exc
