"""On-screen text extraction via the system ``tesseract`` binary.

tesseract is invoked as a subprocess (the same pattern used for ffprobe/ffmpeg),
so no new Python dependency is required. Only images are read; nothing is written
back to the source media.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from video_learning.core.errors import AnalysisToolMissingError, OcrError

_OCR_TIMEOUT_SECONDS = 120


class TesseractOcr:
    """Runs tesseract over an image file and returns the recognized text."""

    def __init__(self, lang: str = "eng") -> None:
        self._lang = lang

    def extract_text(self, image: Path) -> str:
        if shutil.which("tesseract") is None:
            raise AnalysisToolMissingError(
                "tesseract was not found on PATH. Install it first, e.g. `brew install tesseract`."
            )
        command = ["tesseract", str(image), "stdout", "-l", self._lang]
        try:
            result = subprocess.run(  # noqa: S603
                command, capture_output=True, text=True, timeout=_OCR_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as exc:
            raise OcrError(
                f"tesseract timed out reading text from '{image.name}'."
            ) from exc
        except OSError as exc:
            raise AnalysisToolMissingError(f"Could not execute tesseract: {exc}") from exc

        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            hint = detail[-1] if detail else "no details reported by tesseract"
            raise OcrError(f"tesseract failed on '{image.name}': {hint}")
        return result.stdout
