"""Unit tests for TesseractOcr with a mocked subprocess boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from video_learning.adapters.tesseract_ocr import TesseractOcr
from video_learning.core.errors import AnalysisToolMissingError, OcrError

WHICH = "video_learning.adapters.tesseract_ocr.shutil.which"
RUN = "video_learning.adapters.tesseract_ocr.subprocess.run"


def _completed(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["tesseract"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "frame_000.png"
    path.write_bytes(b"\x89PNG")
    return path


def test_returns_stdout_text(image: Path) -> None:
    with (
        patch(WHICH, return_value="/usr/local/bin/tesseract"),
        patch(RUN, return_value=_completed("Qoder terminal\n")),
    ):
        assert TesseractOcr().extract_text(image) == "Qoder terminal\n"


def test_missing_binary_raises(image: Path) -> None:
    with (
        patch(WHICH, return_value=None),
        pytest.raises(AnalysisToolMissingError, match="brew install tesseract"),
    ):
        TesseractOcr().extract_text(image)


def test_nonzero_exit_raises_ocr_error(image: Path) -> None:
    with (
        patch(WHICH, return_value="/usr/local/bin/tesseract"),
        patch(RUN, return_value=_completed("", returncode=1, stderr="boom")),
        pytest.raises(OcrError, match="boom"),
    ):
        TesseractOcr().extract_text(image)


def test_timeout_raises_ocr_error(image: Path) -> None:
    with (
        patch(WHICH, return_value="/usr/local/bin/tesseract"),
        patch(RUN, side_effect=subprocess.TimeoutExpired(cmd=["tesseract"], timeout=120)),
        pytest.raises(OcrError, match="timed out"),
    ):
        TesseractOcr().extract_text(image)
