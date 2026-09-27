"""Adapter: local audio transcription via the whisper.cpp ``whisper-cli`` binary.

whisper.cpp is invoked as a subprocess and its JSON output is parsed (never its
human-readable stdout). Segment offsets are reported in milliseconds and are
converted to float seconds deterministically.

Discovery is strictly via PATH (``shutil.which``): no Homebrew prefix, pyenv
shim, or other architecture-specific path is hard-coded, so the same code works
on Intel and Apple Silicon macOS. ``shutil.which`` alone is *not* trusted: a
shim can exist on PATH yet exit 127, which is detected and reported clearly.

NOTE ON VERIFICATION: whisper.cpp is not installed in this environment, so the
exact ``whisper-cli`` flags and JSON schema below follow whisper.cpp's
documented ``--output-json`` format and MUST be re-verified against the
installed version on first real use. No undocumented flags are invented.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from video_learning.core.errors import (
    TranscriptionToolMissingError,
    WhisperExecutionError,
)
from video_learning.core.transcript import (
    CapabilityItem,
    TranscriptionCapability,
    TranscriptSegment,
)

EXECUTABLE = "whisper-cli"
BACKEND_NAME = "whisper.cpp"
RECOMMENDED_MODEL = "ggml-base.en"
MODEL_ENV_VAR = "VLS_WHISPER_MODEL"

_PROBE_TIMEOUT_SECONDS = 15
_TRANSCRIBE_TIMEOUT_SECONDS = 3600


def resolve_model(explicit: Path | None) -> Path | None:
    """Return an existing ggml model path from ``--model`` or the env var.

    Never downloads anything and never hard-codes a machine-specific location.
    Returns ``None`` when no usable model is configured.
    """
    if explicit is not None:
        return explicit if explicit.is_file() else None
    configured = os.environ.get(MODEL_ENV_VAR)
    if configured:
        candidate = Path(configured).expanduser()
        return candidate if candidate.is_file() else None
    return None


class WhisperCpp:
    """Runs whisper.cpp and maps its JSON output onto ``TranscriptSegment``s."""

    # -- capability ---------------------------------------------------------

    def detect_capability(self, model: Path | None) -> TranscriptionCapability:
        backend = self._detect_backend()
        model_path = resolve_model(model)
        if model_path is not None:
            model_item = CapabilityItem(model_path.stem, "available", str(model_path))
            using_model = model_path.stem
        else:
            model_item = CapabilityItem(
                RECOMMENDED_MODEL,
                "missing",
                f"model not found — pass --model PATH or set {MODEL_ENV_VAR}",
            )
            using_model = "—"
        using_language = "English" if model_item.label.endswith(".en") else "auto"
        return TranscriptionCapability(
            backend=backend,
            model=model_item,
            recommended=f"{BACKEND_NAME} + {RECOMMENDED_MODEL}",
            recommended_detail="Local • English",
            using_backend=BACKEND_NAME,
            using_model=using_model,
            using_language=using_language,
            ready=backend.state == "available" and model_path is not None,
        )

    def _detect_backend(self) -> CapabilityItem:
        path = shutil.which(EXECUTABLE)
        if path is None:
            return CapabilityItem(
                BACKEND_NAME,
                "missing",
                f"{EXECUTABLE} not found on PATH — install whisper.cpp "
                "(e.g. `brew install whisper-cpp`)",
            )
        # A shim can exist on PATH yet fail to run (exit 127). Probe cheaply.
        try:
            completed = subprocess.run(  # noqa: S603
                [EXECUTABLE, "--help"],
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return CapabilityItem(
                BACKEND_NAME, "unusable", f"found at {path} but --help timed out"
            )
        except OSError as exc:
            return CapabilityItem(
                BACKEND_NAME, "unusable", f"found at {path} but could not run ({exc})"
            )
        if completed.returncode == 127:
            return CapabilityItem(
                BACKEND_NAME,
                "unusable",
                f"found at {path} but exited 127 (broken shim — not installed "
                "in the active environment?)",
            )
        if completed.returncode != 0:
            return CapabilityItem(
                BACKEND_NAME,
                "unusable",
                f"found at {path} but --help exited {completed.returncode}",
            )
        return CapabilityItem(BACKEND_NAME, "available", path)

    # -- transcription ------------------------------------------------------

    def transcribe(
        self,
        wav_path: Path,
        model: Path,
        out_dir: Path,
        language: str | None = None,
    ) -> tuple[list[TranscriptSegment], str | None]:
        """Transcribe ``wav_path``; return ordered segments and the language."""
        if shutil.which(EXECUTABLE) is None:
            raise TranscriptionToolMissingError(
                f"{EXECUTABLE} was not found on PATH. Install whisper.cpp "
                "(e.g. `brew install whisper-cpp`)."
            )
        prefix = out_dir / "transcript"
        command = [
            EXECUTABLE,
            "-m", str(model),
            "-f", str(wav_path),
            "-l", language or "auto",
            "-oj",
            "-of", str(prefix),
        ]
        self._run(command)
        json_path = Path(f"{prefix}.json")
        if not json_path.exists():
            raise WhisperExecutionError(
                f"whisper.cpp produced no JSON output ('{json_path.name}')."
            )
        return self._parse(json_path.read_text(encoding="utf-8"))

    def _run(self, command: list[str]) -> None:
        try:
            result = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                timeout=_TRANSCRIBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise WhisperExecutionError(
                f"whisper.cpp timed out after {_TRANSCRIBE_TIMEOUT_SECONDS}s."
            ) from exc
        except OSError as exc:
            raise WhisperExecutionError(
                f"Could not execute {EXECUTABLE}: {exc}"
            ) from exc
        if result.returncode == 127:
            raise TranscriptionToolMissingError(
                f"{EXECUTABLE} exited 127 (broken shim — found on PATH but not "
                "installed in the active environment). Install whisper.cpp "
                "(e.g. `brew install whisper-cpp`)."
            )
        if result.returncode != 0:
            lines = (result.stderr or "").strip().splitlines()
            detail = lines[-1] if lines else f"exit code {result.returncode}"
            raise WhisperExecutionError(f"whisper.cpp failed: {detail}")

    def _parse(self, raw: str) -> tuple[list[TranscriptSegment], str | None]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WhisperExecutionError(
                f"whisper.cpp output was not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise WhisperExecutionError("whisper.cpp output was not a JSON object.")
        language = _detected_language(data)
        entries = data.get("transcription") or []
        if not isinstance(entries, list):
            raise WhisperExecutionError(
                "whisper.cpp output 'transcription' was not a list."
            )
        segments: list[TranscriptSegment] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise WhisperExecutionError(
                    "whisper.cpp transcript entry was not an object."
                )
            offsets = entry.get("offsets")
            if not isinstance(offsets, dict):
                raise WhisperExecutionError(
                    "whisper.cpp transcript entry had no 'offsets' object."
                )
            start = _ms_to_seconds(offsets.get("from"))
            end = _ms_to_seconds(offsets.get("to"))
            if start is None or end is None:
                raise WhisperExecutionError(
                    "whisper.cpp transcript entry had non-numeric offsets."
                )
            text = entry.get("text")
            segments.append(
                TranscriptSegment(
                    start_seconds=start,
                    end_seconds=end,
                    text=text.strip() if isinstance(text, str) else "",
                )
            )
        return segments, language


def _detected_language(data: dict[str, Any]) -> str | None:
    for key in ("result", "params"):
        block = data.get(key)
        if isinstance(block, dict):
            language = block.get("language")
            if isinstance(language, str) and language:
                return language
    return None


def _ms_to_seconds(raw: Any) -> float | None:
    """Convert a whisper.cpp millisecond offset to float seconds (exact)."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw) / 1000.0
