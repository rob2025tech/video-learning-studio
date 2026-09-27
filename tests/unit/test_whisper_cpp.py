"""Unit tests for the whisper.cpp adapter: discovery, capability, JSON parsing.

All subprocess and filesystem boundaries are mocked; no model is downloaded and
no real ``whisper-cli`` is required, so the suite stays deterministic/offline.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from video_learning.adapters import whisper_cpp
from video_learning.adapters.whisper_cpp import WhisperCpp, resolve_model
from video_learning.core.errors import (
    TranscriptionToolMissingError,
    WhisperExecutionError,
)
from video_learning.core.transcript import TranscriptSegment

WHICH = "video_learning.adapters.whisper_cpp.shutil.which"
RUN = "video_learning.adapters.whisper_cpp.subprocess.run"

RunResult = subprocess.CompletedProcess[str]


def _completed(returncode: int = 0, stderr: str = "") -> RunResult:
    return subprocess.CompletedProcess(
        args=["whisper-cli"], returncode=returncode, stdout="", stderr=stderr
    )


def _fake_whisper_run(payload: object) -> Callable[..., RunResult]:
    """A subprocess.run stand-in that writes whisper.cpp's JSON output file."""
    text = payload if isinstance(payload, str) else json.dumps(payload)

    def fake_run(command: list[str], **kwargs: object) -> RunResult:
        prefix = command[command.index("-of") + 1]
        Path(f"{prefix}.json").write_text(text, encoding="utf-8")
        return _completed(0)

    return fake_run


# -- model resolution -------------------------------------------------------


def test_resolve_model_explicit_existing(tmp_path: Path) -> None:
    model = tmp_path / "ggml-base.en.bin"
    model.write_bytes(b"x")
    assert resolve_model(model) == model


def test_resolve_model_explicit_missing(tmp_path: Path) -> None:
    assert resolve_model(tmp_path / "nope.bin") is None


def test_resolve_model_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "ggml-base.en.bin"
    model.write_bytes(b"x")
    monkeypatch.setenv(whisper_cpp.MODEL_ENV_VAR, str(model))
    assert resolve_model(None) == model


def test_resolve_model_env_points_at_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(whisper_cpp.MODEL_ENV_VAR, str(tmp_path / "gone.bin"))
    assert resolve_model(None) is None


def test_resolve_model_nothing_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(whisper_cpp.MODEL_ENV_VAR, raising=False)
    assert resolve_model(None) is None


# -- backend discovery / capability ----------------------------------------


def test_capability_backend_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(whisper_cpp.MODEL_ENV_VAR, raising=False)
    with patch(WHICH, return_value=None):
        cap = WhisperCpp().detect_capability(None)
    assert cap.backend.state == "missing"
    assert cap.ready is False
    assert "brew install whisper-cpp" in (cap.backend.detail or "")


def test_capability_backend_and_model_available(tmp_path: Path) -> None:
    model = tmp_path / "ggml-base.en.bin"
    model.write_bytes(b"x")
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(RUN, return_value=_completed(0)),
    ):
        cap = WhisperCpp().detect_capability(model)
    assert cap.backend.state == "available"
    assert cap.model.state == "available"
    assert cap.model.label == "ggml-base.en"
    assert cap.ready is True
    assert cap.using_language == "English"
    assert cap.recommended == "whisper.cpp + ggml-base.en"


def test_capability_backend_broken_shim_127(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(whisper_cpp.MODEL_ENV_VAR, raising=False)
    with (
        patch(WHICH, return_value="/Users/x/.pyenv/shims/whisper-cli"),
        patch(RUN, return_value=_completed(127)),
    ):
        cap = WhisperCpp().detect_capability(None)
    assert cap.backend.state == "unusable"
    assert "127" in (cap.backend.detail or "")
    assert cap.ready is False


def test_capability_backend_probe_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(whisper_cpp.MODEL_ENV_VAR, raising=False)
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(RUN, side_effect=OSError("exec format error")),
    ):
        cap = WhisperCpp().detect_capability(None)
    assert cap.backend.state == "unusable"


def test_capability_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(whisper_cpp.MODEL_ENV_VAR, raising=False)
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(RUN, return_value=_completed(0)),
    ):
        cap = WhisperCpp().detect_capability(None)
    assert cap.model.state == "missing"
    assert cap.model.label == whisper_cpp.RECOMMENDED_MODEL
    assert cap.ready is False
    assert "VLS_WHISPER_MODEL" in cap.blocking_message()


# -- transcription ----------------------------------------------------------


def test_transcribe_missing_binary_raises(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value=None),
        pytest.raises(TranscriptionToolMissingError, match="brew install whisper-cpp"),
    ):
        WhisperCpp().transcribe(
            tmp_path / "audio.wav", tmp_path / "ggml-base.en.bin", tmp_path
        )


def test_transcribe_command_construction_and_parse(
    tmp_path: Path, whisper_cpp_json: str
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(command: list[str], **kwargs: object) -> RunResult:
        captured["command"] = command
        prefix = command[command.index("-of") + 1]
        Path(f"{prefix}.json").write_text(whisper_cpp_json, encoding="utf-8")
        return _completed(0)

    model = tmp_path / "ggml-base.en.bin"
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(RUN, side_effect=fake_run),
    ):
        segments, language = WhisperCpp().transcribe(
            tmp_path / "audio.wav", model, tmp_path
        )

    command = captured["command"]
    assert command[0] == "whisper-cli"
    assert command[command.index("-m") + 1] == str(model)
    assert command[command.index("-f") + 1] == str(tmp_path / "audio.wav")
    assert "-oj" in command
    assert command[command.index("-of") + 1] == str(tmp_path / "transcript")
    assert language == "en"
    assert len(segments) == 2


def test_transcribe_ms_to_seconds_and_order(
    tmp_path: Path, whisper_cpp_json: str
) -> None:
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(RUN, side_effect=_fake_whisper_run(whisper_cpp_json)),
    ):
        segments, _ = WhisperCpp().transcribe(
            tmp_path / "a.wav", tmp_path / "ggml-base.en.bin", tmp_path
        )

    # 0 ms -> 0.0 s, 3840 ms -> 3.84 s, 7200 ms -> 7.2 s; source order kept.
    assert segments == [
        TranscriptSegment(
            0.0, 3.84, "Today we're going to build a small command line tool."
        ),
        TranscriptSegment(3.84, 7.2, "It reads a video and prints what it sees."),
    ]
    starts = [s.start_seconds for s in segments]
    assert starts == sorted(starts)


def test_transcribe_empty(tmp_path: Path) -> None:
    payload = {"result": {"language": "en"}, "transcription": []}
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(RUN, side_effect=_fake_whisper_run(payload)),
    ):
        segments, language = WhisperCpp().transcribe(
            tmp_path / "a.wav", tmp_path / "m.bin", tmp_path
        )
    assert segments == []
    assert language == "en"


def test_transcribe_broken_shim_127(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value="/Users/x/.pyenv/shims/whisper-cli"),
        patch(RUN, return_value=_completed(127)),
        pytest.raises(TranscriptionToolMissingError, match="127"),
    ):
        WhisperCpp().transcribe(tmp_path / "a.wav", tmp_path / "m.bin", tmp_path)


def test_transcribe_nonzero_exit(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(RUN, return_value=_completed(1, stderr="model load failed")),
        pytest.raises(WhisperExecutionError, match="model load failed"),
    ):
        WhisperCpp().transcribe(tmp_path / "a.wav", tmp_path / "m.bin", tmp_path)


def test_transcribe_timeout(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(
            RUN,
            side_effect=subprocess.TimeoutExpired(cmd=["whisper-cli"], timeout=3600),
        ),
        pytest.raises(WhisperExecutionError, match="timed out"),
    ):
        WhisperCpp().transcribe(tmp_path / "a.wav", tmp_path / "m.bin", tmp_path)


def test_transcribe_no_json_written(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(RUN, return_value=_completed(0)),  # succeeds but writes no JSON
        pytest.raises(WhisperExecutionError, match="no JSON output"),
    ):
        WhisperCpp().transcribe(tmp_path / "a.wav", tmp_path / "m.bin", tmp_path)


def test_transcribe_malformed_json(tmp_path: Path) -> None:
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(RUN, side_effect=_fake_whisper_run("not json {")),
        pytest.raises(WhisperExecutionError, match="not valid JSON"),
    ):
        WhisperCpp().transcribe(tmp_path / "a.wav", tmp_path / "m.bin", tmp_path)


def test_transcribe_entry_without_offsets(tmp_path: Path) -> None:
    payload = {
        "result": {"language": "en"},
        "transcription": [{"text": "no offsets here"}],
    }
    with (
        patch(WHICH, return_value="/opt/bin/whisper-cli"),
        patch(RUN, side_effect=_fake_whisper_run(payload)),
        pytest.raises(WhisperExecutionError, match="offsets"),
    ):
        WhisperCpp().transcribe(tmp_path / "a.wav", tmp_path / "m.bin", tmp_path)


def test_parse_is_deterministic(whisper_cpp_json: str) -> None:
    adapter = WhisperCpp()
    assert adapter._parse(whisper_cpp_json) == adapter._parse(whisper_cpp_json)
