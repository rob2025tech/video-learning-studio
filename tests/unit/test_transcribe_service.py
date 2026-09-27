"""Unit tests for TranscribeService: orchestration with fakes, no binaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_learning.core.errors import (
    FileNotFoundError_,
    TranscriptionModelMissingError,
)
from video_learning.core.models import AudioStreamInfo, MediaInfo
from video_learning.core.transcript import TranscriptSegment
from video_learning.services.inspect_service import InspectService
from video_learning.services.transcribe_service import TranscribeService


class FakeProbe:
    def __init__(self, media: MediaInfo) -> None:
        self._media = media
        self.calls: list[Path] = []

    def probe(self, path: Path) -> MediaInfo:
        self.calls.append(path)
        return self._media


class FakeAudio:
    def __init__(self) -> None:
        self.calls: list[tuple[MediaInfo, Path]] = []

    def extract(self, media: MediaInfo, out_dir: Path) -> Path:
        self.calls.append((media, out_dir))
        wav = out_dir / "audio.wav"
        wav.write_bytes(b"RIFF")
        return wav


class FakeWhisper:
    def __init__(
        self, segments: list[TranscriptSegment], language: str | None
    ) -> None:
        self._segments = segments
        self._language = language
        self.calls: list[tuple[Path, Path, Path]] = []

    def transcribe(
        self,
        wav_path: Path,
        model: Path,
        out_dir: Path,
        language: str | None = None,
    ) -> tuple[list[TranscriptSegment], str | None]:
        self.calls.append((wav_path, model, out_dir))
        return list(self._segments), self._language


def _service(
    tmp_path: Path, segments: list[TranscriptSegment], language: str | None = "en"
) -> tuple[TranscribeService, FakeProbe, FakeAudio, FakeWhisper]:
    media = MediaInfo(
        path=tmp_path / "clip.mov",
        filename="clip.mov",
        duration_seconds=10.0,
        audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2),
    )
    probe = FakeProbe(media)
    audio = FakeAudio()
    whisper = FakeWhisper(segments, language)
    service = TranscribeService(
        inspect_service=InspectService(probe=probe),
        audio_extractor=audio,  # type: ignore[arg-type]
        whisper=whisper,  # type: ignore[arg-type]
    )
    return service, probe, audio, whisper


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "ggml-base.en.bin"
    model.write_bytes(b"x")
    return model


def _video(tmp_path: Path) -> Path:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    return video


def test_composes_probe_audio_whisper(tmp_path: Path) -> None:
    video = _video(tmp_path)
    segments = [TranscriptSegment(0.0, 3.84, "Hello world")]
    service, probe, audio, whisper = _service(tmp_path, segments)

    result = service.transcribe(video, model=_model(tmp_path))

    assert probe.calls == [video]  # went through InspectService -> MediaProbe once
    assert len(audio.calls) == 1
    assert len(whisper.calls) == 1
    assert result.segments == tuple(segments)
    assert result.segment_count == 1


def test_records_provenance(tmp_path: Path) -> None:
    video = _video(tmp_path)
    service, _, _, _ = _service(tmp_path, [TranscriptSegment(0.0, 1.0, "hi")], "en")

    result = service.transcribe(video, model=_model(tmp_path))

    assert result.backend == "whisper.cpp"
    assert result.model == "ggml-base.en"
    assert result.language == "en"
    assert result.source == str(video)


def test_language_defaults_to_unknown(tmp_path: Path) -> None:
    video = _video(tmp_path)
    service, _, _, _ = _service(tmp_path, [], language=None)

    result = service.transcribe(video, model=_model(tmp_path))

    assert result.language == "unknown"


def test_to_dict_shape_and_guarantees(tmp_path: Path) -> None:
    video = _video(tmp_path)
    service, _, _, _ = _service(tmp_path, [TranscriptSegment(0.0, 3.84, "Hello")])

    payload = service.transcribe(video, model=_model(tmp_path)).to_dict()

    assert payload["segments"] == [
        {"start_seconds": 0.0, "end_seconds": 3.84, "text": "Hello"}
    ]
    assert payload["segment_count"] == 1
    assert payload["backend"] == "whisper.cpp"
    for key in (
        "source",
        "model",
        "language",
        "renamed",
        "applied",
        "source_modified",
    ):
        assert key in payload
    assert payload["renamed"] is False
    assert payload["applied"] is False
    assert payload["source_modified"] is False


def test_temp_audio_is_cleaned_up(tmp_path: Path) -> None:
    video = _video(tmp_path)
    service, _, audio, _ = _service(tmp_path, [TranscriptSegment(0.0, 1.0, "hi")])

    service.transcribe(video, model=_model(tmp_path))

    # The WAV was written into a TemporaryDirectory that is now gone.
    _, out_dir = audio.calls[0]
    assert not out_dir.exists()


def test_missing_model_raises_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLS_WHISPER_MODEL", raising=False)
    video = _video(tmp_path)
    service, probe, audio, whisper = _service(tmp_path, [])

    with pytest.raises(TranscriptionModelMissingError, match="VLS_WHISPER_MODEL"):
        service.transcribe(video)

    assert probe.calls == []
    assert audio.calls == []
    assert whisper.calls == []


def test_explicit_missing_model_raises(tmp_path: Path) -> None:
    video = _video(tmp_path)
    service, _, _, _ = _service(tmp_path, [])

    with pytest.raises(TranscriptionModelMissingError, match="not found at"):
        service.transcribe(video, model=tmp_path / "gone.bin")


def test_model_resolved_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLS_WHISPER_MODEL", str(_model(tmp_path)))
    video = _video(tmp_path)
    service, _, _, _ = _service(tmp_path, [TranscriptSegment(0.0, 1.0, "hi")])

    result = service.transcribe(video)

    assert result.model == "ggml-base.en"


def test_missing_file_raises(tmp_path: Path) -> None:
    service, probe, _, _ = _service(tmp_path, [])

    with pytest.raises(FileNotFoundError_):
        service.transcribe(tmp_path / "nope.mov", model=_model(tmp_path))

    assert probe.calls == []
