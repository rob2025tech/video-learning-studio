"""Unit tests for ReportService/ReportResult/ReportStore (Stage 0G).

Subprocess-heavy collaborators are faked, so the suite never needs ffmpeg,
tesseract, or whisper.cpp. Covers report construction, the three evidence
sections, the structured transcription status (ok / no-audio / unavailable),
deterministic serialization, and non-destructive artifact writing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_learning.core.errors import (
    FileNotFoundError_,
    ReportArtifactError,
    TranscriptionError,
    TranscriptionToolMissingError,
)
from video_learning.core.keywords import KeywordSuggestion, PhraseSuggestion
from video_learning.core.models import AudioStreamInfo, MediaInfo, VideoStreamInfo
from video_learning.core.transcript import (
    CapabilityItem,
    TranscriptionCapability,
    TranscriptResult,
    TranscriptSegment,
)
from video_learning.services.analyze_service import AnalyzeResult, FrameSnapshot
from video_learning.services.report_service import REPORT_SCHEMA, ReportService, ReportStore


class FakeInspect:
    def __init__(self, media: MediaInfo) -> None:
        self._media = media
        self.calls: list[Path] = []

    def inspect(self, path: Path) -> MediaInfo:
        self.calls.append(path)
        return self._media


class FakeAnalyze:
    def __init__(self, result: AnalyzeResult) -> None:
        self._result = result
        self.calls: list[Path] = []

    def analyze(self, path: Path) -> AnalyzeResult:
        self.calls.append(path)
        return self._result


class FakeTranscribe:
    def __init__(
        self,
        result: TranscriptResult | None = None,
        error: TranscriptionError | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[Path, Path | None]] = []

    def transcribe(self, path: Path, *, model: Path | None = None) -> TranscriptResult:
        self.calls.append((path, model))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class FakeWhisper:
    def __init__(self, capability: TranscriptionCapability) -> None:
        self._capability = capability
        self.calls: list[Path | None] = []

    def detect_capability(self, model: Path | None) -> TranscriptionCapability:
        self.calls.append(model)
        return self._capability


def _capability(*, ready: bool) -> TranscriptionCapability:
    backend = CapabilityItem(
        "whisper.cpp",
        "available" if ready else "missing",
        None if ready else "whisper-cli not found on PATH — install whisper.cpp",
    )
    model = CapabilityItem(
        "ggml-base.en",
        "available" if ready else "missing",
        None if ready else "model not found — pass --model PATH or set VLS_WHISPER_MODEL",
    )
    return TranscriptionCapability(
        backend=backend,
        model=model,
        recommended="whisper.cpp + ggml-base.en",
        recommended_detail="Local • English",
        using_backend="whisper.cpp",
        using_model="ggml-base.en" if ready else "—",
        using_language="English" if ready else "auto",
        ready=ready,
    )


def _media(tmp_path: Path, *, has_audio: bool = True) -> MediaInfo:
    return MediaInfo(
        path=tmp_path / "clip.mov",
        filename="clip.mov",
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
        duration_seconds=100.0,
        video=VideoStreamInfo(
            codec="h264", width=1920, height=1080, frame_rate=30.0, pixel_format="yuv420p"
        ),
        audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2, language="und")
        if has_audio
        else None,
    )


def _analysis(tmp_path: Path) -> AnalyzeResult:
    return AnalyzeResult(
        source=tmp_path / "clip.mov",
        keywords=[KeywordSuggestion("Qoder", 3), KeywordSuggestion("video", 2)],
        phrases=[PhraseSuggestion("pyproject.toml", 2)],
        snapshots=[
            FrameSnapshot(15.2, "video-learning-studio\npyproject.toml"),
            FrameSnapshot(45.6, "Qoder changelog"),
        ],
        frames_analyzed=2,
        ocr_text_chars=48,
    )


def _empty_analysis(tmp_path: Path) -> AnalyzeResult:
    return AnalyzeResult(
        source=tmp_path / "clip.mov",
        keywords=[],
        phrases=[],
        snapshots=[],
        frames_analyzed=0,
        ocr_text_chars=0,
    )


def _transcript(*segments: TranscriptSegment) -> TranscriptResult:
    return TranscriptResult(
        source="clip.mov",
        backend="whisper.cpp",
        model="ggml-base.en",
        language="en",
        segments=segments,
    )


def _service(
    tmp_path: Path,
    *,
    has_audio: bool = True,
    transcript: TranscriptResult | None = None,
    error: TranscriptionError | None = None,
    ready: bool = True,
    analysis: AnalyzeResult | None = None,
) -> tuple[ReportService, FakeInspect, FakeAnalyze, FakeTranscribe, FakeWhisper]:
    inspect = FakeInspect(_media(tmp_path, has_audio=has_audio))
    analyze = FakeAnalyze(analysis if analysis is not None else _analysis(tmp_path))
    transcribe = FakeTranscribe(result=transcript, error=error)
    whisper = FakeWhisper(_capability(ready=ready))
    service = ReportService(
        inspect_service=inspect,  # type: ignore[arg-type]
        analyze_service=analyze,  # type: ignore[arg-type]
        transcribe_service=transcribe,  # type: ignore[arg-type]
        whisper=whisper,  # type: ignore[arg-type]
    )
    return service, inspect, analyze, transcribe, whisper


# -- construction ------------------------------------------------------------


def test_build_combines_all_evidence(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    segments = (
        TranscriptSegment(17.4, 23.1, "Today we build a studio"),
        TranscriptSegment(60.0, 64.0, "Later we review"),
    )
    service, inspect, analyze, transcribe, _ = _service(tmp_path, transcript=_transcript(*segments))

    report = service.build(video)

    assert inspect.calls == [video]
    assert analyze.calls == [video]
    assert len(transcribe.calls) == 1
    assert report.transcription_status == "ok"
    assert report.transcription is not None
    # 2 OCR snapshots + 2 speech segments merged into one timeline.
    assert len(report.timeline) == 4


def test_report_schema_and_top_level_shape(tmp_path: Path) -> None:
    service, *_ = _service(tmp_path, transcript=_transcript(TranscriptSegment(0.0, 1.0, "hi")))

    payload = service.build(tmp_path / "clip.mov").to_dict()

    assert payload["schema"] == REPORT_SCHEMA
    assert payload["source"] == str(tmp_path / "clip.mov")
    assert set(payload) == {
        "schema",
        "source",
        "metadata",
        "ocr",
        "transcription",
        "timeline",
        "renamed",
        "applied",
        "source_modified",
    }
    assert payload["renamed"] is False
    assert payload["applied"] is False
    assert payload["source_modified"] is False


# -- metadata ----------------------------------------------------------------


def test_metadata_section_is_stable_and_excludes_volatile_filesystem_values(
    tmp_path: Path,
) -> None:
    service, *_ = _service(tmp_path, transcript=_transcript())

    meta = service.build(tmp_path / "clip.mov").to_dict()["metadata"]

    assert meta["filename"] == "clip.mov"
    assert meta["duration_seconds"] == 100.0
    assert meta["has_audio"] is True
    assert meta["video"] == {
        "codec": "h264",
        "width": 1920,
        "height": 1080,
        "frame_rate": 30.0,
        "pixel_format": "yuv420p",
    }
    assert meta["audio"] == {
        "codec": "aac",
        "sample_rate": 48000,
        "channels": 2,
        "language": "und",
    }
    # mtime/size would make the report nondeterministic, so they are omitted.
    assert "modified_at" not in meta
    assert "size_bytes" not in meta
    assert "size_human" not in meta


# -- OCR ---------------------------------------------------------------------


def test_ocr_section_preserves_complete_text(tmp_path: Path) -> None:
    service, *_ = _service(tmp_path, transcript=_transcript())

    ocr = service.build(tmp_path / "clip.mov").to_dict()["ocr"]

    assert ocr["frames_analyzed"] == 2
    assert ocr["ocr_text_chars"] == 48
    # Complete OCR text is preserved verbatim (including the newline), not trimmed.
    assert ocr["snapshots"][0] == {
        "timestamp_seconds": 15.2,
        "ocr_text": "video-learning-studio\npyproject.toml",
    }
    assert ocr["phrases"] == [{"phrase": "pyproject.toml", "count": 2}]
    assert ocr["keywords"][0] == {"term": "Qoder", "count": 3}


def test_empty_ocr_yields_speech_only_timeline(tmp_path: Path) -> None:
    service, *_ = _service(
        tmp_path,
        transcript=_transcript(TranscriptSegment(5.0, 8.0, "only speech")),
        analysis=_empty_analysis(tmp_path),
    )

    report = service.build(tmp_path / "clip.mov")
    payload = report.to_dict()

    assert payload["ocr"]["snapshots"] == []
    assert payload["ocr"]["frames_analyzed"] == 0
    assert len(report.timeline) == 1
    assert payload["timeline"][0]["type"] == "speech"


# -- transcription -----------------------------------------------------------


def test_transcription_section_provenance(tmp_path: Path) -> None:
    service, *_ = _service(
        tmp_path, transcript=_transcript(TranscriptSegment(17.4, 23.1, "Today we build"))
    )

    section = service.build(tmp_path / "clip.mov").to_dict()["transcription"]

    assert section == {
        "status": "ok",
        "detail": None,
        "backend": "whisper.cpp",
        "model": "ggml-base.en",
        "language": "en",
        "segments": [{"start_seconds": 17.4, "end_seconds": 23.1, "text": "Today we build"}],
        "segment_count": 1,
    }


def test_empty_transcript_is_still_ok(tmp_path: Path) -> None:
    service, *_ = _service(tmp_path, transcript=_transcript())

    report = service.build(tmp_path / "clip.mov")
    section = report.to_dict()["transcription"]

    assert report.transcription_status == "ok"
    assert section["status"] == "ok"
    assert section["segments"] == []
    assert section["segment_count"] == 0
    # Timeline carries only the OCR points.
    assert [event.to_dict()["type"] for event in report.timeline] == ["ocr", "ocr"]


def test_no_audio_skips_transcription_entirely(tmp_path: Path) -> None:
    service, _, _, transcribe, whisper = _service(tmp_path, has_audio=False)

    report = service.build(tmp_path / "clip.mov")

    assert report.transcription_status == "no-audio"
    assert report.transcription is None
    assert report.transcription_detail is None
    # No capability probe and no transcription attempt when there is no audio.
    assert transcribe.calls == []
    assert whisper.calls == []
    assert report.to_dict()["transcription"] == {
        "status": "no-audio",
        "detail": None,
        "backend": None,
        "model": None,
        "language": None,
        "segments": [],
        "segment_count": 0,
    }
    # OCR/metadata remain useful.
    assert len(report.timeline) == 2


def test_unavailable_backend_records_blocking_message(tmp_path: Path) -> None:
    service, _, _, transcribe, whisper = _service(tmp_path, ready=False)

    report = service.build(tmp_path / "clip.mov")

    assert report.transcription_status == "unavailable"
    assert report.transcription is None
    assert report.transcription_detail is not None
    assert "whisper-cli not found" in report.transcription_detail
    # Capability was probed, but transcription was never attempted.
    assert whisper.calls == [None]
    assert transcribe.calls == []
    section = report.to_dict()["transcription"]
    assert section["status"] == "unavailable"
    assert section["backend"] is None
    assert section["segments"] == []
    assert section["segment_count"] == 0


def test_runtime_failure_is_explicit_and_deterministic(tmp_path: Path) -> None:
    # Capability ready but transcribe() raises: the report stays explicit, and the
    # detail is a fixed string — never the (nondeterministic) exception text.
    service, _, _, _, _ = _service(
        tmp_path,
        ready=True,
        error=TranscriptionToolMissingError("boom /tmp/random-1234"),
    )

    report = service.build(tmp_path / "clip.mov")

    assert report.transcription_status == "unavailable"
    assert report.transcription is None
    assert report.transcription_detail == "transcription was attempted but failed at runtime"
    assert "boom" not in (report.transcription_detail or "")


# -- timeline ----------------------------------------------------------------


def test_timeline_is_chronological_and_keeps_types_distinct(tmp_path: Path) -> None:
    segments = (
        TranscriptSegment(17.4, 23.1, "Today we build"),
        TranscriptSegment(60.0, 64.0, "Later we review"),
    )
    service, *_ = _service(tmp_path, transcript=_transcript(*segments))

    report = service.build(tmp_path / "clip.mov")

    times = [event.sort_key[0] for event in report.timeline]
    assert times == sorted(times)
    # 15.2 ocr, 17.4 speech, 45.6 ocr, 60.0 speech
    assert [event.to_dict()["type"] for event in report.timeline] == [
        "ocr",
        "speech",
        "ocr",
        "speech",
    ]


# -- determinism -------------------------------------------------------------


def test_to_dict_serialization_is_deterministic(tmp_path: Path) -> None:
    service, *_ = _service(
        tmp_path, transcript=_transcript(TranscriptSegment(0.0, 1.5, "hi there"))
    )
    report = service.build(tmp_path / "clip.mov")

    first = json.dumps(report.to_dict(), indent=2)
    second = json.dumps(report.to_dict(), indent=2)

    assert first == second


# -- errors ------------------------------------------------------------------


def test_inspect_failure_propagates(tmp_path: Path) -> None:
    class Boom:
        def inspect(self, path: Path) -> MediaInfo:
            raise FileNotFoundError_(f"File not found: {path}")

    whisper = FakeWhisper(_capability(ready=True))
    service = ReportService(
        inspect_service=Boom(),  # type: ignore[arg-type]
        analyze_service=FakeAnalyze(_analysis(tmp_path)),  # type: ignore[arg-type]
        transcribe_service=FakeTranscribe(result=_transcript()),  # type: ignore[arg-type]
        whisper=whisper,  # type: ignore[arg-type]
    )

    with pytest.raises(FileNotFoundError_):
        service.build(tmp_path / "nope.mov")


# -- store -------------------------------------------------------------------


def test_store_writes_deterministic_schema_tagged_artifact(tmp_path: Path) -> None:
    service, *_ = _service(
        tmp_path, transcript=_transcript(TranscriptSegment(0.0, 1.0, "hi"))
    )
    report = service.build(tmp_path / "clip.mov")
    store = ReportStore()
    out = tmp_path / "reports" / "report.json"

    returned = store.save(report, out)

    assert returned == out
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == REPORT_SCHEMA
    # Byte-identical on a second write (determinism).
    first_bytes = out.read_bytes()
    store.save(report, out)
    assert out.read_bytes() == first_bytes


def test_store_creates_missing_parent_directories(tmp_path: Path) -> None:
    service, *_ = _service(tmp_path, transcript=_transcript())
    report = service.build(tmp_path / "clip.mov")
    out = tmp_path / "a" / "b" / "c" / "report.json"

    ReportStore().save(report, out)

    assert out.exists()


def test_store_never_touches_the_source_video(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"source-bytes")
    before = video.stat()
    service, *_ = _service(tmp_path, transcript=_transcript())
    report = service.build(video)

    ReportStore().save(report, tmp_path / "report.json")

    after = video.stat()
    assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime)
    assert video.read_bytes() == b"source-bytes"


def test_store_write_error_raises_report_artifact_error(tmp_path: Path) -> None:
    service, *_ = _service(tmp_path, transcript=_transcript())
    report = service.build(tmp_path / "clip.mov")
    # A path whose "parent" is an existing file cannot be created/written.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")

    with pytest.raises(ReportArtifactError):
        ReportStore().save(report, blocker / "report.json")
