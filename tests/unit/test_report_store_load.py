"""Write-then-read round-trip tests for the report artifact (Multimodal Search v1, Step 3).

The artifact is written and read by the production code paths, so these pin the
serialization pair against the real on-disk shape instead of a fixture that could drift
from it. Nothing here touches a video: the ``source`` recorded in the artifact names a
path that never exists, which is only possible because reading a report never opens it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from video_learning.core.errors import ReportArtifactError
from video_learning.core.keywords import KeywordSuggestion, PhraseSuggestion
from video_learning.core.models import AudioStreamInfo, MediaInfo, VideoStreamInfo
from video_learning.core.scene import VisualChange
from video_learning.core.segments import BoundaryCandidate, Segment, Segmentation, SpeechBoundary
from video_learning.core.timeline import OcrEvent, SpeechEvent, build_timeline
from video_learning.core.transcript import TranscriptResult, TranscriptSegment
from video_learning.services.analyze_service import AnalyzeResult, FrameSnapshot
from video_learning.services.report_service import REPORT_SCHEMA, ReportResult, ReportStore
from video_learning.services.study_service import StudyEvidenceService


def _report(video: Path) -> ReportResult:
    media = MediaInfo(
        path=video,
        filename=video.name,
        container_format="mov",
        duration_seconds=60.0,
        video=VideoStreamInfo(
            codec="h264", width=1920, height=1080, frame_rate=30.0, pixel_format="yuv420p"
        ),
        audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2, language="und"),
    )
    analysis = AnalyzeResult(
        source=video,
        keywords=[KeywordSuggestion("Qoder", 3)],
        phrases=[PhraseSuggestion("pyproject.toml", 2)],
        snapshots=[FrameSnapshot(15.2, "video-learning-studio pyproject.toml")],
        frames_analyzed=1,
        ocr_text_chars=34,
    )
    transcription = TranscriptResult(
        source=str(video),
        backend="whisper.cpp",
        model="ggml-base.en",
        language="en",
        segments=(
            TranscriptSegment(17.4, 23.1, "Today we build a local search index"),
            TranscriptSegment(31.0, 36.5, "and rank the evidence by timestamp"),
        ),
    )
    timeline = build_timeline(
        [
            OcrEvent(15.2, "video-learning-studio pyproject.toml"),
            OcrEvent(33.0, "SearchIndex records"),
        ],
        [
            SpeechEvent(17.4, 23.1, "Today we build a local search index"),
            SpeechEvent(31.0, 36.5, "and rank the evidence by timestamp"),
        ],
    )
    segmentation = Segmentation(
        visual_changes=(VisualChange(30.0, 27.5),),
        speech_boundaries=(SpeechBoundary(27.05, 7.9),),
        boundaries=(
            BoundaryCandidate(
                timestamp_seconds=27.05,
                reason="combined",
                visual_changes=(VisualChange(30.0, 27.5),),
                speech_boundaries=(SpeechBoundary(27.05, 7.9),),
            ),
        ),
        segments=(Segment(0.0, 27.05, "combined"), Segment(27.05, 60.0, None)),
        video_duration_seconds=60.0,
    )
    return ReportResult(
        source=video,
        media=media,
        analysis=analysis,
        transcription=transcription,
        transcription_status="ok",
        transcription_detail=None,
        segmentation=segmentation,
        timeline=timeline,
    )


def _without_segments(report: ReportResult) -> ReportResult:
    """The documented unknown-duration shape: evidence kept, no segments assembled.

    Nothing in this shape compares a timestamp against a segment window, so it is where
    a wrong-typed number would otherwise survive rehydration all the way to a renderer.
    """
    return replace(report, segmentation=Segmentation((), (), (), (), None))


def _write(tmp_path: Path, report: ReportResult) -> Path:
    out = tmp_path / "report.json"
    ReportStore().save(report, out)
    return out


def _rewrite(path: Path, mutate: dict[str, object]) -> None:
    """Persist ``mutate`` merged over the artifact already stored at ``path``."""
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(mutate)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_load_evidence_matches_the_service_derivation(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"  # deliberately never created
    report = _report(video)
    out = _write(tmp_path, report)

    loaded = ReportStore().load_evidence(out)

    assert loaded.to_dict() == StudyEvidenceService().derive(report).to_dict()
    assert loaded.source == video


def test_load_evidence_carries_a_no_audio_transcription_state(tmp_path: Path) -> None:
    video = tmp_path / "clip.mov"
    report = replace(
        _report(video),
        transcription=None,
        transcription_status="no-audio",
        transcription_detail=None,
    )
    out = _write(tmp_path, report)

    loaded = ReportStore().load_evidence(out)

    assert loaded.transcription_status == "no-audio"
    assert loaded.transcription_available is False
    assert loaded.to_dict() == StudyEvidenceService().derive(report).to_dict()


def test_load_evidence_ignores_sections_it_never_reads(tmp_path: Path) -> None:
    """Only the segmentation, timeline and transcription status are rehydrated.

    ``metadata`` is a flattened, volatile-field-free projection with no ``MediaInfo``
    counterpart and ``ocr`` is evidence the derivation does not consume, so a reader
    must not need either one — and the duration below must come from the segmentation.
    """
    out = _write(tmp_path, _report(tmp_path / "clip.mov"))
    _rewrite(out, {"metadata": "not even an object", "ocr": None})

    assert ReportStore().load_evidence(out).video_duration_seconds == 60.0


def test_load_evidence_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReportArtifactError, match="Could not read report artifact"):
        ReportStore().load_evidence(tmp_path / "nope.json")


def test_load_evidence_reports_invalid_json(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    out.write_text("{not json", encoding="utf-8")

    with pytest.raises(ReportArtifactError, match="is not valid JSON"):
        ReportStore().load_evidence(out)


def test_load_evidence_reports_invalid_utf8(tmp_path: Path) -> None:
    """A byte-level corruption is reported against the artifact, not raised as a traceback.

    Everything except the one corrupted byte is what the writer produced, so the encoding
    is the only fault. ``UnicodeDecodeError`` is a ``ValueError`` rather than an ``OSError``,
    which is exactly why it needs its own handling on the read.
    """
    out = _write(tmp_path, _report(tmp_path / "clip.mov"))
    out.write_bytes(out.read_bytes().replace(b"clip.mov", b"clip\xff.mov", 1))

    with pytest.raises(ReportArtifactError, match="Could not read report artifact") as exc:
        ReportStore().load_evidence(out)
    assert str(out) in str(exc.value)
    assert isinstance(exc.value.__cause__, UnicodeDecodeError)


def test_load_evidence_reports_a_json_array(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    out.write_text("[]", encoding="utf-8")

    with pytest.raises(ReportArtifactError, match="is not a JSON object"):
        ReportStore().load_evidence(out)


def test_load_evidence_reports_a_foreign_schema(tmp_path: Path) -> None:
    out = _write(tmp_path, _report(tmp_path / "clip.mov"))
    _rewrite(out, {"schema": "video-learning.report/v2"})

    with pytest.raises(ReportArtifactError, match="Unsupported report artifact schema") as exc:
        ReportStore().load_evidence(out)
    assert REPORT_SCHEMA in str(exc.value)


def test_load_evidence_reports_a_missing_section(tmp_path: Path) -> None:
    out = _write(tmp_path, _report(tmp_path / "clip.mov"))
    document = json.loads(out.read_text(encoding="utf-8"))
    del document["timeline"]
    out.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReportArtifactError, match="is missing the") as exc:
        ReportStore().load_evidence(out)
    assert "'timeline'" in str(exc.value)


def test_load_evidence_reports_a_malformed_boundary_reason(tmp_path: Path) -> None:
    out = _write(tmp_path, _report(tmp_path / "clip.mov"))
    document = json.loads(out.read_text(encoding="utf-8"))
    document["segments"]["segments"][0]["boundary_reason"] = "chapter"
    out.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReportArtifactError, match="cannot interpret") as exc:
        ReportStore().load_evidence(out)
    assert "chapter" in str(exc.value)


def test_load_evidence_refuses_an_artifact_from_other_assembly_rules(tmp_path: Path) -> None:
    """A threshold the writer disagrees with is refused, not adopted."""
    out = _write(tmp_path, _report(tmp_path / "clip.mov"))
    document = json.loads(out.read_text(encoding="utf-8"))
    document["segments"]["merge_tolerance_seconds"] = 4.0
    out.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReportArtifactError, match="merge_tolerance_seconds"):
        ReportStore().load_evidence(out)


def test_load_evidence_rejects_an_unknown_timeline_event(tmp_path: Path) -> None:
    out = _write(tmp_path, _report(tmp_path / "clip.mov"))
    document = json.loads(out.read_text(encoding="utf-8"))
    document["timeline"].append({"type": "caption", "start_seconds": 1.0, "text": "hi"})
    out.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReportArtifactError, match="cannot interpret") as exc:
        ReportStore().load_evidence(out)
    assert "caption" in str(exc.value)


def test_load_evidence_keeps_evidence_when_no_segments_were_assembled(tmp_path: Path) -> None:
    """The empty-segments shape is legitimate, and stays the derivation's own answer."""
    report = _without_segments(_report(tmp_path / "clip.mov"))
    out = _write(tmp_path, report)

    loaded = ReportStore().load_evidence(out)

    assert loaded.video_duration_seconds is None
    assert loaded.segments == ()
    assert len(loaded.unattributed.ocr_events) == 2
    assert len(loaded.unattributed.transcript_events) == 2
    assert loaded.to_dict() == StudyEvidenceService().derive(report).to_dict()


@pytest.mark.parametrize("value", ["60", True])
def test_load_evidence_rejects_a_non_numeric_duration(value: object, tmp_path: Path) -> None:
    """A quoted or boolean duration is refused, not carried into the evidence."""
    out = _write(tmp_path, _without_segments(_report(tmp_path / "clip.mov")))
    document = json.loads(out.read_text(encoding="utf-8"))
    document["segments"]["video_duration_seconds"] = value
    out.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReportArtifactError, match="video_duration_seconds") as exc:
        ReportStore().load_evidence(out)
    assert "cannot interpret" in str(exc.value)


@pytest.mark.parametrize("value", ["15.2", True])
@pytest.mark.parametrize(
    ("kind", "key"),
    [
        ("ocr", "timestamp_seconds"),
        ("speech", "start_seconds"),
        ("speech", "end_seconds"),
    ],
)
def test_load_evidence_rejects_a_non_numeric_timestamp(
    kind: str, key: str, value: object, tmp_path: Path
) -> None:
    """Both ends of a speech interval are the same numeric schema, so both are checked."""
    out = _write(tmp_path, _without_segments(_report(tmp_path / "clip.mov")))
    document = json.loads(out.read_text(encoding="utf-8"))
    event = next(item for item in document["timeline"] if item["type"] == kind)
    event[key] = value
    out.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReportArtifactError, match=key) as exc:
        ReportStore().load_evidence(out)
    assert "cannot interpret" in str(exc.value)


def test_load_evidence_accepts_integer_seconds(tmp_path: Path) -> None:
    """JSON has one number type, so ``60`` is as valid as ``60.0`` for a float field."""
    out = _write(tmp_path, _without_segments(_report(tmp_path / "clip.mov")))
    document = json.loads(out.read_text(encoding="utf-8"))
    document["segments"]["video_duration_seconds"] = 60
    ocr = next(item for item in document["timeline"] if item["type"] == "ocr")
    ocr["timestamp_seconds"] = 15
    speech = next(item for item in document["timeline"] if item["type"] == "speech")
    speech["start_seconds"] = 17
    speech["end_seconds"] = 23
    out.write_text(json.dumps(document), encoding="utf-8")

    loaded = ReportStore().load_evidence(out)

    assert loaded.video_duration_seconds == 60.0
    assert loaded.unattributed.ocr_events[0].timestamp_seconds == 15.0
    assert loaded.unattributed.transcript_events[0].start_seconds == 17.0
    assert loaded.unattributed.transcript_events[0].end_seconds == 23.0
