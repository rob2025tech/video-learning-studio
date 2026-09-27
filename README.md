# video-learning-studio

Local-first CLI for turning videos stored on your Mac into a personal learning system.

## Stage 0

Implements `inspect` (media metadata via `ffprobe`) and `analyze` (content-based
keyword suggestions via on-screen text — proposal-only; it never renames or
modifies files).

```
video-learning inspect "/path/to/video.mov"
video-learning inspect "/path/to/video.mov" --json

video-learning analyze "/path/to/video.mov"
video-learning analyze "/path/to/video.mov" --json
```

`analyze` samples a few frames with `ffmpeg`, reads their on-screen text with
`tesseract`, and ranks the words it actually observed into suggested keywords.
It suggests descriptive content only — it never generates a date/time prefix,
never infers a device name, and never rewrites your existing filename. Those are
yours to keep.

No new Python dependencies are introduced: `ffmpeg` and `tesseract` are used as
external system binaries via subprocess, the same pattern as `ffprobe`.

## Stage 0F — OCR snapshot persistence

`save-ocr` persists the OCR evidence to a local, auditable JSON artifact so a
human can later trace any suggested phrase or keyword back through
`video -> snapshot timestamp -> original OCR text`.

```
video-learning save-ocr "/path/to/video.mov" --out "/path/to/evidence.json"
video-learning save-ocr "/path/to/video.mov" --out "/path/to/evidence.json" --json
```

It runs the same read-only analysis as `analyze` (`ffmpeg` + `tesseract`) and
writes the **complete** OCR text of every timestamped snapshot alongside the
deterministic phrases/keywords already derived from that same text. The result is
a single schema-tagged JSON document (`video-learning.ocr-snapshots/v1`) that is:

- **deterministic** — stable ordering with no wall-clock timestamps or random
  IDs, so the same video yields byte-identical output;
- **non-destructive** — the source video is never renamed, moved, or modified;
- **complete** — the full OCR text is preserved, never replaced by suggestions.

No database, cloud storage, or new Python dependency is introduced; the artifact
is an ordinary local file. `--out` is required, and its parent directories are
created if missing. The complete OCR text and its derived suggestions come from
the same OCR pass, so nothing is re-extracted.

## Stage 0G — Metadata / Report

`report` combines everything Stage 0 knows about one video into a single
deterministic, local, auditable report: media metadata (Stage 0A), timestamped
OCR evidence (Stage 0D/0F), timestamped audio transcription (Stage 0E), and a
**unified chronological timeline** that interleaves them.

```
video-learning report "/path/to/video.mov"
video-learning report "/path/to/video.mov" --out "/path/to/report.json"
video-learning report "/path/to/video.mov" --json
video-learning report "/path/to/video.mov" --model "/path/to/ggml-base.en.bin"
```

It is a thin orchestration layer that **reuses** `InspectService`,
`AnalyzeService`, and `TranscribeService` (plus their existing result models) —
it is not a second implementation of OCR or transcription, and it introduces no
database, index, or provider framework.

The report contains:

- **Metadata** — duration, dimensions, frame rate, video/audio codecs, audio
  presence, sample rate, and channels, taken from the existing inspection model.
  Volatile filesystem values (mtime, size) are omitted so output stays
  deterministic.
- **OCR evidence** — every snapshot's timestamp and **complete** OCR text, plus
  the phrases/keywords already derived from that same text (reused verbatim, no
  second representation).
- **Transcription evidence** — backend, model, language, and segments with
  `start_seconds`/`end_seconds`/`text` (reused verbatim).
- **Unified timeline** — OCR and speech as **distinct** event types: OCR is a
  point (`{"type": "ocr", "timestamp_seconds": …, "text": …}`) and speech is an
  interval (`{"type": "speech", "start_seconds": …, "end_seconds": …, "text": …}`).
  Events are sorted chronologically with deterministic tie-breaking (an OCR point
  sorts before a speech interval at the same time; equal keys keep input order).
  No semantic alignment between OCR and speech is attempted.

With `--out PATH` the report is written as a schema-tagged JSON artifact
(`video-learning.report/v1`); `--json` prints that same report
(`result.to_dict()`, with no CLI-only fields). The terminal report is concise:
OCR timeline entries show a short preview while the artifact retains the complete
OCR text.

Transcription follows the existing Stage 0E capability/error semantics and is
never faked:

- `no-audio` — the video has no audio stream, so transcription is skipped and the
  metadata + OCR report is still produced;
- `unavailable` — audio is present but the local whisper.cpp backend/model is
  missing or unusable; the report records an explicit, deterministic reason (the
  Stage 0E blocking message) and no provenance is invented;
- `ok` — transcription ran; backend/model/language/segments are included.

Models are never downloaded and there is no cloud fallback. The report is
deterministic (stable ordering, no wall-clock timestamps, no random IDs), so the
same video with the same available backends/models yields a byte-identical
artifact, and it is non-destructive: the source video is only ever read, and the
only file written is the caller-chosen `--out` artifact.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- FFmpeg/ffprobe installed and on PATH (`brew install ffmpeg`)
- tesseract installed and on PATH (`brew install tesseract`) — required for `analyze`

## Development

```
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```
