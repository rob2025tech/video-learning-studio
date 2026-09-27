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
