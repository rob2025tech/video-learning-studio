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
