"""Segmentation use case (Stage 0I): evidence -> deterministic video segments.

A thin orchestration layer that composes one adapter with the pure domain logic:

    FfmpegSceneDetector -> visual-change evidence
    transcript segments -> speech-gap evidence
    merge (Phase 1) + minimum-duration promotion (Phase 2) -> final segments

It reuses the ``MediaInfo`` and ``TranscriptResult`` the caller already has, so it
never re-runs inspection, analysis, or transcription. It performs no semantic
interpretation: the result is evidence plus deterministic boundaries, not
chapters. The source video is only ever read.
"""

from __future__ import annotations

from collections.abc import Sequence

from video_learning.adapters.ffmpeg_scene import FfmpegSceneDetector
from video_learning.core.models import MediaInfo
from video_learning.core.segments import (
    Segmentation,
    assemble_segments,
    detect_speech_boundaries,
    merge_boundaries,
)
from video_learning.core.transcript import TranscriptResult, TranscriptSegment


class SegmentService:
    """Use case: assemble deterministic segments from visual + speech evidence."""

    def __init__(self, scene_detector: FfmpegSceneDetector) -> None:
        self._scene = scene_detector

    def segment(
        self, media: MediaInfo, transcription: TranscriptResult | None
    ) -> Segmentation:
        """Build a :class:`Segmentation` from evidence this video already has.

        ``transcription`` may be ``None`` (no audio, or the backend was
        unavailable); that simply yields no speech-gap evidence. Zero visual
        changes and zero speech boundaries is valid and, with a known duration,
        produces one full-duration segment.

        The final segment must end at the EXACT authoritative duration, so the
        only duration source is ``media.duration_seconds`` (ffprobe, Stage 0A); it
        is never estimated from evidence timestamps. When ffprobe reported no
        duration (``None``) that guarantee cannot be met, so no final segments are
        assembled — but every piece of evidence is retained and the ``None``
        duration is recorded, keeping the report explicit rather than implying
        segmentation completed normally.
        """
        visual_changes = self._scene.detect(media)
        speech_segments: Sequence[TranscriptSegment] = (
            transcription.segments if transcription is not None else ()
        )
        speech_boundaries = detect_speech_boundaries(speech_segments)
        boundaries = merge_boundaries(visual_changes, speech_boundaries)

        # Authoritative duration only — never estimated from evidence timestamps.
        duration = media.duration_seconds
        segments = assemble_segments(boundaries, duration) if duration is not None else []
        return Segmentation(
            visual_changes=tuple(visual_changes),
            speech_boundaries=tuple(speech_boundaries),
            boundaries=tuple(boundaries),
            segments=tuple(segments),
            video_duration_seconds=duration,
        )
