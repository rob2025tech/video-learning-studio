"""Study-evidence use case (LearnLens Task 1): report -> per-segment evidence.

A thin orchestration adapter, mirroring the project's "one service per use case"
pattern (``SegmentService``, ``ReportService``). It performs no subprocess work
and no I/O: it unpacks the evidence a :class:`ReportResult` already carries
(``source``, ``segmentation``, the unified ``timeline`` and the transcription
status/detail) and delegates the deterministic binding to the pure
:func:`~video_learning.core.study.build_study_evidence`.

It introduces no new provider abstraction (there is only one implementation) and
changes nothing about the existing report schema or behavior.
"""

from __future__ import annotations

from video_learning.core.study import StudyEvidence, build_study_evidence
from video_learning.services.report_service import ReportResult


class StudyEvidenceService:
    """Use case: derive structured learning evidence from an existing report."""

    def derive(self, report: ReportResult) -> StudyEvidence:
        """Return the :class:`StudyEvidence` for ``report`` (read-only).

        The report's own ``source`` is threaded through as provenance, so the
        derived artifact names the video it describes like every other artifact.
        """
        return build_study_evidence(
            segmentation=report.segmentation,
            timeline=report.timeline,
            transcription_status=report.transcription_status,
            transcription_detail=report.transcription_detail,
            source=report.source,
        )
