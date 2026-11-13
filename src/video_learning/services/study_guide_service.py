"""Study-guide use case (LearnLens Task 2): video -> per-segment study guide.

A thin orchestration adapter, mirroring the project's "one service per use case"
pattern (``ReportService``, ``StudyEvidenceService``). It performs no analysis of
its own and no I/O: it reuses the existing pipeline end to end and delegates the
deterministic projection to the pure
:func:`~video_learning.core.study_guide.build_study_guide`.

    ReportService.build      -> ReportResult    (Stages 0A/0D/0E/0F/0I, once)
    StudyEvidenceService.derive -> StudyEvidence (Task 1 binding)
    build_study_guide        -> StudyGuide       (Task 2 projection)

:meth:`build` calls the report service exactly once, so a video is never analyzed
twice for one guide, and it accepts the same ``model`` override
``ReportService.build`` does. :meth:`from_report` is the same projection for a
report the caller already holds.

Constructing :class:`~video_learning.services.report_service.ReportService` (and
its adapter dependencies) stays the caller's job, as it is for every other service
here. Building the guide in-process from the live pipeline — rather than reloading
a persisted artifact — is deliberate: no ``from_dict`` support was added to any
existing model for this task, and the existing report schema is untouched.

Out of scope, and not introduced here: CLI commands, persistence, quiz generation,
and UI integration. The source video is only ever read.
"""

from __future__ import annotations

from pathlib import Path

from video_learning.core.study_guide import StudyGuide, build_study_guide
from video_learning.services.report_service import ReportResult, ReportService
from video_learning.services.study_service import StudyEvidenceService


class StudyGuideService:
    """Use case: project the existing evidence layer into a study guide."""

    def __init__(
        self,
        report_service: ReportService,
        study_service: StudyEvidenceService | None = None,
    ) -> None:
        self._report = report_service
        self._study = study_service or StudyEvidenceService()

    def build(self, path: Path, *, model: Path | None = None) -> StudyGuide:
        """Return the study guide for the video at ``path`` (read-only)."""
        return self.from_report(self._report.build(path, model=model))

    def from_report(self, report: ReportResult) -> StudyGuide:
        """Return the study guide for a report that was already built."""
        return build_study_guide(self._study.derive(report))
