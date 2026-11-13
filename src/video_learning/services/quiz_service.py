"""Quiz use case (LearnLens Task 3, Phase A): video -> evidence-grounded quiz.

A thin orchestration adapter, mirroring the project's "one service per use case"
pattern (``ReportService``, ``StudyEvidenceService``, ``StudyGuideService``). It
performs no analysis of its own and no I/O: it reuses the existing pipeline end to
end and delegates the derivation to the pure
:func:`~video_learning.core.quiz.build_study_quiz`.

    ReportService.build         -> ReportResult  (Stages 0A/0D/0E/0F/0I, once)
    StudyEvidenceService.derive -> StudyEvidence (Task 1 binding)
    build_study_quiz            -> StudyQuiz     (Task 3 Phase A derivation)

:meth:`build` calls the report service exactly once, so a video is never analyzed
twice for one quiz, and it accepts the same ``model`` override
``ReportService.build`` does. :meth:`from_report` is the same derivation for a report
the caller already holds, and :meth:`from_evidence` — a static helper, since it needs
no dependency — is the entry point when Task 1 evidence was already derived, which is
what a consumer holding a guide would use.

Deriving from :class:`~video_learning.core.study.StudyEvidence` rather than from
:class:`~video_learning.core.study_guide.StudyGuide` is deliberate: the guide is a
by-reference projection of the same segment records and adds no attribution the quiz
needs, so reading the evidence directly avoids depending on a presentation layer.

Constructing :class:`~video_learning.services.report_service.ReportService` (and its
adapter dependencies) stays the caller's job, as it is for every other service here.
Building the quiz in-process from the live pipeline — rather than reloading a
persisted artifact — is deliberate: no ``from_dict`` support was added to any existing
model for this task, and the existing report schema is untouched.

Out of scope, and not introduced here: grading, answer normalization, feedback, CLI
commands, persistence, and UI integration. The source video is only ever read.
"""

from __future__ import annotations

from pathlib import Path

from video_learning.core.quiz import StudyQuiz, build_study_quiz
from video_learning.core.study import StudyEvidence
from video_learning.services.report_service import ReportResult, ReportService
from video_learning.services.study_service import StudyEvidenceService


class StudyQuizService:
    """Use case: derive a quiz from the existing evidence layer."""

    def __init__(
        self,
        report_service: ReportService,
        study_service: StudyEvidenceService | None = None,
    ) -> None:
        self._report = report_service
        self._study = study_service or StudyEvidenceService()

    @staticmethod
    def from_evidence(evidence: StudyEvidence) -> StudyQuiz:
        """Return the quiz for Task 1 evidence that was already derived."""
        return build_study_quiz(evidence)

    def build(self, path: Path, *, model: Path | None = None) -> StudyQuiz:
        """Return the quiz for the video at ``path`` (read-only)."""
        return self.from_report(self._report.build(path, model=model))

    def from_report(self, report: ReportResult) -> StudyQuiz:
        """Return the quiz for a report that was already built."""
        return build_study_quiz(self._study.derive(report))
