from __future__ import annotations

from tabcaddy.domain.models import (
    DatasetSource,
    DiffComparisonType,
    DiffLevel,
    DiffReport,
    DiffSummary,
    ProfileMode,
)
from tabcaddy.infrastructure.diff_support import compare_analyses
from tabcaddy.infrastructure.mixed_differ import _content_hash


class FileDiffer:
    def __init__(self, generate_analysis) -> None:
        self._generate_analysis = generate_analysis

    def diff(
        self, left: DatasetSource, right: DatasetSource, level: DiffLevel
    ) -> DiffReport:
        profile_mode = (
            ProfileMode.DEEP if level != DiffLevel.METADATA else ProfileMode.STANDARD
        )
        left_analysis = self._generate_analysis.run(left, profile_mode).analysis
        right_analysis = self._generate_analysis.run(right, profile_mode).analysis
        report = compare_analyses(left_analysis, right_analysis, level)
        report.summary = DiffSummary(
            comparison_type=DiffComparisonType.FILE,
            content_status=(
                "identical"
                if _content_hash(left.path) == _content_hash(right.path)
                else "modified"
            ),
        )
        return report
