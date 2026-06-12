from __future__ import annotations

from tabcaddy.diff.common import analyze_pair
from tabcaddy.diff.comparison import compare_analyses
from tabcaddy.diff.hash_utils import content_hash
from tabcaddy.domain.models import (
    DatasetSource,
    DiffComparisonType,
    DiffLevel,
    DiffReport,
    DiffSummary,
)


class FileDiffer:
    def __init__(self, generate_analysis) -> None:
        self._generate_analysis = generate_analysis

    def diff(
        self, left: DatasetSource, right: DatasetSource, level: DiffLevel
    ) -> DiffReport:
        left_analysis, right_analysis = analyze_pair(
            self._generate_analysis,
            left,
            right,
            level,
        )
        report = compare_analyses(left_analysis, right_analysis, level)
        report.summary = DiffSummary(
            comparison_type=DiffComparisonType.FILE,
            content_status=(
                "identical"
                if content_hash(left.path) == content_hash(right.path)
                else "modified"
            ),
        )
        return report