from __future__ import annotations

from tabcaddy.diff.common import analyze_pair
from tabcaddy.diff.comparison import compare_analyses
from tabcaddy.diff.hash_utils import content_hash
from tabcaddy.diff.row_level import compare_rows_by_key
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
        self,
        left: DatasetSource,
        right: DatasetSource,
        level: DiffLevel,
        *,
        key_columns: tuple[str, ...] = (),
        row_examples: int = 20,
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
        if key_columns:
            row_result = compare_rows_by_key(
                left.path,
                right.path,
                key_columns,
                max_examples=row_examples,
            )
            report.row_diff_summary = row_result.summary
            report.row_change_examples = row_result.updated_examples
            report.row_added_key_samples = row_result.added_key_samples
            report.row_removed_key_samples = row_result.removed_key_samples
        return report
