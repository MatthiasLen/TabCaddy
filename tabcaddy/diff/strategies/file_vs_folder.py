from __future__ import annotations

from tabcaddy.diff.common import analyze_pair
from tabcaddy.diff.comparison import compare_analyses
from tabcaddy.diff.matching import MatchStatus, resolve_file_folder_match
from tabcaddy.diff.row_level import compare_rows_by_key
from tabcaddy.domain.models import (
    DatasetSource,
    DiffComparisonType,
    DiffLevel,
    DiffReport,
    DiffSummary,
    SourceType,
)


class MixedDiffer:
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
        file_source, folder_source = self._split_sources(left, right)
        match = resolve_file_folder_match(file_source, folder_source)

        if match.status is MatchStatus.MISSING:
            report = DiffReport()
            report.summary = DiffSummary(
                comparison_type=DiffComparisonType.FILE_FOLDER,
                match_status="missing",
            )
            return report

        if match.status is MatchStatus.AMBIGUOUS:
            report = DiffReport()
            report.summary = DiffSummary(
                comparison_type=DiffComparisonType.FILE_FOLDER,
                match_status="ambiguous",
                candidate_paths=match.candidate_paths,
            )
            return report

        if match.status is MatchStatus.UNMODIFIED:
            report = DiffReport()
            report.summary = DiffSummary(
                comparison_type=DiffComparisonType.FILE_FOLDER,
                match_status="unmodified",
                matched_path=match.matched_path,
            )
            return report

        if match.matched_file is None:
            raise ValueError("Modified file-folder match requires a matched file")

        matched_source = DatasetSource(
            path=match.matched_file, source_type=SourceType.FILE
        )
        # Preserve original left/right order from CLI arguments to ensure output semantics
        # match the user's command (e.g., 'diff <folder> <file>' shows folder vs file in that order)
        if left.source_type == SourceType.FILE:
            left_analysis, right_analysis = analyze_pair(
                self._generate_analysis,
                left,
                matched_source,
                level,
            )
        else:
            left_analysis, right_analysis = analyze_pair(
                self._generate_analysis,
                matched_source,
                right,
                level,
            )
        report = compare_analyses(left_analysis, right_analysis, level)
        report.summary = DiffSummary(
            comparison_type=DiffComparisonType.FILE_FOLDER,
            match_status=MatchStatus.MODIFIED.value,
            matched_path=match.matched_path,
        )
        if key_columns:
            if left.source_type == SourceType.FILE:
                row_left_path = left.path
                row_right_path = matched_source.path
            else:
                row_left_path = matched_source.path
                row_right_path = right.path
            row_result = compare_rows_by_key(
                row_left_path,
                row_right_path,
                key_columns,
                max_examples=row_examples,
                source_path=match.matched_path,
            )
            report.row_diff_summary = row_result.summary
            report.row_change_examples = row_result.updated_examples
            report.row_added_key_samples = row_result.added_key_samples
            report.row_removed_key_samples = row_result.removed_key_samples
        return report

    def _split_sources(
        self, left: DatasetSource, right: DatasetSource
    ) -> tuple[DatasetSource, DatasetSource]:
        if (
            left.source_type == SourceType.FILE
            and right.source_type == SourceType.FOLDER
        ):
            return left, right
        if (
            left.source_type == SourceType.FOLDER
            and right.source_type == SourceType.FILE
        ):
            return right, left
        raise ValueError("MixedDiffer requires one file source and one folder source.")
