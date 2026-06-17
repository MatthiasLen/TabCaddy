from __future__ import annotations

from tabcaddy.diff.common import analyze_pair
from tabcaddy.diff.comparison import compare_analyses
from tabcaddy.diff.folder_inventory import build_relative_file_index, diff_folder_inventory
from tabcaddy.diff.row_level import compare_rows_by_key
from tabcaddy.domain.models import (
    DatasetSource,
    DiffComparisonType,
    DiffLevel,
    DiffReport,
    RowDiffSummary,
    DiffSummary,
)


class FolderDiffer:
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
        inventory = diff_folder_inventory(left, right)
        report = DiffReport()

        for file_name in inventory.added_files:
            report.file_changes.append(f"Added file: {file_name}")
        for file_name in inventory.removed_files:
            report.file_changes.append(f"Removed file: {file_name}")
        for file_name in inventory.modified_files:
            report.file_changes.append(f"Modified file: {file_name}")

        report.summary = DiffSummary(
            comparison_type=DiffComparisonType.FOLDER,
            matching_files=inventory.matching_files,
            modified_files=len(inventory.modified_files),
            only_in_left=inventory.only_in_left,
            only_in_right=inventory.only_in_right,
        )

        if level == DiffLevel.METADATA:
            return report

        left_analysis, right_analysis = analyze_pair(
            self._generate_analysis,
            left,
            right,
            level,
        )
        content_report = compare_analyses(left_analysis, right_analysis, level)
        report.metadata_changes.extend(content_report.metadata_changes)
        report.schema_changes.extend(content_report.schema_changes)
        report.statistics_changes.extend(content_report.statistics_changes)
        report.warnings.extend(content_report.warnings)

        if key_columns:
            left_index = build_relative_file_index(left)
            right_index = build_relative_file_index(right)
            common_paths = sorted(set(left_index) & set(right_index))
            compared_files = 0
            added_rows = 0
            removed_rows = 0
            updated_rows = 0
            unchanged_rows = 0

            for relative_path in common_paths:
                result = compare_rows_by_key(
                    left_index[relative_path],
                    right_index[relative_path],
                    key_columns,
                    max_examples=row_examples,
                    source_path=relative_path,
                )
                compared_files += 1
                added_rows += result.summary.added_rows
                removed_rows += result.summary.removed_rows
                updated_rows += result.summary.updated_rows
                unchanged_rows += result.summary.unchanged_rows

                remaining_examples = row_examples - len(report.row_change_examples)
                if remaining_examples > 0:
                    report.row_change_examples.extend(
                        result.updated_examples[:remaining_examples]
                    )

                remaining_added = row_examples - len(report.row_added_key_samples)
                if remaining_added > 0:
                    report.row_added_key_samples.extend(
                        result.added_key_samples[:remaining_added]
                    )

                remaining_removed = row_examples - len(report.row_removed_key_samples)
                if remaining_removed > 0:
                    report.row_removed_key_samples.extend(
                        result.removed_key_samples[:remaining_removed]
                    )

            report.row_diff_summary = RowDiffSummary(
                key_columns=list(key_columns),
                added_rows=added_rows,
                removed_rows=removed_rows,
                updated_rows=updated_rows,
                unchanged_rows=unchanged_rows,
                compared_files=compared_files,
            )
        return report
