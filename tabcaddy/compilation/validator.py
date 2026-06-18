from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import polars as pl

from tabcaddy.shared.dataset_io import read_dataframe


_SOURCE_FILE_COLUMN = "_source_file"
_MAX_PATH_PREVIEW = 5


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status_messages: list[str] = field(default_factory=list)
    selected_file_count: int = 0
    skipped_file_count: int = 0


class ValidateCompiledDataset:
    def run(
        self,
        *,
        source_root: Path,
        selected_files: list[Path],
        skipped_files: list[str],
        compiled_parts: list[Path],
        expected_columns: set[str],
        progress_reporter: Callable[[str], None] | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        status_messages: list[str] = []

        def _status(message: str) -> None:
            status_messages.append(message)
            if progress_reporter is not None:
                progress_reporter(message)

        _status("Validation started: preparing compiled dataset checks.")

        if skipped_files:
            warnings.append(
                "Excluded files from compilation (schema mismatch): "
                f"{self._preview_paths(sorted(skipped_files))}"
            )
            _status(
                "Validation note: "
                f"{len(skipped_files)} schema-mismatched files were excluded by compile."
            )

        if not compiled_parts:
            errors.append("Compiled dataset does not contain any parquet parts.")
            _status("Validation failed: compiled dataset contains no parquet parts.")
            return ValidationResult(
                passed=False,
                errors=errors,
                warnings=warnings,
                status_messages=status_messages,
                selected_file_count=len(selected_files),
                skipped_file_count=len(skipped_files),
            )

        _status("Validation step 1/3: checking compiled schema.")
        observed_columns = self._read_compiled_columns(compiled_parts)
        missing_columns, unexpected_columns = self._validate_columns(
            expected_columns, observed_columns, errors
        )
        _status(
            "Schema summary: "
            f"expected={len(expected_columns)}, observed={len(observed_columns)}, "
            f"missing={len(missing_columns)}, unexpected={len(unexpected_columns)}."
        )

        if _SOURCE_FILE_COLUMN in observed_columns:
            _status("Validation step 2/3: checking source-file coverage.")
            coverage = self._validate_file_coverage(
                source_root,
                selected_files,
                compiled_parts,
                errors,
            )
            _status(
                "Coverage summary: "
                f"selected={coverage['selected_count']}, "
                f"observed={coverage['observed_count']}, "
                f"missing={coverage['missing_count']}, "
                f"unexpected={coverage['unexpected_count']}."
            )
        else:
            _status(
                "Validation step 2/3: skipped source-file coverage; "
                "required _source_file column is missing."
            )

        _status("Validation step 3/3: checking row counts.")
        expected_row_count, observed_row_count = self._validate_row_count(
            selected_files, compiled_parts, errors
        )
        _status(
            "Row-count summary: "
            f"expected={expected_row_count}, observed={observed_row_count}, "
            f"delta={observed_row_count - expected_row_count}."
        )

        if errors:
            _status("Validation finished: failed.")
        else:
            _status("Validation finished: passed.")

        return ValidationResult(
            passed=not errors,
            errors=errors,
            warnings=warnings,
            status_messages=status_messages,
            selected_file_count=len(selected_files),
            skipped_file_count=len(skipped_files),
        )

    def _read_compiled_columns(self, compiled_parts: list[Path]) -> set[str]:
        schema = pl.scan_parquet([str(path) for path in compiled_parts]).collect_schema()
        return set(schema.names())

    def _validate_columns(
        self,
        expected_columns: set[str],
        observed_columns: set[str],
        errors: list[str],
    ) -> tuple[list[str], list[str]]:
        missing_columns = sorted(expected_columns - observed_columns)
        unexpected_columns = sorted(observed_columns - expected_columns)

        if missing_columns:
            errors.append(
                "Compiled dataset is missing expected columns: "
                f"{', '.join(missing_columns)}"
            )
        if unexpected_columns:
            errors.append(
                "Compiled dataset contains unexpected columns: "
                f"{', '.join(unexpected_columns)}"
            )

        return missing_columns, unexpected_columns

    def _validate_file_coverage(
        self,
        source_root: Path,
        selected_files: list[Path],
        compiled_parts: list[Path],
        errors: list[str],
    ) -> dict[str, int]:
        expected_paths = {
            path.relative_to(source_root).as_posix() for path in selected_files
        }

        observed_paths_series = (
            pl.scan_parquet([str(path) for path in compiled_parts])
            .select(pl.col(_SOURCE_FILE_COLUMN).cast(pl.String))
            .unique()
            .collect()
            .to_series()
        )
        observed_paths = {str(item) for item in observed_paths_series.to_list() if item}

        missing_paths = sorted(expected_paths - observed_paths)
        unexpected_paths = sorted(observed_paths - expected_paths)

        if missing_paths:
            errors.append(
                "Compiled dataset is missing rows for selected files: "
                f"{self._preview_paths(missing_paths)}"
            )
        if unexpected_paths:
            errors.append(
                "Compiled dataset contains rows from non-selected files: "
                f"{self._preview_paths(unexpected_paths)}"
            )

        return {
            "selected_count": len(expected_paths),
            "observed_count": len(observed_paths),
            "missing_count": len(missing_paths),
            "unexpected_count": len(unexpected_paths),
        }

    def _validate_row_count(
        self,
        selected_files: list[Path],
        compiled_parts: list[Path],
        errors: list[str],
    ) -> tuple[int, int]:
        expected_row_count = sum(read_dataframe(path).height for path in selected_files)
        observed_row_count = int(
            pl.scan_parquet([str(path) for path in compiled_parts])
            .select(pl.len())
            .collect()
            .item()
        )

        if expected_row_count != observed_row_count:
            errors.append(
                "Compiled dataset row count mismatch: "
                f"expected {expected_row_count}, observed {observed_row_count}."
            )

        return expected_row_count, observed_row_count

    def _preview_paths(self, paths: list[str]) -> str:
        if len(paths) <= _MAX_PATH_PREVIEW:
            return ", ".join(paths)

        shown = ", ".join(paths[:_MAX_PATH_PREVIEW])
        return f"{shown} (+{len(paths) - _MAX_PATH_PREVIEW} more)"
