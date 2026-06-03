from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import file_digest

from tabcaddy.domain.models import DatasetSource, DiffLevel, DiffReport, ProfileMode
from tabcaddy.infrastructure.diff_support import compare_analyses
from tabcaddy.infrastructure.source_resolver import iter_dataset_files


def _content_hash(path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


class FolderDiffer:
    def __init__(self, generate_analysis) -> None:
        self._generate_analysis = generate_analysis

    def diff(
        self, left: DatasetSource, right: DatasetSource, level: DiffLevel
    ) -> DiffReport:
        left_files = {
            str(path.relative_to(left.path)): path for path in iter_dataset_files(left)
        }
        right_files = {
            str(path.relative_to(right.path)): path
            for path in iter_dataset_files(right)
        }

        report = DiffReport(
            metadata_changes=[],
            schema_changes=[],
            statistics_changes=[],
        )

        # Always report file additions/removals
        for file_name in sorted(right_files.keys() - left_files.keys()):
            report.metadata_changes.append(f"Added file: {file_name}")
        for file_name in sorted(left_files.keys() - right_files.keys()):
            report.metadata_changes.append(f"Removed file: {file_name}")

        # For matching files, detect modifications (fast path via file stats + hash)
        modified_files = self._find_modified_files(
            left_files, right_files, sorted(left_files.keys() & right_files.keys())
        )
        for file_name in modified_files:
            report.metadata_changes.append(f"Modified file: {file_name}")

        # Only generate analysis if needed for content comparison
        if level != DiffLevel.METADATA:
            profile_mode = (
                ProfileMode.DEEP if level != DiffLevel.METADATA else ProfileMode.STANDARD
            )
            left_analysis = self._generate_analysis.run(left, profile_mode)
            right_analysis = self._generate_analysis.run(right, profile_mode)
            content_report = compare_analyses(left_analysis, right_analysis, level)
            report.metadata_changes.extend(content_report.metadata_changes)
            report.schema_changes = content_report.schema_changes
            report.statistics_changes = content_report.statistics_changes
            report.warnings = content_report.warnings

        return report

    def _find_modified_files(
        self, left_files: dict, right_files: dict, common_files: list
    ) -> list[str]:
        """Find modified files using parallel hashing for efficiency."""
        modified = []
        if not common_files:
            return modified

        # Collect files that differ in size or mtime as candidates for hashing
        hash_needed = []
        for file_name in common_files:
            left_path = left_files[file_name]
            right_path = right_files[file_name]
            left_stat = left_path.stat()
            right_stat = right_path.stat()

            # Quick checks: size or modification time differ
            if left_stat.st_size != right_stat.st_size:
                modified.append(file_name)
            elif left_stat.st_mtime != right_stat.st_mtime:
                # Modification times differ—queue for hash verification
                hash_needed.append((file_name, left_path, right_path))

        # Parallelize hash computation for files requiring content verification
        if hash_needed:
            modified.extend(self._check_file_hashes(hash_needed))

        return modified

    def _check_file_hashes(self, file_tuples: list) -> list[str]:
        """Compute hashes in parallel and return modified file names."""
        modified = []
        hash_cache = {}

        # Pre-compute hashes for all files in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all hash jobs
            future_to_path = {
                executor.submit(_content_hash, path): path
                for _, left_path, right_path in file_tuples
                for path in (left_path, right_path)
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                hash_cache[str(path)] = future.result()

        # Compare cached hashes
        for file_name, left_path, right_path in file_tuples:
            if (
                hash_cache.get(str(left_path))
                != hash_cache.get(str(right_path))
            ):
                modified.append(file_name)

        return modified
