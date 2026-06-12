from __future__ import annotations

from pathlib import Path

from tabcaddy.analysis.sources import SUPPORTED_FILE_SUFFIXES
from tabcaddy.merge.common import (
    PlannedOperation,
    build_directory_index,
    iter_supported_files,
    match_indexed_file,
    supports_merge_pair,
)


class MergePlanner:
    def plan(
        self,
        source: Path,
        target: Path,
        out: Path | None,
        inplace: bool,
        ignore_filetype: bool,
    ) -> list[PlannedOperation]:
        if source.is_file() and target.is_file():
            if not supports_merge_pair(source, target, ignore_filetype):
                raise ValueError(
                    "Source and target file types must match unless --ignore-filetype is provided."
                )
            if out is None or out.is_dir():
                raise ValueError(
                    "File-to-file merge requires --out to point to a file."
                )
            return [
                PlannedOperation(
                    source=source,
                    target=target,
                    destination=out,
                    output_directory=False,
                    kind="merge",
                )
            ]

        if source.is_dir() and target.is_file():
            raise ValueError("Folder-to-file merge is not supported.")

        if source.is_file() and target.is_dir():
            output_directory = self._is_output_directory_path(out)
            target_index = build_directory_index(target, ignore_filetype)
            matched = match_indexed_file(source, target_index, ignore_filetype)
            return [
                PlannedOperation(
                    source=source,
                    target=matched,
                    destination=self._resolve_destination(
                        source=source,
                        source_root=source.parent,
                        target_root=target,
                        matched_target=matched,
                        out=out,
                        inplace=inplace,
                        output_directory=output_directory,
                    ),
                    output_directory=output_directory,
                    kind="merge" if matched is not None else "source_only",
                )
            ]

        self._validate_folder_output(out)

        target_index = build_directory_index(
            target,
            ignore_filetype,
            relative_to=target,
        )
        operations: list[PlannedOperation] = []
        matched_targets: set[Path] = set()
        for file_path in iter_supported_files(source):
            matched = match_indexed_file(
                file_path,
                target_index,
                ignore_filetype,
                relative_to=source,
            )
            if matched is not None:
                matched_targets.add(matched)
            operations.append(
                PlannedOperation(
                    source=file_path,
                    target=matched,
                    destination=self._resolve_destination(
                        source=file_path,
                        source_root=source,
                        target_root=target,
                        matched_target=matched,
                        out=out,
                        inplace=inplace,
                        output_directory=True,
                    ),
                    output_directory=True,
                    kind="merge" if matched is not None else "source_only",
                )
            )

        if not inplace:
            if out is None:
                raise ValueError("Provide --out unless --inplace is selected.")
            for target_file in sorted(set(target_index.values()) - matched_targets):
                operations.append(
                    PlannedOperation(
                        source=target_file,
                        target=None,
                        destination=out / target_file.relative_to(target),
                        output_directory=True,
                        kind="target_passthrough",
                    )
                )
        return operations

    def _validate_folder_output(self, out: Path | None) -> None:
        if out is not None and out.exists() and not out.is_dir():
            raise ValueError(
                "Folder-to-folder merge requires --out to point to a directory."
            )
        if (
            out is not None
            and not out.exists()
            and out.suffix.lower() in SUPPORTED_FILE_SUFFIXES
        ):
            raise ValueError(
                "Folder-to-folder merge requires --out to point to a directory."
            )

    def _resolve_destination(
        self,
        source: Path,
        source_root: Path,
        target_root: Path,
        matched_target: Path | None,
        out: Path | None,
        inplace: bool,
        output_directory: bool,
    ) -> Path:
        if matched_target is not None:
            if inplace:
                return matched_target
            if out is None:
                raise ValueError("Provide --out unless --inplace is selected.")
            if not output_directory:
                return out
            return out / matched_target.relative_to(target_root)

        relative_path = source.relative_to(source_root)
        if inplace:
            return target_root / relative_path
        if out is None:
            raise ValueError("Provide --out unless --inplace is selected.")
        if output_directory:
            return out / relative_path
        if relative_path.parent == Path("."):
            return out
        raise ValueError(
            "Folder-to-folder merge requires --out to point to a directory."
        )

    def _is_output_directory_path(self, out: Path | None) -> bool:
        if out is None:
            return False
        if out.exists():
            return out.is_dir()
        return out.suffix.lower() not in SUPPORTED_FILE_SUFFIXES
