from __future__ import annotations

from pathlib import Path

from tabcaddy.application.merge.common import resolve_existing_path
from tabcaddy.application.merge.executor import MergeExecutor
from tabcaddy.application.merge.planner import MergePlanner
from tabcaddy.application.merge.validator import MergeValidator


class MergeDatasets:
    def __init__(
        self,
        planner: MergePlanner | None = None,
        validator: MergeValidator | None = None,
        executor: MergeExecutor | None = None,
    ) -> None:
        self._planner = planner or MergePlanner()
        self._validator = validator or MergeValidator()
        self._executor = executor or MergeExecutor()

    def run(
        self,
        source: Path,
        target: Path,
        out: Path | None,
        inplace: bool,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> list[Path]:
        source_path = resolve_existing_path(source, role="source")
        target_path = resolve_existing_path(target, role="target")
        output_path = out.expanduser().resolve() if out is not None else None

        if inplace == (output_path is not None):
            raise ValueError("Provide exactly one of --out or --inplace.")

        operations = self._planner.plan(
            source=source_path,
            target=target_path,
            out=output_path,
            inplace=inplace,
            ignore_filetype=ignore_filetype,
        )
        prepared_operations = self._validator.prepare_operations(
            operations=operations,
            out=output_path,
            inplace=inplace,
            on_columns=on_columns,
            ignore_filetype=ignore_filetype,
        )

        written: list[Path] = []
        for operation in prepared_operations:
            self._executor.execute(operation, on_columns=on_columns, inplace=inplace)
            written.append(operation.destination)
        return written