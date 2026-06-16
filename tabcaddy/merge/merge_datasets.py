from __future__ import annotations

from pathlib import Path

from tabcaddy.merge.common import SchemaEvolution, resolve_merge_path
from tabcaddy.merge.executor import MergeExecutor
from tabcaddy.merge.planner import MergePlanner
from tabcaddy.merge.validator import MergeValidator


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
        schema_evolution: str = "strict",
    ) -> list[Path]:
        source_path = resolve_merge_path(source, role="source")
        target_path = resolve_merge_path(target, role="target")
        output_path = out.expanduser().resolve() if out is not None else None

        if inplace == (output_path is not None):
            raise ValueError("Provide exactly one of --out or --inplace.")

        resolved_schema_evolution = self._resolve_schema_evolution(schema_evolution)
        if resolved_schema_evolution == "allow-additive" and ignore_filetype:
            raise ValueError(
                "--schema-evolution allow-additive is not supported with --ignore-filetype."
            )

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
            schema_evolution=resolved_schema_evolution,
        )

        transaction_root: Path | None = None
        if output_path is not None and len(prepared_operations) > 1:
            transaction_root = output_path
        elif inplace and target_path.is_dir():
            transaction_root = target_path

        return self._executor.execute_all(
            prepared_operations,
            on_columns=on_columns,
            transaction_root=transaction_root,
        )

    def preview(
        self,
        source: Path,
        target: Path,
        out: Path | None,
        inplace: bool,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
        schema_evolution: str = "strict",
    ) -> tuple[list[str], bool]:
        source_path = resolve_merge_path(source, role="source")
        target_path = resolve_merge_path(target, role="target")
        output_path = out.expanduser().resolve() if out is not None else None

        if inplace == (output_path is not None):
            raise ValueError("Provide exactly one of --out or --inplace.")

        resolved_schema_evolution = self._resolve_schema_evolution(schema_evolution)
        if resolved_schema_evolution == "allow-additive" and ignore_filetype:
            raise ValueError(
                "--schema-evolution allow-additive is not supported with --ignore-filetype."
            )

        operations = self._planner.plan(
            source=source_path,
            target=target_path,
            out=output_path,
            inplace=inplace,
            ignore_filetype=ignore_filetype,
        )
        return self._validator.preview_operations(
            operations=operations,
            out=output_path,
            inplace=inplace,
            on_columns=on_columns,
            ignore_filetype=ignore_filetype,
            schema_evolution=resolved_schema_evolution,
        )

    def _resolve_schema_evolution(self, schema_evolution: str) -> SchemaEvolution:
        lowered = schema_evolution.lower()
        if lowered in {"strict", "allow-additive"}:
            return lowered
        raise ValueError(f"Unsupported schema evolution mode: {schema_evolution}")
