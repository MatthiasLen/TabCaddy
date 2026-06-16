from __future__ import annotations

from pathlib import Path

from tabcaddy.merge.common import MergeStrategy, SchemaEvolution, resolve_merge_path
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
        strategy: str,
        ignore_filetype: bool,
        schema_evolution: str,
    ) -> list[Path]:
        (
            source_path,
            target_path,
            output_path,
            merge_strategy,
            schema_policy,
        ) = self._resolve_merge_inputs(
            source=source,
            target=target,
            out=out,
            inplace=inplace,
            on_columns=on_columns,
            strategy=strategy,
            ignore_filetype=ignore_filetype,
            schema_evolution=schema_evolution,
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
            strategy=merge_strategy,
            ignore_filetype=ignore_filetype,
            schema_evolution=schema_policy,
        )

        transaction_root: Path | None = None
        if output_path is not None and len(prepared_operations) > 1:
            transaction_root = output_path
        elif inplace and target_path.is_dir():
            transaction_root = target_path

        return self._executor.execute_all(
            prepared_operations,
            on_columns=on_columns,
            strategy=merge_strategy,
            transaction_root=transaction_root,
        )

    def preview(
        self,
        source: Path,
        target: Path,
        out: Path | None,
        inplace: bool,
        on_columns: tuple[str, ...],
        strategy: str,
        ignore_filetype: bool,
        schema_evolution: str,
    ) -> tuple[list[str], bool]:
        (
            source_path,
            target_path,
            output_path,
            merge_strategy,
            schema_policy,
        ) = self._resolve_merge_inputs(
            source=source,
            target=target,
            out=out,
            inplace=inplace,
            on_columns=on_columns,
            strategy=strategy,
            ignore_filetype=ignore_filetype,
            schema_evolution=schema_evolution,
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
            strategy=merge_strategy,
            ignore_filetype=ignore_filetype,
            schema_evolution=schema_policy,
        )

    def _resolve_strategy(self, strategy: str) -> MergeStrategy:
        lowered = strategy.lower()
        if lowered in {"append", "upsert"}:
            return lowered
        raise ValueError(f"Unsupported merge strategy: {strategy}")

    def _resolve_schema_evolution(self, schema_evolution: str) -> SchemaEvolution:
        lowered = schema_evolution.lower()
        if lowered in {"strict", "allow-additive"}:
            return lowered
        raise ValueError(f"Unsupported schema evolution mode: {schema_evolution}")

    def _resolve_merge_inputs(
        self,
        source: Path,
        target: Path,
        out: Path | None,
        inplace: bool,
        on_columns: tuple[str, ...],
        strategy: str,
        ignore_filetype: bool,
        schema_evolution: str,
    ) -> tuple[Path, Path, Path | None, MergeStrategy, SchemaEvolution]:
        source_path = resolve_merge_path(source, role="source")
        target_path = resolve_merge_path(target, role="target")
        output_path = out.expanduser().resolve() if out is not None else None

        if inplace == (output_path is not None):
            raise ValueError("Provide exactly one of --out or --inplace.")

        merge_strategy = self._resolve_strategy(strategy)
        if merge_strategy == "upsert" and not on_columns:
            raise ValueError("--strategy upsert requires at least one --on column.")

        schema_policy = self._resolve_schema_evolution(schema_evolution)
        if schema_policy == "allow-additive" and ignore_filetype:
            raise ValueError(
                "--schema-evolution allow-additive is not supported with "
                "--ignore-filetype in v1. Remove one of the flags."
            )

        return (
            source_path,
            target_path,
            output_path,
            merge_strategy,
            schema_policy,
        )
