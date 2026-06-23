from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
import typer
from rich.text import Text

from tabcaddy.analysis import GenerateAnalysis, resolve_source
from tabcaddy.compilation import CompileDataset
from tabcaddy.diff import DiffDatasets
from tabcaddy.domain.models import DiffLevel
from tabcaddy.domain.models import ProfileMode
from tabcaddy.merge import MergeDatasets
from tabcaddy.plot import PlotDataset
from tabcaddy.preview import HeadDataset
from tabcaddy.rendering.console import create_console
from tabcaddy.rendering.console import resolve_render_profile
from tabcaddy.rendering.views.diff import build_diff_view
from tabcaddy.rendering.views.head import build_file_head_view, build_folder_head_view
from tabcaddy.rendering.views.plot import build_plot_view
from tabcaddy.rendering.views.schema import build_schema_view
from tabcaddy.rendering.views.summary import build_summary_view
from tabcaddy.transforms import ScaffoldTransform, TransformDataset


app = typer.Typer(
    add_completion=False,
    help="Explore, compile, transform, and compare datasets.",
    no_args_is_help=True,
)


@app.callback()
def root() -> None:
    """TabCaddy command line interface."""


@app.command(
    help="Display a summary of the dataset, including file counts and schema overview"
)
def summary(
    source: Path,
    profile: Literal["quick", "standard", "deep"] = typer.Option(
        "standard", "--profile"
    ),
) -> None:
    source = Path(source).expanduser().resolve()
    console = create_console()
    render = resolve_render_profile(console)

    try:
        result = GenerateAnalysis().run(resolve_source(source), ProfileMode(profile))
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    console.print(build_summary_view(result.analysis, render=render))


@app.command(help="Display the schema of a dataset")
def schema(
    source: Path,
) -> None:
    source = Path(source).expanduser().resolve()
    console = create_console()
    render = resolve_render_profile(console)

    try:
        result = GenerateAnalysis().run(resolve_source(source), ProfileMode.QUICK)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    console.print(build_schema_view(result.analysis, result.files, render=render))


@app.command(
    name="compile", help="Compile a dataset into a standardized Parquet format"
)
def compile_dataset(
    folder: Path,
    output: Path = typer.Option(
        Path("compiled_dataset"),
        "--output",
        help="Output path for the compiled dataset",
    ),
    schema_index: int | None = typer.Option(
        None, "--schema", help="Index of schema to compile when multiple are detected"
    ),
    interactive: bool = typer.Option(False, "--interactive"),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Validate compiled output against selected source files.",
    ),
) -> None:
    console = create_console()
    render = resolve_render_profile(console)
    compiler = CompileDataset()
    try:
        source = resolve_source(folder)
        selected_schema = schema_index
        selection_preview = None

        if interactive and selected_schema is None:
            selection_preview = compiler.preview_selection(source)
            if len(selection_preview.analysis.schemas) > 1:
                console.print(
                    f"Multiple schemas detected ({len(selection_preview.analysis.schemas)}): "
                )
                for index, sch in enumerate(
                    selection_preview.analysis.schemas, start=1
                ):
                    console.print(
                        f"  [cyan]Schema {index}[/cyan]: {len(sch.columns)} columns, observed in {sch.occurrence_count} files"
                    )
                selected_schema = typer.prompt("Choose schema number", type=int)

        output_path, skipped, warnings, validation_result, coverage = compiler.run(
            source,
            output,
            selected_schema,
            precomputed_selection=selection_preview,
            validate=validate,
            validation_progress=(
                lambda message: console.print(
                    message,
                    style="cyan",
                    markup=False,
                )
                if validate
                else None
            ),
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if coverage.selected_files < coverage.total_supported_files:
        console.print(
            "[yellow]Compilation coverage:[/yellow] "
            f"compiled {coverage.selected_files} of {coverage.total_supported_files} supported files."
        )
        if coverage.unreadable_files > 0:
            console.print(
                "[yellow]Not compiled due to read/inspect errors:[/yellow] "
                f"{coverage.unreadable_files} files. See warnings below."
            )
        if coverage.skipped_schema_files > 0:
            console.print(
                "[yellow]Not compiled due to schema selection:[/yellow] "
                f"{coverage.skipped_schema_files} files."
            )
    else:
        console.print(
            "[green]Compilation coverage:[/green] "
            f"all {coverage.selected_files} supported files were compiled."
        )

    if skipped:
        console.print(
            f"Skipped {len(skipped)} files from non-selected schemas.", style="yellow"
        )
    if warnings:
        warning_text = Text(
            "\n".join(f"- {warning}" for warning in warnings), style="yellow"
        )
        console.print(
            render.panel(
                warning_text,
                title="Warnings",
                border_style="yellow",
            )
        )
    console.print(f"Compiled dataset written to [green]{output_path}[/green]")

    if validation_result is not None:
        for warning in validation_result.warnings:
            console.print(warning, style="yellow", markup=False)

        if validation_result.passed:
            console.print(
                "[green]Validation passed.[/green] "
                f"Verified {validation_result.selected_file_count} selected files."
            )
            return

        for error in validation_result.errors:
            console.print(error, style="red", markup=False)
        raise typer.Exit(code=1)


@app.command(help="Transform a dataset using a specified transform script")
def transform(
    input_path: Path,
    transform_path: Path,
    output_path: Path | None = typer.Argument(None),
    workers: int = typer.Option(1, "--workers", min=1),
) -> None:
    console = create_console()
    render = resolve_render_profile(console)
    try:
        source = resolve_source(input_path)
        destination, warnings = TransformDataset().run(
            source, transform_path, output_path, workers
        )
    except Exception as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    console.print(f"Transformed files written to [green]{destination}[/green]")
    if warnings:
        warning_text = Text(
            "\n".join(f"- {warning}" for warning in warnings), style="yellow"
        )
        console.print(
            render.panel(
                warning_text,
                title="Warnings",
                border_style="yellow",
            )
        )


@app.command(
    "scaffold-transform", help="Generate a Python transform scaffold for a dataset"
)
def scaffold_transform(
    source: Path,
    output: Path = typer.Option(Path("transform_template.py"), "--output"),
) -> None:
    console = create_console()
    try:
        destination = ScaffoldTransform().run(resolve_source(source), output)
    except FileExistsError as error:
        raise typer.BadParameter(str(error), param_hint="--output") from error
    console.print(f"Transform scaffold written to [green]{destination}[/green]")


@app.command(help="Compare two datasets and report differences")
def diff(
    left: Path,
    right: Path,
    level: DiffLevel = typer.Option(
        DiffLevel.FULL,
        "--level",
        help="Comparison depth: metadata (file changes only), statistics (+ column stats), full (+ schema details)",
    ),
    on: list[str] | None = typer.Option(
        None,
        "--on",
        help="One or more key columns used for row-level explainable diff output.",
    ),
    row_examples: int = typer.Option(
        20,
        "--row-examples",
        min=1,
        help="Maximum number of row-level examples to show per section.",
    ),
) -> None:
    console = create_console()
    render = resolve_render_profile(console)
    generator = GenerateAnalysis()
    try:
        left = Path(left).expanduser().resolve()
        right = Path(right).expanduser().resolve()
        if level != DiffLevel.FULL and on:
            raise ValueError("Row-level key diff requires --level full.")
        report = DiffDatasets(generator).run(
            resolve_source(left),
            resolve_source(right),
            level,
            key_columns=tuple(on or ()),
            row_examples=row_examples,
        )
    except pl.exceptions.PolarsError as error:
        console.print(f"Failed to read input data for diff: {error}")
        raise typer.Exit(code=1) from error
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(str(error))
        raise typer.Exit(code=1) from error

    console.print(build_diff_view(report, level=level, render=render))


@app.command(help="Preview the first rows of a file or folder")
def head(
    source: Path,
    n: int = typer.Option(
        10, "--n", "-n", help="Number of rows (file) or files (folder) to show"
    ),
    show_meta: bool = typer.Option(False, "--showmeta", help="Show metadata columns"),
) -> None:
    console = create_console()
    render = resolve_render_profile(console)
    try:
        source = Path(source).expanduser().resolve()
        result = HeadDataset().run(resolve_source(source), n)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if result.is_folder:
        console.print(
            build_folder_head_view(
                [(r.path, r.df) for r in result.frames],
                render=render,
                show_meta=show_meta,
            )
        )
    else:
        frame = result.frames[0]
        console.print(
            build_file_head_view(
                frame.df, frame.path, render=render, show_meta=show_meta
            )
        )


@app.command(help="Plot one column against another using line or scatter charts")
def plot(
    source: Path,
    column_x: str,
    column_y: str,
    kind: Literal["auto", "line", "scatter"] = typer.Option(
        "auto",
        "--kind",
        help=(
            "Chart kind. auto chooses line for temporal x and for numeric x "
            "only when values are monotonic and unique."
        ),
    ),
    aggregate_x: Literal["mean", "median", "min", "max", "sum", "count"]
    | None = typer.Option(
        None,
        "--aggregate-x",
        help="Optional y-aggregation applied per x value before plotting.",
    ),
    fail_on_x_duplicates: bool = typer.Option(
        False,
        "--fail-on-x-duplicates",
        help="Fail when duplicate x-values are present.",
    ),
    fail_on_unsorted_x: bool = typer.Option(
        False,
        "--fail-on-unsorted-x",
        help="Fail instead of auto-sorting x-values for line plots.",
    ),
    folder_max_files: int = typer.Option(
        5,
        "--folder-max-files",
        help=(
            "Maximum number of files to plot from a folder input. "
            "Use a larger value to include more files."
        ),
    ),
) -> None:
    console = create_console()
    render = resolve_render_profile(console)

    try:
        result = PlotDataset().run(
            resolve_source(source),
            column_x,
            column_y,
            kind=kind,
            aggregate_x=aggregate_x,
            fail_on_x_duplicates=fail_on_x_duplicates,
            fail_on_unsorted_x=fail_on_unsorted_x,
            folder_max_files=folder_max_files,
        )
    except pl.exceptions.PolarsError as error:
        console.print(f"Failed to read input data for plot: {error}")
        raise typer.Exit(code=1) from error
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    console.print(build_plot_view(result, render=render))


@app.command(
    help=(
        "Merge files or folders with schema validation and conflict checks. "
        "Use --strategy append (default) or --strategy upsert with --on keys. "
        "Compiled datasets are not supported."
    )
)
def merge(
    source: Path,
    target: Path,
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output file or folder. Required unless --inplace is provided.",
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Modify the resolved target path in place using an atomic replace.",
    ),
    on: list[str] | None = typer.Option(
        None,
        "--on",
        help="One or more key columns used for conflict-aware key merges.",
    ),
    strategy: Literal["append", "upsert"] = typer.Option(
        "append",
        "--strategy",
        help=(
            "Merge strategy. append keeps target rows and adds source rows not already "
            "present. upsert requires --on and replaces matching target keys with source rows."
        ),
    ),
    schema_evolution: Literal["strict", "allow-additive"] = typer.Option(
        "strict",
        "--schema-evolution",
        help=(
            "Schema policy. strict requires identical column layouts. "
            "allow-additive keeps target columns and appends source-only columns."
        ),
    ),
    ignore_filetype: bool = typer.Option(
        False,
        "--ignore-filetype",
        help="Allow folder matching across CSV, Parquet, Feather, and Arrow extensions.",
    ),
    dry: bool = typer.Option(
        False,
        "--dry",
        help="Preview the merge plan without writing any files.",
    ),
) -> None:
    console = create_console()
    merge_datasets = MergeDatasets()
    written: list[Path] = []

    try:
        if dry:
            lines, has_issues = merge_datasets.preview(
                source,
                target,
                out,
                inplace,
                tuple(on or ()),
                strategy,
                ignore_filetype,
                schema_evolution,
            )
        else:
            written = merge_datasets.run(
                source,
                target,
                out,
                inplace,
                tuple(on or ()),
                strategy,
                ignore_filetype,
                schema_evolution,
            )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if dry:
        console.print("Dry-run merge plan")
        for line in lines:
            console.print(line)
        if has_issues:
            raise typer.Exit(code=1)
        return

    if len(written) == 1:
        console.print(f"Merged dataset written to [green]{written[0]}[/green]")
        return

    output_root = (
        out.expanduser().resolve() if out is not None else target.expanduser().resolve()
    )
    console.print(f"Merged {len(written)} files into [green]{output_root}[/green]")


def main() -> None:
    app()
