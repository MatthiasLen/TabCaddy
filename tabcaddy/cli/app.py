from __future__ import annotations

from pathlib import Path

import typer

from tabcaddy.application.compile_dataset import CompileDataset
from tabcaddy.application.diff_datasets import DiffDatasets
from tabcaddy.application.generate_analysis import GenerateAnalysis
from tabcaddy.application.head_dataset import HeadDataset
from tabcaddy.application.merge import MergeDatasets
from tabcaddy.application.scaffold_transform import ScaffoldTransform
from tabcaddy.application.transform_dataset import TransformDataset
from tabcaddy.domain.models import DiffLevel
from tabcaddy.domain.models import ProfileMode
from tabcaddy.infrastructure.analysis_builder import AnalysisBuilder
from tabcaddy.infrastructure.source_resolver import resolve_source
from tabcaddy.rendering.console import create_console
from tabcaddy.rendering.console import resolve_render_profile
from tabcaddy.rendering.views.diff import build_diff_view
from tabcaddy.rendering.views.head import build_file_head_view, build_folder_head_view
from tabcaddy.rendering.views.schema import build_schema_view
from tabcaddy.rendering.views.summary import build_summary_view


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
    profile: ProfileMode = typer.Option(ProfileMode.STANDARD, "--profile"),
) -> None:
    source = Path(source).expanduser().resolve()
    console = create_console()
    render = resolve_render_profile(console)

    result = GenerateAnalysis().run(resolve_source(source), profile)
    console.print(build_summary_view(result.analysis, render=render))


@app.command(help="Display the schema of a dataset")
def schema(
    source: Path,
) -> None:
    source = Path(source).expanduser().resolve()
    console = create_console()
    render = resolve_render_profile(console)

    result = GenerateAnalysis().run(resolve_source(source), ProfileMode.QUICK)
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
) -> None:
    console = create_console()
    source = resolve_source(folder)
    selected_schema = schema_index

    if interactive and selected_schema is None:
        preview = AnalysisBuilder().build(source, ProfileMode.QUICK)
        if len(preview.analysis.schemas) > 1:
            console.print(
                f"Multiple schemas detected ({len(preview.analysis.schemas)}): "
            )
            for index, sch in enumerate(preview.analysis.schemas, start=1):
                console.print(
                    f"  [cyan]Schema {index}[/cyan]: {len(sch.columns)} columns, observed in {sch.occurrence_count} files"
                )
            selected_schema = typer.prompt("Choose schema number", type=int)

    output_path, skipped = CompileDataset().run(source, output, selected_schema)

    console.print(f"Compiled dataset written to [green]{output_path}[/green]")
    if skipped:
        console.print(
            f"Skipped {len(skipped)} files from non-selected schemas.", style="yellow"
        )


@app.command(help="Transform a dataset using a specified transform script")
def transform(
    input_path: Path,
    transform_path: Path,
    output_path: Path | None = typer.Argument(None),
    workers: int = typer.Option(1, "--workers", min=1),
) -> None:
    console = create_console()
    source = resolve_source(input_path)
    destination = TransformDataset().run(source, transform_path, output_path, workers)
    console.print(f"Transformed files written to [green]{destination}[/green]")


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
) -> None:
    console = create_console()
    render = resolve_render_profile(console)
    generator = GenerateAnalysis()

    left = Path(left).expanduser().resolve()
    right = Path(right).expanduser().resolve()

    report = DiffDatasets(generator).run(
        resolve_source(left), resolve_source(right), level
    )
    console.print(build_diff_view(report, level=level, render=render))


@app.command(help="Preview the first rows of a file or folder")
def head(
    source: Path,
    n: int = typer.Option(
        10, "--n", "-n", help="Number of rows (file) or files (folder) to show"
    ),
    show_meta: bool = typer.Option(False, "--showmeta", help="Show metadata columns"),
) -> None:
    source = Path(source).expanduser().resolve()
    console = create_console()
    render = resolve_render_profile(console)
    result = HeadDataset().run(resolve_source(source), n)
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


@app.command(help="Merge files or folders with schema validation and conflict checks")
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
    ignore_filetype: bool = typer.Option(
        False,
        "--ignore-filetype",
        help="Allow folder matching across CSV, Parquet, Feather, and Arrow extensions.",
    ),
) -> None:
    console = create_console()

    try:
        written = MergeDatasets().run(
            source=source,
            target=target,
            out=out,
            inplace=inplace,
            on_columns=tuple(on or ()),
            ignore_filetype=ignore_filetype,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if len(written) == 1:
        console.print(f"Merged dataset written to [green]{written[0]}[/green]")
        return

    console.print(
        f"Merged {len(written)} files into [green]{written[0].parent}[/green]"
    )


def main() -> None:
    app()
