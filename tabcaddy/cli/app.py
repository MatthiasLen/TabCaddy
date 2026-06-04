from __future__ import annotations

from pathlib import Path

import typer

from tabcaddy.application.compile_dataset import CompileDataset
from tabcaddy.application.diff_datasets import DiffDatasets
from tabcaddy.application.generate_analysis import GenerateAnalysis
from tabcaddy.application.scaffold_transform import ScaffoldTransform
from tabcaddy.application.transform_dataset import TransformDataset
from tabcaddy.domain.models import DiffLevel
from tabcaddy.domain.models import ProfileMode
from tabcaddy.infrastructure.analysis_builder import AnalysisBuilder
from tabcaddy.infrastructure.source_resolver import resolve_source
from tabcaddy.rendering.console import create_console
from tabcaddy.rendering.console import resolve_render_profile
from tabcaddy.rendering.views.diff import build_diff_view
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


@app.command()
def summary(
    source: Path,
    profile: ProfileMode = typer.Option(ProfileMode.STANDARD, "--profile"),
) -> None:
    source = Path(source).expanduser().resolve()
    console = create_console()
    render = resolve_render_profile(console)
    analysis = GenerateAnalysis().run(resolve_source(source), profile)
    console.print(build_summary_view(analysis, render=render))


@app.command()
def schema(
    source: Path,
    profile: ProfileMode = typer.Option(ProfileMode.STANDARD, "--profile"),
) -> None:
    source = Path(source).expanduser().resolve()
    console = create_console()
    render = resolve_render_profile(console)
    scan = AnalysisBuilder().build(resolve_source(source), profile)
    console.print(build_schema_view(scan.analysis, scan.files, render=render))


@app.command(name="compile")
def compile_dataset(
    folder: Path,
    output: Path = typer.Option(Path("compiled_dataset"), "--output"),
    schema_index: int | None = typer.Option(None, "--schema"),
    interactive: bool = typer.Option(False, "--interactive"),
) -> None:
    console = create_console()
    source = resolve_source(folder)
    selected_schema = schema_index
    if interactive and selected_schema is None:
        preview = AnalysisBuilder().build(source, ProfileMode.QUICK)
        if len(preview.analysis.schemas) > 1:
            selected_schema = typer.prompt(
                "Multiple schemas detected. Choose schema number", type=int
            )
    output_path, skipped = CompileDataset().run(source, output, selected_schema)
    console.print(f"Compiled dataset written to [green]{output_path}[/green]")
    if skipped:
        console.print(
            f"Skipped {len(skipped)} files from non-selected schemas.", style="yellow"
        )


@app.command()
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


@app.command("scaffold-transform")
def scaffold_transform(
    source: Path,
    output: Path = typer.Option(Path("transform_template.py"), "--output"),
) -> None:
    console = create_console()
    destination = ScaffoldTransform().run(resolve_source(source), output)
    console.print(f"Transform scaffold written to [green]{destination}[/green]")


@app.command()
def diff(
    left: Path,
    right: Path,
    level: DiffLevel = typer.Option(DiffLevel.FULL, "--level"),
) -> None:
    console = create_console()
    render = resolve_render_profile(console)
    generator = GenerateAnalysis()

    left = Path(left).expanduser().resolve()
    right = Path(right).expanduser().resolve()

    report = DiffDatasets(generator).run(
        resolve_source(left), resolve_source(right), level
    )
    console.print(build_diff_view(report, render=render))


def main() -> None:
    app()
