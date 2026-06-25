## TabCaddy

[![CI](https://github.com/MatthiasLen/TabCaddy/actions/workflows/ci.yml/badge.svg)](https://github.com/MatthiasLen/TabCaddy/actions/workflows/ci.yml)

TabCaddy is a dataset-centric CLI for tabular data engineering workflows. It helps you move from raw files to reproducible dataset operations in the terminal.

Use it to:

- inspect data
- detect schema drift and dominant schema groups
- compile heterogeneous raw data into a reusable Parquet dataset
- scaffold and run Python transforms with Polars
- diff dataset versions at metadata, statistics, or full levels
- merge incoming drops into an archive with conflict-aware validation

TabCaddy works with single files, directory trees, and compiled TabCaddy datasets.

### Installation

Requirements:

- Python 3.11+

Install with pip:

```bash
pip install tabcaddy
```

Install as a standalone CLI with uv:

```bash
uv tool install tabcaddy
```

Add to a project environment with uv:

```bash
uv add tabcaddy
```

### Supported Sources

- `.csv`
- `.feather`
- `.arrow`
- `.parquet`
- folders containing supported files
- compiled datasets created by `tabcaddy compile`

### Command Map

- `summary`: profile counts, schemas, stats, and warnings
- `head`: preview rows from files, folders, or compiled datasets
- `schema`: inspect schema groups and drift-focused schema diagnostics
- `plot`: plot one column against another as a line or scatter chart
- `compile`: materialize a selected schema into a compiled Parquet dataset
- `scaffold-transform`: generate a transform starter from observed schemas
- `transform`: apply Python transforms to file, folder, or compiled inputs
- `diff`: compare files/folders/compiled datasets
- `merge`: combine source data into a target with validation and conflict rules

### Quick Start

Typical curation flow (inspect, clean, compile):

```bash
tabcaddy summary data/
tabcaddy head data/ --n 5
tabcaddy schema data/
tabcaddy scaffold-transform data/
tabcaddy transform data/ transform_template.py cleaned_data/
tabcaddy compile cleaned_data/ --interactive
```

Typical incremental ingest flow (clean, merge, compile):

```bash
tabcaddy scaffold-transform incoming/
tabcaddy transform incoming/ transform_template.py incoming_cleaned/
# optional: preview merge plan without writing output
tabcaddy merge incoming_cleaned/ archive/ --out merged_archive --on id --dry
tabcaddy merge incoming_cleaned/ archive/ --out merged_archive --on id
tabcaddy compile merged_archive/ --interactive
```

Compile before transforming when you want to lock onto a single schema first, or when the transform input is already compiled.

### Transform Workflow Example

If you are using `scaffold-transform` and `transform` for the first time, use this loop:

1. generate a starter script from the source you want to clean
2. replace the scaffold examples with your Polars logic
3. run the transform over the file, folder, or compiled dataset
4. inspect or compile the transformed output

Generate a scaffold from the raw folder:

```bash
tabcaddy scaffold-transform source_data/ --output transform_source_data.py
```

The scaffold includes observed schema comments and ready-to-edit examples. A typical edited transform looks like this:

```python
import polars as pl

def transform(df: pl.DataFrame, context=None) -> pl.DataFrame:
    # In this example, the transformation fills missing `status` values, casts
    # `amount` to a numeric type, and adds the source filename as a new column.

    if "status" in df.columns:
        df = df.with_columns(pl.col("status").fill_null("unknown"))

    if "amount" in df.columns:
        df = df.with_columns(pl.col("amount").cast(pl.Float64))

    if context is not None:
        df = df.with_columns(pl.lit(context.file_name).alias("SOURCE_FILE"))

    return df
```

All imports in transform scripts must be declared at module top-level. The same rule applies to local helper modules imported by the transform. Imports inside `transform(...)`, helper functions, or other nested scopes are rejected.

Then apply it and inspect the result:

```bash
tabcaddy transform source_data/ transform_source_data.py transformed_data/ --workers 4
tabcaddy summary transformed_data/
tabcaddy head transformed_data/ --n 5
```

If you omit `transformed_data/`, TabCaddy creates a sibling output path with `_transformed` appended.

### Command Reference

`summary`

```bash
tabcaddy summary <source> [--profile quick|standard|deep]
```

Best default entry point for understanding a source.

- `quick`: counts only
- `standard`: metadata, schema overview, lightweight statistics, warnings
- `deep`: adds histograms, uniqueness estimates, and column hashes

Example:

```bash
tabcaddy summary data/ --profile deep
```

`head`

```bash
tabcaddy head <source> [--n N] [--showmeta]
```

Previews rows without loading the full dataset into a notebook.

- file input: first `N` rows
- compiled dataset input: first `N` rows from compiled Parquet data
- folder input: first row from each of the first `N` files

Use `--showmeta` to include metadata columns in output.

`schema`

```bash
tabcaddy schema <source>
```

Focused schema analysis for groups, type changes, and non-dominant files. It always runs quick schema analysis.

`plot`

```bash
tabcaddy plot <source> <column> [<column> ...] [--kind auto|line|scatter|histogram] [--aggregate-x mean|median|min|max|sum|count] [--interpolation linear|nearest] [--fail-on-x-duplicates] [--fail-on-x-unsorted] [--n N] [--filter "COLUMN OP VALUE"]
```

Plots columns in the terminal as line, scatter, or histogram charts.

- line/scatter mode column layout: first column is `x`, remaining columns are one or more `y` series
- histogram mode column layout: one or more target columns; each target renders a separate histogram section
- `x` in line/scatter: numeric (`Int`, `Float`, `Decimal`) or temporal (`Date`, `Datetime`, `Time`, `Duration`); categorical/string x is accepted for scatter only
- `y` in line/scatter: one or more y-columns; each must be numeric, boolean (`true=1`, `false=0`), or castable to `Float64`; strings and nested types are not supported
- `--kind auto` picks `line` for temporal `x` only when x-values are unique; if temporal duplicates exist it picks `scatter`; for numeric `x`, it picks `line` only when values are monotonic and unique; otherwise it picks `scatter`
- `--kind histogram` is explicit (not auto-selected) and uses the same numeric histogram policy as deep summary profiling
- `--filter` takes a single expression argument, for example `--filter "event_date>=2026-01-01"`; `OP` must be one of `==`, `!=`, `>`, `>=`, `<`, `<=`; column names may include spaces and digits (for example `part description==A`, `2026_value>=2`), and column names containing operator characters must be wrapped in brackets (for example `[temp>=setpoint]==1`)
- for temporal columns, use ISO-8601 literals: `Date` uses `YYYY-MM-DD`; `Datetime` uses `YYYY-MM-DDTHH:MM:SS` (timezone accepted when present)
- `--interpolation` controls line rendering interpolation; defaults to `linear` and also supports `nearest`
- line plots fail on duplicate `x` by default unless `--aggregate-x` is provided
- line plots auto-sort `x` by default; use `--fail-on-x-unsorted` for strict mode
- `--aggregate-x`, `--interpolation`, `--fail-on-x-duplicates`, and `--fail-on-x-unsorted` are line/scatter-only options
- for folder input with line/scatter, omitted `--n` plots the first file (`N=1`); when provided, `--n` limits plotting to the first `N` files
- for folder input with histogram, omitted `--n` aggregates across the full folder history; when provided, `--n` switches to per-file stacked histogram sections for the first `N` files
- multiple columns render as stacked sections
- rows with null values or non-numeric `y` values are dropped and reported as warnings

`compile`

```bash
tabcaddy compile <folder> [--output compiled_dataset] [--schema N] [--interactive] [--validate]
```

Compiles a folder into a standardized Parquet-backed dataset.

- use `--schema N` to choose a schema directly
- use `--interactive` to inspect detected schemas and select one at the prompt
- files from non-selected schemas are skipped and reported
- use `--validate` to verify the compiled output against the selected source files
- compile output includes a coverage summary, for example `compiled X of Y supported files`
- unreadable/corrupt files are not compiled; they are counted in coverage and listed in warnings

`--validate` checks selected-schema columns, `_source_file` coverage, and total row counts. If some source files are corrupt or unreadable, compile still succeeds when possible and the coverage summary makes the partial result explicit.

`scaffold-transform`

```bash
tabcaddy scaffold-transform <source> [--output transform_template.py]
```

Generates a Python transform scaffold based on observed schemas.

- output is a ready-to-edit Python file that uses Polars
- the scaffold includes comments for each observed schema group and example transforms
- default loop: scaffold once, edit the script, then run `tabcaddy transform`

`transform`

```bash
tabcaddy transform <input> <transform.py> [output_path] [--workers N]
```

Applies a Python transform to a file, folder, or compiled dataset.

- if `output_path` is omitted, TabCaddy creates one by appending `_transformed`
- compiled input produces compiled output with refreshed `metadata.json` and `data/`
- folder and compiled inputs can use `--workers N` for parallel execution
- for single-file input, `output_path` may be a file path such as `cleaned.csv`

Supported signatures:

```python
def transform(df):
    return df
def transform(df, context):
    return df
```

Import policy:

- import installed packages and sibling modules at module top-level only
- imports inside `transform`, local helper functions, conditionals, loops, or other nested scopes are not supported

`context` fields:

- `file_name`
- `file_path`
- `schema` (list of `{name, dtype}` entries)
- `metadata.row_count`
- `metadata.schema_hash`

`diff`

```bash
tabcaddy diff <left> <right> [--level metadata|statistics|full] [--on COLUMN ...] [--row-examples N]
```

Supported comparisons: file vs file, folder vs folder, file vs folder (either side), and compiled dataset vs compiled dataset.

Unsupported combinations (for example file vs compiled dataset) are rejected.

For file-vs-folder comparisons, matching is filename-based across the folder tree:

- no match: `missing`
- one unique exact-content match: `unmodified`
- one filename match with content change: `modified`
- multiple candidates: `ambiguous`

Levels:

- `metadata`: high-level file and dataset changes
- `statistics`: metadata plus column-stat changes
- `full`: metadata, schema, statistics, and optional key-aware row-level explainability

Key-aware row-level explainability at `full` level:

- provide one or more `--on` columns to compare records by business key
- output includes row counts by class: added, removed, updated, unchanged
- output can include updated-row examples with field-level before/after deltas
- `--row-examples` limits displayed examples while preserving aggregate counts
- key columns must exist on both sides and be unique per side for row-level comparison

Example: `tabcaddy diff customer_left.csv customer_right.csv --level full --on customer_id --row-examples 25`

`merge`

```bash
tabcaddy merge <source> <target> (--out <path> | --inplace) [--on COLUMN ...] [--strategy append|upsert] [--schema-evolution strict|allow-additive] [--ignore-filetype] [--dry]
```

Merges source rows into matching target files and preserves the target layout.

Use `--dry` to preview matches, output destinations, schema issues, casts, and expected conflicts without writing output.

Core rules:

- supports file-to-file, file-to-folder, and folder-to-folder merges
- folder-to-file merge is not supported
- provide exactly one of `--out` or `--inplace`
- compiled datasets are rejected (merge does not rebuild compiled metadata)
- folder matching is by relative path

Strategy and keys:

- default `append`: keeps target rows and appends source rows not already present
- `upsert`: requires `--on` and replaces matching target keys with source rows
- `--on` is optional in append mode and enables conflict-aware duplicate-key validation

Schema behavior:

- default `strict`: identical column layout required
- `allow-additive`: union columns (target order first, then source-only), fill missing values with nulls
- `allow-additive` is not supported with `--ignore-filetype` in v1

File type behavior:

- when both source and target are files, file types must match unless `--ignore-filetype` is set
- with `--ignore-filetype`, matching ignores extension and uses relative path plus stem
- ambiguous ignore-filetype matches fail fast before any write
- dtype mismatches are rejected unless a valid CSV-to-binary cast is possible under ignore-filetype mode

Output and safety:

- file-to-file merge supports `--out <file>` or `--inplace`
- folder-to-folder merge requires `--out` directory or `--inplace`
- non-inplace folder merge carries unmatched target files into output unchanged
- non-inplace merge does not overwrite existing output files
- folder merges are transactional; inplace writes use atomic replacement per destination

Examples:

```bash
# Preview a merge plan
tabcaddy merge incoming/ archive/ --out merged_archive/ --on customer_id --dry

# Append mode (default)
tabcaddy merge incoming/ archive/ --out merged_archive/ --strategy append

# Upsert mode
tabcaddy merge incoming/ archive/ --out merged_archive/ --strategy upsert --on customer_id
```

### Help

Show all commands:

```bash
tabcaddy --help
```

Show command-specific help with `tabcaddy <command> --help`, for example:

```bash
tabcaddy plot --help
tabcaddy merge --help
```
