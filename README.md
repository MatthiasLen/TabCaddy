## TabCaddy

[![CI](https://github.com/MatthiasLen/TabCaddy/actions/workflows/ci.yml/badge.svg)](https://github.com/MatthiasLen/TabCaddy/actions/workflows/ci.yml)

TabCaddy is a command-line tool for exploring, compiling, transforming, and comparing tabular datasets.

It works with single files, whole folders, and previously compiled datasets, so you can move from quick inspection to repeatable data workflows without switching tools.

### Install

TabCaddy currently requires Python 3.13 or newer.

Install with `pip`:

```bash
pip install tabcaddy
```

Install with `uv` as a standalone CLI:

```bash
uv tool install tabcaddy
```

If you prefer to add it to a project environment, you can also use:

```bash
uv add tabcaddy
```

### Supported Inputs

- `.csv`
- `.feather`
- `.arrow`
- folders containing supported files
- compiled TabCaddy datasets created by `tabcaddy compile`

### Quick Start

Inspect a single dataset:

```bash
tabcaddy summary trades.feather
```

Inspect a folder of files:

```bash
tabcaddy summary data/
```

Look at schema drift across a folder:

```bash
tabcaddy schema data/
```

Compile a folder into a reusable parquet dataset:

```bash
tabcaddy compile data/
```

Compare two datasets:

```bash
tabcaddy diff old_data/ new_data/
```

### What `summary` Shows

`tabcaddy summary` is the main entry point for understanding a dataset. Depending on the selected profile, it can show:

- file, row, column, and schema counts
- schema overview and schema distribution
- per-column statistics such as null rate, min, max, and mean
- date ranges for temporal columns
- warnings such as schema drift
- deep-profile histograms, uniqueness estimates, and column hashes

### Profiles

Use `--profile` with `summary` or `schema` to control how much work TabCaddy does.

- `quick`: counts only, for fast inspection
- `standard`: metadata, schema overview, lightweight statistics, and warnings
- `deep`: adds uniqueness estimates, histograms, and column hashes

Example:

```bash
tabcaddy summary data/ --profile deep
```

### Commands

`summary`

```bash
tabcaddy summary <source> [--profile quick|standard|deep]
```

Use this for everyday inspection of a file, folder, or compiled dataset.

`schema`

```bash
tabcaddy schema <source> [--profile quick|standard|deep]
```

Use this when you care specifically about schema groups, type changes, and files that violate the dominant schema.

`compile`

```bash
tabcaddy compile <folder> [--output compiled_dataset] [--schema N] [--interactive]
```

Compile a folder of compatible files into a parquet-backed TabCaddy dataset. If multiple schemas are present, you can either pick one explicitly with `--schema` or let TabCaddy ask you in `--interactive` mode.

`transform`

```bash
tabcaddy transform <input> <transform.py> [output_folder] [--workers N]
```

Apply a Python transform to a single file or a folder of files. If you omit the output folder, TabCaddy creates one automatically by appending `_transformed`.

When the input is a compiled dataset, output is written as a compiled dataset as well (`data/*.parquet` plus refreshed `metadata.json`).

Supported transform signatures:

```python
def transform(df):
    return df
```

```python
def transform(df, context):
    return df
```

The optional `context` includes:

- `file_name`
- `file_path`
- `schema`
- `metadata.row_count`
- `metadata.schema_hash`

`scaffold-transform`

```bash
tabcaddy scaffold-transform <source> [--output transform_template.py]
```

Generate a starter transform file with observed schemas and example snippets.

`diff`

```bash
tabcaddy diff <left> <right> [--level metadata|statistics|full]
```

Compare two files, two folders, or two compiled datasets.

- `metadata`: high-level changes only
- `statistics`: metadata plus statistics changes
- `full`: metadata, schema, and statistics changes

`merge`

```bash
tabcaddy merge <source> <target> (--out <path> | --inplace) [--on COLUMN ...] [--ignore-filetype]
```

Merge appends source rows onto the matching target file and writes the result using the target file layout. Matching is by relative path inside folders; with `--ignore-filetype`, matching ignores the extension and uses relative path plus file stem instead.

Rows are merged vertically, not column-by-column: both files must have the same columns in the same order, and the merged output keeps that schema. Exact duplicate rows are removed. Rows with the same key are not updated or combined.

`--on` can be repeated to define a composite key such as `--on customer_id --on order_id`. Those columns are only used to detect conflicts after the row merge: if two rows share the same key but differ in any other column, the merge fails. If the non-key values are identical, the duplicate collapses to one row through normal exact-row deduplication.

Use `--ignore-filetype` to merge matching CSV and binary files by path when their column layout is compatible; when merging CSV into a binary target, TabCaddy casts the CSV columns to the target types and fails if that cast is not possible.

### Typical Workflows

Inspect a folder before deciding whether to compile it:

```bash
tabcaddy summary raw_exports/
tabcaddy schema raw_exports/
```

Compile the dominant schema into a reusable dataset:

```bash
tabcaddy compile raw_exports/ --interactive
```

Generate a transform scaffold, edit it, then apply it:

```bash
tabcaddy scaffold-transform raw_exports/
tabcaddy transform raw_exports/ transform_template.py cleaned_exports/
```

Compare two compiled snapshots:

```bash
tabcaddy diff compiled_dataset_2025_12 compiled_dataset_2026_01 --level full
```

Merge an incoming folder into an archive without modifying the original target:

```bash
tabcaddy merge incoming/ archive/ --out merged_archive --on id
```

Use a composite key by repeating `--on`:

```bash
tabcaddy merge incoming/ archive/ --out merged_archive --on customer_id --on order_id
```

### Help

See all commands:

```bash
tabcaddy --help
```

See help for a specific command:

```bash
tabcaddy summary --help
tabcaddy compile --help
tabcaddy transform --help
```
