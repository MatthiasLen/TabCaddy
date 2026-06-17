## TabCaddy

[![CI](https://github.com/MatthiasLen/TabCaddy/actions/workflows/ci.yml/badge.svg)](https://github.com/MatthiasLen/TabCaddy/actions/workflows/ci.yml)

TabCaddy is a dataset-centric CLI for tabular data engineering workflows. It helps you move from raw files to reproducible dataset operations without leaving the terminal.

Use it to:

- profile files and folders quickly
- inspect sample rows before modeling
- detect schema drift and dominant schema groups
- compile heterogeneous raw data into a reusable Parquet dataset
- scaffold and run Python transforms
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

Note: compiling before transforming is still useful when you want to lock onto one schema first, or when the transform input is already a compiled dataset.

### Command Reference

`summary`

```bash
tabcaddy summary <source> [--profile quick|standard|deep]
```

Best default entry point for understanding a source.

- `quick`: counts only
- `standard`: metadata, schema overview, lightweight statistics, and warnings
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

Focused schema analysis for schema groups, type changes, and non-dominant files. This command always uses quick schema analysis and does not take `--profile`.

`compile`

```bash
tabcaddy compile <folder> [--output compiled_dataset] [--schema N] [--interactive]
```

Compiles a folder into a standardized Parquet-backed dataset.

- use `--schema N` to choose a schema directly
- use `--interactive` to inspect detected schemas and select one at the prompt
- files from non-selected schemas are skipped and reported

`scaffold-transform`

```bash
tabcaddy scaffold-transform <source> [--output transform_template.py]
```

Generates a Python transform scaffold based on observed schemas.

`transform`

```bash
tabcaddy transform <input> <transform.py> [output_path] [--workers N]
```

Applies a Python transform to a file, folder, or compiled dataset.

- if `output_path` is omitted, TabCaddy creates one by appending `_transformed`
- compiled input produces compiled output with refreshed `metadata.json` and `data/`

Supported signatures:

```python
def transform(df):
    return df
```

```python
def transform(df, context):
    return df
```

`context` fields:

- `file_name`
- `file_path`
- `schema`
- `metadata.row_count`
- `metadata.schema_hash`

`diff`

```bash
tabcaddy diff <left> <right> [--level metadata|statistics|full]
```

Supported comparisons:

- file vs file
- folder vs folder
- file vs folder (either side)
- compiled dataset vs compiled dataset

Unsupported combinations (for example file vs compiled dataset) are rejected.

For file-vs-folder comparisons, matching is filename-based across the folder tree:

- no match: `missing`
- one unique exact-content match: `unmodified`
- one filename match with content change: `modified`
- multiple candidates: `ambiguous`

Levels:

- `metadata`: high-level file and dataset changes
- `statistics`: metadata plus column-stat changes
- `full`: metadata, schema, and statistics changes

`merge`

```bash
tabcaddy merge <source> <target> (--out <path> | --inplace) [--on COLUMN ...] [--strategy append|upsert] [--schema-evolution strict|allow-additive] [--ignore-filetype] [--dry]
```

Merges source rows into matching target files and preserves the target layout.

Use `--dry` to preview matched and unmatched files, output destinations, schema issues, casts, and expected conflicts without writing output.

Core rules:

- supports file-to-file, file-to-folder, and folder-to-folder merges
- folder-to-file merge is not supported
- provide exactly one of `--out` or `--inplace`
- compiled datasets are rejected (merge does not rebuild compiled metadata)
- folder matching is by relative path

Strategy and keys:

- default `append`: keeps target rows and appends source rows not already present
- `upsert`: requires `--on` and replaces matching target keys with source rows
- `--on` is optional in append mode, but enables conflict-aware duplicate-key validation

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

- file-to-file merge requires `--out` to point to a file
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

Show command-specific help:

```bash
tabcaddy summary --help
tabcaddy schema --help
tabcaddy scaffold-transform --help
tabcaddy head --help
tabcaddy compile --help
tabcaddy transform --help
tabcaddy diff --help
tabcaddy merge --help
```
