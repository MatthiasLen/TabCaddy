## TabCaddy

[![CI](https://github.com/MatthiasLen/TabCaddy/actions/workflows/ci.yml/badge.svg)](https://github.com/MatthiasLen/TabCaddy/actions/workflows/ci.yml)

TabCaddy is a command-line tool for getting from raw tabular files to repeatable dataset workflows quickly.

Use it to:

- inspect files or folders
- preview rows without opening a notebook
- spot schema drift
- compile mixed raw files into a reusable Parquet dataset
- scaffold and run Python-based transforms
- compare dataset versions
- merge incoming files into an existing archive with conflict checks

It works with single files, whole folders, and TabCaddy compiled datasets.

### Install

TabCaddy currently requires Python 3.11 or newer.

Install with `pip`:

```bash
pip install tabcaddy
```

Install with `uv` as a standalone CLI:

```bash
uv tool install tabcaddy
```

Add it to a project environment with `uv`:

```bash
uv add tabcaddy
```

### Supported Inputs

- `.csv`
- `.feather`
- `.arrow`
- `.parquet`
- folders containing supported files
- compiled TabCaddy datasets created by `tabcaddy compile`

### Quick Start

Most users will want one of these flows.

Clean raw files, then compile:

```bash
tabcaddy summary data/
tabcaddy head data/ --n 5
tabcaddy schema data/
tabcaddy scaffold-transform data/
tabcaddy transform data/ transform_template.py cleaned_data/
tabcaddy compile cleaned_data/ --interactive
```

Clean an incoming drop, merge it into a raw archive, then compile:

```bash
tabcaddy scaffold-transform incoming/
tabcaddy transform incoming/ transform_template.py incoming_cleaned/
tabcaddy merge incoming_cleaned/ archive/ --out merged_archive --on id
tabcaddy compile merged_archive --interactive
```

`compile` before `transform` is still valid when you want to lock onto one schema first, or when you are transforming an existing compiled dataset.

Mental model:

- `summary`: what is in here?
- `head`: what do the rows look like?
- `schema`: how many shapes of data are present?
- `compile`: turn a raw folder into a reusable dataset
- `scaffold-transform`: generate a starter transform script
- `transform`: apply a transform to a file, folder, or compiled dataset
- `diff`: compare two versions of data
- `merge`: append incoming data onto an existing file or folder with validation

### Commands

`summary`

```bash
tabcaddy summary <source> [--profile quick|standard|deep]
```

Use this as the default entry point for understanding a file, folder, or compiled dataset.

Depending on profile, it can show:

- file, row, column, and schema counts
- schema overview and schema distribution
- per-column statistics such as null rate, min, max, and mean
- date ranges for temporal columns
- warnings such as schema drift
- deep-profile histograms, uniqueness estimates, and column hashes

Profiles:

- `quick`: counts only
- `standard`: metadata, schema overview, lightweight statistics, and warnings
- `deep`: adds deeper statistics such as histograms, uniqueness estimates, and hashes

Example:

```bash
tabcaddy summary data/ --profile deep
```

`head`

```bash
tabcaddy head <source> [--n N] [--showmeta]
```

Preview rows without loading the full dataset into a notebook.

- for a file, it shows the first `N` rows
- for a compiled dataset, it shows the first `N` rows across the compiled Parquet data
- for a folder, it shows the first row from each of the first `N` files

Use `--showmeta` to include metadata columns in the rendered output.

`schema`

```bash
tabcaddy schema <source>
```

Use this when you care specifically about schema groups, type changes, and files that do not match the dominant schema.

This command always runs the quick schema analysis path. It does not take `--profile`.

`compile`

```bash
tabcaddy compile <folder> [--output compiled_dataset] [--schema N] [--interactive]
```

Compile a folder of supported files into a standardized Parquet-backed TabCaddy dataset.

If multiple schemas are present:

- use `--schema N` to choose one explicitly
- use `--interactive` to preview detected schemas and choose one at the prompt

Non-selected schemas are skipped and reported after the compile completes.

`scaffold-transform`

```bash
tabcaddy scaffold-transform <source> [--output transform_template.py]
```

Generate a starter Python transform file based on the observed dataset schemas.

`transform`

```bash
tabcaddy transform <input> <transform.py> [output_path] [--workers N]
```

Apply a Python transform to a file, folder, or compiled dataset.

If `output_path` is omitted, TabCaddy creates one automatically by appending `_transformed`.

When the input is a compiled dataset, output is written as another compiled dataset with refreshed `metadata.json` and Parquet parts under `data/`.

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

`diff`

```bash
tabcaddy diff <left> <right> [--level metadata|statistics|full]
```

Compare two files, two folders, a file against a folder, or two compiled datasets.

Supported source combinations:

- file vs file
- folder vs folder
- file vs folder (either side)
- compiled dataset vs compiled dataset

Other combinations, such as file vs compiled dataset or folder vs compiled dataset, are rejected.

For file-vs-folder diffs, TabCaddy matches by filename across the folder tree:

- if no match is found: summary reports `missing`
- if one unique exact-content match can be identified: summary reports `unmodified`
- if exactly one filename match exists but content differs: summary reports `modified` and includes diff details
- if multiple candidates remain: summary reports `ambiguous` and lists candidate paths

- `metadata`: high-level file and dataset changes only
- `statistics`: metadata plus column statistics changes
- `full`: metadata, schema, and statistics changes

`merge`

```bash
tabcaddy merge <source> <target> (--out <path> | --inplace) [--on COLUMN ...] [--strategy append|upsert] [--schema-evolution strict|allow-additive] [--ignore-filetype] [--dry]
```

Merge appends source rows onto matching target files and writes the result using the target file layout.
Use `--dry` to preview matched files, unmatched files, destination paths, schema issues, casts, and expected conflicts without writing anything.

Important behavior:

- supports file-to-file, file-to-folder, and folder-to-folder merges
- folder-to-file merge is not supported
- you must provide exactly one of `--out` or `--inplace`
- compiled datasets are rejected explicitly because merge does not rebuild compiled metadata
- matching inside folders is by relative path, not just filename
- when `source` and `target` are both files, their file types must match unless `--ignore-filetype` is provided
- default `--strategy append` preserves target rows and appends only source rows that are not already present in target
- `--strategy upsert` requires `--on`; matching target keys are replaced by source rows (source wins)
- `--on` is optional for append; when provided, merge checks for conflicting duplicate keys after append
- default `--schema-evolution strict` requires source and target to have the same column layout
- `--schema-evolution allow-additive` unions columns (target order first, then source-only columns), filling missing values with nulls
- `--schema-evolution allow-additive` is not supported with `--ignore-filetype` in v1

How output paths work:

- file-to-file merge requires `--out` to point to a file
- file-to-folder merge can write a single merged file or an output directory tree, depending on `--out`
- folder-to-folder merge requires `--out` to point to a directory unless you use `--inplace`
- non-inplace folder-to-folder merge carries unmatched target files through into the output directory unchanged
- non-inplace merge will not overwrite an existing output file

How matching works:

- file-to-folder merge looks for a matching target file in the target tree
- folder-to-folder merge matches files by relative path within the source and target trees
- with `--ignore-filetype`, matching ignores the extension and uses relative path plus file stem
- if `--ignore-filetype` would match more than one target file, merge fails before writing anything

Schema and type rules:

- in strict mode, source and target must have the same columns in the same order
- merge key columns passed with `--on` must exist in both inputs
- without `--ignore-filetype`, differing column dtypes are rejected
- with `--ignore-filetype`, TabCaddy can cast CSV source columns into a binary target schema when the cast is valid
- if a CSV-to-binary cast is not possible, merge fails without writing partial output

Use `--on` one or more times to define key columns for conflict detection:

```bash
tabcaddy merge incoming/ archive/ --out merged_archive --on customer_id --on order_id
```

Strategy examples:

```bash
# Append mode (default): preserve target rows, add source rows not already present
tabcaddy merge incoming/ archive/ --out merged_archive --strategy append

# Upsert mode: replace matching target keys with source rows (source wins)
tabcaddy merge incoming/ archive/ --out merged_archive --strategy upsert --on customer_id
```

Conflict behavior in append mode when `--on` is provided:

- if two rows share the same key and differ in non-key values, merge fails instead of silently choosing one
- key-only schemas are also checked: duplicate keys are treated as conflicts
- with `--dry`, TabCaddy exits non-zero when the preview detects a blocking issue

Preview a batch merge before writing:

```bash
tabcaddy merge incoming/ archive/ --out merged_archive --on customer_id --dry
```

Safety behavior:

- folder merges are transactional: if one file fails validation or write, TabCaddy does not leave behind a partially merged directory tree
- `--inplace` writes use atomic replacement per destination after staging

### Help

See all commands:

```bash
tabcaddy --help
```

See help for a specific command:

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
