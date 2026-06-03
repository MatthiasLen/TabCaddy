# TabCaddy v1 — End-to-End Implementation Blueprint

## Status

**Authoritative implementation plan**

This document is intended to be consumed directly by an implementation agent and serves as the architectural specification for TabCaddy v1.

All implementation decisions should follow this document unless a clear defect is discovered during implementation.

---

# 1. Product Definition

## Goal

TabCaddy is a command-line toolkit for:

* Exploring datasets
* Understanding schemas
* Detecting schema drift
* Transforming collections of files
* Compiling datasets
* Comparing datasets

Supported inputs:

* CSV files
* Feather files
* Folders containing CSV/Feather files
* Previously compiled TabCaddy datasets

The tool should scale from:

```text id="x26nnx"
single file
```

to

```text id="4jb8e9"
thousands of files
hundreds of millions of rows
```

without requiring architectural changes.

---

# 2. Core Design Philosophy

## Dataset-Centric Architecture

Users think about datasets.

Not files.

Not tables.

Not storage formats.

Every command should therefore operate on a unified dataset abstraction.

---

## Single Analysis Model

The entire system revolves around a single canonical analysis object:

```python id="qv6k8q"
DatasetAnalysis
```

Every command either:

* Produces a DatasetAnalysis
* Consumes a DatasetAnalysis
* Compares two DatasetAnalysis objects
* Renders a DatasetAnalysis

This avoids duplicated scanning logic and overlapping report types.

---

## Strict Separation Of Concerns

The system is divided into:

```text id="2z9j9p"
CLI
Application
Domain
Infrastructure
Rendering
```

Each layer has a clearly defined responsibility.

---

# 3. Technology Stack

## CLI

```text id="4k2qgv"
Typer
```

Reason:

* Type-safe
* Excellent developer experience
* Good help generation

---

## Data Engine

```text id="lk7z5s"
Polars
PyArrow
```

Reason:

* Excellent Feather support
* Excellent Parquet support
* Fast
* Memory efficient
* Lazy execution support

---

## Rendering

```text id="c94fxc"
Rich
asciichartpy (line charts only)
Custom Unicode chart renderers
```

---

## Validation

```text id="7rjlwm"
Pydantic
```

---

## Testing

```text id="cn6sdq"
pytest
```

---

# Appendix to 3: Console Visualization Strategy

TabCaddy should not rely on a single charting library.

Different dataset insights require different visual representations.

## Primary Visualizations

### Rich Tables

Rich tables are the default visualization mechanism and should be used whenever structured data needs to be presented.

Examples:

- Column statistics
- Schema summaries
- Diff reports
- Quality diagnostics
- Dataset metadata

### Unicode Bar Charts

Implement lightweight custom Unicode-based bar charts for categorical distributions and proportional metrics.

Examples:

- Schema distribution
- Null-rate comparison
- File distribution by schema
- Top categorical values

Example:

```text
Schema Distribution

Schema A  ██████████████████ 92%
Schema B  ██                 7%
Schema C  ▏                 1%
```

These charts should be implemented as internal rendering utilities and should not require external plotting dependencies.

## Secondary Visualizations

### Line Charts

Use `asciichartpy` exclusively for time-series and trend visualization.

Examples:

- Rows over time
- Average value evolution
- Dataset growth over time
- Numeric trend analysis

Line charts should not be used for categorical distributions, schema summaries, or histogram-like visualizations.

## Rendering Structure

Add:

```text
rendering/

    charts/

        bar_chart.py
        line_chart.py
```

Responsibilities:

- `bar_chart.py` → Unicode bar chart rendering
- `line_chart.py` → asciichartpy integration

This keeps visualization concerns isolated from business logic and allows additional chart types to be added later without affecting the application architecture.


# 4. Repository Layout

```text id="zzxt40"
tabcaddy/

├── cli/
│
├── application/
│
├── domain/
│
├── infrastructure/
│
├── rendering/
│
├── tests/
│
├── config/
│
└── pyproject.toml
```

---

# 5. Domain Layer

The domain layer contains only business concepts.

No Polars.

No Rich.

No filesystem access.

---

# DatasetSource

Represents a user-supplied source.

```python id="pfjn1u"
@dataclass(frozen=True)
class DatasetSource:
    path: Path
    source_type: SourceType
```

---

# SourceType

```python id="a4a2cc"
class SourceType(Enum):
    FILE
    FOLDER
    COMPILED_DATASET
```

---

# ColumnDefinition

```python id="c1qjlwm"
@dataclass(frozen=True)
class ColumnDefinition:
    name: str
    dtype: str
```

---

# SchemaSignature

Represents one schema.

```python id="3rly2e"
@dataclass
class SchemaSignature:
    columns: list[ColumnDefinition]
    hash: str
    occurrence_count: int
```

Hash must be deterministic.

Hash input:

```text id="45nsh7"
column_name:dtype
```

ordered by column position.

SHA256 recommended.

---

# DatasetMetadata

Represents dataset identity and provenance.

```python id="wpnuvx"
@dataclass
class DatasetMetadata:
    version: int

    created_at: datetime

    row_count: int

    column_count: int

    source_file_count: int

    schema_hash: str | None

    column_hashes: dict[str, str] | None
```

Notes:

* `column_hashes` may be omitted in lightweight analysis modes.
* Used by diff and validation workflows.

---

# ColumnStatistics

```python id="8rz8pn"
@dataclass
class ColumnStatistics:
    dtype: str

    null_rate: float

    unique_estimate: int | None

    min_value: Any | None

    max_value: Any | None

    mean: float | None

    median: float | None

    stddev: float | None
```

---

# DatasetStatistics

```python id="x1qh1h"
@dataclass
class DatasetStatistics:
    columns: dict[str, ColumnStatistics]
```

---

# DatasetAnalysis

The central domain object.

```python id="e5jrgw"
@dataclass
class DatasetAnalysis:
    metadata: DatasetMetadata

    schemas: list[SchemaSignature]

    statistics: DatasetStatistics | None

    warnings: list[str]
```

All dataset exploration is represented through this object.

---

# DiffReport

```python id="t0bpcq"
@dataclass
class DiffReport:
    metadata_changes: list[str]

    schema_changes: list[str]

    statistics_changes: list[str]

    warnings: list[str]
```

---

# 6. Application Layer

Application layer contains use cases.

No filesystem implementation.

No Rich rendering.

---

# GenerateAnalysis

Purpose:

Generate a DatasetAnalysis.

```python id="mbv04g"
GenerateAnalysis(
    source,
    profile_mode
)
```

Returns:

```python id="4ecfqi"
DatasetAnalysis
```

This use case powers:

* summary
* schema
* compile
* scaffold-transform
* diff

---

# CompileDataset

Purpose:

Compile compatible files into a TabCaddy dataset.

```python id="4d8c7r"
CompileDataset(...)
```

Responsibilities:

* Detect schemas
* Validate selected schema
* Stream rows
* Write Parquet dataset
* Generate metadata

---

# TransformDataset

Purpose:

Apply user transformations.

```python id="smt66q"
TransformDataset(...)
```

Responsibilities:

* Load transform
* Validate transform signature
* Iterate matching files
* Write transformed outputs

---

# DiffDatasets

Purpose:

Compare datasets.

```python id="tkglnz"
DiffDatasets(...)
```

---

# ScaffoldTransform

Purpose:

Generate transform template.

```python id="0s56mr"
ScaffoldTransform(...)
```

---

# 7. Infrastructure Layer

Contains all external integrations.

---

# Source Resolver

```text id="95pb4i"
source_resolver.py
```

Purpose:

```python id="8pq5jz"
resolve_source(path)
```

Returns:

```python id="cz3q8z"
DatasetSource
```

---

# Readers

```text id="ovaf4z"
csv_reader.py

feather_reader.py

parquet_dataset_reader.py
```

Responsibilities:

* Reading
* Lazy scanning where possible

---

# Writers

```text id="e2gw5z"
csv_writer.py

feather_writer.py

parquet_dataset_writer.py
```

---

# Schema Analyzer

```text id="yv71h5"
schema_analyzer.py
```

Responsibilities:

* Extract schemas
* Build schema signatures
* Detect schema drift
* Group matching schemas

---

# Analysis Builder

```text id="d4v1kv"
analysis_builder.py
```

Responsibilities:

Generate:

```python id="8v7mjk"
DatasetAnalysis
```

---

# Metadata Builder

```text id="6tk7h0"
metadata_builder.py
```

Responsibilities:

Generate:

```python id="n4a4ut"
DatasetMetadata
```

---

# Cache Manager

```text id="7h1xq0"
cache_manager.py
```

Responsibilities:

Store and retrieve analyses.

---

# Transform Loader

```text id="x4k3ic"
transform_loader.py
```

Responsibilities:

* Load user transform module
* Validate callable

---

# Differ Implementations

```text id="fzz5lh"
file_differ.py

folder_differ.py

compiled_dataset_differ.py
```

---

# 8. Rendering Layer

Rendering only.

No business logic.

---

# Structure

```text id="1f3lf0"
rendering/

├── console.py
│
└── views/

    summary.py
    schema.py
    diff.py
```

---

# Rich Standards

Use:

```text id="z9wnzh"
Panel
Table
Tree
Rule
Progress
```

Avoid excessive visual complexity.

Output should remain readable on narrow terminals.

---

# Color Standards

```text id="x2q1ae"
Blue    Information
Green   Success
Yellow  Warning
Red     Error
Cyan    Metadata
```

---

# 9. Dataset Formats

## Supported Inputs

```text id="hnvs4l"
.csv
.feather
.arrow
```

---

# Compiled Dataset Format

Always use:

```text id="9xk0ul"
Parquet Dataset
```

---

# Dataset Layout

```text id="q79g5x"
compiled_dataset/

metadata.json

data/

    part-000.parquet
    part-001.parquet
    part-002.parquet
```

Rationale:

Future-proof layout.

Allows future additions:

```text id="d3n38z"
reports/
cache/
exports/
```

without changing dataset structure.

---

# 10. CLI Commands

---

# summary

## Purpose

Primary dataset exploration command.

This command replaces the need for a separate inspect command.

---

## Usage

```bash id="93z3bc"
tabcaddy summary <source>
```

Examples:

```bash id="kh3l5s"
tabcaddy summary trades.feather

tabcaddy summary data/

tabcaddy summary compiled_dataset/
```

---

## Output

### Metadata

```text id="p0wsw0"
files
rows
columns
schemas
```

### Schema Overview

### Statistics

### Date Ranges

### Warnings

---

## Analysis Profiles

### Quick

```bash id="4f7cga"
--profile quick
```

Computes:

```text id="csyblh"
row counts
file counts
schema counts
```

Only.

---

### Standard

Default.

Computes:

```text id="2ohwsl"
metadata
schemas
lightweight statistics
warnings
```

---

### Deep

```bash id="vlx1sm"
--profile deep
```

Computes:

```text id="c4b8ij"
full statistics
uniqueness estimates
histograms
column hashes
```

Potentially expensive.

---

# schema

## Purpose

Advanced schema diagnostics.

---

## Usage

```bash id="5wzv08"
tabcaddy schema <source>
```

---

## Output

### Schema Groups

### Schema Drift

### Type Changes

### Files Violating Dominant Schema

### Occurrence Counts

This command exists specifically for schema debugging and diagnostics.

---

# compile

## Purpose

Compile compatible files into a unified analytical dataset.

---

## Usage

```bash id="s5jq91"
tabcaddy compile <folder>
```

---

## Workflow

### Scan

Analyze all files.

### Group

Group schemas.

### Decision

If one schema:

Compile automatically.

If multiple schemas:

Display:

```text id="5nk5w4"
Schema 1 (412 files)
Schema 2 (7 files)
Schema 3 (2 files)
```

User reruns:

```bash id="0f4y4x"
tabcaddy compile data/ --schema 1
```

---

## Interactive Mode

```bash id="f7dbho"
tabcaddy compile data/ --interactive
```

Optional.

---

## Output

```text id="ktod8c"
compiled_dataset/
```

containing:

```text id="3v7f3l"
metadata.json

data/
```

---

# transform

## Purpose

Apply user-defined transformations.

---

## Usage

```bash id="6o7o7u"
tabcaddy transform \
    input_folder \
    transform.py \
    [output_folder]
```

---

## Transform Signatures

Supported:

```python id="v2o6qy"
def transform(df):
    return df
```

or

```python id="x0uv6m"
def transform(df, context):
    return df
```

---

# Transform Context

Provides:

```python id="e88pwq"
file_name

file_path

schema

metadata
```

Metadata includes:

```python id="jgntr6"
row_count

schema_hash
```

---

## Processing Model

V1 supports:

```text id="r4ob8z"
file-level transformations
```

only.

Architecture should leave room for future dataset-level transforms.

---

## Parallel Processing

```bash id="ejxylo"
--workers N
```

Supported.

---

## Default Output Folder

```text id="v4irdd"
<input_folder>_transformed
```

---

# scaffold-transform

## Purpose

Generate transform template.

---

## Usage

```bash id="w6ohxh"
tabcaddy scaffold-transform <source>
```

---

## Generated Template Includes

### Observed Schemas

### Column Types

### Identity Transform

### Example Transform Snippets

---

# diff

## Purpose

Compare datasets.

---

# Comparison Levels

Supported:

```text id="qovm9v"
metadata
statistics
full
```

---

## Usage

```bash id="6ktj0v"
tabcaddy diff A B
```

---

# Internal Architecture

```python id="w0zypi"
Differ
```

Implementations:

```python id="6u5frs"
FileDiffer

FolderDiffer

CompiledDatasetDiffer
```

---

# File Diff

```bash id="9g42j4"
tabcaddy diff fileA.feather fileB.feather
```

Shows:

* Metadata changes
* Schema changes
* Statistics changes

---

# Folder Diff

```bash id="mjlwm7"
tabcaddy diff folderA folderB
```

Shows:

* Added files
* Removed files
* Modified files
* Schema drift
* Dataset statistics changes

---

# Compiled Dataset Diff

```bash id="74trbp"
tabcaddy diff compiledA compiledB
```

Shows:

* Metadata changes
* Provenance changes
* Schema changes
* Statistics changes

---

# 11. Cache System

## Location

```text id="7vzl17"
.tabcaddy/cache/
```

---

## Cache Unit

Cache complete analyses.

```text id="dgt87m"
<analysis_hash>.json
```

Store:

```text id="7s8jlt"
metadata
schemas
statistics
warnings
```

together.

---

## Invalidation

Based on:

```text id="aqr0iy"
path
size
mtime
```

plus schema hash changes.

---

# 12. Testing Strategy

## Unit Tests

Cover:

* schema hashing
* schema grouping
* metadata generation
* statistics generation
* diff logic
* cache logic

---

## Integration Tests

Use real:

* CSV files
* Feather files
* Compiled datasets

---

## Snapshot Tests

Validate Rich output.

---

# 13. Version 1 Deliverables

Mandatory commands:

```text id="g3n7u9"
summary

schema

compile

transform

scaffold-transform

diff
```

Mandatory core domain objects:

```text id="onp4ib"
DatasetSource

DatasetAnalysis

DatasetMetadata

SchemaSignature

DiffReport
```

Mandatory infrastructure services:

```text id="mcp72y"
SourceResolver

SchemaAnalyzer

AnalysisBuilder

MetadataBuilder

CacheManager

TransformLoader
```

This architecture intentionally favors simplicity over abstraction density while remaining scalable enough to support future additions such as validation pipelines, SQL querying, report generation, and DuckDB integration.
