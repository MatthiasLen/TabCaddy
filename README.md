## TabCaddy

[![CI](https://github.com/MatthiasLen/TabCaddy/actions/workflows/ci.yml/badge.svg)](https://github.com/MatthiasLen/TabCaddy/actions/workflows/ci.yml)

TabCaddy is a dataset-centric CLI for exploring, compiling, transforming, and diffing CSV, Feather, and compiled parquet datasets.

### Commands

- `tabcaddy summary <source>`
- `tabcaddy schema <source>`
- `tabcaddy compile <folder> [--schema N]`
- `tabcaddy transform <input> <transform.py> [output]`
- `tabcaddy scaffold-transform <source> [--output transform_template.py]`
- `tabcaddy diff <left> <right> [--level metadata|statistics|full]`

### Profiles

- `quick`: metadata and schema counts
- `standard`: metadata, schema overview, lightweight statistics
- `deep`: full statistics, uniqueness estimates, histograms, and column hashes

Run with the local virtual environment:

```powershell
.\.venv\Scripts\python -m tabcaddy --help
```

### Development Checks

Install the dev tools and register the hooks:

```powershell
uv sync --group dev
uv run pre-commit install
```

Run the same checks locally that GitHub Actions runs:

```powershell
uv run pre-commit run --all-files
```

Run the test suite:

```powershell
.\.venv\Scripts\python -m pytest
```
