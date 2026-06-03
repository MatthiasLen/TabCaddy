## TabCaddy

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


