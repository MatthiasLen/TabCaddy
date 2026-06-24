from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def tc(tabcaddy_cmd: list[str], args: list[str], cwd: Path) -> None:
    run([*tabcaddy_cmd, *args], cwd)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_batch(root: Path, files: dict[str, list[dict[str, Any]]]) -> None:
    for rel_path, rows in files.items():
        if not rows:
            continue
        write_csv(root / rel_path, rows, list(rows[0].keys()))


def build_sample_data(workflow_root: Path) -> tuple[Path, Path]:
    raw_batch = workflow_root / "raw_batch"
    incoming_batch = workflow_root / "incoming_batch"

    # 1) Create the initial raw batch with mild schema drift to mimic real-world ingest noise.
    write_batch(
        raw_batch,
        {
            "service/orders_2026_01.csv": [
                {
                    "order_id": 1001,
                    "customer_id": "C-001",
                    "event_date": "2026-01-03",
                    "units": 2,
                    "unit_price": 120.0,
                    "channel": "online",
                },
                {
                    "order_id": 1002,
                    "customer_id": "C-002",
                    "event_date": "2026-01-07",
                    "units": 1,
                    "unit_price": 430.0,
                    "channel": "partner",
                },
                {
                    "order_id": 1003,
                    "customer_id": "C-003",
                    "event_date": "2026-01-10",
                    "units": 3,
                    "unit_price": 80.0,
                    "channel": "online",
                },
            ],
            "service/orders_2026_02.csv": [
                {
                    "order_id": 2001,
                    "customer_id": "C-003",
                    "event_date": "2026-02-02",
                    "units": 2,
                    "unit_price": 95.0,
                    "channel": "online",
                    "discount_pct": 0.05,
                },
                {
                    "order_id": 2002,
                    "customer_id": "C-004",
                    "event_date": "2026-02-08",
                    "units": 1,
                    "unit_price": 210.0,
                    "channel": "field",
                    "discount_pct": 0.10,
                },
            ],
        },
    )

    # 2) Create an incoming batch containing updates and new rows for a later merge.
    write_batch(
        incoming_batch,
        {
            "service/orders_2026_01.csv": [
                {
                    "order_id": 1002,
                    "customer_id": "C-002",
                    "event_date": "2026-01-07",
                    "units": 2,
                    "unit_price": 430.0,
                    "channel": "partner",
                },
                {
                    "order_id": 1004,
                    "customer_id": "C-005",
                    "event_date": "2026-01-15",
                    "units": 1,
                    "unit_price": 500.0,
                    "channel": "online",
                },
            ],
            "service/orders_2026_02.csv": [
                {
                    "order_id": 2001,
                    "customer_id": "C-003",
                    "event_date": "2026-02-02",
                    "units": 2,
                    "unit_price": 95.0,
                    "channel": "online",
                    "discount_pct": 0.15,
                },
                {
                    "order_id": 2003,
                    "customer_id": "C-006",
                    "event_date": "2026-02-11",
                    "units": 5,
                    "unit_price": 70.0,
                    "channel": "partner",
                    "discount_pct": 0.00,
                },
            ],
        },
    )

    return raw_batch, incoming_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end TabCaddy workflow covering key TabCaddy CLI capabilities."
    )
    parser.add_argument(
        "--workspace",
        default="workflow_demo",
        help="Directory where workflow artifacts should be created.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep existing workspace contents instead of recreating it.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    workflow_root = (repo_root / args.workspace).resolve()
    workflow_root.mkdir(parents=True, exist_ok=True)

    if not args.keep and any(workflow_root.iterdir()):
        shutil.rmtree(workflow_root)
        workflow_root.mkdir(parents=True, exist_ok=True)

    tabcaddy_cmd = [sys.executable, "-m", "tabcaddy"]

    # 1) & 2) Create the initial raw and incoming batch of CSV files to simulate a real-world data ingest scenario.
    raw_batch, incoming_batch = build_sample_data(workflow_root)

    # 3) Summarize raw ingest data to get a quick profile before any cleaning work.
    tc(tabcaddy_cmd, ["summary", str(raw_batch), "--profile", "standard"], repo_root)

    # 4) Preview rows to inspect values and spot obvious quality issues fast.
    tc(tabcaddy_cmd, ["head", str(raw_batch), "--n", "5", "--showmeta"], repo_root)

    # 5) Run dedicated schema analysis to confirm drift and dominant schema groups.
    tc(tabcaddy_cmd, ["schema", str(raw_batch)], repo_root)

    transform_script = workflow_root / "transform_orders.py"

    # 6) Scaffold a transform template so the workflow starts from TabCaddy's generated contract.
    tc(
        tabcaddy_cmd,
        ["scaffold-transform", str(raw_batch), "--output", str(transform_script)],
        repo_root,
    )

    # 7) Replace the scaffold with a realistic transform that standardizes names and computes features.
    transform_script.write_text(
        """from __future__ import annotations

import polars as pl


def transform(df: pl.DataFrame, context=None) -> pl.DataFrame:
    working = df
    if \"channel\" in working.columns:
        working = working.with_columns(
            pl.col(\"channel\").cast(pl.String).str.to_lowercase().str.strip_chars().alias(\"channel\")
        )
    if \"discount_pct\" not in working.columns:
        working = working.with_columns(pl.lit(0.0).alias(\"discount_pct\"))
    if {\"units\", \"unit_price\"}.issubset(working.columns):
        working = working.with_columns(
            (pl.col(\"units\").cast(pl.Float64) * pl.col(\"unit_price\").cast(pl.Float64)).alias(\"gross_revenue\")
        )
        working = working.with_columns(
            (pl.col(\"gross_revenue\") * (1 - pl.col(\"discount_pct\").cast(pl.Float64))).alias(\"net_revenue\")
        )
    source_file = getattr(context, \"file_name\", \"unknown\") if context is not None else \"unknown\"
    return working.with_columns(pl.lit(source_file).alias(\"source_file\"))
""",
        encoding="utf-8",
    )

    cleaned_batch = workflow_root / "cleaned_batch"
    incoming_cleaned = workflow_root / "incoming_cleaned"

    # 8) Apply the transform to the initial raw folder to create the curated dataset candidate.
    tc(
        tabcaddy_cmd,
        ["transform", str(raw_batch), str(transform_script), str(cleaned_batch)],
        repo_root,
    )

    # 9) Compile curated files into a reusable TabCaddy compiled dataset for downstream tasks.
    # Use --validate so compile also demonstrates provenance/row-count integrity checks.
    compiled_initial = workflow_root / "compiled_initial"
    tc(
        tabcaddy_cmd,
        [
            "compile",
            str(cleaned_batch),
            "--output",
            str(compiled_initial),
            "--validate",
        ],
        repo_root,
    )

    # 10) Inspect compiled rows to validate the curated output produced from the compile stage.
    tc(tabcaddy_cmd, ["head", str(compiled_initial), "--n", "5"], repo_root)

    # 11) Plot transformed revenue trends to include visual diagnostics in the demo.
    tc(
        tabcaddy_cmd,
        [
            "plot",
            str(cleaned_batch / "service" / "orders_2026_01.csv"),
            "event_date",
            "net_revenue",
            "--kind",
            "line",
            "--interpolation",
            "nearest",
        ],
        repo_root,
    )

    # 12) Plot price vs net revenue as a scatter view to quickly inspect margin behavior.
    tc(
        tabcaddy_cmd,
        [
            "plot",
            str(cleaned_batch / "service" / "orders_2026_02.csv"),
            "unit_price",
            "net_revenue",
            "--kind",
            "scatter",
        ],
        repo_root,
    )

    # 13) Transform the incoming batch with the same business logic to keep processing consistent.
    tc(
        tabcaddy_cmd,
        [
            "transform",
            str(incoming_batch),
            str(transform_script),
            str(incoming_cleaned),
        ],
        repo_root,
    )

    # 14) Diff curated baseline vs curated incoming at full depth with key-based row diagnostics.
    tc(
        tabcaddy_cmd,
        [
            "diff",
            str(cleaned_batch),
            str(incoming_cleaned),
            "--level",
            "full",
            "--on",
            "order_id",
            "--row-examples",
            "5",
        ],
        repo_root,
    )

    # 15) Run a dry merge plan first to preview matches, conflicts, and schema behavior safely.
    merged_folder = workflow_root / "merged_archive"
    merge_args = [
        "merge",
        str(incoming_cleaned),
        str(cleaned_batch),
        "--out",
        str(merged_folder),
        "--on",
        "order_id",
        "--strategy",
        "upsert",
        "--schema-evolution",
        "allow-additive",
    ]
    tc(tabcaddy_cmd, [*merge_args, "--dry"], repo_root)

    # 16) Execute the real merge after validation to produce the consolidated archive.
    tc(tabcaddy_cmd, merge_args, repo_root)

    # 17) Compile the merged archive to publish a new reusable dataset version.
    compiled_merged = workflow_root / "compiled_merged"
    tc(
        tabcaddy_cmd,
        ["compile", str(merged_folder), "--output", str(compiled_merged), "--validate"],
        repo_root,
    )

    # 18) Diff old vs new compiled datasets to quantify impact of the incoming release.
    # Compiled-vs-compiled comparisons do not support row-level key options.
    tc(
        tabcaddy_cmd,
        ["diff", str(compiled_initial), str(compiled_merged), "--level", "full"],
        repo_root,
    )

    # 19) Produce a deep summary for reporting and handoff to analytics stakeholders.
    tc(tabcaddy_cmd, ["summary", str(compiled_merged), "--profile", "deep"], repo_root)

    # 20) End with a compact trend chart over the final compiled dataset for quick storytelling.
    tc(
        tabcaddy_cmd,
        [
            "plot",
            str(compiled_merged),
            "event_date",
            "net_revenue",
            "gross_revenue",
            "--kind",
            "line",
            "--aggregate-x",
            "mean",
            "--interpolation",
            "nearest",
        ],
        repo_root,
    )

    print("\nWorkflow complete. Artifacts are available under:")
    print(workflow_root)


if __name__ == "__main__":
    main()
