from __future__ import annotations

import argparse
from pathlib import Path

from tabcaddy.test_support.synthetic_assets import (
    generate_synthetic_test_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CSV and Feather datasets for TabCaddy regression tests."
    )
    parser.add_argument(
        "output", help="Directory where the synthetic assets should be written."
    )
    parser.add_argument(
        "--rows",
        "-r",
        type=int,
        default=10,
        help="Number of rows to generate per file.",
    )
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=None,
        help="Number of telemetry files to generate in each telemetry group.",
    )
    args = parser.parse_args()

    layout = generate_synthetic_test_assets(
        Path(args.output), n=args.rows, num_files=args.num_files
    )
    print(f"Synthetic assets written to {layout.output_root}")
    print(f"- baseline: {layout.baseline_root}")
    print(f"- variant: {layout.variant_root}")
    print(f"- manifest: {layout.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
