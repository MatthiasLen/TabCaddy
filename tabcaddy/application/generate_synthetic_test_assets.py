from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class SyntheticAssetLayout:
    output_root: Path
    baseline_root: Path
    variant_root: Path


def generate_synthetic_test_assets(
    output_root: Path, n: int = 10
) -> SyntheticAssetLayout:
    if n < 1:
        raise ValueError("n must be at least 1")

    output_root = output_root.expanduser().resolve()
    baseline_root = output_root / "baseline"
    variant_root = output_root / "variant"

    for root in (baseline_root, variant_root):
        if root.exists():
            shutil.rmtree(root)

    _write_bundle(baseline_root, variant=False, n=n, seed=20260422)
    _write_bundle(variant_root, variant=True, n=n, seed=20260207)

    manifest = {
        "rows_per_file": n,
        "baseline": _bundle_manifest(baseline_root),
        "variant": _bundle_manifest(variant_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return SyntheticAssetLayout(
        output_root=output_root,
        baseline_root=baseline_root,
        variant_root=variant_root,
    )


def _bundle_manifest(root: Path) -> dict[str, object]:
    files = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    return {"root": str(root), "file_count": len(files), "files": files}


def _write_bundle(root: Path, variant: bool, n: int, seed: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    version_specs = [
        (
            "1000001-10000",
            datetime(2023, 7, 17, 5, 20, 24, 855855, tzinfo=timezone.utc),
            0,
            0,
            0,
        ),
        (
            "100002-100209",
            datetime(2025, 2, 5, 15, 27, 24, 528528, tzinfo=timezone.utc),
            1,
            8,
            3,
        ),
    ]
    for serial, start, version_seed, voltage_shift, current_shift in version_specs:
        frame = _build_version_telemetry_frame(
            start=start,
            version_seed=version_seed,
            voltage_shift=voltage_shift,
            current_shift=current_shift,
            n=n,
            rng=rng,
        )
        if variant and serial == "100002-100209":
            frame = frame.with_columns(
                [
                    (pl.col("DELTA_TIME") + 5000).alias("DELTA_TIME"),
                    pl.when(pl.col("index") == 3)
                    .then(2)
                    .otherwise(pl.col("VERSION"))
                    .alias("VERSION"),
                ]
            )
        _write_feather(root / "telemetry" / "version" / f"{serial}_SUDS.feather", frame)

    mcu_specs = [
        (
            "1000001-10000",
            datetime(2023, 7, 17, 5, 20, 24, 855855, tzinfo=timezone.utc),
            0,
            0,
        ),
        (
            "100002-100209",
            datetime(2025, 2, 5, 15, 27, 24, 528528, tzinfo=timezone.utc),
            11,
            4,
        ),
        (
            "100003-100075",
            datetime(2024, 3, 1, 9, 42, 10, 101010, tzinfo=timezone.utc),
            6,
            7,
        ),
    ]
    if variant:
        mcu_specs.append(
            (
                "100250-100999",
                datetime(2026, 1, 8, 11, 5, 0, 222222, tzinfo=timezone.utc),
                14,
                5,
            )
        )
    for serial, start, voltage_shift, current_shift in mcu_specs:
        frame = _build_mcu_telemetry_frame(
            serial=serial,
            start=start,
            voltage_shift=voltage_shift,
            current_shift=current_shift,
            n=n,
            rng=rng,
        )
        if variant and serial == "100002-100209":
            frame = frame.with_columns(
                [
                    pl.when(pl.col("index") == 2)
                    .then(pl.col("VOLTAGE") + 40)
                    .otherwise(pl.col("VOLTAGE"))
                    .alias("VOLTAGE"),
                    pl.when(pl.col("index") == 4)
                    .then(pl.col("CURRENT") + 12)
                    .otherwise(pl.col("CURRENT"))
                    .alias("CURRENT"),
                ]
            )
        _write_feather(root / "telemetry" / "mcu" / f"{serial}_SUDS.feather", frame)

    parts = _build_swcal_tools_parts_frame(variant=variant, n=n, rng=rng)
    products = _build_products_consumed_frame(variant=variant, n=n, rng=rng)
    _write_csv(root / "service" / "PRED_MAINT_SWCalToolsParts.csv", parts)
    _write_csv(root / "service" / "PRED_MAINT_ProductsConsumed.csv", products)


def _build_version_telemetry_frame(
    start: datetime,
    version_seed: int,
    voltage_shift: int,
    current_shift: int,
    n: int,
    rng: random.Random,
) -> pl.DataFrame:
    base_rows = _build_common_telemetry_rows(
        start=start,
        voltage_shift=voltage_shift,
        current_shift=current_shift,
        n=n,
        rng=rng,
    )
    return pl.DataFrame(
        [
            {
                **row,
                "VERSION": _maybe_missing(
                    rng,
                    (version_seed + row["index"] + rng.randint(0, 1)) % 2,
                    0.06,
                ),
            }
            for row in base_rows
        ]
    )


def _build_mcu_telemetry_frame(
    serial: str,
    start: datetime,
    voltage_shift: int,
    current_shift: int,
    n: int,
    rng: random.Random,
) -> pl.DataFrame:
    base_rows = _build_common_telemetry_rows(
        start=start,
        voltage_shift=voltage_shift,
        current_shift=current_shift,
        n=n,
        rng=rng,
    )
    return pl.DataFrame(
        [{**row, "SN": _maybe_missing(rng, serial, 0.04)} for row in base_rows]
    )


def _build_common_telemetry_rows(
    start: datetime,
    voltage_shift: int,
    current_shift: int,
    n: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    timestamp = start
    for index in range(n):
        delta_time = rng.randint(250, 9_500)
        if index > 0 and rng.random() < 0.2:
            delta_time = rng.randint(120_000, 3_600_000)
        if index > 0:
            timestamp += timedelta(milliseconds=delta_time)
        suds = 1 if rng.random() < 0.55 else 0
        voltage = _clamp(
            rng.randint(70, 240) + voltage_shift + rng.randint(-15, 15), 0, 255
        )
        current = _clamp(
            rng.randint(60, 160) + current_shift + rng.randint(-12, 12), 0, 255
        )
        inj_progress = _clamp((index // max(1, n // 4)) + rng.randint(-1, 1), 0, 7)
        rows.append(
            {
                "index": index,
                "DATE": _maybe_missing(rng, timestamp, 0.04),
                "SUDS": _maybe_missing(rng, suds, 0.03),
                "VOLTAGE": _maybe_missing(rng, voltage, 0.08),
                "CURRENT": _maybe_missing(rng, current, 0.08),
                "DELTA_TIME": _maybe_missing(rng, delta_time, 0.05),
                "INJ_PROGRESS": _maybe_missing(rng, inj_progress, 0.05),
            }
        )
    return rows


def _build_swcal_tools_parts_frame(
    variant: bool, n: int, rng: random.Random
) -> pl.DataFrame:
    country_options = [
        ("SE", "Soland", ["LUMENPORT", "BRIGHTHOLD", "MISTRAL"]),
        ("DE", "Deltora", ["IRONVALE", "EMBERN"]),
        ("US", "Auroria", ["SKYFORGE", "RIVERGATE"]),
    ]
    if variant:
        country_options.append(("GB", "Glassmere", ["MOONHAVEN", "STARWICK"]))
    work_types = ["RuneCare - Ritual Tune", "Flux Repair"]
    symptom_options = [
        ("null", "null"),
        ("Flowcraft", "MIST"),
        ("Sparkwork", "ARC"),
    ]
    product_lines = ["Nebulon", "Aetheris", "Quorix"]
    line_types = ["Spellware", "Tuning Tools"]
    part_descriptions = [
        "Nebulon 1.11.2 spell patch",
        "Oriole 288 tuning prism",
        "Fixture, pressure tune, Nebulon",
        "Arc diagnostic patch",
        "Mist valve calibration kit",
    ]

    rows = []
    for index in range(n):
        country_code, country, cities = rng.choice(country_options)
        symptom_category, symptom_code = rng.choice(symptom_options)
        work_date = datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(
            days=index * 2
        )
        rows.append(
            {
                "COUNTRY_CODE": country_code,
                "COUNTRY": country,
                "CITY": _maybe_missing(rng, rng.choice(cities), 0.06),
                "WORK_ORDER": f"W-{613861 + index:09d}",
                "WORK_ORDER_TYPE": rng.choice(work_types),
                "SYMPTOM_CATEGORY": _maybe_missing(rng, symptom_category, 0.12),
                "SYMPTOM_CODE": _maybe_missing(rng, symptom_code, 0.12),
                "DATE": int(work_date.strftime("%Y%m%d")),
                "PRODUCT_LINE": rng.choice(product_lines),
                "SYSTEM_EQUNR": 1701000021849000 + index * 137 + rng.randint(0, 90),
                "SYSTEM_SERIAL_ID": 90000820 + index * 3 + rng.randint(0, 2),
                "SYSTEM_MATERIAL_ID": 88620000 + rng.randint(0, 9999),
                "LINE_TYPE": rng.choice(line_types),
                "PART_MATERIAL_ID": 86830000 + rng.randint(0, 59999),
                "PART_DESCRIPTION": _maybe_missing(
                    rng, rng.choice(part_descriptions), 0.08
                ),
                "SERIAL_NUMBER": _maybe_missing(
                    rng,
                    rng.choice(["A", "07L-1075", "408849", "PATCH-01", "KIT-220"]),
                    0.2,
                ),
            }
        )
    return pl.DataFrame(rows)


def _build_products_consumed_frame(
    variant: bool, n: int, rng: random.Random
) -> pl.DataFrame:
    work_types = ["RuneCare - Ritual Tune", "Flux Repair"]
    symptom_options = [
        ("null", "null"),
        ("Flowcraft", "MIST"),
        ("Sparkwork", "ARC"),
    ]
    product_lines = ["Nebulon", "Aetheris", "Quorix"]
    tracking_types = ["Open Stock", "Batch Marked", "Rune Serialized"]
    product_descriptions = [
        "Breeze filter 60",
        "Mist sensor service kit",
        "Drift base plate",
        "Arc sensing harness",
        "Pressure tuning kit",
    ]
    if variant:
        product_lines.append("Velatrix")
        product_descriptions.append("Flux controller retrofit kit")

    rows = []
    for index in range(n):
        symptom_category, symptom_code = rng.choice(symptom_options)
        tracking_type = rng.choice(tracking_types)
        work_date = datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(
            days=index * 2
        )
        serial_number = None
        batch_number = None
        if tracking_type == "Batch Marked":
            batch_number = (
                f"HC{rng.randint(10, 99):02d}X{rng.choice(['LY', 'QZ', 'RT'])}"
            )
        elif tracking_type == "Rune Serialized":
            serial_number = f"HS-{rng.randint(100, 999)}"
        rows.append(
            {
                "WORK_ORDER": f"W-{613861 + index:09d}",
                "ID": f"0WOSe0000092{rng.choice(['cnB', 'coC', 'zzZ'])}{index:02d}AQ",
                "WORK_ORDER_TYPE": rng.choice(work_types),
                "SYMPTOM_CATEGORY": _maybe_missing(rng, symptom_category, 0.12),
                "SYMPTOM_CODE": _maybe_missing(rng, symptom_code, 0.12),
                "DATE": int(work_date.strftime("%Y%m%d")),
                "PRODUCT_LINE": rng.choice(product_lines),
                "SYSTEM_EQUNR": 1701000021849000 + index * 173 + rng.randint(0, 120),
                "SYSTEM_SERIAL_ID": 90000820 + index * 4 + rng.randint(0, 3),
                "SYSTEM_MATERIAL_ID": 88620000 + rng.randint(0, 9999),
                "PRODUCT_CONSUMED": 86170000 + rng.randint(0, 69999),
                "PRODUCT_DESCRIPTION": _maybe_missing(
                    rng, rng.choice(product_descriptions), 0.08
                ),
                "TRACKING_TYPE": tracking_type,
                "SERIAL_NUMBER": serial_number,
                "BATCH_NUMBER": batch_number,
                "PRODUCT_CONSUMED_ID": f"PC-{19541 + index}",
            }
        )
    return pl.DataFrame(rows)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _maybe_missing(
    rng: random.Random, value: object, probability: float
) -> object | None:
    if rng.random() < probability:
        return None
    return value


def _write_csv(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)


def _write_feather(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_ipc(path)
