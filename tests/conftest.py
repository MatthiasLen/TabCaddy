from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _write_dataset(path: Path, frame: pl.DataFrame) -> None:
    if path.suffix == ".csv":
        frame.write_csv(path)
    else:
        frame.write_ipc(path)


@pytest.fixture
def homogeneous_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "homogeneous"
    folder.mkdir()
    _write_dataset(
        folder / "a.csv",
        pl.DataFrame(
            {
                "id": [1, 2],
                "value": [10.0, 20.0],
                "ts": [date(2024, 1, 1), date(2024, 1, 2)],
            }
        ),
    )
    _write_dataset(
        folder / "b.feather",
        pl.DataFrame(
            {
                "id": [3, 4],
                "value": [30.0, 40.0],
                "ts": [date(2024, 1, 3), date(2024, 1, 4)],
            }
        ),
    )
    return folder


@pytest.fixture
def drift_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "drift"
    folder.mkdir()
    _write_dataset(
        folder / "a.csv",
        pl.DataFrame({"id": [1, 2], "value": [10.0, 20.0]}),
    )
    _write_dataset(
        folder / "b.feather",
        pl.DataFrame({"id": [3, 4], "value": [30.0, 40.0]}),
    )
    _write_dataset(
        folder / "c.csv",
        pl.DataFrame({"id": [5], "value": [50.0], "extra": ["x"]}),
    )
    return folder
