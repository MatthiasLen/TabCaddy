from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


def _resolve_version() -> str:
    try:
        return version("tabcaddy")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        if pyproject_path.exists():
            with pyproject_path.open("rb") as handle:
                return str(tomllib.load(handle)["project"]["version"])
        return "unknown"


__version__ = _resolve_version()

__all__ = ["__version__"]
