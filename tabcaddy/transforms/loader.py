from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from pydantic import BaseModel


class TransformMetadata(BaseModel):
    row_count: int
    schema_hash: str


@dataclass(frozen=True)
class TransformContext:
    file_name: str
    file_path: str
    schema: list[dict[str, str]]
    metadata: TransformMetadata


class TransformLoader:
    def load(self, path: Path) -> tuple[Callable[..., Any], bool]:
        module = self._load_module(path)
        transform = getattr(module, "transform", None)
        if transform is None or not callable(transform):
            raise ValueError(
                "Transform module must define a callable named 'transform'."
            )
        signature = inspect.signature(transform)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) not in {1, 2}:
            raise ValueError("Transform signature must be (df) or (df, context).")
        return transform, len(positional) == 2

    def _load_module(self, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Unable to load transform module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


__all__ = ["TransformContext", "TransformLoader", "TransformMetadata"]
