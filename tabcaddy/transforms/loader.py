from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import sys
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
        module, transform_root = self._load_module(path)
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

        for parameter in signature.parameters.values():
            if parameter.kind == parameter.VAR_POSITIONAL:
                raise ValueError(
                    "Transform signature cannot include *args; supported signatures are "
                    "(df) or (df, context)."
                )
            if (
                parameter.kind == parameter.KEYWORD_ONLY
                and parameter.default is inspect.Parameter.empty
            ):
                raise ValueError(
                    "Transform signature cannot include required keyword-only "
                    f"parameter '{parameter.name}'."
                )
        return self._with_transform_path(transform, transform_root), len(
            positional
        ) == 2

    def _load_module(self, path: Path) -> tuple[ModuleType, Path]:
        resolved_path = path.expanduser().resolve()
        transform_root = resolved_path.parent
        module_hash = hashlib.sha1(str(resolved_path).encode("utf-8")).hexdigest()
        module_name = f"_tabcaddy_transform_{module_hash}"
        spec = importlib.util.spec_from_file_location(module_name, resolved_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Unable to load transform module: {path}")

        local_names = self._discover_local_import_names(transform_root)
        self._evict_conflicting_modules(local_names, transform_root)

        module = importlib.util.module_from_spec(spec)
        transform_dir = str(transform_root)
        module_names_before = set(sys.modules)
        added_to_path = False
        if transform_dir not in sys.path:
            sys.path.insert(0, transform_dir)
            added_to_path = True

        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        finally:
            if added_to_path and transform_dir in sys.path:
                sys.path.remove(transform_dir)

        self._purge_new_local_modules(
            module_names_before=module_names_before,
            transform_root=transform_root,
            exclude_names={module_name},
        )
        return module, transform_root

    def _with_transform_path(
        self, transform: Callable[..., Any], transform_root: Path
    ) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            local_names = self._discover_local_import_names(transform_root)
            self._evict_conflicting_modules(local_names, transform_root)

            transform_dir = str(transform_root)
            module_names_before = set(sys.modules)
            added_to_path = False
            if transform_dir not in sys.path:
                sys.path.insert(0, transform_dir)
                added_to_path = True
            try:
                return transform(*args, **kwargs)
            finally:
                if added_to_path and transform_dir in sys.path:
                    sys.path.remove(transform_dir)
                self._purge_new_local_modules(
                    module_names_before=module_names_before,
                    transform_root=transform_root,
                    exclude_names=set(),
                )

        return wrapped

    def _purge_new_local_modules(
        self,
        module_names_before: set[str],
        transform_root: Path,
        exclude_names: set[str],
    ) -> None:
        for name, candidate in list(sys.modules.items()):
            if name in exclude_names or name in module_names_before:
                continue
            if self._module_is_within_root(candidate, transform_root):
                sys.modules.pop(name, None)

    def _evict_conflicting_modules(
        self, local_names: set[str], transform_root: Path
    ) -> None:
        for name in local_names:
            candidate = sys.modules.get(name)
            if candidate is None:
                continue
            if self._module_is_within_root(candidate, transform_root):
                continue
            sys.modules.pop(name, None)

    def _discover_local_import_names(self, transform_root: Path) -> set[str]:
        names: set[str] = set()
        for child in transform_root.iterdir():
            if child.name.startswith("."):
                continue
            if child.is_file() and child.suffix == ".py" and child.stem != "__init__":
                names.add(child.stem)
            if child.is_dir() and (child / "__init__.py").exists():
                names.add(child.name)
        return names

    def _module_is_within_root(self, module: ModuleType, transform_root: Path) -> bool:
        origin = self._module_origin(module)
        if origin is None:
            return False
        root = str(transform_root)
        origin_str = str(origin)
        try:
            return os.path.commonpath([origin_str, root]) == root
        except ValueError:
            return False

    def _module_origin(self, module: ModuleType) -> Path | None:
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str):
            try:
                return Path(module_file).resolve()
            except OSError:
                return None

        module_spec = getattr(module, "__spec__", None)
        module_origin = getattr(module_spec, "origin", None)
        if isinstance(module_origin, str):
            try:
                return Path(module_origin).resolve()
            except OSError:
                return None

        return None


__all__ = ["TransformContext", "TransformLoader", "TransformMetadata"]
