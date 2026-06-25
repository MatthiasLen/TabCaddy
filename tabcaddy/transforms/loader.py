from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

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
    _SIGNATURE_HINT = (
        "(df) or (df, context), with optional keyword-only parameters that have "
        "defaults."
    )

    def load(self, path: Path) -> tuple[Callable[..., Any], bool]:
        module = self._load_module(path)
        candidate = getattr(module, "transform", None)
        if candidate is None or not callable(candidate):
            raise ValueError(
                "Transform module must define a callable named 'transform'."
            )
        transform = cast(Callable[..., Any], candidate)
        signature = inspect.signature(transform)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) not in {1, 2}:
            raise ValueError(f"Transform signature must be {self._SIGNATURE_HINT}")

        for parameter in signature.parameters.values():
            if parameter.kind == parameter.VAR_POSITIONAL:
                raise ValueError(
                    "Transform signature cannot include *args; supported signatures are "
                    f"{self._SIGNATURE_HINT}"
                )
            if (
                parameter.kind == parameter.KEYWORD_ONLY
                and parameter.default is inspect.Parameter.empty
            ):
                raise ValueError(
                    "Transform signature cannot include required keyword-only "
                    f"parameter '{parameter.name}'."
                )
        return transform, len(positional) == 2

    def _load_module(self, path: Path) -> ModuleType:
        resolved_path = path.expanduser().resolve()
        transform_root = resolved_path.parent
        self._validate_transform_sources(transform_root)
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
            )
        return module

    def _validate_transform_sources(self, transform_root: Path) -> None:
        for path in self._iter_transform_source_files(transform_root):
            self._validate_top_level_imports(path, transform_root)

    def _validate_top_level_imports(self, path: Path, transform_root: Path) -> None:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as error:
            message = error.msg or "invalid syntax"
            location = self._describe_source_path(path, transform_root)
            if error.lineno is not None:
                raise ValueError(
                    f"Invalid syntax in {location} at line {error.lineno}: {message}"
                ) from error
            raise ValueError(f"Invalid syntax in {location}: {message}") from error

        violations = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for nested in ast.walk(node):
                if isinstance(nested, (ast.Import, ast.ImportFrom)):
                    line = getattr(nested, "lineno", "?")
                    violations.append(line)

        if violations:
            lines = ", ".join(str(line) for line in sorted(set(violations)))
            location = self._describe_source_path(path, transform_root)
            raise ValueError(
                "All imports in transform scripts and local helper modules must be "
                "module top-level, not inside function or class bodies. "
                f"Move imports to the top of {location} (line(s): {lines})."
            )

    def _iter_transform_source_files(self, transform_root: Path):
        for path in sorted(transform_root.rglob("*.py")):
            if any(
                part.startswith(".") or part == "__pycache__" for part in path.parts
            ):
                continue
            yield path

    def _describe_source_path(self, path: Path, transform_root: Path) -> str:
        if path == transform_root / "transform.py":
            return "transform.py"
        try:
            return str(path.relative_to(transform_root))
        except ValueError:
            return str(path)

    def _purge_new_local_modules(
        self,
        module_names_before: set[str],
        transform_root: Path,
    ) -> None:
        for name, candidate in list(sys.modules.items()):
            if name in module_names_before:
                continue
            if self._path_is_within_root(
                self._module_origin(candidate), transform_root
            ):
                sys.modules.pop(name, None)

    def _evict_conflicting_modules(
        self, local_names: set[str], transform_root: Path
    ) -> None:
        for name in local_names:
            candidate = sys.modules.get(name)
            if candidate is None:
                continue
            origin = self._module_origin(candidate)
            if origin is None:
                continue
            if self._path_is_within_root(origin, transform_root):
                continue
            self._evict_module_and_descendants(name)

    def _evict_module_and_descendants(self, module_name: str) -> None:
        descendant_prefix = f"{module_name}."
        for name in list(sys.modules):
            if name == module_name or name.startswith(descendant_prefix):
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

    def _path_is_within_root(self, origin: Path | None, transform_root: Path) -> bool:
        if origin is None:
            return False
        try:
            origin.relative_to(transform_root)
            return True
        except ValueError:
            return False

    def _module_origin(self, module: ModuleType) -> Path | None:
        module_file = getattr(module, "__file__", None)
        resolved = self._resolve_absolute_path(module_file)
        if resolved is not None:
            return resolved

        module_spec = getattr(module, "__spec__", None)
        return self._resolve_absolute_path(getattr(module_spec, "origin", None))

    def _resolve_absolute_path(self, value: object) -> Path | None:
        if not isinstance(value, str) or value in {"built-in", "frozen"}:
            return None
        try:
            resolved = Path(value)
            if not resolved.is_absolute():
                return None
            return resolved.resolve()
        except OSError:
            return None


__all__ = ["TransformContext", "TransformLoader", "TransformMetadata"]
