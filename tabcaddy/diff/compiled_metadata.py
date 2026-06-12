from __future__ import annotations

import json
from json import JSONDecodeError

from tabcaddy.domain.models import DatasetSource


def load_compiled_provenance(source: DatasetSource) -> dict | None:
    metadata_path = source.path / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    compiled = payload.get("compiled")
    return compiled if isinstance(compiled, dict) else None
