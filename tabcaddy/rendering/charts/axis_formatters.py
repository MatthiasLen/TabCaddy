from __future__ import annotations

from datetime import UTC, datetime
import math


def format_numeric_axis(value: float) -> str:
    if not math.isfinite(value):
        return str(value)

    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))

    return f"{value:.6g}"


def format_epoch_seconds_utc(value: float) -> str:
    if not math.isfinite(value):
        return str(value)

    try:
        dt = datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return f"{value:.3g}"

    if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.strftime("%Y-%m-%d")

    return dt.strftime("%Y-%m-%d %H:%M:%SZ")
