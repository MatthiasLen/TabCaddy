from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl


_FILTER_OPERATORS_BY_LENGTH = ("==", "!=", ">=", "<=", ">", "<")
_FILTER_COLUMN_OPERATOR_CHARS = frozenset("=<>!")
_FILTER_OPERATORS = {
    "==": lambda lhs, rhs: lhs == rhs,
    "!=": lambda lhs, rhs: lhs != rhs,
    ">": lambda lhs, rhs: lhs > rhs,
    ">=": lambda lhs, rhs: lhs >= rhs,
    "<": lambda lhs, rhs: lhs < rhs,
    "<=": lambda lhs, rhs: lhs <= rhs,
}


class PlotFiltering:
    def build_predicate(self, expression: str, *, schema: pl.Schema) -> pl.Expr:
        column, operator, raw_value = self.parse_expression(expression)
        if column not in schema:
            raise ValueError(f"Column not found in --filter: {column}")
        column_dtype = schema[column]
        return _FILTER_OPERATORS[operator](
            pl.col(column),
            self.parse_value(raw_value, dtype=column_dtype),
        )

    def parse_expression(self, expression: str) -> tuple[str, str, str]:
        stripped_expression = expression.strip()
        if not stripped_expression:
            raise ValueError(
                "Invalid --filter expression. Expected format: COLUMN OP VALUE "
                "with OP in ==, !=, >, >=, <, <=."
            )

        bracket_depth = 0
        for index, char in enumerate(stripped_expression):
            if char == "[":
                bracket_depth += 1
                continue
            if char == "]":
                if bracket_depth == 0:
                    raise ValueError(
                        "Invalid --filter expression. Expected format: COLUMN OP VALUE "
                        "with OP in ==, !=, >, >=, <, <=."
                    )
                bracket_depth -= 1
                continue
            if bracket_depth > 0:
                continue

            for operator in _FILTER_OPERATORS_BY_LENGTH:
                if not stripped_expression.startswith(operator, index):
                    continue

                raw_column = stripped_expression[:index].strip()
                raw_value = stripped_expression[index + len(operator) :].strip()
                if not raw_column or not raw_value:
                    raise ValueError(
                        "Invalid --filter expression. Expected format: COLUMN OP VALUE "
                        "with OP in ==, !=, >, >=, <, <=."
                    )

                if raw_column.startswith("["):
                    if not raw_column.endswith("]"):
                        raise ValueError(
                            "Invalid --filter expression. Expected format: COLUMN OP VALUE "
                            "with OP in ==, !=, >, >=, <, <=."
                        )
                    column = raw_column[1:-1].strip()
                    if not column:
                        raise ValueError(
                            "Invalid --filter expression. Expected format: COLUMN OP VALUE "
                            "with OP in ==, !=, >, >=, <, <=."
                        )
                    return column, operator, raw_value

                if "[" in raw_column or "]" in raw_column:
                    raise ValueError(
                        "Invalid --filter expression. Expected format: COLUMN OP VALUE "
                        "with OP in ==, !=, >, >=, <, <=."
                    )
                if any(token in _FILTER_COLUMN_OPERATOR_CHARS for token in raw_column):
                    raise ValueError(
                        "Invalid --filter expression. Expected format: COLUMN OP VALUE "
                        "with OP in ==, !=, >, >=, <, <=."
                    )
                return raw_column, operator, raw_value

        if bracket_depth != 0:
            raise ValueError(
                "Invalid --filter expression. Expected format: COLUMN OP VALUE "
                "with OP in ==, !=, >, >=, <, <=."
            )
        raise ValueError(
            "Invalid --filter expression. Expected format: COLUMN OP VALUE "
            "with OP in ==, !=, >, >=, <, <=."
        )

    def parse_value(
        self, value: str, *, dtype: pl.DataType
    ) -> bool | int | float | str | date | datetime | time:
        stripped = value.strip()
        if (
            len(stripped) >= 2
            and stripped[0] == stripped[-1]
            and stripped[0]
            in {
                '"',
                "'",
            }
        ):
            stripped = stripped[1:-1]

        if self._is_string_like_dtype(dtype):
            return stripped

        if dtype == pl.Date:
            try:
                return date.fromisoformat(stripped)
            except ValueError as error:
                raise ValueError(
                    "Invalid --filter value for Date column. Expected ISO-8601 date "
                    "(YYYY-MM-DD)."
                ) from error

        if self._is_time_dtype(dtype):
            try:
                parsed_time = time.fromisoformat(stripped)
            except ValueError as error:
                raise ValueError(
                    "Invalid --filter value for Time column. Expected ISO-8601 "
                    "time (HH:MM[:SS[.ffffff]])."
                ) from error
            if parsed_time.tzinfo is not None:
                raise ValueError(
                    "Invalid --filter value for Time column. Time literals with "
                    "timezone offsets are not supported."
                )
            return parsed_time

        if self._is_datetime_dtype(dtype):
            try:
                parsed = datetime.fromisoformat(stripped)
            except ValueError as error:
                raise ValueError(
                    "Invalid --filter value for Datetime column. Expected ISO-8601 "
                    "datetime (for example 2026-01-01T12:34:56 or 2026-01-01)."
                ) from error
            timezone_name = getattr(dtype, "time_zone", None)
            if timezone_name:
                if parsed.tzinfo is None:
                    if timezone_name == "UTC":
                        return parsed.replace(tzinfo=UTC)
                    try:
                        return parsed.replace(tzinfo=ZoneInfo(timezone_name))
                    except (ValueError, TypeError, KeyError) as error:
                        raise ValueError(
                            "Invalid timezone metadata for Datetime filter comparison: "
                            f"{timezone_name}"
                        ) from error
                if timezone_name == "UTC":
                    return parsed.astimezone(UTC)
                try:
                    return parsed.astimezone(ZoneInfo(timezone_name))
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(
                        "Invalid timezone metadata for Datetime filter comparison: "
                        f"{timezone_name}"
                    ) from error
            if parsed.tzinfo is not None:
                return parsed.astimezone(UTC).replace(tzinfo=None)
            return parsed

        if dtype == pl.Boolean:
            lowered = stripped.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
            raise ValueError(
                "Invalid --filter value for Boolean column. Expected true or false."
            )

        if self._is_numeric_dtype(dtype):
            if "Decimal" in str(dtype):
                try:
                    return Decimal(stripped)
                except (ArithmeticError, ValueError):
                    pass
            for parser in (int, float):
                try:
                    return parser(stripped)
                except ValueError:
                    continue
            raise ValueError(
                "Invalid --filter value for numeric column. Expected a numeric literal."
            )

        lowered = stripped.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

        for parser in (int, float):
            try:
                return parser(stripped)
            except ValueError:
                continue
        return stripped

    def _is_numeric_dtype(self, dtype: pl.DataType) -> bool:
        probe = getattr(dtype, "is_numeric", None)
        if callable(probe):
            return bool(probe())
        if probe is not None:
            return bool(probe)
        return any(token in str(dtype) for token in ("Int", "UInt", "Float", "Decimal"))

    def _is_datetime_dtype(self, dtype: pl.DataType) -> bool:
        if dtype == pl.Datetime:
            return True
        base_type = getattr(dtype, "base_type", None)
        if callable(base_type):
            try:
                return base_type() == pl.Datetime
            except TypeError:
                return False
        return "Datetime" in str(dtype)

    def _is_time_dtype(self, dtype: pl.DataType) -> bool:
        if dtype == pl.Time:
            return True
        base_type = getattr(dtype, "base_type", None)
        if callable(base_type):
            try:
                return base_type() == pl.Time
            except TypeError:
                return False
        return "Time" in str(dtype)

    def _is_string_like_dtype(self, dtype: pl.DataType) -> bool:
        probe = getattr(dtype, "is_string", None)
        if callable(probe):
            return bool(probe())
        if probe is not None:
            return bool(probe)
        return any(
            token in str(dtype)
            for token in (
                "String",
                "Utf8",
                "Categorical",
                "Enum",
            )
        )
