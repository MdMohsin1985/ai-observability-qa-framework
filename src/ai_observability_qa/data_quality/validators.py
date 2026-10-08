from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

MAX_REPORTED_ROWS = 20


@dataclass(frozen=True)
class DataQualityResult:
    """Result for a single data-quality validation."""

    check_name: str
    passed: bool
    details: str


def validate_required_columns(
    dataframe: pd.DataFrame, required_columns: list[str]
) -> DataQualityResult:
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        return DataQualityResult(
            check_name="required_columns",
            passed=False,
            details=f"Missing required columns: {missing_columns}",
        )

    return DataQualityResult(
        check_name="required_columns",
        passed=True,
        details="All required columns are present.",
    )


def validate_column_types(
    dataframe: pd.DataFrame, expected_types: dict[str, str]
) -> DataQualityResult:
    mismatches = {}

    for column, expected_type in expected_types.items():
        if column not in dataframe.columns:
            mismatches[column] = {
                "expected": expected_type,
                "actual": "missing",
            }
            continue

        actual_type = str(dataframe[column].dtype)
        if not _dtype_matches(actual_type, expected_type):
            mismatches[column] = {
                "expected": expected_type,
                "actual": actual_type,
            }

    if mismatches:
        return DataQualityResult(
            check_name="column_types",
            passed=False,
            details=f"Column type mismatches: {mismatches}",
        )

    return DataQualityResult(
        check_name="column_types",
        passed=True,
        details="All checked columns have expected pandas dtypes.",
    )


def _dtype_matches(actual_type: str, expected_type: str) -> bool:
    if actual_type == expected_type:
        return True

    string_type_aliases = {"object", "str", "string"}
    return actual_type in string_type_aliases and expected_type in string_type_aliases


def validate_non_null_columns(
    dataframe: pd.DataFrame, non_null_columns: list[str]
) -> DataQualityResult:
    null_counts = {}

    for column in non_null_columns:
        if column not in dataframe.columns:
            null_counts[column] = "missing"
            continue

        null_count = int(dataframe[column].isna().sum())
        if null_count > 0:
            null_counts[column] = null_count

    if null_counts:
        return DataQualityResult(
            check_name="non_null_columns",
            passed=False,
            details=f"Null values found: {null_counts}",
        )

    return DataQualityResult(
        check_name="non_null_columns",
        passed=True,
        details="All checked columns are non-null.",
    )


def validate_numeric_ranges(
    dataframe: pd.DataFrame, numeric_ranges: dict[str, dict[str, int | float]]
) -> DataQualityResult:
    violations = {}

    for column, limits in numeric_ranges.items():
        if column not in dataframe.columns:
            violations[column] = "missing"
            continue

        minimum = limits.get("min")
        maximum = limits.get("max")
        invalid_mask = pd.Series(False, index=dataframe.index)

        if minimum is not None:
            invalid_mask |= dataframe[column] < minimum

        if maximum is not None:
            invalid_mask |= dataframe[column] > maximum

        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            invalid_rows = _format_invalid_rows(dataframe, column, invalid_mask)
            violations[column] = {
                "invalid_count": invalid_count,
                "min": minimum,
                "max": maximum,
                "rows": invalid_rows,
            }

    if violations:
        return DataQualityResult(
            check_name="numeric_ranges",
            passed=False,
            details=f"Numeric range violations: {violations}",
        )

    return DataQualityResult(
        check_name="numeric_ranges",
        passed=True,
        details="All checked numeric values are within configured ranges.",
    )


def _format_invalid_rows(
    dataframe: pd.DataFrame, column: str, invalid_mask: pd.Series
) -> list[dict[str, Any]]:
    invalid_records = []
    invalid_dataframe = dataframe.loc[invalid_mask, [column]].head(MAX_REPORTED_ROWS)

    for dataframe_index, row in invalid_dataframe.iterrows():
        invalid_records.append(
            {
                "csv_row": int(dataframe.index.get_loc(dataframe_index)) + 2,
                "dataframe_index": dataframe_index,
                "value": _to_python_value(row[column]),
            }
        )

    return invalid_records


def _to_python_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()

    return value


def validate_allowed_values(
    dataframe: pd.DataFrame, allowed_values: dict[str, list[Any]]
) -> DataQualityResult:
    violations = {}

    for column, allowed in allowed_values.items():
        if column not in dataframe.columns:
            violations[column] = "missing"
            continue

        invalid_values = sorted(
            value for value in dataframe[column].dropna().unique() if value not in allowed
        )
        if invalid_values:
            violations[column] = invalid_values

    if violations:
        return DataQualityResult(
            check_name="allowed_values",
            passed=False,
            details=f"Unexpected categorical values: {violations}",
        )

    return DataQualityResult(
        check_name="allowed_values",
        passed=True,
        details="All checked values are in the allowed sets.",
    )


def validate_min_rows(dataframe: pd.DataFrame, min_rows: int) -> DataQualityResult:
    row_count = len(dataframe)

    if row_count < min_rows:
        return DataQualityResult(
            check_name="min_rows",
            passed=False,
            details=f"Expected at least {min_rows} rows, found {row_count}.",
        )

    return DataQualityResult(
        check_name="min_rows",
        passed=True,
        details=f"Dataset has {row_count} rows.",
    )
