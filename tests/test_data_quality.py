import json
from pathlib import Path

import pandas as pd
import pytest

from ai_observability_qa.data_quality import (
    validate_allowed_values,
    validate_column_types,
    validate_min_rows,
    validate_non_null_columns,
    validate_numeric_ranges,
    validate_required_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "test_data" / "sample_customer_churn.csv"
DIRTY_DATASET_PATH = PROJECT_ROOT / "test_data" / "sample_customer_churn_dirty.csv"
RULES_PATH = PROJECT_ROOT / "config" / "data_quality_rules.json"


def load_rules() -> dict:
    with RULES_PATH.open(encoding="utf-8") as rules_file:
        return json.load(rules_file)


def format_data_quality_failures(results) -> str:
    failed_results = [result for result in results if not result.passed]
    lines = ["Data quality validation failed:"]

    for result in failed_results:
        lines.append(f"- {result.check_name}: {result.details}")

    return "\n".join(lines)


def test_sample_dataset_passes_configured_data_quality_rules():
    """Verify clean benchmark dataset satisfies all configured data quality expectations."""
    dataframe = pd.read_csv(DATASET_PATH)
    rules = load_rules()

    results = [
        validate_min_rows(dataframe, rules["min_rows"]),
        validate_required_columns(dataframe, rules["required_columns"]),
        validate_column_types(dataframe, rules["column_types"]),
        validate_non_null_columns(dataframe, rules["non_null_columns"]),
        validate_numeric_ranges(dataframe, rules["numeric_ranges"]),
        validate_allowed_values(dataframe, rules["allowed_values"]),
    ]

    failed_results = [result for result in results if not result.passed]
    if failed_results:
        pytest.fail(format_data_quality_failures(failed_results), pytrace=False)


def test_dirty_dataset_fails_configured_data_quality_rules():
    """Verify that intentional schema and range defects are actively caught and reported."""
    dataframe = pd.read_csv(DIRTY_DATASET_PATH)
    rules = load_rules()

    results = [
        validate_numeric_ranges(dataframe, rules["numeric_ranges"]),
        validate_allowed_values(dataframe, rules["allowed_values"]),
    ]

    failed_results = [result for result in results if not result.passed]
    assert len(failed_results) == 2, "Expected range and allowed value checks to flag dirty data"

    failed_checks = {r.check_name for r in failed_results}
    assert "numeric_ranges" in failed_checks
    assert "allowed_values" in failed_checks


def test_required_column_validation_reports_missing_columns():
    dataframe = pd.DataFrame({"age": [35], "churned": [0]})

    result = validate_required_columns(
        dataframe,
        required_columns=["age", "monthly_charges", "churned"],
    )

    assert result.passed is False
    assert "monthly_charges" in result.details


def test_numeric_range_validation_reports_out_of_range_values():
    dataframe = pd.DataFrame({"age": [17, 42, 101]})

    result = validate_numeric_ranges(
        dataframe,
        numeric_ranges={"age": {"min": 18, "max": 100}},
    )

    assert result.passed is False
    assert "invalid_count" in result.details
    assert "'csv_row': 2" in result.details
    assert "'csv_row': 4" in result.details
    assert "'value': 17" in result.details
    assert "'value': 101" in result.details
