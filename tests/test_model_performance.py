from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from ai_observability_qa.model_performance import (
    ModelPerformanceResult,
    calculate_classification_metrics,
    calculate_sliced_metrics,
    validate_error_rates,
    validate_global_performance,
    validate_sliced_performance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "test_data" / "churn_model_predictions.csv"
DEGRADED_DATASET_PATH = PROJECT_ROOT / "test_data" / "churn_model_predictions_degraded.csv"
RULES_PATH = PROJECT_ROOT / "config" / "model_performance_rules.json"


def load_rules() -> dict:
    with RULES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def format_performance_failures(results: list[ModelPerformanceResult]) -> str:
    failures = [r for r in results if not r.passed]
    lines = ["Model Performance validation failed:"]
    for failure in failures:
        lines.append(f"- {failure.check_name}: {failure.details}")
    return "\n".join(lines)


def test_model_meets_global_performance_thresholds():
    """Verify that model predictions satisfy global QA performance criteria."""
    dataframe = pd.read_csv(DATASET_PATH)
    rules = load_rules()

    assert len(dataframe) >= rules["min_sample_size"], (
        f"Validation sample size {len(dataframe)} is smaller than required minimum {rules['min_sample_size']}."
    )

    metrics = calculate_classification_metrics(
        y_true=dataframe["churned"],
        y_pred=dataframe["prediction"],
        y_prob=dataframe["probability_churn"],
    )

    results = validate_global_performance(metrics, rules["thresholds"])
    failures = [r for r in results if not r.passed]

    if failures:
        pytest.fail(format_performance_failures(failures), pytrace=False)


def test_model_meets_error_rate_limits():
    """Verify operational QA bounds on false positive rate and false negative rate."""
    dataframe = pd.read_csv(DATASET_PATH)
    rules = load_rules()

    metrics = calculate_classification_metrics(
        y_true=dataframe["churned"],
        y_pred=dataframe["prediction"],
        y_prob=dataframe["probability_churn"],
    )

    error_limits = rules.get("error_rate_limits", {})
    results = validate_error_rates(
        metrics,
        max_false_positive_rate=error_limits.get("max_false_positive_rate"),
        max_false_negative_rate=error_limits.get("max_false_negative_rate"),
    )

    failures = [r for r in results if not r.passed]
    if failures:
        pytest.fail(format_performance_failures(failures), pytrace=False)


def test_model_slice_performance_across_segments():
    """Verify performance consistency across customer subgroups to avoid hidden degradation."""
    dataframe = pd.read_csv(DATASET_PATH)
    rules = load_rules()
    slice_config = rules["slice_evaluations"]

    sliced_metrics = calculate_sliced_metrics(
        dataframe=dataframe,
        slice_column=slice_config["slice_column"],
        y_true_column="churned",
        y_pred_column="prediction",
        y_prob_column="probability_churn",
    )

    results = validate_sliced_performance(
        sliced_metrics=sliced_metrics,
        slice_rules=slice_config["thresholds"],
        slice_column=slice_config["slice_column"],
        min_slice_samples=slice_config["min_slice_samples"],
    )

    failures = [r for r in results if not r.passed]
    if failures:
        pytest.fail(format_performance_failures(failures), pytrace=False)


def test_degraded_model_fails_performance_gate():
    """Ensure that QA test suite actively catches model performance regression."""
    dataframe = pd.read_csv(DEGRADED_DATASET_PATH)
    rules = load_rules()

    metrics = calculate_classification_metrics(
        y_true=dataframe["churned"],
        y_pred=dataframe["prediction"],
        y_prob=dataframe["probability_churn"],
    )

    results = validate_global_performance(metrics, rules["thresholds"])
    failures = [r for r in results if not r.passed]

    # Degraded model MUST fail the QA gate
    assert len(failures) > 0, "Expected degraded model to fail performance thresholds, but it passed."
    failed_checks = [f.check_name for f in failures]
    assert any("accuracy" in check or "f1_score" in check for check in failed_checks)


def test_slice_validation_flags_insufficient_sample_size():
    """Ensure slices with insufficient samples are flagged rather than producing misleading metrics."""
    low_sample_df = pd.DataFrame({
        "contract_type": ["month-to-month", "month-to-month", "month-to-month"],
        "churned": [1, 0, 1],
        "prediction": [1, 0, 1],
        "probability_churn": [0.9, 0.1, 0.8],
    })

    sliced_metrics = calculate_sliced_metrics(
        low_sample_df, "contract_type", "churned", "prediction", "probability_churn"
    )

    results = validate_sliced_performance(
        sliced_metrics=sliced_metrics,
        slice_rules={"accuracy": {"min": 0.70}},
        slice_column="contract_type",
        min_slice_samples=10,
    )

    assert len(results) == 1
    assert results[0].passed is False
    assert "insufficient for QA confidence" in results[0].details


def test_mismatched_prediction_lengths_raises_error():
    """Defensive check for length mismatch between ground truth and predictions."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_classification_metrics(y_true=[1, 0, 1], y_pred=[1, 0])

