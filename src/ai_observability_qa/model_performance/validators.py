from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import pandas as pd

from ai_observability_qa.model_performance.metrics import (
    calculate_classification_metrics,
    calculate_sliced_metrics,
)


@dataclass(frozen=True)
class ModelPerformanceResult:
    """Result for a single model performance validation check."""

    check_name: str
    passed: bool
    details: str
    metric_name: str | None = None
    actual_value: float | None = None
    expected_threshold: str | None = None


def validate_metric_threshold(
    metric_name: str,
    actual_value: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> ModelPerformanceResult:
    """Validate that a single numeric metric satisfies min/max threshold criteria."""
    if math.isnan(actual_value):
        return ModelPerformanceResult(
            check_name=f"threshold_{metric_name}",
            passed=False,
            details=f"Metric '{metric_name}' is NaN and cannot be validated.",
            metric_name=metric_name,
            actual_value=actual_value,
            expected_threshold=f"min={min_value}, max={max_value}",
        )

    failures: list[str] = []

    if min_value is not None and actual_value < min_value:
        failures.append(f"actual {actual_value:.4f} < min threshold {min_value:.4f}")

    if max_value is not None and actual_value > max_value:
        failures.append(f"actual {actual_value:.4f} > max threshold {max_value:.4f}")

    if failures:
        expected = []
        if min_value is not None:
            expected.append(f">={min_value}")
        if max_value is not None:
            expected.append(f"<={max_value}")

        return ModelPerformanceResult(
            check_name=f"threshold_{metric_name}",
            passed=False,
            details=f"Metric '{metric_name}' failed threshold: {'; '.join(failures)}.",
            metric_name=metric_name,
            actual_value=actual_value,
            expected_threshold=" and ".join(expected),
        )

    return ModelPerformanceResult(
        check_name=f"threshold_{metric_name}",
        passed=True,
        details=f"Metric '{metric_name}' satisfied thresholds (actual: {actual_value:.4f}).",
        metric_name=metric_name,
        actual_value=actual_value,
        expected_threshold=f"min={min_value}, max={max_value}",
    )


def validate_global_performance(
    metrics: Mapping[str, float],
    threshold_rules: Mapping[str, Mapping[str, float]],
) -> list[ModelPerformanceResult]:
    """Validate overall model metrics against configured min/max thresholds.

    Args:
        metrics: Computed metrics dictionary.
        threshold_rules: Map of metric name to {"min": ..., "max": ...}.
    """
    results: list[ModelPerformanceResult] = []

    for metric_name, rule in threshold_rules.items():
        if metric_name not in metrics:
            results.append(
                ModelPerformanceResult(
                    check_name=f"threshold_{metric_name}",
                    passed=False,
                    details=f"Expected metric '{metric_name}' was not found in computed metrics.",
                    metric_name=metric_name,
                )
            )
            continue

        min_val = rule.get("min")
        max_val = rule.get("max")
        actual_val = metrics[metric_name]

        results.append(
            validate_metric_threshold(
                metric_name=metric_name,
                actual_value=actual_val,
                min_value=min_val,
                max_value=max_val,
            )
        )

    return results


def validate_sliced_performance(
    sliced_metrics: Mapping[Any, Mapping[str, float]],
    slice_rules: Mapping[str, Mapping[str, float]],
    slice_column: str,
    min_slice_samples: int = 5,
) -> list[ModelPerformanceResult]:
    """Validate that individual data slices/segments meet minimum performance standards.

    Ensures that high overall performance does not mask unacceptable degradation
    in critical subgroups.
    """
    results: list[ModelPerformanceResult] = []

    for slice_val, metrics in sliced_metrics.items():
        sample_count = int(metrics.get("total_samples", 0))
        if sample_count < min_slice_samples:
            results.append(
                ModelPerformanceResult(
                    check_name=f"slice_sample_size_{slice_column}_{slice_val}",
                    passed=False,
                    details=(
                        f"Slice '{slice_column}={slice_val}' has only {sample_count} samples "
                        f"(minimum required: {min_slice_samples}) - insufficient for QA confidence."
                    ),
                )
            )
            continue

        for metric_name, rule in slice_rules.items():
            if metric_name not in metrics:
                continue

            min_val = rule.get("min")
            max_val = rule.get("max")
            actual_val = metrics[metric_name]

            res = validate_metric_threshold(
                metric_name=f"{slice_column}:{slice_val}:{metric_name}",
                actual_value=actual_val,
                min_value=min_val,
                max_value=max_val,
            )
            results.append(res)

    return results


def validate_error_rates(
    metrics: Mapping[str, float],
    max_false_positive_rate: float | None = None,
    max_false_negative_rate: float | None = None,
) -> list[ModelPerformanceResult]:
    """Validate specific QA operational boundaries on FPR and FNR."""
    results: list[ModelPerformanceResult] = []

    if max_false_positive_rate is not None:
        fpr = metrics.get("false_positive_rate", 0.0)
        results.append(
            validate_metric_threshold(
                metric_name="false_positive_rate",
                actual_value=fpr,
                max_value=max_false_positive_rate,
            )
        )

    if max_false_negative_rate is not None:
        fnr = metrics.get("false_negative_rate", 0.0)
        results.append(
            validate_metric_threshold(
                metric_name="false_negative_rate",
                actual_value=fnr,
                max_value=max_false_negative_rate,
            )
        )

    return results

