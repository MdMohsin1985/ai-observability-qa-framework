from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class FeatureDriftResult:
    """Statistical drift test result for a single feature."""

    feature_name: str
    method: str
    test_statistic: float
    p_value: float | None
    threshold: float
    drift_detected: bool
    details: str


@dataclass(frozen=True)
class DatasetDriftResult:
    """Aggregated dataset-level drift result across all feature evaluations."""

    total_features_tested: int
    drifted_features_count: int
    drift_share: float
    drift_share_threshold: float
    dataset_drift_detected: bool
    feature_results: list[FeatureDriftResult]
    details: str


def calculate_psi(
    baseline: Sequence[float] | np.ndarray | pd.Series,
    current: Sequence[float] | np.ndarray | pd.Series,
    num_bins: int | None = None,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between baseline and current distributions.

    Benchmarks:
    - PSI < 0.10: Stable / No significant distribution change.
    - 0.10 <= PSI < 0.20: Moderate distribution shift (warning).
    - PSI >= 0.20: Significant data drift (critical alert).
    """
    baseline_arr = np.asarray(baseline, dtype=float)
    current_arr = np.asarray(current, dtype=float)

    baseline_clean = baseline_arr[~np.isnan(baseline_arr)]
    current_clean = current_arr[~np.isnan(current_arr)]

    if len(baseline_clean) == 0 or len(current_clean) == 0:
        return 0.0

    if num_bins is None:
        num_bins = 5 if len(baseline_clean) < 300 else 10

    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.percentile(baseline_clean, quantiles * 100)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    base_counts, _ = np.histogram(baseline_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(current_clean, bins=bin_edges)

    base_pct = base_counts / len(baseline_clean)
    curr_pct = curr_counts / len(current_clean)

    base_pct = np.where(base_pct == 0, epsilon, base_pct)
    curr_pct = np.where(curr_pct == 0, epsilon, curr_pct)

    psi_value = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(psi_value)


def detect_numerical_drift_ks(
    baseline: Sequence[float] | pd.Series,
    current: Sequence[float] | pd.Series,
    feature_name: str,
    p_value_threshold: float = 0.05,
) -> FeatureDriftResult:
    """Two-sample Kolmogorov-Smirnov (KS) test for numerical distribution shift.

    Null hypothesis H0: Both samples are drawn from the same continuous distribution.
    If p-value < threshold, reject H0 (drift detected).
    """
    b_clean = pd.Series(baseline).dropna().values
    c_clean = pd.Series(current).dropna().values

    res = stats.ks_2samp(b_clean, c_clean)
    stat = float(res.statistic)
    p_val = float(res.pvalue)

    drift_detected = p_val < p_value_threshold
    details = (
        f"KS stat = {stat:.4f}, p-value = {p_val:.4f} "
        f"({'drift detected' if drift_detected else 'stable'}, threshold p < {p_value_threshold})"
    )

    return FeatureDriftResult(
        feature_name=feature_name,
        method="ks_test",
        test_statistic=stat,
        p_value=p_val,
        threshold=p_value_threshold,
        drift_detected=drift_detected,
        details=details,
    )


def detect_numerical_drift_psi(
    baseline: Sequence[float] | pd.Series,
    current: Sequence[float] | pd.Series,
    feature_name: str,
    psi_threshold: float = 0.20,
) -> FeatureDriftResult:
    """Population Stability Index (PSI) test on continuous features."""
    psi = calculate_psi(baseline, current)
    drift_detected = psi >= psi_threshold

    details = (
        f"PSI = {psi:.4f} "
        f"({'drift detected' if drift_detected else 'stable'}, threshold PSI >= {psi_threshold})"
    )

    return FeatureDriftResult(
        feature_name=feature_name,
        method="psi",
        test_statistic=psi,
        p_value=None,
        threshold=psi_threshold,
        drift_detected=drift_detected,
        details=details,
    )


def detect_categorical_drift_chisquare(
    baseline: Sequence[Any] | pd.Series,
    current: Sequence[Any] | pd.Series,
    feature_name: str,
    p_value_threshold: float = 0.05,
) -> FeatureDriftResult:
    """Two-way Chi-Square test of independence on categorical distributions."""
    b_series = pd.Series(baseline).dropna()
    c_series = pd.Series(current).dropna()

    all_categories = sorted(list(set(b_series.unique()).union(set(c_series.unique()))))

    b_counts = b_series.value_counts().reindex(all_categories, fill_value=0).values
    c_counts = c_series.value_counts().reindex(all_categories, fill_value=0).values

    contingency_table = np.array([b_counts, c_counts])
    res = stats.chi2_contingency(contingency_table)

    stat = float(res.statistic)
    p_val = float(res.pvalue)

    drift_detected = p_val < p_value_threshold
    details = (
        f"Chi2 stat = {stat:.4f}, p-value = {p_val:.4f} "
        f"({'drift detected' if drift_detected else 'stable'}, threshold p < {p_value_threshold})"
    )

    return FeatureDriftResult(
        feature_name=feature_name,
        method="chi_square",
        test_statistic=stat,
        p_value=p_val,
        threshold=p_value_threshold,
        drift_detected=drift_detected,
        details=details,
    )

