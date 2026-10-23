from __future__ import annotations

from typing import Any, Mapping
import pandas as pd

from ai_observability_qa.drift.detectors import (
    DatasetDriftResult,
    FeatureDriftResult,
    detect_categorical_drift_chisquare,
    detect_numerical_drift_ks,
    detect_numerical_drift_psi,
)


def validate_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_rules: Mapping[str, Any],
) -> list[FeatureDriftResult]:
    """Validate drift across all configured numerical and categorical features."""
    results: list[FeatureDriftResult] = []

    # 1. Numerical Features
    num_features = drift_rules.get("numerical_features", {})
    for feat_name, config in num_features.items():
        if feat_name not in baseline_df.columns or feat_name not in current_df.columns:
            continue

        method = config.get("method", "ks_test")
        threshold = config.get("threshold", 0.05 if method == "ks_test" else 0.20)

        if method == "psi":
            res = detect_numerical_drift_psi(
                baseline=baseline_df[feat_name],
                current=current_df[feat_name],
                feature_name=feat_name,
                psi_threshold=threshold,
            )
        else:
            res = detect_numerical_drift_ks(
                baseline=baseline_df[feat_name],
                current=current_df[feat_name],
                feature_name=feat_name,
                p_value_threshold=threshold,
            )
        results.append(res)

    # 2. Categorical Features
    cat_features = drift_rules.get("categorical_features", {})
    for feat_name, config in cat_features.items():
        if feat_name not in baseline_df.columns or feat_name not in current_df.columns:
            continue

        threshold = config.get("threshold", 0.05)
        res = detect_categorical_drift_chisquare(
            baseline=baseline_df[feat_name],
            current=current_df[feat_name],
            feature_name=feat_name,
            p_value_threshold=threshold,
        )
        results.append(res)

    return results


def validate_dataset_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_rules: Mapping[str, Any],
) -> DatasetDriftResult:
    """Validate dataset-level drift based on aggregated feature test results."""
    feature_results = validate_feature_drift(baseline_df, current_df, drift_rules)

    total_tested = len(feature_results)
    drifted_count = sum(1 for f in feature_results if f.drift_detected)
    drift_share = (drifted_count / total_tested) if total_tested > 0 else 0.0

    threshold_share = drift_rules.get("max_drifted_features_ratio", 0.33)
    dataset_drift_detected = drift_share > threshold_share

    details = (
        f"{drifted_count}/{total_tested} features drifted ({drift_share:.1%}). "
        f"Allowed threshold: {threshold_share:.1%}. "
        f"Status: {'DATASET DRIFT DETECTED' if dataset_drift_detected else 'DATASET STABLE'}."
    )

    return DatasetDriftResult(
        total_features_tested=total_tested,
        drifted_features_count=drifted_count,
        drift_share=drift_share,
        drift_share_threshold=threshold_share,
        dataset_drift_detected=dataset_drift_detected,
        feature_results=feature_results,
        details=details,
    )


def format_drift_failures(dataset_result: DatasetDriftResult) -> str:
    """Format readable diagnostic failure message detailing drifted features."""
    lines = [f"Data Drift Validation Failed: {dataset_result.details}"]
    drifted_features = [f for f in dataset_result.feature_results if f.drift_detected]
    for feat in drifted_features:
        lines.append(f"- Feature '{feat.feature_name}' ({feat.method}): {feat.details}")
    return "\n".join(lines)

