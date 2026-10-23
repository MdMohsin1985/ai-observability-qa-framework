from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from ai_observability_qa.drift import (
    calculate_psi,
    detect_categorical_drift_chisquare,
    detect_numerical_drift_ks,
    detect_numerical_drift_psi,
    format_drift_failures,
    validate_dataset_drift,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = PROJECT_ROOT / "config" / "data_drift_rules.json"
BASELINE_PATH = PROJECT_ROOT / "test_data" / "baseline_churn_reference.csv"
STABLE_PATH = PROJECT_ROOT / "test_data" / "production_churn_current_stable.csv"
DRIFTED_PATH = PROJECT_ROOT / "test_data" / "production_churn_current_drifted.csv"


def load_rules() -> dict:
    with RULES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def test_stable_production_data_passes_dataset_drift_gate():
    """Verify that stable production inference data passes dataset drift checks."""
    baseline_df = pd.read_csv(BASELINE_PATH)
    stable_df = pd.read_csv(STABLE_PATH)
    rules = load_rules()

    result = validate_dataset_drift(baseline_df, stable_df, rules)

    assert result.dataset_drift_detected is False, format_drift_failures(result)
    assert result.drifted_features_count == 0


def test_drifted_production_data_triggers_dataset_drift_gate():
    """Verify that significant feature distribution shift triggers dataset drift alerts."""
    baseline_df = pd.read_csv(BASELINE_PATH)
    drifted_df = pd.read_csv(DRIFTED_PATH)
    rules = load_rules()

    result = validate_dataset_drift(baseline_df, drifted_df, rules)

    assert result.dataset_drift_detected is True
    assert result.drifted_features_count > 0
    assert result.drift_share > rules["max_drifted_features_ratio"]


def test_ks_test_detects_age_distribution_shift():
    """Verify Kolmogorov-Smirnov test detects significant numerical distribution shift."""
    baseline_df = pd.read_csv(BASELINE_PATH)
    drifted_df = pd.read_csv(DRIFTED_PATH)

    res = detect_numerical_drift_ks(
        baseline=baseline_df["age"],
        current=drifted_df["age"],
        feature_name="age",
        p_value_threshold=0.05,
    )

    assert res.drift_detected is True
    assert res.p_value is not None
    assert res.p_value < 0.05


def test_psi_detects_tenure_distribution_shift():
    """Verify Population Stability Index (PSI) detects severe compression in tenure."""
    baseline_df = pd.read_csv(BASELINE_PATH)
    drifted_df = pd.read_csv(DRIFTED_PATH)

    res = detect_numerical_drift_psi(
        baseline=baseline_df["tenure_months"],
        current=drifted_df["tenure_months"],
        feature_name="tenure_months",
        psi_threshold=0.20,
    )

    assert res.drift_detected is True
    assert res.test_statistic >= 0.20


def test_chi_square_detects_contract_type_categorical_drift():
    """Verify Chi-Square contingency test detects categorical composition shifts."""
    baseline_df = pd.read_csv(BASELINE_PATH)
    drifted_df = pd.read_csv(DRIFTED_PATH)

    res = detect_categorical_drift_chisquare(
        baseline=baseline_df["contract_type"],
        current=drifted_df["contract_type"],
        feature_name="contract_type",
        p_value_threshold=0.05,
    )

    assert res.drift_detected is True
    assert res.p_value is not None
    assert res.p_value < 0.05


def test_psi_identical_distributions_yields_zero():
    """Verify mathematical consistency: PSI equals 0 for identical datasets."""
    sample_data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    psi = calculate_psi(sample_data, sample_data)
    assert pytest.approx(psi, abs=1e-5) == 0.0

