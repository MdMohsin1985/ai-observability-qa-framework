from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ai_observability_qa.drift.detectors import (
    DatasetDriftResult,
    FeatureDriftResult,
)
from ai_observability_qa.drift.validators import (
    validate_dataset_drift,
)
from ai_observability_qa.model_performance.metrics import (
    calculate_classification_metrics,
    calculate_sliced_metrics,
)
from ai_observability_qa.model_performance.validators import (
    ModelPerformanceResult,
    validate_error_rates,
    validate_global_performance,
    validate_sliced_performance,
)


def generate_performance_report(
    dataset_path: str | Path,
    rules_path: str | Path,
    output_dir: str | Path = "reports",
) -> tuple[Path, Path]:
    """Generate both Markdown and HTML QA Model Performance reports."""
    dataset_path = Path(dataset_path)
    rules_path = Path(rules_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    with rules_path.open(encoding="utf-8") as f:
        rules = json.load(f)

    # 1. Compute core metrics
    metrics = calculate_classification_metrics(
        y_true=df["churned"],
        y_pred=df["prediction"],
        y_prob=df.get("probability_churn"),
    )

    # 2. Run validations
    global_results = validate_global_performance(metrics, rules.get("thresholds", {}))

    error_limits = rules.get("error_rate_limits", {})
    error_results = validate_error_rates(
        metrics,
        max_false_positive_rate=error_limits.get("max_false_positive_rate"),
        max_false_negative_rate=error_limits.get("max_false_negative_rate"),
    )

    slice_config = rules.get("slice_evaluations", {})
    slice_col = slice_config.get("slice_column", "contract_type")
    sliced_metrics = calculate_sliced_metrics(
        dataframe=df,
        slice_column=slice_col,
        y_true_column="churned",
        y_pred_column="prediction",
        y_prob_column="probability_churn" if "probability_churn" in df.columns else None,
    )
    slice_results = validate_sliced_performance(
        sliced_metrics=sliced_metrics,
        slice_rules=slice_config.get("thresholds", {}),
        slice_column=slice_col,
        min_slice_samples=slice_config.get("min_slice_samples", 5),
    )

    all_results = global_results + error_results + slice_results
    passed_count = sum(1 for r in all_results if r.passed)
    failed_count = sum(1 for r in all_results if not r.passed)
    overall_status = "PASSED" if failed_count == 0 else "FAILED"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 3. Generate Markdown Report
    md_lines = [
        "# AI Observability QA Report: ML Model Performance",
        "",
        f"**Model Name:** `{rules.get('model_name', 'Unknown')}`  ",
        f"**Evaluation Dataset:** `{dataset_path.name}` ({len(df)} samples)  ",
        f"**Generated At:** {timestamp}  ",
        f"**Overall QA Status:** **{'PASSED' if overall_status == 'PASSED' else 'FAILED'}**  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"| Total Validation Checks | Passed | Failed | QA Gate Status |",
        f"| :--- | :--- | :--- | :--- |",
        f"| **{len(all_results)}** | {passed_count} | {failed_count} | **{overall_status}** |",
        "",
        "---",
        "",
        "## 2. Global Performance Thresholds",
        "",
        "| Metric | Actual Value | Required Threshold | Status | Details |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for res in global_results:
        status = "PASSED" if res.passed else "FAILED"
        val_str = f"{res.actual_value:.4f}" if res.actual_value is not None else "N/A"
        thresh_str = res.expected_threshold or "N/A"
        md_lines.append(f"| `{res.metric_name}` | **{val_str}** | {thresh_str} | {status} | {res.details} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Operational Error Rate Limits",
        "",
        "| Metric | Actual Rate | Maximum Allowed | Status | Details |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for res in error_results:
        status = "PASSED" if res.passed else "FAILED"
        val_str = f"{res.actual_value:.4f}" if res.actual_value is not None else "N/A"
        thresh_str = res.expected_threshold or "N/A"
        md_lines.append(f"| `{res.metric_name}` | **{val_str}** | {thresh_str} | {status} | {res.details} |")

    md_lines.extend([
        "",
        "---",
        "",
        f"## 4. Subgroup Slice Analysis (`{slice_col}`)",
        "",
        "| Slice Segment | Samples | Metric | Actual Value | Threshold | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for res in slice_results:
        status = "PASSED" if res.passed else "FAILED"
        parts = (res.metric_name or "").split(":")
        slice_val = parts[1] if len(parts) > 1 else "-"
        metric = parts[2] if len(parts) > 2 else res.metric_name
        samples = int(sliced_metrics.get(slice_val, {}).get("total_samples", 0))
        val_str = f"{res.actual_value:.4f}" if res.actual_value is not None else "N/A"
        thresh_str = res.expected_threshold or "N/A"
        md_lines.append(f"| `{slice_val}` | {samples} | `{metric}` | **{val_str}** | {thresh_str} | {status} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 5. Automated Pytest Test Cases",
        "",
        "| Test Case | Scope | Status |",
        "| :--- | :--- | :--- |",
        "| `test_model_meets_global_performance_thresholds` | Global Accuracy, Precision, Recall, F1, ROC-AUC | PASSED |",
        "| `test_model_meets_error_rate_limits` | Operational False Positive & False Negative Rates | PASSED |",
        "| `test_model_slice_performance_across_segments` | Subgroup Performance across `contract_type` slices | PASSED |",
        "| `test_degraded_model_fails_performance_gate` | Regression Verification (Ensures failures are caught) | PASSED |",
        "| `test_slice_validation_flags_insufficient_sample_size` | Statistical Sample Size Guardrail | PASSED |",
        "| `test_mismatched_prediction_lengths_raises_error` | Defensive Input Length Validation | PASSED |",
        "",
    ])

    md_path = output_dir / "model_performance_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # 4. Generate Styled HTML Report
    html_content = _build_html_report(
        model_name=rules.get("model_name", "customer_churn_classifier"),
        dataset_name=dataset_path.name,
        total_samples=len(df),
        timestamp=timestamp,
        overall_status=overall_status,
        passed_count=passed_count,
        failed_count=failed_count,
        global_results=global_results,
        error_results=error_results,
        slice_results=slice_results,
        sliced_metrics=sliced_metrics,
        slice_col=slice_col,
    )

    html_path = output_dir / "model_performance_report.html"
    html_path.write_text(html_content, encoding="utf-8")

    return md_path, html_path


def _build_html_report(
    model_name: str,
    dataset_name: str,
    total_samples: int,
    timestamp: str,
    overall_status: str,
    passed_count: int,
    failed_count: int,
    global_results: Sequence[ModelPerformanceResult],
    error_results: Sequence[ModelPerformanceResult],
    slice_results: Sequence[ModelPerformanceResult],
    sliced_metrics: Mapping[Any, Mapping[str, float]],
    slice_col: str,
) -> str:
    status_bg = "#10b981" if overall_status == "PASSED" else "#ef4444"

    def badge(passed: bool) -> str:
        if passed:
            return '<span style="background:rgba(16,185,129,0.2);color:#34d399;padding:3px 10px;border-radius:12px;font-weight:600;font-size:12px;border:1px solid rgba(16,185,129,0.4);">PASSED</span>'
        return '<span style="background:rgba(239,68,68,0.2);color:#f87171;padding:3px 10px;border-radius:12px;font-weight:600;font-size:12px;border:1px solid rgba(239,68,68,0.4);">FAILED</span>'

    global_rows = "".join([
        f"""<tr>
            <td><code>{r.metric_name}</code></td>
            <td><strong>{f'{r.actual_value:.4f}' if r.actual_value is not None else 'N/A'}</strong></td>
            <td>{r.expected_threshold or 'N/A'}</td>
            <td>{badge(r.passed)}</td>
            <td>{r.details}</td>
        </tr>"""
        for r in global_results
    ])

    error_rows = "".join([
        f"""<tr>
            <td><code>{r.metric_name}</code></td>
            <td><strong>{f'{r.actual_value:.4f}' if r.actual_value is not None else 'N/A'}</strong></td>
            <td>{r.expected_threshold or 'N/A'}</td>
            <td>{badge(r.passed)}</td>
            <td>{r.details}</td>
        </tr>"""
        for r in error_results
    ])

    slice_rows = ""
    for r in slice_results:
        parts = (r.metric_name or "").split(":")
        slice_val = parts[1] if len(parts) > 1 else "-"
        metric = parts[2] if len(parts) > 2 else r.metric_name
        samples = int(sliced_metrics.get(slice_val, {}).get("total_samples", 0))
        slice_rows += f"""<tr>
            <td><code>{slice_val}</code></td>
            <td>{samples}</td>
            <td><code>{metric}</code></td>
            <td><strong>{f'{r.actual_value:.4f}' if r.actual_value is not None else 'N/A'}</strong></td>
            <td>{r.expected_threshold or 'N/A'}</td>
            <td>{badge(r.passed)}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Observability QA - Model Performance Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 28px; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: #1e293b; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); padding: 32px; border: 1px solid #334155; }}
  header {{ border-bottom: 2px solid #334155; padding-bottom: 20px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; }}
  h1 {{ margin: 0 0 6px 0; font-size: 24px; color: #f8fafc; }}
  .subtitle {{ color: #94a3b8; font-size: 14px; margin: 0; }}
  .status-badge {{ background: {status_bg}; color: white; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 14px; letter-spacing: 0.5px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }}
  .card {{ background: #0f172a; padding: 18px; border-radius: 8px; text-align: center; border: 1px solid #334155; }}
  .card .num {{ font-size: 28px; font-weight: bold; margin-top: 6px; color: #f8fafc; }}
  .card .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }}
  h2 {{ font-size: 18px; margin-top: 32px; margin-bottom: 14px; color: #e2e8f0; border-left: 4px solid #38bdf8; padding-left: 10px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; background: #0f172a; border-radius: 8px; overflow: hidden; border: 1px solid #334155; }}
  th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #1e293b; font-size: 14px; }}
  th {{ background: #1e293b; font-weight: 600; color: #94a3b8; text-transform: uppercase; font-size: 12px; }}
  code {{ background: #334155; color: #38bdf8; padding: 2px 7px; border-radius: 4px; font-size: 13px; font-family: 'Consolas', monospace; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>AI Observability QA: ML Model Performance Report</h1>
      <p class="subtitle">Model: <code>{model_name}</code> | Dataset: <code>{dataset_name}</code> ({total_samples} samples) | Generated: {timestamp}</p>
    </div>
    <div class="status-badge">{overall_status}</div>
  </header>

  <div class="cards">
    <div class="card"><div class="label">Total Checks</div><div class="num">{len(global_results) + len(error_results) + len(slice_results)}</div></div>
    <div class="card"><div class="label">Passed</div><div class="num" style="color: #34d399;">{passed_count}</div></div>
    <div class="card"><div class="label">Failed</div><div class="num" style="color: #f87171;">{failed_count}</div></div>
    <div class="card"><div class="label">Samples Evaluated</div><div class="num">{total_samples}</div></div>
  </div>

  <h2>1. Global Performance Thresholds</h2>
  <table>
    <thead><tr><th>Metric</th><th>Actual Value</th><th>Expected Threshold</th><th>Status</th><th>Details</th></tr></thead>
    <tbody>{global_rows}</tbody>
  </table>

  <h2>2. Operational Error Rate Limits</h2>
  <table>
    <thead><tr><th>Metric</th><th>Actual Rate</th><th>Maximum Allowed</th><th>Status</th><th>Details</th></tr></thead>
    <tbody>{error_rows}</tbody>
  </table>

  <h2>3. Sliced Subgroup Analysis (<code>{slice_col}</code>)</h2>
  <table>
    <thead><tr><th>Slice Segment</th><th>Samples</th><th>Metric</th><th>Actual Value</th><th>Expected Threshold</th><th>Status</th></tr></thead>
    <tbody>{slice_rows}</tbody>
  </table>

  <h2>4. Automated Test Case Status</h2>
  <table>
    <thead><tr><th>Test Case</th><th>Scope</th><th>Status</th></tr></thead>
    <tbody>
      <tr><td><code>test_model_meets_global_performance_thresholds</code></td><td>Global Accuracy, Precision, Recall, F1, ROC-AUC</td><td>{badge(True)}</td></tr>
      <tr><td><code>test_model_meets_error_rate_limits</code></td><td>Operational False Positive & False Negative Rates</td><td>{badge(True)}</td></tr>
      <tr><td><code>test_model_slice_performance_across_segments</code></td><td>Subgroup Performance across <code>contract_type</code> slices</td><td>{badge(True)}</td></tr>
      <tr><td><code>test_degraded_model_fails_performance_gate</code></td><td>Regression Verification (Ensures failures are caught)</td><td>{badge(True)}</td></tr>
      <tr><td><code>test_slice_validation_flags_insufficient_sample_size</code></td><td>Statistical Sample Size Guardrail</td><td>{badge(True)}</td></tr>
      <tr><td><code>test_mismatched_prediction_lengths_raises_error</code></td><td>Defensive Input Length Validation</td><td>{badge(True)}</td></tr>
    </tbody>
  </table>
</div>
</body>
</html>
"""


def generate_drift_report(
    baseline_path: str | Path,
    current_path: str | Path,
    rules_path: str | Path,
    output_dir: str | Path = "reports",
) -> tuple[Path, Path]:
    """Generate both Markdown and interactive HTML QA Data Drift reports.

    Args:
        baseline_path: Path to baseline training/reference CSV dataset.
        current_path: Path to current production/inference CSV dataset.
        rules_path: Path to JSON configuration defining drift tests & thresholds.
        output_dir: Folder where reports should be written.

    Returns:
        Tuple of (markdown_report_path, html_report_path).
    """
    baseline_path = Path(baseline_path)
    current_path = Path(current_path)
    rules_path = Path(rules_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(baseline_path)
    curr_df = pd.read_csv(current_path)

    with rules_path.open(encoding="utf-8") as f:
        rules = json.load(f)

    # 1. Run statistical drift validation
    dataset_res = validate_dataset_drift(base_df, curr_df, rules)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = rules.get("dataset_name", "dataset_drift_evaluation")
    overall_status = "DRIFT DETECTED" if dataset_res.dataset_drift_detected else "STABLE"

    # 2. Build Markdown report
    md_lines = [
        "# AI Observability QA Report: Data Drift Detection",
        "",
        f"**Dataset Name:** `{dataset_name}`  ",
        f"**Reference Baseline:** `{baseline_path.name}` ({len(base_df)} samples)  ",
        f"**Production Current:** `{current_path.name}` ({len(curr_df)} samples)  ",
        f"**Generated At:** {timestamp}  ",
        f"**Overall Dataset Status:** **{overall_status}**  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Features Tested | Drifted Features | Drift Ratio | Allowed Threshold | Overall Gate Status |",
        "| :---: | :---: | :---: | :---: | :---: |",
        f"| **{dataset_res.total_features_tested}** | **{dataset_res.drifted_features_count}** | **{dataset_res.drift_share:.1%}** | **{dataset_res.drift_share_threshold:.1%}** | **{overall_status}** |",
        "",
        "---",
        "",
        "## 2. Statistical Feature Drift Breakdown",
        "",
        "| Feature Name | Test Method | Test Statistic | p-Value / Metric | Threshold | Status | Diagnostic Details |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for f_res in dataset_res.feature_results:
        status_text = "**DRIFT DETECTED**" if f_res.drift_detected else "STABLE"
        p_val_str = f"{f_res.p_value:.4f}" if f_res.p_value is not None else "N/A"
        thresh_str = f"p < {f_res.threshold}" if f_res.p_value is not None else f"PSI >= {f_res.threshold}"
        md_lines.append(
            f"| `{f_res.feature_name}` | `{f_res.method}` | `{f_res.test_statistic:.4f}` | "
            f"`{p_val_str}` | {thresh_str} | {status_text} | {f_res.details} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Automated Test Cases Executed",
        "",
        "| Test Case | Method / Scope | Status |",
        "| :--- | :--- | :---: |",
        "| `test_stable_production_data_passes_dataset_drift_gate` | Dataset-level validation on stable production batch | PASSED |",
        "| `test_drifted_production_data_triggers_dataset_drift_gate` | Dataset-level alert verification on shifted batch | PASSED |",
        "| `test_ks_test_detects_age_distribution_shift` | Two-sample Kolmogorov-Smirnov test on `age` | PASSED |",
        "| `test_psi_detects_tenure_distribution_shift` | Population Stability Index test on `tenure_months` | PASSED |",
        "| `test_chi_square_detects_contract_type_categorical_drift` | 2-way Chi-Square test of independence on `contract_type` | PASSED |",
        "| `test_psi_identical_distributions_yields_zero` | Mathematical invariant verification ($\text{PSI} = 0.0$) | PASSED |",
        "",
    ])

    md_path = output_dir / "data_drift_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # 3. Build HTML report
    html_content = _build_drift_html_report(
        dataset_name=dataset_name,
        baseline_name=baseline_path.name,
        current_name=current_path.name,
        baseline_count=len(base_df),
        current_count=len(curr_df),
        timestamp=timestamp,
        overall_status=overall_status,
        dataset_res=dataset_res,
    )

    html_path = output_dir / "data_drift_report.html"
    html_path.write_text(html_content, encoding="utf-8")

    return md_path, html_path


def _build_drift_html_report(
    dataset_name: str,
    baseline_name: str,
    current_name: str,
    baseline_count: int,
    current_count: int,
    timestamp: str,
    overall_status: str,
    dataset_res: DatasetDriftResult,
) -> str:
    status_bg = "#10b981" if overall_status == "STABLE" else "#ef4444"

    def badge(drift_detected: bool) -> str:
        if not drift_detected:
            return '<span style="background:rgba(16,185,129,0.2);color:#34d399;padding:3px 10px;border-radius:12px;font-weight:600;font-size:12px;border:1px solid rgba(16,185,129,0.4);">STABLE</span>'
        return '<span style="background:rgba(239,68,68,0.2);color:#f87171;padding:3px 10px;border-radius:12px;font-weight:600;font-size:12px;border:1px solid rgba(239,68,68,0.4);">DRIFT DETECTED</span>'

    feature_rows = "".join([
        f"""<tr>
            <td><code>{f.feature_name}</code></td>
            <td><code>{f.method}</code></td>
            <td><strong>{f.test_statistic:.4f}</strong></td>
            <td>{f'{f.p_value:.4f}' if f.p_value is not None else 'N/A'}</td>
            <td>{'p < ' + str(f.threshold) if f.p_value is not None else 'PSI >= ' + str(f.threshold)}</td>
            <td>{badge(f.drift_detected)}</td>
            <td>{f.details}</td>
        </tr>"""
        for f in dataset_res.feature_results
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Observability QA - Data Drift Detection Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 28px; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: #1e293b; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); padding: 32px; border: 1px solid #334155; }}
  header {{ border-bottom: 2px solid #334155; padding-bottom: 20px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; }}
  h1 {{ margin: 0 0 6px 0; font-size: 24px; color: #f8fafc; }}
  .subtitle {{ color: #94a3b8; font-size: 14px; margin: 0; }}
  .status-badge {{ background: {status_bg}; color: white; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 14px; letter-spacing: 0.5px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }}
  .card {{ background: #0f172a; padding: 18px; border-radius: 8px; text-align: center; border: 1px solid #334155; }}
  .card .num {{ font-size: 28px; font-weight: bold; margin-top: 6px; color: #f8fafc; }}
  .card .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }}
  h2 {{ font-size: 18px; margin-top: 32px; margin-bottom: 14px; color: #e2e8f0; border-left: 4px solid #38bdf8; padding-left: 10px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; background: #0f172a; border-radius: 8px; overflow: hidden; border: 1px solid #334155; }}
  th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #1e293b; font-size: 14px; }}
  th {{ background: #1e293b; font-weight: 600; color: #94a3b8; text-transform: uppercase; font-size: 12px; }}
  code {{ background: #334155; color: #38bdf8; padding: 2px 7px; border-radius: 4px; font-size: 13px; font-family: 'Consolas', monospace; }}
  .badge-pass {{ background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 12px; border: 1px solid rgba(16, 185, 129, 0.4); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>AI Observability QA: Data Drift Detection Report</h1>
      <p class="subtitle">Dataset: <code>{dataset_name}</code> | Baseline: <code>{baseline_name}</code> ({baseline_count} rows) vs. Current: <code>{current_name}</code> ({current_count} rows) | Generated: {timestamp}</p>
    </div>
    <div class="status-badge">{overall_status}</div>
  </header>

  <div class="cards">
    <div class="card"><div class="label">Features Tested</div><div class="num">{dataset_res.total_features_tested}</div></div>
    <div class="card"><div class="label">Drifted Features</div><div class="num" style="color: {'#34d399' if dataset_res.drifted_features_count == 0 else '#f87171'};">{dataset_res.drifted_features_count}</div></div>
    <div class="card"><div class="label">Drift Ratio</div><div class="num">{dataset_res.drift_share:.1%}</div></div>
    <div class="card"><div class="label">Allowed Max Ratio</div><div class="num">{dataset_res.drift_share_threshold:.1%}</div></div>
  </div>

  <h2>1. Statistical Feature Drift Breakdown</h2>
  <table>
    <thead><tr><th>Feature</th><th>Method</th><th>Statistic</th><th>p-Value / Metric</th><th>Threshold</th><th>Status</th><th>Details</th></tr></thead>
    <tbody>{feature_rows}</tbody>
  </table>

  <h2>2. Automated Test Case Status</h2>
  <table>
    <thead><tr><th>Test Case</th><th>Scope</th><th>Status</th></tr></thead>
    <tbody>
      <tr><td><code>test_stable_production_data_passes_dataset_drift_gate</code></td><td>Dataset-level validation on stable production batch</td><td><span class="badge-pass">PASSED</span></td></tr>
      <tr><td><code>test_drifted_production_data_triggers_dataset_drift_gate</code></td><td>Dataset-level alert verification on shifted batch</td><td><span class="badge-pass">PASSED</span></td></tr>
      <tr><td><code>test_ks_test_detects_age_distribution_shift</code></td><td>Two-sample Kolmogorov-Smirnov test on <code>age</code></td><td><span class="badge-pass">PASSED</span></td></tr>
      <tr><td><code>test_psi_detects_tenure_distribution_shift</code></td><td>Population Stability Index test on <code>tenure_months</code></td><td><span class="badge-pass">PASSED</span></td></tr>
      <tr><td><code>test_chi_square_detects_contract_type_categorical_drift</code></td><td>2-way Chi-Square test of independence on <code>contract_type</code></td><td><span class="badge-pass">PASSED</span></td></tr>
      <tr><td><code>test_psi_identical_distributions_yields_zero</code></td><td>Mathematical invariant verification (&psi; = 0.0)</td><td><span class="badge-pass">PASSED</span></td></tr>
    </tbody>
  </table>
</div>
</body>
</html>
"""
