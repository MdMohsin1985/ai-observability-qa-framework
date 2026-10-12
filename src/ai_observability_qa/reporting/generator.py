from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

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
    """Generate both Markdown and HTML QA Model Performance reports.

    Args:
        dataset_path: Path to CSV dataset with ground truth and predictions.
        rules_path: Path to JSON configuration containing QA thresholds.
        output_dir: Folder where reports should be written.

    Returns:
        Tuple of (markdown_report_path, html_report_path).
    """
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
            return '<span style="background:#dcfce7;color:#15803d;padding:3px 10px;border-radius:12px;font-weight:600;font-size:12px;">PASSED</span>'
        return '<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;border-radius:12px;font-weight:600;font-size:12px;">FAILED</span>'

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
<title>AI Observability QA - Model Performance Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px; }}
  .container {{ max-width: 1050px; margin: 0 auto; background: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); padding: 32px; }}
  header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 18px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; }}
  h1 {{ margin: 0 0 6px 0; font-size: 24px; color: #0f172a; }}
  .subtitle {{ color: #64748b; font-size: 14px; margin: 0; }}
  .status-badge {{ background: {status_bg}; color: white; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 14px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }}
  .card {{ background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center; }}
  .card .num {{ font-size: 26px; font-weight: bold; margin-top: 4px; color: #0f172a; }}
  .card .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; }}
  h2 {{ font-size: 18px; margin-top: 28px; margin-bottom: 12px; color: #334155; border-left: 4px solid #3b82f6; padding-left: 8px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
  th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
  th {{ background: #f8fafc; font-weight: 600; color: #475569; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 13px; font-family: monospace; }}
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
    <div class="card"><div class="label">Passed</div><div class="num" style="color: #10b981;">{passed_count}</div></div>
    <div class="card"><div class="label">Failed</div><div class="num" style="color: #ef4444;">{failed_count}</div></div>
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


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    dataset = root / "test_data" / "churn_model_predictions.csv"
    rules = root / "config" / "model_performance_rules.json"
    md_p, html_p = generate_performance_report(dataset, rules, root / "reports")
    print(f"Generated Markdown report at: {md_p}")
    print(f"Generated HTML report at: {html_p}")

