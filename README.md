# AI Observability QA Framework

A practical QA framework for validating and monitoring AI and ML systems.

The project is intentionally modular so capabilities can be added gradually without turning the repository into a collection of one-off scripts. The current focus is **Session 3: Data Quality Testing**.

## Planned Capabilities

- ML model performance testing
- Data quality testing
- Data drift detection
- Model drift and performance degradation checks
- AI observability with logs, metrics, and traces
- Monitoring and alert validation
- LLM observability
- Explainability checks using SHAP or LIME
- Bias and fairness testing
- Robustness and adversarial testing
- MLOps and model versioning checks
- Traceability, auditability, and compliance evidence
- Automated QA reporting

## Current Scope

This starting version includes a lightweight data-quality testing foundation:

- A Python package under `src/`
- Pytest-based automated tests under `tests/`
- A small sample ML dataset under `test_data/`
- JSON configuration for dataset expectations under `config/`
- A `reports/` folder for future generated QA outputs

The project does **not** yet implement data drift detection, model monitoring, observability dashboards, LLM observability, explainability, fairness testing, or alert validation.

## Project Structure

```text
ai-observability-qa-framework/
├── config/
│   └── data_quality_rules.json
├── reports/
│   └── .gitkeep
├── src/
│   └── ai_observability_qa/
│       └── data_quality/
│           └── validators.py
├── test_data/
│   └── sample_customer_churn.csv
├── tests/
│   └── test_data_quality.py
├── requirements.txt
├── pytest.ini
└── README.md
```

## Getting Started

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Run the tests:

```bash
pytest
python -m pytest --junitxml=reports/test-results.xml
```

## Data Quality Testing

The initial data-quality checks validate common dataset expectations:

- Required columns are present
- Columns have expected pandas dtypes
- Required fields do not contain null values
- Numeric values stay within configured ranges
- Categorical values stay within allowed sets
- Dataset row count meets a minimum threshold
