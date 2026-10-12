# AI Observability QA Framework

A modular, production-style QA framework for validating, testing, and monitoring AI and Machine Learning systems across their lifecycle.

Each capability in the framework is self-contained with its own rules, evaluation logic, automated test suites, and dedicated documentation.

---

## Capabilities & Roadmap

| Capability | Status | Documentation | Focus Areas |
| :--- | :---: | :--- | :--- |
| **Data Quality Testing** | Complete | [Data Quality Guide](docs/capabilities/data_quality.md) | Schema validation, pandas dtypes, null constraints, numeric ranges, category checks |
| **ML Model Performance Testing** | Complete | [Model Performance Guide](docs/capabilities/model_performance.md) | Classification metrics (Accuracy, F1, ROC-AUC), FPR/FNR bounds, subgroup slices, regression gates |
| **Data Drift Detection** | Next Up | *Planned* | KS-test, Population Stability Index (PSI), Chi-Square test for feature shift |
| **Model Drift & Degradation** | Planned | *Planned* | Tracking prediction drift and ground-truth concept decay over time |
| **AI Observability (Logs, Metrics, Traces)** | Planned | *Planned* | OpenTelemetry instrumentation, inference latency, token tracing |
| **Monitoring & Alert Validation** | Planned | *Planned* | Threshold triggers, alert rule verification, synthetic degradation alarms |
| **LLM Observability & Evaluation** | Planned | *Planned* | Prompt injection, hallucination checks, RAG evaluation, token cost tracking |
| **Explainability (SHAP / LIME)** | Planned | *Planned* | Feature attribution consistency, model interpretability checks |
| **Bias & Fairness Testing** | Planned | *Planned* | Disparate impact, demographic parity, equalized odds across protected classes |
| **Robustness & Adversarial Testing** | Planned | *Planned* | Data perturbation, noise injection, boundary and stress testing |
| **MLOps & Model Versioning** | Planned | *Planned* | Model registry checks, artifact metadata, reproducibility verification |
| **Traceability, Auditability & Compliance** | Planned | *Planned* | Audit logs, compliance evidence, data lineage tracking |
| **Automated QA Reporting** | Active | [Reports Hub](reports/) | Markdown summaries, interactive HTML dashboards, JUnit XML |

---

## Project Structure

```text
ai-observability-qa-framework/
├── config/                                 # JSON rule configurations
│   ├── data_quality_rules.json
│   └── model_performance_rules.json
├── docs/                                   # Capability-specific documentation & assets
│   ├── capabilities/
│   │   ├── data_quality.md
│   │   └── model_performance.md
│   └── images/
│       ├── data_quality_flowchart.png
│       └── model_performance_flowchart.png
├── reports/                                # Generated QA outputs
│   ├── test-results.xml
│   ├── model-performance-results.xml
│   ├── model_performance_report.md
│   └── model_performance_report.html
├── src/
│   └── ai_observability_qa/
│       ├── __init__.py
│       ├── data_quality/                   # Data Quality Validation
│       │   ├── __init__.py
│       │   └── validators.py
│       ├── model_performance/              # ML Model Performance Testing
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── validators.py
│       └── reporting/                      # Automated report generator
│           ├── __init__.py
│           └── generator.py
├── test_data/                              # Benchmark & test datasets
│   ├── sample_customer_churn.csv
│   ├── churn_model_predictions.csv
│   └── churn_model_predictions_degraded.csv
├── tests/                                  # Pytest automated test suites
│   ├── test_data_quality.py
│   └── test_model_performance.py
├── pytest.ini
├── requirements.txt
└── README.md                               # Framework overview & capability index
```

---

## Getting Started

### 1. Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### 2. Running Automated Tests

```bash
# Run the entire test suite
pytest

# Run a specific capability suite
pytest tests/test_data_quality.py
pytest tests/test_model_performance.py

# Generate HTML test report
pytest tests/test_model_performance.py --html=reports/model_performance_pytest_report.html --self-contained-html

# Generate JUnit XML test report
python -m pytest tests/test_model_performance.py --junitxml=reports/model-performance-results.xml
```

---

## Detailed Capability Guides

For in-depth explanations of inputs, rules, logic engines, flowcharts, and test designs, refer to each capability guide:

- **[Data Quality Testing Guide](docs/capabilities/data_quality.md)** — Schema, dtypes, nulls, ranges, and allowed categories.
- **[Model Performance Testing Guide](docs/capabilities/model_performance.md)** — Metrics, subgroup slices, error bounds, and regression gates.
