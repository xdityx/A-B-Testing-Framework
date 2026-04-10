# A/B Testing Framework

A production-grade A/B testing framework built in Python — designed for designing, analyzing, and reporting on controlled experiments with statistical rigor.

## 🎯 Project Overview

This framework provides an end-to-end toolkit for A/B testing, covering:

| Module | Description | Status |
|--------|-------------|--------|
| **Experiment Design** | Power analysis, sample size calculation, MDE estimation | ✅ Complete |
| **Data Simulation** | Generate realistic A/B test datasets | 🔜 In Progress |
| **Statistical Tests** | Frequentist (z-test, t-test) & Bayesian analysis | 🔜 Planned |
| **Diagnostics** | SRM checks, novelty effects, AA validation | 🔜 Planned |
| **Reporting** | Interactive HTML dashboards via Plotly + Jinja2 | 🔜 Planned |
| **MLflow Tracking** | Experiment logging & comparison | 🔜 Planned |

## 📐 Architecture

```
ab_testing_framework/
├── src/
│   ├── experiment_design.py    # Power analysis & sample size calculators
│   ├── data_simulation.py      # Simulated A/B test data generation
│   ├── statistical_tests.py    # Frequentist & Bayesian hypothesis tests
│   ├── diagnostics.py          # Pre/post-experiment health checks
│   └── reporting.py            # HTML report generation
├── tests/                      # Pytest test suite
├── notebooks/                  # Jupyter notebook walkthroughs
├── mlflow_tracking/            # MLflow experiment logging
├── requirements.txt
└── Dockerfile
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/ab_testing_framework.git
cd ab_testing_framework

# Install dependencies
pip install -r requirements.txt

# Run the test suite
pytest tests/ -v
```

## 📊 Experiment Design (Stage 1)

The `experiment_design` module provides three core utilities for planning A/B tests:

### Sample Size Calculation

Calculate the required sample size per variant given a baseline conversion rate and minimum detectable effect:

```python
from src.experiment_design import calculate_sample_size

# How many users per variant to detect a 5% relative lift
# on a 10% baseline conversion rate?
n = calculate_sample_size(baseline_rate=0.10, mde=0.05)
# → 57,756 users per variant
```

### Minimum Detectable Effect

Given a fixed sample size, determine the smallest effect you can reliably detect:

```python
from src.experiment_design import calculate_mde

mde = calculate_mde(baseline_rate=0.10, sample_size=57756)
# → ~0.05 (5% relative lift)
```

### Experiment Duration

Estimate how many days the experiment needs to run:

```python
from src.experiment_design import experiment_duration_days

days = experiment_duration_days(
    sample_size_per_variant=57756,
    daily_traffic=100000,
    traffic_split=0.50
)
# → ~1.16 days
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `baseline_rate` | Control group conversion rate (0–1) | Required |
| `mde` | Minimum detectable effect as relative lift | Required |
| `alpha` | Significance level (Type I error rate) | `0.05` |
| `power` | Statistical power (1 − Type II error rate) | `0.80` |

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## 🛠️ Tech Stack

- **Python 3.11+**
- **NumPy / Pandas** — Data manipulation
- **SciPy / Statsmodels** — Statistical computations
- **Plotly** — Interactive visualizations
- **Jinja2** — HTML report templating
- **MLflow** — Experiment tracking
- **Pytest** — Testing framework

## 📄 License

MIT

---

*Built as part of a Data Science portfolio to demonstrate experimentation expertise.*
