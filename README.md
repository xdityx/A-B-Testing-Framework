# A/B Testing Framework

A production-grade A/B testing framework built in Python — designed for designing, analyzing, and reporting on controlled experiments with statistical rigor.

## 🎯 Project Overview

This framework provides an end-to-end toolkit for A/B testing, covering:

| Module | Description | Status |
|--------|-------------|--------|
| **Experiment Design** | Power analysis, sample size calculation, MDE estimation | ✅ Complete |
| **Data Simulation** | Generate realistic A/B test datasets | ✅ Complete |
| **Statistical Tests** | Frequentist (z-test, t-test) & Bayesian analysis | ✅ Complete |
| **Diagnostics** | SRM checks, novelty effects, AA validation | ✅ Complete |
| **Reporting** | Interactive HTML dashboards via Plotly + Jinja2 | ✅ Complete |
| **MLflow Tracking** | Experiment logging & comparison | ✅ Complete |

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

## 🎲 Data Simulation (Stage 2)

The `data_simulation` module generates realistic experiment datasets with
binary conversion and continuous revenue metrics, timestamps, and day-of-week
traffic patterns.

### Simulate a Standard A/B Test

```python
from src.data_simulation import simulate_ab_test

df = simulate_ab_test(
    n_control=5000,
    n_treatment=5000,
    control_rate=0.10,
    treatment_rate=0.12,
    avg_revenue_per_converter=45.0,
    duration_days=30,
    seed=42,
)
```

Returns a DataFrame with columns:
`user_id` · `variant` · `converted` · `revenue` · `timestamp` · `day_of_week`

### Simulate an A/A Test

```python
from src.data_simulation import simulate_aa_test

df = simulate_aa_test(n_per_variant=5000, rate=0.10, seed=42)
```

Both variants share the same underlying rate — used to validate
no false positives exist before running a real experiment.

### Simulate a Sample Ratio Mismatch

```python
from src.data_simulation import simulate_srm_scenario

df = simulate_srm_scenario(
    total_users=10000,
    intended_split=0.50,
    actual_control_split=0.60,
)
```

Generates an intentionally imbalanced dataset to test SRM detection in
the diagnostics module.

---

## 📈 Statistical Tests (Stage 3)

The `statistical_tests` module provides frequentist and Bayesian hypothesis
testing for both binary (conversion) and continuous (revenue) metrics.

### Frequentist — Z-Test for Proportions

```python
from src.statistical_tests import z_test_proportions

result = z_test_proportions(
    control_converted=1000,
    control_total=10000,
    treatment_converted=1200,
    treatment_total=10000,
)
# result.significant   → True
# result.p_value       → 0.000017
# result.lift          → 0.20  (20% relative lift)
# result.ci_lower      → 0.013
# result.ci_upper      → 0.027
```

### Frequentist — Welch's T-Test for Continuous Metrics

```python
from src.statistical_tests import t_test_continuous
import numpy as np

rng = np.random.default_rng(42)
control   = rng.normal(45.0, 10.0, 5000)
treatment = rng.normal(50.0, 10.0, 5000)

result = t_test_continuous(control, treatment)
# result.significant   → True
# result.lift          → ~0.11  (11% lift in revenue)
# result.ci_lower/upper → CI on mean difference
```

### Bayesian — Beta-Binomial Analysis

```python
from src.statistical_tests import bayesian_ab_test

result = bayesian_ab_test(
    control_converted=800,
    control_total=10000,
    treatment_converted=1000,
    treatment_total=10000,
)
# result.prob_treatment_wins  → 0.997  (P(B > A))
# result.expected_loss        → 0.0001 (risk of choosing treatment)
# result.ci_lower / ci_upper  → 95% credible interval on difference
```

The Bayesian model uses a **Beta-Binomial conjugate** with a uniform
prior `Beta(1, 1)` by default, updated with observed data via 100,000
Monte Carlo samples from the posterior.

---

## 🩺 Diagnostics (Stage 4)

The `diagnostics` module ensures experiment data is trustworthy before interpreting results.

### Sample Ratio Mismatch (SRM) Check

Detects if the actual traffic split significantly deviates from the intended split, which usually indicates logging bugs or faulty randomization.

```python
from src.diagnostics import check_srm

result = check_srm(
    control_users=5150,
    treatment_users=4850,
    expected_control_proportion=0.50
)
# result.srm_detected → True (p < 0.001)
# result.chi2_stat    → 9.0
```

### Novelty Effect Detection

Detects if the treatment variant's success is just a temporary buzz that decays over time.

```python
from src.diagnostics import check_novelty_effect
import pandas as pd

daily_rates = pd.Series([0.12, 0.11, 0.10, 0.08, 0.07])
result = check_novelty_effect(daily_rates)
# result.novelty_detected → True
# result.trend_slope      → -0.012
```

### Multiple Testing Correction

Controls the Family-Wise Error Rate (FWER) when running multiple simultaneous hypothesis tests using the Holm-Bonferroni method.

```python
from src.diagnostics import holm_bonferroni_correction

p_values = [0.01, 0.04, 0.03, 0.001]
is_significant = holm_bonferroni_correction(p_values, alpha=0.05)
# → [True, False, False, True]
```

---

## 📋 Reporting (Stage 5)

The `reporting` module generates a self-contained HTML dashboard with
embedded interactive Plotly charts — no server required, just open the file
in any browser.

### Generate a Report

```python
from src.data_simulation import simulate_ab_test
from src.statistical_tests import z_test_proportions, bayesian_ab_test
from src.diagnostics import check_srm, check_novelty_effect
from src.reporting import generate_report
import pandas as pd

# 1. Simulate data
df = simulate_ab_test(5000, 5000, 0.10, 0.12, seed=42)

# 2. Run analyses
control = df[df["variant"] == "control"]
treatment = df[df["variant"] == "treatment"]

freq = z_test_proportions(
    control["converted"].sum(), len(control),
    treatment["converted"].sum(), len(treatment),
)
bayes = bayesian_ab_test(
    control["converted"].sum(), len(control),
    treatment["converted"].sum(), len(treatment),
)
srm = check_srm(len(control), len(treatment))

daily = df.copy()
daily["date"] = daily["timestamp"].dt.date
treatment_daily = daily[daily["variant"] == "treatment"].groupby("date")["converted"].mean()
novelty = check_novelty_effect(treatment_daily)

# 3. Generate report
path = generate_report(df, freq, bayes, srm, novelty)
# → reports/ab_test_report.html
```

The report includes:
- **Summary cards** — Lift, P-Value, P(B > A), Ship/Hold decision
- **Conversion rate chart** — Grouped bar comparison
- **Revenue distribution** — Box plot for converters
- **Daily trend** — Line chart tracking conversion over time
- **Bayesian posterior** — Probability of winning visualization
- **Diagnostic alerts** — SRM and novelty status indicators

---

## 📦 MLflow Tracking (Stage 6)

The `mlflow_tracking` module logs experiment parameters, metrics, and
artifacts to MLflow for reproducibility and comparison across runs.

### Log an Experiment

```python
from mlflow_tracking.log_experiment import log_experiment

run_id = log_experiment(
    experiment_name="checkout_button_test",
    baseline_rate=0.10,
    mde=0.05,
    sample_size_per_variant=5000,
    freq_result=freq,
    bayesian_result=bayes,
    srm_result=srm,
    report_path="reports/ab_test_report.html",
)
```

**Logged to MLflow:**
- **Parameters** — baseline_rate, mde, sample_size, alpha
- **Metrics** — p_value, lift, P(B > A), expected_loss, SRM status
- **Tags** — ship/hold decision, SRM pass/fail
- **Artifacts** — HTML report (if provided)

### Compare Past Experiments

```python
from mlflow_tracking.log_experiment import compare_experiments

df = compare_experiments("checkout_button_test")
```

---

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
