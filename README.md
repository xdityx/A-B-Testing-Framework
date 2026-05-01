# A/B Testing Framework

[![CI](https://github.com/xdityx/A-B-Testing-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/xdityx/A-B-Testing-Framework/actions/workflows/ci.yml)

A production-grade A/B testing framework built in Python — designed for designing, analyzing, and reporting on controlled experiments with statistical rigor.

---
## 🚀 Live Demo

**Try the interactive web interface:** [A/B Testing Framework on Streamlit Cloud](https://a-b-testing-framework-mkbmuqqzyng7ivqk25qrae.streamlit.app/)

Explore experiment design, run simulations, analyze results, and see Bayesian posteriors in action.

---

## 🎯 Project Overview

This framework provides an end-to-end toolkit for A/B testing, covering:

| Module | Description | Status |
|--------|-------------|--------|
| **Experiment Design** | Power analysis, sample size calculation, MDE estimation | ✅ Complete |
| **Data Simulation** | Generate realistic A/B test datasets | ✅ Complete |
| **Statistical Tests** | Frequentist (z-test, t-test) & Bayesian analysis | ✅ Complete |
| **Diagnostics** | SRM checks, novelty effects, AA validation | ✅ Complete |
| **Reporting** | Interactive HTML dashboards via Plotly + Jinja2 | ✅ Complete |
| **Decision Integration** | Composite SHIP/HOLD/INCONCLUSIVE decision rule | ✅ Complete |
| **Sequential Testing** | Wald's SPRT for early stopping | ✅ Complete |
| **Causal Validation** | SUTVA checks, assignment balance, confounding risk | ✅ Complete |
| **Heterogeneous Effects** | Per-cohort analysis, segment-level decisions | ✅ Complete |
| **Sensitivity Analysis** | Robustness to baseline uncertainty | ✅ Complete |
| **MLflow Tracking** | Experiment logging & comparison | ✅ Complete |

## 📐 Architecture

```
ab_testing_framework/
├── src/
│   ├── experiment_design.py     # Power analysis & sample size calculators
│   ├── data_simulation.py       # Simulated A/B test data generation
│   ├── statistical_tests.py     # Frequentist & Bayesian hypothesis tests
│   ├── diagnostics.py           # Pre/post-experiment health checks
│   ├── reporting.py             # HTML report generation
│   ├── experiment_decision.py   # Composite SHIP/HOLD/INCONCLUSIVE rule
│   ├── sequential_testing.py    # Wald's SPRT for early stopping
│   ├── causal_validation.py     # SUTVA & balance validation
│   ├── heterogeneous_effects.py # Per-cohort treatment effect analysis
│   ├── sensitivity.py           # Baseline uncertainty robustness
│   └── templates/               # Jinja2 HTML templates
├── tests/                       # Pytest test suite (49 tests)
├── notebooks/                   # End-to-end 9-stage showcase notebook
├── mlflow_tracking/             # MLflow experiment logging
├── pyproject.toml               # Project metadata & tool config
├── requirements.txt
└── Dockerfile
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/xdityx/A-B-Testing-Framework.git
cd A-B-Testing-Framework

# Install dependencies
pip install -r requirements.txt

# Run the test suite
python -m pytest tests/ -v
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
# result.kendall_tau      → -1.0
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

## 🎯 Decision Integration (Stage 6)

The `experiment_decision` module fuses frequentist, Bayesian, SRM, and novelty
signals into a single actionable recommendation with calibrated confidence
and structured reasoning.

### Composite Decision Rule

```python
from src.experiment_decision import make_decision

decision = make_decision(
    freq_result=freq,
    bayes_result=bayes,
    srm_result=srm,
    novelty_result=novelty,
)
```

**Decision logic:**

- **SHIP** — SRM pass, no novelty, and strong evidence (p < 0.05 or P(B>A) > 0.95). Risk: LOW.
- **HOLD** — SRM fail, or weak evidence on both frequentist and Bayesian axes. Risk: HIGH.
- **INCONCLUSIVE** — Borderline p-value (0.05–0.10), moderate Bayesian probability (0.80–0.95), or novelty detected alongside significance. Risk: MEDIUM.

### Sample Output

```
Recommendation:   SHIP
Confidence:       0.98
Risk level:       LOW

Reasoning:
  - SRM: PASSED
  - Novelty: NOT DETECTED
  - Frequentist: p=0.0018, lift=19.68%
  - Bayesian: P(B>A)=0.9990, EL=0.0000
```

The reasoning list provides a full audit trail — every diagnostic signal is
recorded regardless of the final recommendation.

---

## ⏱️ Sequential Testing (Stage 7)

The `sequential_testing` module implements Wald's Sequential Probability Ratio
Test (SPRT), allowing experimenters to monitor results as data arrives and
stop early when the evidence is decisive — typically saving 30–50% of sample
size compared with fixed-horizon tests.

### Early Stopping with SPRT

```python
from src.sequential_testing import sequential_z_test

# Check after each day's data arrives
result = sequential_z_test(
    control_converted=80,
    control_total=1000,
    treatment_converted=120,
    treatment_total=1000,
    baseline_rate=0.10,
    mde=0.20,
    alpha=0.05,
    beta=0.20,
)
```

**How it works:** Two boundaries are derived from the desired error rates —
`log_A = log((1 − β) / α)` (upper) and `log_B = log(β / (1 − α))` (lower).
At each check-in the cumulative log-likelihood ratio is compared against
these boundaries.

### Sample Output

```
Day  1 | n=  888 | log_LR=  +3.57 | STOP_FOR_TREATMENT
Day  3 | n= 2100 | log_LR=  +1.20 | CONTINUE
Day  5 | n= 3600 | log_LR=  +2.80 | STOP_FOR_TREATMENT
```

| Field | Description |
|-------|-------------|
| `can_stop` | `True` when a boundary has been crossed |
| `stop_reason` | `CONTINUE`, `STOP_FOR_TREATMENT`, `STOP_FOR_CONTROL`, `INCONCLUSIVE` |
| `samples_needed` | Estimated additional observations to reach the upper boundary |

---

## 📊 Causal Validation (Stage 8)

The `causal_validation` module checks whether the experiment's causal
structure is sound *before* any results are interpreted. Two assumptions
matter most: SUTVA (stable unit treatment value assumption) and assignment
balance.

### Pre-Analysis Checks

```python
from src.causal_validation import validate_experiment_assumptions

result = validate_experiment_assumptions(
    df,
    randomization_level="user",
    metric_level="user",
    expected_split=0.50,
    tolerance=0.05,
)
```

### Sample Output

```
SUTVA OK:              True
Assignment balanced:   True
Confounding risk:      LOW
Notes:                 No issues detected
```

**What it catches:**

- **SUTVA violation** — Randomized at session level but measuring at user level? A single user can appear in both arms, breaking independence.
- **Assignment imbalance** — A 60/40 split when 50/50 was intended signals a randomization bug. The module flags imbalance beyond a configurable tolerance.
- **Confounding risk** — Graded LOW / MEDIUM / HIGH based on the combination of SUTVA and balance checks, with actionable notes (e.g., "Cluster at session level to fix SUTVA").

---

## 🔍 Heterogeneous Effects (Stage 9)

The `heterogeneous_effects` module segments experiment results by any
categorical column to identify where the treatment effect is strongest,
absent, or even harmful.

### Per-Cohort Analysis

```python
from src.heterogeneous_effects import analyze_by_cohort, summarize_cohorts

cohorts = analyze_by_cohort(df, cohort_column="device_type")
print(summarize_cohorts(cohorts))
```

### Sample Output

```markdown
| Cohort  | N (ctrl) | N (treat) | Lift   | p-value | Sig? |
|---------|----------|-----------|--------|---------|------|
| desktop | 1,724    | 1,769     | +28.9% | 0.0126  | Yes  |
| mobile  | 2,772    | 2,701     | +14.7% | 0.0673  | No   |
| tablet  | 504      | 530       | +23.2% | 0.2730  | No   |
```

**Why it matters:** An overall +20% lift can hide the fact that the
treatment only works on desktop. Shipping a mobile experience that
doesn't convert wastes engineering effort and may actively hurt
metrics. Cohort analysis turns a single go/no-go into a nuanced
rollout plan.

---

## 🛡️ Sensitivity Analysis (Stage 10)

The `sensitivity` module stress-tests the shipping decision by re-running
the z-test under perturbed baseline assumptions — answering the question
"If our observed baseline were off by ±2%, would we still ship?"

### Robustness Check

```python
from src.sensitivity import sensitivity_analysis, display_sensitivity

result = sensitivity_analysis(
    control_rate_observed=0.10,
    treatment_rate_observed=0.14,
    n_control=5000,
    n_treatment=5000,
    baseline_uncertainty=0.02,
)
print(display_sensitivity(result))
```

### Sample Output

```
| Baseline          | Significant |
|-------------------|-------------|
| -2% (8.00%)       | Yes         |
| ±0% (10.00%)      | Yes         |
| +2% (12.00%)      | Yes         |

Recommendation: ROBUST
Decision holds across baseline uncertainty
```

**Recommendation mapping:**

- **ROBUST** — All scenarios significant. Ship with confidence.
- **CONDITIONAL** — Two of three significant. Decision depends on accurate baseline measurement.
- **FRAGILE** — One or zero significant. Validate the baseline offline before committing.

---

## 📦 MLflow Tracking (Stage 11)

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

## 📓 Notebook Showcase (Stage 12)

The `notebooks/ab_test_showcase.py` file is a full end-to-end walkthrough
using a real-world scenario: **testing a 1-click checkout button on an
e-commerce site** ($50k implementation cost, $2M/year baseline revenue).

Open it in VS Code as a Jupyter notebook (percent format) or convert with
`jupytext`.

**Covers all 9 analysis stages:**
1. Experiment design & power analysis
2. Data simulation with timestamps, revenue, day-of-week patterns
3. Causal validation (SUTVA, balance)
4. Pre-experiment diagnostics (SRM, novelty)
5. Sequential monitoring (SPRT with boundary plot)
6. Frequentist & Bayesian analysis with posterior visualization
7. Heterogeneous effects by device cohort
8. Sensitivity analysis across baseline uncertainty
9. Composite decision with confidence and reasoning

```bash
# Run in VS Code: open the file → select "Run Cell" on each # %% block
# Or convert to .ipynb:
pip install jupytext
jupytext --to notebook notebooks/ab_test_showcase.py
```

---

## 🐳 Docker (Stage 13)

Run the full test suite in an isolated container:

```bash
docker build -t ab-testing-framework .
docker run ab-testing-framework
```

---

## ⚙️ CI/CD (Stage 14)

Automated testing and linting via GitHub Actions on every push and PR.

**Pipeline runs:**
- 🧪 **Tests** — Full pytest suite across Python 3.11 and 3.12
- 📊 **Coverage** — Coverage report uploaded as artifact
- 🧹 **Lint** — Ruff linter + formatter checks
- 📦 **Cache** — pip dependency caching for fast runs

See the [Actions tab](https://github.com/xdityx/A-B-Testing-Framework/actions) for run history.

---

## 🧪 Running Tests

The test suite contains **49 tests** covering all 11 source modules.

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing -v

# Run via Docker
docker run ab-testing-framework
```

```
tests/test_statistical_tests.py    12 passed
tests/test_diagnostics.py           5 passed
tests/test_data_simulation.py       7 passed
tests/test_experiment_decision.py   8 passed
tests/test_sequential_testing.py    8 passed
tests/test_causal_validation.py     3 passed
tests/test_heterogeneous_effects.py 2 passed
tests/test_sensitivity.py           4 passed
─────────────────────────────────────────────
49 passed in 0.46s
```

## 🛠️ Tech Stack

- **Python 3.11+**
- **NumPy / Pandas** — Data manipulation
- **SciPy / Statsmodels** — Statistical computations
- **Plotly** — Interactive visualizations
- **Jinja2** — HTML report templating
- **MLflow** — Experiment tracking
- **Pytest** — Testing framework

## 📈 Interview-Ready Depth

This framework goes well beyond "run a z-test and check p < 0.05." It
demonstrates the full decision-making pipeline that experimentation teams
at top companies actually use:

- **Composite decision rule** — Integrates frequentist p-values, Bayesian posterior probabilities, SRM diagnostics, and novelty checks into a single SHIP/HOLD/INCONCLUSIVE recommendation with calibrated confidence. No more eyeballing four different outputs and hoping they agree.
- **Sequential testing** — Wald's SPRT lets you stop experiments early when the evidence is overwhelming, saving 30–50% of traffic for the next test. Interviewers ask about peeking problems — this is the principled answer.
- **Causal validation** — SUTVA checks and assignment balance verification catch the silent experiment-killers (wrong randomization unit, broken traffic split) before they corrupt your results.
- **Heterogeneous effects** — Per-cohort analysis reveals that a +20% overall lift might be +30% on desktop and 0% on mobile. The difference between a blanket rollout and a targeted one is real revenue.
- **Sensitivity analysis** — Stress-tests the decision against plausible baseline uncertainty. A ROBUST result means the call holds even if your baseline estimate is off by ±2%. A FRAGILE one means you're one measurement error away from a wrong decision.

## 📄 License

MIT

---

*Production-ready framework for designing, executing, and interpreting controlled experiments with statistical rigor and causal validity.*
