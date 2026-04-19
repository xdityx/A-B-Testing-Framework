# %% [markdown]
# # A/B Test Analysis: 1-Click Checkout Button
#
# **Scenario**: Our e-commerce platform is testing a new "1-Click Checkout"
# button to reduce friction in the purchase flow. We hypothesize that
# simplifying checkout will increase conversion rates.
#
# **Control (A)**: Current multi-step checkout
# **Treatment (B)**: New 1-click checkout button
#
# This notebook walks through the full experiment lifecycle:
# 1. Experiment Design & Power Analysis
# 2. Data Simulation
# 3. Pre-experiment Diagnostics
# 4. Frequentist Analysis
# 5. Bayesian Analysis
# 6. Reporting & Decision

# %%
import sys

sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.experiment_design import (
    calculate_mde,
    calculate_sample_size,
    experiment_duration_days,
)
from src.data_simulation import simulate_aa_test, simulate_ab_test
from src.diagnostics import check_novelty_effect, check_srm
from src.reporting import generate_report
from src.statistical_tests import (
    bayesian_ab_test,
    t_test_continuous,
    z_test_proportions,
)

# %% [markdown]
# ## 1. Experiment Design
#
# Before running any experiment, we need to determine:
# - How many users do we need?
# - What's the smallest effect we can detect?
# - How long will the experiment run?

# %%
# Our current checkout conversion rate is 3.2%
BASELINE_RATE = 0.032

# We want to detect at least a 10% relative improvement (3.2% → 3.52%)
MDE = 0.10

# Standard thresholds
ALPHA = 0.05
POWER = 0.80

sample_size = calculate_sample_size(BASELINE_RATE, MDE, ALPHA, POWER)
print(f"Required sample size per variant: {sample_size:,}")

# What's the MDE with our available traffic?
actual_mde = calculate_mde(BASELINE_RATE, sample_size, ALPHA, POWER)
print(f"Minimum detectable effect: {actual_mde:.2%} relative lift")

# How long with 50,000 daily visitors?
DAILY_TRAFFIC = 50_000
days = experiment_duration_days(sample_size, DAILY_TRAFFIC, traffic_split=0.50)
print(f"Estimated experiment duration: {days:.1f} days")

# %% [markdown]
# ### Power Analysis Visualization
#
# Let's visualize how sample size changes with different MDEs
# to understand the tradeoff between sensitivity and duration.

# %%
mde_range = np.arange(0.02, 0.25, 0.01)
sizes = [calculate_sample_size(BASELINE_RATE, m, ALPHA, POWER) for m in mde_range]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=mde_range * 100,
        y=sizes,
        mode="lines+markers",
        line=dict(color="#636efa", width=2),
        marker=dict(size=4),
    )
)
fig.add_vline(
    x=MDE * 100,
    line_dash="dash",
    line_color="#ef553b",
    annotation_text=f"Our MDE ({MDE:.0%})",
)
fig.update_layout(
    title="Sample Size vs. Minimum Detectable Effect",
    xaxis_title="MDE (% relative lift)",
    yaxis_title="Sample Size per Variant",
    template="plotly_dark",
)
fig.show()

# %% [markdown]
# ## 2. Data Simulation
#
# Since we're demonstrating the framework, we simulate realistic
# experiment data. In production, this would come from your event
# pipeline.

# %%
# Simulate: treatment has a true 12% relative lift (3.2% → 3.584%)
TREATMENT_RATE = BASELINE_RATE * (1 + 0.12)  # true effect slightly above MDE

df = simulate_ab_test(
    n_control=sample_size,
    n_treatment=sample_size,
    control_rate=BASELINE_RATE,
    treatment_rate=TREATMENT_RATE,
    avg_revenue_per_converter=67.50,
    duration_days=int(np.ceil(days)),
    seed=42,
)

print(f"Total users: {len(df):,}")
print(f"Control: {len(df[df['variant'] == 'control']):,}")
print(f"Treatment: {len(df[df['variant'] == 'treatment']):,}")
print("\nSample data:")
df.head(10)

# %% [markdown]
# ## 3. Pre-Experiment Diagnostics
#
# Before interpreting results, we check for data quality issues.

# %% [markdown]
# ### 3a. Sample Ratio Mismatch (SRM)
#
# If the actual traffic split differs significantly from 50/50,
# something is wrong with randomization.

# %%
control = df[df["variant"] == "control"]
treatment = df[df["variant"] == "treatment"]

srm_result = check_srm(len(control), len(treatment))
print(f"SRM detected: {srm_result.srm_detected}")
print(f"Actual split: {srm_result.actual_split:.4f}")
print(f"Chi² statistic: {srm_result.chi2_stat:.4f}")
print(f"P-value: {srm_result.p_value:.4f}")

if srm_result.srm_detected:
    print("\n⚠️ WARNING: Sample Ratio Mismatch detected! Investigate before proceeding.")
else:
    print("\n✅ No SRM detected. Traffic split looks healthy.")

# %% [markdown]
# ### 3b. Novelty Effect Check
#
# Is the treatment effect decaying over time? If so, the initial
# lift may just be users reacting to something "new."

# %%
daily = df.copy()
daily["date"] = daily["timestamp"].dt.date
treatment_daily = daily[daily["variant"] == "treatment"].groupby("date")[
    "converted"
].mean()

novelty_result = check_novelty_effect(treatment_daily)
print(f"Novelty effect detected: {novelty_result.novelty_detected}")
print(f"Trend slope: {novelty_result.trend_slope:.6f}")
print(f"P-value: {novelty_result.p_value:.4f}")

if novelty_result.novelty_detected:
    print("\n⚠️ WARNING: Conversion rate is declining over time.")
else:
    print("\n✅ No significant novelty decay detected.")

# %% [markdown]
# ## 4. Frequentist Analysis
#
# We use a two-sided z-test for proportions to determine if the
# difference in conversion rates is statistically significant.

# %%
freq_result = z_test_proportions(
    control_converted=int(control["converted"].sum()),
    control_total=len(control),
    treatment_converted=int(treatment["converted"].sum()),
    treatment_total=len(treatment),
    alpha=ALPHA,
)

print("=" * 50)
print("FREQUENTIST RESULTS")
print("=" * 50)
print(f"Control conversion rate:   {freq_result.control_rate:.4%}")
print(f"Treatment conversion rate: {freq_result.treatment_rate:.4%}")
print(f"Relative lift:             {freq_result.lift:.2%}")
print(f"Z-statistic:               {freq_result.z_stat:.4f}")
print(f"P-value:                   {freq_result.p_value:.6f}")
print(
    "95% CI on difference:      "
    f"[{freq_result.ci_lower:.4%}, {freq_result.ci_upper:.4%}]"
)
print(f"Significant at α={ALPHA}:   {freq_result.significant}")

# %% [markdown]
# ### Revenue Analysis
#
# Beyond conversion, did 1-click checkout affect average order value?

# %%
control_revenue = control[control["converted"] == 1]["revenue"].values
treatment_revenue = treatment[treatment["converted"] == 1]["revenue"].values

revenue_result = t_test_continuous(control_revenue, treatment_revenue, alpha=ALPHA)

print("=" * 50)
print("REVENUE ANALYSIS (Welch's t-test)")
print("=" * 50)
print(f"Control mean revenue:   ${revenue_result.control_mean:.2f}")
print(f"Treatment mean revenue: ${revenue_result.treatment_mean:.2f}")
print(f"Lift:                   {revenue_result.lift:.2%}")
print(f"P-value:                {revenue_result.p_value:.6f}")
print(f"Significant:            {revenue_result.significant}")

# %% [markdown]
# ## 5. Bayesian Analysis
#
# The Bayesian approach gives us direct probability statements:
# "What's the probability that treatment is better than control?"

# %%
bayes_result = bayesian_ab_test(
    control_converted=int(control["converted"].sum()),
    control_total=len(control),
    treatment_converted=int(treatment["converted"].sum()),
    treatment_total=len(treatment),
    n_samples=200_000,
    seed=42,
)

print("=" * 50)
print("BAYESIAN RESULTS")
print("=" * 50)
print(f"P(Treatment > Control):  {bayes_result.prob_treatment_wins:.2%}")
print(f"Expected loss:           {bayes_result.expected_loss:.6f}")
print(
    "95% credible interval:   "
    f"[{bayes_result.ci_lower:.4%}, {bayes_result.ci_upper:.4%}]"
)

# %% [markdown]
# ### Posterior Visualization

# %%
# Recreate posterior samples for visualization
rng = np.random.default_rng(42)
c_conv = int(control["converted"].sum())
c_total = len(control)
t_conv = int(treatment["converted"].sum())
t_total = len(treatment)

control_samples = rng.beta(1 + c_conv, 1 + c_total - c_conv, 200_000)
treatment_samples = rng.beta(1 + t_conv, 1 + t_total - t_conv, 200_000)
lift_samples = (treatment_samples - control_samples) / control_samples

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Posterior Distributions", "Lift Distribution"),
)

fig.add_trace(
    go.Histogram(
        x=control_samples,
        name="Control",
        opacity=0.7,
        marker_color="#ef553b",
        nbinsx=80,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(
        x=treatment_samples,
        name="Treatment",
        opacity=0.7,
        marker_color="#636efa",
        nbinsx=80,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Histogram(
        x=lift_samples * 100,
        name="Lift %",
        opacity=0.7,
        marker_color="#00cc96",
        nbinsx=80,
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig.add_vline(x=0, line_dash="dash", line_color="white", row=1, col=2)

fig.update_layout(
    template="plotly_dark",
    title="Bayesian Posterior Analysis",
    barmode="overlay",
    height=400,
)
fig.update_xaxes(title_text="Conversion Rate", row=1, col=1)
fig.update_xaxes(title_text="Relative Lift (%)", row=1, col=2)
fig.show()

# %% [markdown]
# ## 6. Generate Report & Decision
#
# Generate the full HTML report and make a go/no-go decision.

# %%
report_path = generate_report(
    df=df,
    freq_result=freq_result,
    bayesian_result=bayes_result,
    srm_result=srm_result,
    novelty_result=novelty_result,
    output_path="../reports/ab_test_report.html",
)
print(f"Report saved to: {report_path}")

# %% [markdown]
# ## Final Decision
#
# | Criterion | Result | Status |
# |-----------|--------|--------|
# | SRM Check | No mismatch | ✅ |
# | Novelty Check | No decay | ✅ |
# | Statistical Significance | p < 0.05 | ✅ |
# | P(Treatment > Control) | > 95% | ✅ |
# | Expected Loss | < 0.001 | ✅ |
#
# **Recommendation: SHIP the 1-Click Checkout button.** 🚀
#
# The treatment variant shows a statistically significant improvement
# in conversion rate, confirmed by both frequentist and Bayesian methods,
# with no diagnostic red flags.
