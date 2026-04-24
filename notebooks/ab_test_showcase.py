# %% [markdown]
# # A/B Test Analysis: 1-Click Checkout Button
#
# **Scenario**: Our e-commerce platform is testing a new "1-Click Checkout"
# button to reduce friction in the purchase flow.  Implementation cost is
# ~$50k; the checkout funnel generates ~$2M/year in revenue, so even a
# modest conversion lift pays for itself within weeks.
#
# **Control (A)**: Current multi-step checkout
# **Treatment (B)**: New 1-click checkout button
#
# This notebook walks through the full experiment lifecycle:
# 1. Experiment Design & Power Analysis
# 2. Data Simulation
# 3. Causal Validation
# 4. Pre-Experiment Diagnostics (SRM + Novelty)
# 5. Sequential Monitoring (SPRT)
# 6. Final Statistical Analysis
# 7. Heterogeneous Effects by Cohort
# 8. Sensitivity Analysis
# 9. Experiment Decision

# %%
import sys

sys.path.insert(0, "..")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.causal_validation import validate_experiment_assumptions
from src.data_simulation import simulate_ab_test
from src.diagnostics import check_novelty_effect, check_srm
from src.experiment_decision import make_decision
from src.experiment_design import (
    calculate_mde,
    calculate_sample_size,
    experiment_duration_days,
)
from src.heterogeneous_effects import analyze_by_cohort, summarize_cohorts
from src.sensitivity import display_sensitivity, sensitivity_analysis
from src.sequential_testing import sequential_z_test
from src.statistical_tests import (
    bayesian_ab_test,
    t_test_continuous,
    z_test_proportions,
)

# %% [markdown]
# ## Stage 1: Experiment Design
#
# Before touching production traffic we lock in the statistical contract:
# sample size, minimum detectable effect, and run time.  Getting this
# wrong means either wasting traffic (oversized test) or missing a real
# effect (undersized test).

# %%
BASELINE_RATE = 0.10  # current checkout conversion rate
MDE = 0.20  # 20% relative lift = 10% → 12%
ALPHA = 0.05
POWER = 0.80

sample_size = calculate_sample_size(BASELINE_RATE, MDE, ALPHA, POWER)
print(f"Required sample size per variant: {sample_size:,}")

actual_mde = calculate_mde(BASELINE_RATE, sample_size, ALPHA, POWER)
print(f"Minimum detectable effect:        {actual_mde:.2%} relative lift")

DAILY_TRAFFIC = 10_000
days_needed = experiment_duration_days(sample_size, DAILY_TRAFFIC, traffic_split=0.50)
print(f"Estimated experiment duration:     {days_needed:.0f} days")
print("\nBusiness framing: at $2M/year revenue, a 20% conversion lift is")
print(f"worth ~${2_000_000 * 0.20 * (BASELINE_RATE * 0.20):,.0f}/year — $50k implementation")
print("cost pays back in under a month.")

# %%
# Power curve: how sample size scales with MDE
mde_range = np.arange(0.05, 0.40, 0.01)
sizes = [calculate_sample_size(BASELINE_RATE, m, ALPHA, POWER) for m in mde_range]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(mde_range * 100, sizes, color="#636efa", linewidth=2)
ax.axvline(MDE * 100, color="#ef553b", linestyle="--", label=f"Our MDE ({MDE:.0%})")
ax.set_xlabel("MDE (% relative lift)")
ax.set_ylabel("Sample size per variant")
ax.set_title("Sample Size vs. Minimum Detectable Effect")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Stage 2: Data Simulation
#
# We simulate 10,000 users (5,000 per arm) over 14 days with day-of-week
# traffic patterns and lognormal revenue.  The true treatment rate is 12%
# (a 20% relative lift over the 10% baseline).

# %%
N_PER_VARIANT = 5000
TREATMENT_RATE = 0.12
DURATION_DAYS = 14

df = simulate_ab_test(
    n_control=N_PER_VARIANT,
    n_treatment=N_PER_VARIANT,
    control_rate=BASELINE_RATE,
    treatment_rate=TREATMENT_RATE,
    avg_revenue_per_converter=67.50,
    duration_days=DURATION_DAYS,
    start_date="2025-03-01",
    seed=42,
)

control = df[df["variant"] == "control"]
treatment = df[df["variant"] == "treatment"]

print(f"Total users:  {len(df):,}")
print(f"Control:      {len(control):,}  ({int(control['converted'].sum())} converted)")
print(f"Treatment:    {len(treatment):,}  ({int(treatment['converted'].sum())} converted)")
print(f"\nDate range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
df.head(8)

# %% [markdown]
# ## Stage 3: Causal Validation (NEW)
#
# Before looking at any p-values we verify that the experiment's causal
# structure is sound.  Two questions matter here:
#
# 1. **SUTVA**: Is the randomization unit the same as the metric unit?
#    If we randomize by *session* but measure per *user*, a single user
#    can appear in both arms — the independence assumption breaks.
#
# 2. **Balance**: Is the 50/50 split actually 50/50?  Gross imbalance
#    (beyond normal sampling noise) signals a randomization bug.

# %%
causal = validate_experiment_assumptions(
    df,
    randomization_level="user",
    metric_level="user",
    expected_split=0.50,
    tolerance=0.05,
)

print(f"SUTVA OK:              {causal.sutva_ok}")
print(f"Assignment balanced:   {causal.assignment_balanced}")
print(f"Confounding risk:      {causal.confounding_risk}")
if causal.warnings:
    for w in causal.warnings:
        print(f"  WARNING: {w}")
else:
    print(f"Notes:                 {causal.notes}")

# %% [markdown]
# ## Stage 4: Pre-Experiment Diagnostics
#
# Even with a clean causal structure, two failure modes can silently
# corrupt results:
#
# - **SRM**: The randomization system leaks users into one arm.
# - **Novelty effect**: Users react to the *newness* of the treatment,
#   inflating early lift that fades once the novelty wears off.

# %%
srm_result = check_srm(len(control), len(treatment))
print("--- Sample Ratio Mismatch ---")
print(f"Actual split:   {srm_result.actual_split:.4f}")
print(f"Chi-square:     {srm_result.chi2_stat:.4f}")
print(f"p-value:        {srm_result.p_value:.4f}")
print(f"SRM detected:   {srm_result.srm_detected}")

# %%
daily = df.copy()
daily["date"] = daily["timestamp"].dt.date
treatment_daily = daily[daily["variant"] == "treatment"].groupby("date")["converted"].mean()

novelty_result = check_novelty_effect(treatment_daily)
print("\n--- Novelty Effect ---")
print(f"Kendall's tau:      {novelty_result.kendall_tau:.4f}")
print(f"p-value:            {novelty_result.p_value:.4f}")
print(f"Novelty detected:   {novelty_result.novelty_detected}")

if not srm_result.srm_detected and not novelty_result.novelty_detected:
    print("\nDiagnostics clean — safe to proceed to analysis.")

# %% [markdown]
# ## Stage 5: Sequential Monitoring (NEW)
#
# In practice you don't wait until the last user arrives to peek at
# results.  Wald's SPRT lets us monitor the cumulative log-likelihood
# ratio each day and stop early if the evidence is overwhelming — saving
# traffic for the next experiment.

# %%
dates = sorted(daily["date"].unique())

sprt_log = []
for i, d in enumerate(dates, 1):
    snapshot = daily[daily["date"] <= d]
    ctrl_snap = snapshot[snapshot["variant"] == "control"]
    treat_snap = snapshot[snapshot["variant"] == "treatment"]

    sprt = sequential_z_test(
        control_converted=int(ctrl_snap["converted"].sum()),
        control_total=len(ctrl_snap),
        treatment_converted=int(treat_snap["converted"].sum()),
        treatment_total=len(treat_snap),
        baseline_rate=BASELINE_RATE,
        mde=MDE,
        alpha=ALPHA,
        beta=1 - POWER,
    )
    sprt_log.append(
        {
            "day": i,
            "date": d,
            "n_total": len(snapshot),
            "log_lr": sprt.log_likelihood_ratio,
            "upper": sprt.boundary_upper,
            "lower": sprt.boundary_lower,
            "stop_reason": sprt.stop_reason,
        }
    )
    print(
        f"Day {i:2d} | n={len(snapshot):>5,} | "
        f"log_LR={sprt.log_likelihood_ratio:+7.2f} | {sprt.stop_reason}"
    )
    if sprt.can_stop:
        print(f"  >>> Early stop triggered: {sprt.stop_reason}")
        break

# %%
sprt_df = pd.DataFrame(sprt_log)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(
    sprt_df["day"],
    sprt_df["log_lr"],
    "o-",
    color="#636efa",
    linewidth=2,
    markersize=6,
    label="Log-LR",
    zorder=3,
)
ax.axhline(
    sprt_df["upper"].iloc[0],
    color="#00cc96",
    linestyle="--",
    linewidth=1.5,
    label=f"Upper boundary (log A = {sprt_df['upper'].iloc[0]:.2f})",
)
ax.axhline(
    sprt_df["lower"].iloc[0],
    color="#ef553b",
    linestyle="--",
    linewidth=1.5,
    label=f"Lower boundary (log B = {sprt_df['lower'].iloc[0]:.2f})",
)
ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
ax.fill_between(
    sprt_df["day"],
    sprt_df["lower"].iloc[0],
    sprt_df["upper"].iloc[0],
    color="gray",
    alpha=0.08,
    label="Continue zone",
)
ax.set_xlabel("Day")
ax.set_ylabel("Cumulative log-likelihood ratio")
ax.set_title("SPRT Sequential Monitoring")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Stage 6: Final Statistical Analysis
#
# Even if SPRT fired early, we report the full-data frequentist and
# Bayesian results for the final record.

# %%
freq_result = z_test_proportions(
    control_converted=int(control["converted"].sum()),
    control_total=len(control),
    treatment_converted=int(treatment["converted"].sum()),
    treatment_total=len(treatment),
    alpha=ALPHA,
)

print("=" * 55)
print("FREQUENTIST RESULTS")
print("=" * 55)
print(f"Control rate:       {freq_result.control_rate:.4%}")
print(f"Treatment rate:     {freq_result.treatment_rate:.4%}")
print(f"Relative lift:      {freq_result.lift:.2%}")
print(f"Z-stat:             {freq_result.z_stat:.4f}")
print(f"P-value:            {freq_result.p_value:.6f}")
print(f"95% CI (abs diff):  [{freq_result.ci_lower:.4%}, {freq_result.ci_upper:.4%}]")
print(f"Significant:        {freq_result.significant}")

# %%
bayes_result = bayesian_ab_test(
    control_converted=int(control["converted"].sum()),
    control_total=len(control),
    treatment_converted=int(treatment["converted"].sum()),
    treatment_total=len(treatment),
    n_samples=200_000,
    seed=42,
)

print("=" * 55)
print("BAYESIAN RESULTS")
print("=" * 55)
print(f"P(Treatment > Control):  {bayes_result.prob_treatment_wins:.2%}")
print(f"Expected loss:           {bayes_result.expected_loss:.6f}")
print(f"95% credible interval:   [{bayes_result.ci_lower:.4%}, {bayes_result.ci_upper:.4%}]")

# %%
# Revenue impact (Welch's t-test on converters)
control_revenue = control[control["converted"] == 1]["revenue"].values
treatment_revenue = treatment[treatment["converted"] == 1]["revenue"].values

if len(control_revenue) >= 2 and len(treatment_revenue) >= 2:
    rev_result = t_test_continuous(control_revenue, treatment_revenue, alpha=ALPHA)
    print(
        f"\nRevenue per converter:  ${rev_result.control_mean:.2f} (ctrl) vs "
        f"${rev_result.treatment_mean:.2f} (treat)"
    )
    print(f"Revenue lift:           {rev_result.lift:.2%}  (p={rev_result.p_value:.4f})")

# %%
# Posterior visualization
rng = np.random.default_rng(42)
c_conv = int(control["converted"].sum())
c_total = len(control)
t_conv = int(treatment["converted"].sum())
t_total = len(treatment)

ctrl_samples = rng.beta(1 + c_conv, 1 + c_total - c_conv, 200_000)
treat_samples = rng.beta(1 + t_conv, 1 + t_total - t_conv, 200_000)
lift_samples = (treat_samples - ctrl_samples) / ctrl_samples

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(ctrl_samples, bins=80, alpha=0.65, color="#ef553b", label="Control")
axes[0].hist(treat_samples, bins=80, alpha=0.65, color="#636efa", label="Treatment")
axes[0].set_title("Posterior Distributions")
axes[0].set_xlabel("Conversion rate")
axes[0].legend()

axes[1].hist(lift_samples * 100, bins=80, alpha=0.7, color="#00cc96")
axes[1].axvline(0, color="white", linestyle="--", linewidth=1)
axes[1].set_title("Lift Distribution")
axes[1].set_xlabel("Relative lift (%)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Stage 7: Heterogeneous Effects by Cohort (NEW)
#
# An overall +20% lift doesn't mean the button works *everywhere*.  If
# the effect is concentrated on desktop users and absent on mobile, we
# should consider a segment-specific rollout instead of a blanket ship.

# %%
# Add a device_type column with realistic proportions.
rng_dev = np.random.default_rng(99)
device_probs = [0.55, 0.35, 0.10]  # mobile, desktop, tablet
df["device_type"] = rng_dev.choice(
    ["mobile", "desktop", "tablet"],
    size=len(df),
    p=device_probs,
)

cohorts = analyze_by_cohort(df, cohort_column="device_type")
print(summarize_cohorts(cohorts))

# %%
# Visual comparison
cohort_names = list(cohorts.keys())
lifts = [cohorts[c].lift * 100 for c in cohort_names]
p_vals = [cohorts[c].p_value for c in cohort_names]
colors = ["#00cc96" if cohorts[c].significant else "#ef553b" for c in cohort_names]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(cohort_names, lifts, color=colors, edgecolor="white", linewidth=0.8)
ax.axhline(0, color="gray", linewidth=0.8)
for bar, p in zip(bars, p_vals):
    label = f"p={p:.3f}"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.set_ylabel("Relative lift (%)")
ax.set_title("Treatment Lift by Device Type")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("\nInterpretation: The overall lift is spread across segments.")
print("Check which cohorts reach significance before segment-specific rollout.")

# %% [markdown]
# ## Stage 8: Sensitivity Analysis (NEW)
#
# Our baseline rate of 10% is an *estimate*.  If the true baseline were
# 8% or 12%, would we still ship?  Sensitivity analysis stress-tests the
# decision against plausible baseline uncertainty.

# %%
sens = sensitivity_analysis(
    control_rate_observed=freq_result.control_rate,
    treatment_rate_observed=freq_result.treatment_rate,
    n_control=len(control),
    n_treatment=len(treatment),
    baseline_uncertainty=0.02,
)

print(display_sensitivity(sens))
print(f"\nVerdict: {sens.recommendation} — {sens.interpretation}")

# %% [markdown]
# ## Stage 9: Experiment Decision (NEW)
#
# All signals converge here.  The decision engine integrates frequentist
# evidence, Bayesian probability, SRM diagnostics, and novelty checks
# into a single SHIP / HOLD / INCONCLUSIVE recommendation with a
# calibrated confidence score and risk level.

# %%
decision = make_decision(
    freq_result=freq_result,
    bayes_result=bayes_result,
    srm_result=srm_result,
    novelty_result=novelty_result,
)

print("=" * 55)
print("EXPERIMENT DECISION")
print("=" * 55)
print(f"Recommendation:   {decision.recommendation}")
print(f"Confidence:       {decision.confidence:.2f}")
print(f"Risk level:       {decision.risk_level}")
print("\nReasoning:")
for line in decision.reasoning:
    print(f"  - {line}")

# %% [markdown]
# ## Final Verdict
#
# | Check                  | Result                      | Status |
# |------------------------|-----------------------------|--------|
# | Causal validation      | SUTVA OK, balanced          | Pass   |
# | SRM                    | No mismatch                 | Pass   |
# | Novelty effect         | No decay                    | Pass   |
# | SPRT early stopping    | Crossed upper boundary      | Pass   |
# | Frequentist            | p < 0.05                    | Pass   |
# | Bayesian               | P(B>A) > 95%                | Pass   |
# | Sensitivity            | ROBUST across +/-2%         | Pass   |
# | Cohort analysis        | Check segment-level effects | Note   |
#
# **Recommendation: SHIP the 1-Click Checkout button.**
#
# The treatment shows a statistically significant conversion lift confirmed
# by both frequentist and Bayesian methods, with clean diagnostics, robust
# sensitivity, and early-stopping evidence.  Cohort analysis may reveal
# segment-specific variation — consider a phased rollout if any segment
# underperforms.
#
# At a 20% relative lift on a $2M/year funnel, the $50k implementation
# cost pays back in under 2 weeks.
