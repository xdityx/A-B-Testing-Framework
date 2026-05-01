"""Streamlit web interface for the A/B Testing Framework."""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_simulation import simulate_ab_test
from src.diagnostics import check_novelty_effect, check_srm
from src.experiment_design import (
    calculate_mde,
    calculate_sample_size,
    experiment_duration_days,
)
from src.statistical_tests import bayesian_ab_test, z_test_proportions

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Testing Framework",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — subtle but polished
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Wider metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 16px 20px;
    }
    /* Ship / Hold / Inconclusive badges */
    .badge-ship   { background:#1a7a4a; color:#fff; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1.1rem; }
    .badge-hold   { background:#7a1a1a; color:#fff; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1.1rem; }
    .badge-inc    { background:#5a5a1a; color:#fff; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1.1rem; }
    /* Section header rule */
    hr { border-color: rgba(255,255,255,0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧪 A/B Testing Framework")
st.markdown(
    "A production-grade experimentation platform — "
    "[GitHub](https://github.com/xdityx/A-B-Testing-Framework)"
)
st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — global advanced options
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Global Settings")
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, step=0.01)
    power = st.slider("Statistical power (1−β)", 0.70, 0.99, 0.80, step=0.01)
    st.divider()
    st.markdown(
        "Built with **Python** · scipy · statsmodels · plotly  \n"
        "[Source code](https://github.com/xdityx/A-B-Testing-Framework)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_design, tab_sim, tab_results, tab_bayes = st.tabs(
    ["📐 Experiment Design", "🎲 Simulator", "📊 Analyze Results", "🔮 Bayesian Analysis"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPERIMENT DESIGN
# ══════════════════════════════════════════════════════════════════════════════
with tab_design:
    st.subheader("Experiment Design & Power Analysis")
    st.markdown(
        "Calculate the sample size you need before you run an experiment. "
        "Uses **Cohen's h** effect size for proportions via `statsmodels`."
    )

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        baseline_rate = st.slider(
            "Baseline conversion rate (%)", 1, 60, 10, step=1
        ) / 100
        mde = st.slider(
            "Minimum detectable effect — relative lift (%)", 1, 50, 10, step=1
        ) / 100
        daily_traffic = st.number_input(
            "Daily traffic (total users)", min_value=100, max_value=10_000_000,
            value=50_000, step=1000
        )

        treatment_rate = baseline_rate * (1 + mde)
        st.caption(
            f"Baseline: **{baseline_rate:.1%}** → Expected treatment: "
            f"**{treatment_rate:.1%}** (+{mde:.0%} lift)"
        )

    with col_out:
        try:
            n = calculate_sample_size(baseline_rate, mde, alpha=alpha, power=power)
            days = experiment_duration_days(n, int(daily_traffic))
            mde_for_n = calculate_mde(baseline_rate, n, alpha=alpha, power=power)

            st.metric("Sample size per arm", f"{n:,}")
            st.metric("Total users needed", f"{n * 2:,}")
            st.metric("Estimated duration", f"{math.ceil(days)} days")
            st.metric(
                "Smallest detectable lift (at this n)",
                f"{mde_for_n:.1%}",
            )
        except ValueError as exc:
            st.error(f"Invalid inputs: {exc}")

    st.divider()

    # ── Power curve ──────────────────────────────────────────────────────────
    st.markdown("#### Power Curve — Sample Size vs. Detectable Lift")
    try:
        lifts = np.linspace(0.02, 0.40, 40)
        sizes = []
        for lift in lifts:
            try:
                sizes.append(calculate_sample_size(baseline_rate, float(lift), alpha=alpha, power=power))
            except ValueError:
                sizes.append(None)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lifts * 100,
            y=sizes,
            mode="lines+markers",
            marker=dict(size=4, color="#4f8ef7"),
            line=dict(color="#4f8ef7", width=2),
            name="n per arm",
        ))
        if n:
            fig.add_hline(
                y=n, line_dash="dash", line_color="#f7a84f",
                annotation_text=f"  Current n={n:,}", annotation_position="right",
            )
        fig.update_layout(
            xaxis_title="Relative Lift (%)",
            yaxis_title="Sample Size per Arm",
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Adjust sliders to render power curve.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.subheader("A/B Test Simulator")
    st.markdown(
        "Generate synthetic experiment data with realistic day-of-week patterns "
        "and run a frequentist z-test instantly."
    )

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        sim_baseline = st.slider("Control conversion rate (%)", 1, 60, 8, step=1) / 100
        sim_lift = st.slider("True treatment lift (%)", -20, 50, 15, step=1) / 100
        sim_n = st.number_input("Users per arm", min_value=100, max_value=500_000, value=10_000, step=500)
        sim_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
        run_sim = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    with col_b:
        if run_sim or "sim_result" in st.session_state:
            if run_sim:
                with st.spinner("Simulating…"):
                    try:
                        treatment_rate_sim = sim_baseline * (1 + sim_lift)
                        df_sim = simulate_ab_test(
                            n_control=int(sim_n),
                            n_treatment=int(sim_n),
                            control_rate=sim_baseline,
                            treatment_rate=max(0.001, min(0.999, treatment_rate_sim)),
                            seed=int(sim_seed),
                        )
                        st.session_state["sim_df"] = df_sim
                        st.session_state["sim_baseline"] = sim_baseline

                        ctrl = df_sim[df_sim["variant"] == "control"]
                        treat = df_sim[df_sim["variant"] == "treatment"]
                        result = z_test_proportions(
                            control_converted=int(ctrl["converted"].sum()),
                            control_total=len(ctrl),
                            treatment_converted=int(treat["converted"].sum()),
                            treatment_total=len(treat),
                            alpha=alpha,
                        )
                        st.session_state["sim_result"] = result
                    except Exception as exc:
                        st.error(f"Simulation error: {exc}")
                        st.session_state.pop("sim_result", None)

            if "sim_result" in st.session_state:
                res = st.session_state["sim_result"]
                c1, c2 = st.columns(2)
                c1.metric("Control rate", f"{res.control_rate:.2%}")
                c2.metric("Treatment rate", f"{res.treatment_rate:.2%}")
                c1.metric("Observed lift", f"{res.lift:+.2%}")
                c2.metric("p-value", f"{res.p_value:.4f}")
                c1.metric("Cohen's h", f"{res.cohens_h:.4f}")
                c2.metric(
                    "95% CI (abs. diff)",
                    f"[{res.ci_lower:+.4f}, {res.ci_upper:+.4f}]",
                )

                if res.significant:
                    st.success(f"✅ **Significant at α={alpha}** — effect is real")
                else:
                    st.warning(f"⚠️ **Not significant at α={alpha}** — insufficient evidence")

        else:
            st.info("Configure parameters and click **▶ Run Simulation**.")

    # ── Daily conversion time-series ─────────────────────────────────────────
    if "sim_df" in st.session_state:
        st.divider()
        st.markdown("#### Daily Conversion Rates Over Time")
        df_plot = st.session_state["sim_df"].copy()
        df_plot["date"] = pd.to_datetime(df_plot["timestamp"]).dt.date
        daily = (
            df_plot.groupby(["date", "variant"])["converted"]
            .mean()
            .reset_index()
        )
        ctrl_daily = daily[daily["variant"] == "control"]
        treat_daily = daily[daily["variant"] == "treatment"]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ctrl_daily["date"], y=ctrl_daily["converted"],
            mode="lines+markers", name="Control",
            line=dict(color="#7eb3f7", width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=treat_daily["date"], y=treat_daily["converted"],
            mode="lines+markers", name="Treatment",
            line=dict(color="#f7a84f", width=2),
        ))
        fig2.update_layout(
            xaxis_title="Date",
            yaxis_title="Conversion Rate",
            yaxis_tickformat=".1%",
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # SRM check on simulated data
        with st.expander("🔍 SRM & Novelty Diagnostics"):
            ctrl_n = len(df_plot[df_plot["variant"] == "control"])
            treat_n = len(df_plot[df_plot["variant"] == "treatment"])
            srm = check_srm(ctrl_n, treat_n)
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Control users", f"{ctrl_n:,}")
            col_s2.metric("Treatment users", f"{treat_n:,}")
            col_s3.metric("SRM detected?", "⚠️ YES" if srm.srm_detected else "✅ No")

            treat_daily_rates = (
                df_plot[df_plot["variant"] == "treatment"]
                .groupby("date")["converted"]
                .mean()
            )
            if len(treat_daily_rates) >= 3:
                novelty = check_novelty_effect(treat_daily_rates)
                col_n1, col_n2 = st.columns(2)
                col_n1.metric("Kendall's τ", f"{novelty.kendall_tau:.4f}")
                col_n2.metric(
                    "Novelty decay?",
                    "⚠️ Detected" if novelty.novelty_detected else "✅ None",
                )
            else:
                st.info("Need ≥ 3 days of data for novelty check.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYZE RESULTS (Upload CSV)
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.subheader("Analyze Experiment Results")
    st.markdown(
        "Upload a CSV with columns **`user_id`**, **`variant`** (`control`/`treatment`), "
        "**`converted`** (0/1). The framework will run a z-test and give a decision."
    )

    # ── Sample CSV download ──────────────────────────────────────────────────
    with st.expander("📥 Download sample CSV to try"):
        rng_sample = np.random.default_rng(99)
        n_sample = 500
        sample_df = pd.DataFrame({
            "user_id": range(1, n_sample * 2 + 1),
            "variant": ["control"] * n_sample + ["treatment"] * n_sample,
            "converted": np.concatenate([
                rng_sample.binomial(1, 0.10, n_sample),
                rng_sample.binomial(1, 0.13, n_sample),
            ]),
        })
        csv_bytes = sample_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download sample_ab_data.csv",
            data=csv_bytes,
            file_name="sample_ab_data.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)

            # Validate columns
            required = {"variant", "converted"}
            missing = required - set(df_up.columns)
            if missing:
                st.error(f"Missing columns: {missing}. Need at least `variant` and `converted`.")
                st.stop()

            df_up["variant"] = df_up["variant"].str.strip().str.lower()
            df_up["converted"] = pd.to_numeric(df_up["converted"], errors="coerce")

            ctrl_df = df_up[df_up["variant"] == "control"]
            treat_df = df_up[df_up["variant"] == "treatment"]

            if ctrl_df.empty or treat_df.empty:
                st.error("CSV must have rows for both `control` and `treatment`.")
                st.stop()

            st.success(f"Loaded {len(df_up):,} rows — {len(ctrl_df):,} control, {len(treat_df):,} treatment")

            # ── Run test ────────────────────────────────────────────────────
            ctrl_conv = int(ctrl_df["converted"].sum())
            treat_conv = int(treat_df["converted"].sum())
            result_up = z_test_proportions(
                control_converted=ctrl_conv,
                control_total=len(ctrl_df),
                treatment_converted=treat_conv,
                treatment_total=len(treat_df),
                alpha=alpha,
            )

            # ── Decision logic ──────────────────────────────────────────────
            def make_decision(res: "FrequentistResult", a: float) -> tuple[str, str]:
                if res.p_value < a and res.lift > 0:
                    return "SHIP 🚀", "badge-ship"
                elif res.p_value < a and res.lift < 0:
                    return "HOLD ❌", "badge-hold"
                else:
                    return "INCONCLUSIVE ❓", "badge-inc"

            decision_text, badge_cls = make_decision(result_up, alpha)

            st.divider()
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Control rate", f"{result_up.control_rate:.2%}")
            col_r2.metric("Treatment rate", f"{result_up.treatment_rate:.2%}")
            col_r3.metric("Relative lift", f"{result_up.lift:+.2%}")

            col_r4, col_r5, col_r6 = st.columns(3)
            col_r4.metric("p-value", f"{result_up.p_value:.4f}")
            col_r5.metric("Cohen's h", f"{result_up.cohens_h:.4f}")
            col_r6.metric(
                "95% CI (abs. diff)",
                f"[{result_up.ci_lower:+.4f}, {result_up.ci_upper:+.4f}]",
            )

            st.markdown("#### Recommendation")
            st.markdown(
                f'<span class="{badge_cls}">{decision_text}</span>',
                unsafe_allow_html=True,
            )

            # ── CI waterfall chart ──────────────────────────────────────────
            st.divider()
            st.markdown("#### Effect Size — 95% Confidence Interval")
            fig3 = go.Figure()
            midpoint = result_up.treatment_rate - result_up.control_rate
            fig3.add_trace(go.Scatter(
                x=[result_up.ci_lower, midpoint, result_up.ci_upper],
                y=[0, 0, 0],
                mode="markers+lines",
                marker=dict(size=[8, 14, 8], color=["#7eb3f7", "#f7a84f", "#7eb3f7"]),
                line=dict(color="#7eb3f7", width=3),
                name="95% CI",
            ))
            fig3.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
            fig3.update_layout(
                xaxis_title="Absolute rate difference (treatment − control)",
                yaxis=dict(visible=False),
                template="plotly_dark",
                height=200,
                margin=dict(l=10, r=10, t=10, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

            # SRM check
            srm_up = check_srm(len(ctrl_df), len(treat_df))
            if srm_up.srm_detected:
                st.warning(
                    f"⚠️ **Sample Ratio Mismatch detected!** "
                    f"Expected 50/50 split but got {srm_up.actual_split:.1%} / "
                    f"{1 - srm_up.actual_split:.1%}. Results may be unreliable."
                )

        except Exception as exc:
            st.error(f"Error processing file: {exc}")
    else:
        st.info("Upload a CSV to analyze results, or download the sample file above.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BAYESIAN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_bayes:
    st.subheader("Bayesian A/B Analysis")
    st.markdown(
        "Uses a **Beta-Binomial conjugate model** with Monte Carlo posterior sampling. "
        "Gives you P(treatment > control) — no p-values, no α thresholds."
    )

    col_bay1, col_bay2 = st.columns([1, 1], gap="large")

    with col_bay1:
        bay_ctrl_conv = st.number_input("Control conversions", min_value=0, max_value=10_000_000, value=800)
        bay_ctrl_tot = st.number_input("Control total users", min_value=1, max_value=10_000_000, value=10_000)
        bay_treat_conv = st.number_input("Treatment conversions", min_value=0, max_value=10_000_000, value=960)
        bay_treat_tot = st.number_input("Treatment total users", min_value=1, max_value=10_000_000, value=10_000)
        bay_prior_a = st.slider("Prior α (Beta prior)", 0.5, 10.0, 1.0, step=0.5)
        bay_prior_b = st.slider("Prior β (Beta prior)", 0.5, 10.0, 1.0, step=0.5)
        n_mc = st.select_slider("Monte Carlo samples", options=[10_000, 50_000, 100_000, 500_000], value=100_000)
        run_bayes = st.button("▶ Run Bayesian Test", type="primary", use_container_width=True)

    with col_bay2:
        if run_bayes or "bay_result" in st.session_state:
            if run_bayes:
                with st.spinner("Sampling posteriors…"):
                    try:
                        bay_res = bayesian_ab_test(
                            control_converted=int(bay_ctrl_conv),
                            control_total=int(bay_ctrl_tot),
                            treatment_converted=int(bay_treat_conv),
                            treatment_total=int(bay_treat_tot),
                            prior_alpha=bay_prior_a,
                            prior_beta=bay_prior_b,
                            n_samples=int(n_mc),
                            seed=42,
                        )
                        st.session_state["bay_result"] = bay_res
                        st.session_state["bay_inputs"] = (
                            bay_ctrl_conv, bay_ctrl_tot,
                            bay_treat_conv, bay_treat_tot,
                            bay_prior_a, bay_prior_b, n_mc,
                        )
                    except Exception as exc:
                        st.error(f"Bayesian test error: {exc}")

            if "bay_result" in st.session_state:
                b = st.session_state["bay_result"]
                st.metric("P(treatment > control)", f"{b.prob_treatment_wins:.2%}")
                st.metric("Expected loss (if you ship)", f"{b.expected_loss:.5f}")
                st.metric("Posterior control rate", f"{b.control_rate:.3%}")
                st.metric("Posterior treatment rate", f"{b.treatment_rate:.3%}")
                st.metric(
                    "95% Credible interval",
                    f"[{b.ci_lower:+.4f}, {b.ci_upper:+.4f}]",
                )

                prob = b.prob_treatment_wins
                if prob >= 0.95:
                    st.success("🚀 **Strong evidence — treatment wins.** P ≥ 95%.")
                elif prob >= 0.80:
                    st.info("📈 **Moderate evidence** that treatment is better.")
                elif prob <= 0.05:
                    st.error("❌ **Strong evidence — control wins.**")
                else:
                    st.warning("❓ **Inconclusive** — collect more data.")
        else:
            st.info("Enter counts and click **▶ Run Bayesian Test**.")

    # ── Posterior distribution plot ──────────────────────────────────────────
    if "bay_result" in st.session_state and "bay_inputs" in st.session_state:
        st.divider()
        st.markdown("#### Posterior Distributions — Beta(α_post, β_post)")
        b = st.session_state["bay_result"]
        inps = st.session_state["bay_inputs"]
        ctrl_conv_i, ctrl_tot_i, treat_conv_i, treat_tot_i, prior_a_i, prior_b_i, _ = inps

        ctrl_a_post = prior_a_i + ctrl_conv_i
        ctrl_b_post = prior_b_i + (ctrl_tot_i - ctrl_conv_i)
        treat_a_post = prior_a_i + treat_conv_i
        treat_b_post = prior_b_i + (treat_tot_i - treat_conv_i)

        import scipy.stats as ss
        x = np.linspace(0, 0.20, 500)
        ctrl_pdf = ss.beta.pdf(x, ctrl_a_post, ctrl_b_post)
        treat_pdf = ss.beta.pdf(x, treat_a_post, treat_b_post)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=x, y=ctrl_pdf, mode="lines", name="Control posterior",
            fill="tozeroy", fillcolor="rgba(126,179,247,0.2)",
            line=dict(color="#7eb3f7", width=2),
        ))
        fig4.add_trace(go.Scatter(
            x=x, y=treat_pdf, mode="lines", name="Treatment posterior",
            fill="tozeroy", fillcolor="rgba(247,168,79,0.2)",
            line=dict(color="#f7a84f", width=2),
        ))
        fig4.update_layout(
            xaxis_title="Conversion Rate",
            yaxis_title="Density",
            xaxis_tickformat=".1%",
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Win probability gauge
        st.markdown("#### P(Treatment > Control)")
        fig5 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=b.prob_treatment_wins * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#f7a84f"},
                "steps": [
                    {"range": [0, 50], "color": "rgba(200,50,50,0.3)"},
                    {"range": [50, 80], "color": "rgba(200,200,50,0.3)"},
                    {"range": [80, 100], "color": "rgba(50,200,50,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 95,
                },
            },
        ))
        fig5.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig5, use_container_width=True)
