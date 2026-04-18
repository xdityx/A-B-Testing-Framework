"""HTML reporting utilities for A/B test analysis."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader

from src.diagnostics import NoveltyResult, SRMResult
from src.statistical_tests import BayesianResult, FrequentistResult


def _plotly_json(fig: go.Figure) -> str:
    chart_json = pio.to_json(fig)
    json.loads(chart_json)
    return chart_json


def _variant_order(values: pd.Series) -> list[str]:
    variants = [variant for variant in ["control", "treatment"] if variant in set(values)]
    variants.extend(sorted(set(values) - set(variants)))
    return variants


def _conversion_rate_chart(df: pd.DataFrame) -> str:
    """Create a grouped bar chart of conversion rates by variant."""
    conversion_rates = df.groupby("variant")["converted"].mean()
    fig = go.Figure()

    for variant in _variant_order(conversion_rates.index.to_series()):
        fig.add_trace(
            go.Bar(
                name=variant,
                x=[variant],
                y=[float(conversion_rates.loc[variant])],
            )
        )

    fig.update_layout(
        title="Conversion Rate by Variant",
        template="plotly_dark",
        yaxis_tickformat=".1%",
        barmode="group",
    )
    return _plotly_json(fig)


def _revenue_chart(df: pd.DataFrame) -> str:
    """Create a box plot of converter revenue by variant."""
    converters = df[df["converted"] == 1]
    fig = go.Figure()

    if not converters.empty:
        for variant in _variant_order(converters["variant"]):
            variant_revenue = converters.loc[converters["variant"] == variant, "revenue"]
            fig.add_trace(
                go.Box(
                    name=variant,
                    y=variant_revenue,
                    boxmean=True,
                )
            )

    fig.update_layout(
        title="Revenue Distribution (Converters Only)",
        template="plotly_dark",
    )
    return _plotly_json(fig)


def _daily_trend_chart(df: pd.DataFrame) -> str:
    """Create a line chart of daily conversion rates by variant."""
    chart_df = df.copy()
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
    chart_df["date"] = chart_df["timestamp"].dt.date
    daily_rates = (
        chart_df.groupby(["date", "variant"], as_index=False)["converted"]
        .mean()
        .rename(columns={"converted": "conversion_rate"})
    )

    fig = go.Figure()
    for variant in _variant_order(daily_rates["variant"]):
        variant_daily = daily_rates[daily_rates["variant"] == variant]
        fig.add_trace(
            go.Scatter(
                name=variant,
                x=variant_daily["date"],
                y=variant_daily["conversion_rate"],
                mode="lines+markers",
            )
        )

    fig.update_layout(
        title="Daily Conversion Rate Trend",
        template="plotly_dark",
        yaxis_tickformat=".1%",
    )
    return _plotly_json(fig)


def _bayesian_posterior_chart(bayesian_result: BayesianResult) -> str:
    """Create a bar chart of Bayesian win probabilities."""
    prob_treatment = bayesian_result.prob_treatment_wins
    prob_control = 1.0 - prob_treatment

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Control Wins", "Treatment Wins"],
                y=[prob_control, prob_treatment],
                marker_color=["#ef553b", "#636efa"],
            )
        ]
    )
    fig.update_layout(
        title="Bayesian Probability of Winning",
        template="plotly_dark",
        yaxis_tickformat=".1%",
        yaxis_range=[0, 1],
        shapes=[
            go.layout.Shape(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=0.95,
                y1=0.95,
                line={"color": "#00d97e", "width": 2, "dash": "dash"},
            )
        ],
    )
    return _plotly_json(fig)


def generate_report(
    df: pd.DataFrame,
    freq_result: FrequentistResult,
    bayesian_result: BayesianResult,
    srm_result: SRMResult,
    novelty_result: NoveltyResult,
    output_path: str = "reports/ab_test_report.html",
) -> pathlib.Path:
    """Generate a self-contained HTML report for an A/B test.

    Parameters
    ----------
    df : pandas.DataFrame
        User-level experiment data containing at least ``variant``,
        ``converted``, ``revenue``, and ``timestamp`` columns.
    freq_result : FrequentistResult
        Frequentist proportion test result to summarize in the report.
    bayesian_result : BayesianResult
        Bayesian A/B test result to summarize in the report.
    srm_result : SRMResult
        Sample-ratio-mismatch diagnostic result to summarize in the report.
    novelty_result : NoveltyResult
        Novelty-effect diagnostic result to summarize in the report.
    output_path : str, default="reports/ab_test_report.html"
        Path where the rendered HTML report should be written.

    Returns
    -------
    pathlib.Path
        Path to the generated HTML report.
    """
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    chart_conversion = _conversion_rate_chart(df)
    chart_revenue = _revenue_chart(df)
    chart_daily = _daily_trend_chart(df)
    chart_bayesian = _bayesian_posterior_chart(bayesian_result)

    context = {
        "freq": asdict(freq_result),
        "bayes": asdict(bayesian_result),
        "srm": asdict(srm_result),
        "novelty": asdict(novelty_result),
        "n_control": len(df[df["variant"] == "control"]),
        "n_treatment": len(df[df["variant"] == "treatment"]),
        "chart_conversion": chart_conversion,
        "chart_revenue": chart_revenue,
        "chart_daily": chart_daily,
        "chart_bayesian": chart_bayesian,
    }

    template_dir = pathlib.Path(__file__).parent / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=True,
    )
    template = environment.get_template("report_template.html")
    rendered_html = template.render(**context)
    output.write_text(rendered_html, encoding="utf-8")
    return output
