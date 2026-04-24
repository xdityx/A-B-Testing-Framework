"""Tests for heterogeneous_effects module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.heterogeneous_effects import analyze_by_cohort, summarize_cohorts


def _simulate_cohort_df(seed: int = 42) -> pd.DataFrame:
    """Build a DataFrame where mobile has ~0 % lift and desktop has ~20 %."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    # Mobile: both arms 10 %
    for variant, rate in [("control", 0.10), ("treatment", 0.10)]:
        n = 2000
        converted = rng.binomial(1, rate, size=n)
        for c in converted:
            rows.append({"variant": variant, "device_type": "mobile", "converted": int(c)})
    # Desktop: control 10 %, treatment 12 %
    for variant, rate in [("control", 0.10), ("treatment", 0.12)]:
        n = 3000
        converted = rng.binomial(1, rate, size=n)
        for c in converted:
            rows.append({"variant": variant, "device_type": "desktop", "converted": int(c)})
    return pd.DataFrame(rows)


def test_cohort_analysis_mobile_vs_desktop() -> None:
    """Mobile should be non-sig, desktop should show positive lift."""
    df = _simulate_cohort_df()
    cohorts = analyze_by_cohort(df, cohort_column="device_type")

    assert "mobile" in cohorts
    assert "desktop" in cohorts
    # Mobile has zero true lift → expect non-significant
    assert cohorts["mobile"].significant is False
    # Desktop has positive true lift
    assert cohorts["desktop"].lift > 0.0


def test_cohort_summary_markdown() -> None:
    """summarize_cohorts should produce readable markdown with a table."""
    df = _simulate_cohort_df()
    cohorts = analyze_by_cohort(df, cohort_column="device_type")
    md = summarize_cohorts(cohorts)
    assert "| Cohort" in md
    assert "mobile" in md
    assert "desktop" in md
