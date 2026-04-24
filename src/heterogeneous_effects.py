"""Heterogeneous treatment effect analysis by cohort.

Segments experiment data by a categorical column and runs per-cohort
z-tests to identify where the treatment effect is strongest or absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.statistical_tests import z_test_proportions


@dataclass(frozen=True)
class CohortResult:
    """Per-cohort treatment effect summary.

    Attributes
    ----------
    cohort_name : str
        Name of the segmentation column.
    cohort_value : str
        Specific level within the segmentation column.
    n_control : int
        Observations in the control group for this cohort.
    n_treatment : int
        Observations in the treatment group for this cohort.
    control_rate : float
        Observed conversion rate in control.
    treatment_rate : float
        Observed conversion rate in treatment.
    lift : float
        Relative lift (treatment − control) / control.
    p_value : float
        Two-sided z-test p-value.
    significant : bool
        True when p-value < alpha.
    ci_lower : float
        Lower bound of 95 % CI on the absolute rate difference.
    ci_upper : float
        Upper bound of 95 % CI on the absolute rate difference.
    """

    cohort_name: str
    cohort_value: str
    n_control: int
    n_treatment: int
    control_rate: float
    treatment_rate: float
    lift: float
    p_value: float
    significant: bool
    ci_lower: float
    ci_upper: float


def analyze_by_cohort(
    df: pd.DataFrame,
    cohort_column: str,
    metric_column: str = "converted",
    alpha: float = 0.05,
) -> dict[str, CohortResult]:
    """Run a z-test for each level of *cohort_column*.

    Parameters
    ----------
    df : pandas.DataFrame
        Experiment data with columns ``variant`` (``"control"`` /
        ``"treatment"``), *cohort_column*, and *metric_column* (binary 0/1).
    cohort_column : str
        Categorical column to segment by (e.g. ``"device_type"``).
    metric_column : str, default ``"converted"``
        Binary outcome column.
    alpha : float, default 0.05
        Significance level forwarded to ``z_test_proportions``.

    Returns
    -------
    dict[str, CohortResult]
        Mapping from cohort value to its ``CohortResult``.

    Raises
    ------
    ValueError
        If required columns are missing from *df*.
    """
    for col in ("variant", cohort_column, metric_column):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    results: dict[str, CohortResult] = {}

    for value, group in df.groupby(cohort_column, sort=True):
        ctrl = group[group["variant"] == "control"]
        treat = group[group["variant"] == "treatment"]

        n_control = len(ctrl)
        n_treatment = len(treat)

        if n_control == 0 or n_treatment == 0:
            continue

        control_converted = int(ctrl[metric_column].sum())
        treatment_converted = int(treat[metric_column].sum())

        freq = z_test_proportions(
            control_converted=control_converted,
            control_total=n_control,
            treatment_converted=treatment_converted,
            treatment_total=n_treatment,
            alpha=alpha,
        )

        results[str(value)] = CohortResult(
            cohort_name=cohort_column,
            cohort_value=str(value),
            n_control=n_control,
            n_treatment=n_treatment,
            control_rate=freq.control_rate,
            treatment_rate=freq.treatment_rate,
            lift=freq.lift,
            p_value=freq.p_value,
            significant=freq.significant,
            ci_lower=freq.ci_lower,
            ci_upper=freq.ci_upper,
        )

    return results


def summarize_cohorts(cohorts: dict[str, CohortResult]) -> str:
    """Return a markdown summary of per-cohort treatment effects.

    Parameters
    ----------
    cohorts : dict[str, CohortResult]
        Output of :func:`analyze_by_cohort`.

    Returns
    -------
    str
        Human-readable markdown string.
    """
    if not cohorts:
        return "No cohort results to summarise."

    first = next(iter(cohorts.values()))
    header = f"### Treatment effect by {first.cohort_name}\n\n"
    header += "| Cohort | N (ctrl) | N (treat) | Lift | p-value | Sig? |\n"
    header += "|--------|----------|-----------|------|---------|------|\n"

    rows: list[str] = []
    for value, cr in cohorts.items():
        sig_flag = "Yes" if cr.significant else "No"
        rows.append(
            f"| {value} | {cr.n_control:,} | {cr.n_treatment:,} "
            f"| {cr.lift:+.1%} | {cr.p_value:.4f} | {sig_flag} |"
        )

    return header + "\n".join(rows)
