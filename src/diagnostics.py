"""Diagnostic utilities for experiment quality checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats


@dataclass(frozen=True)
class SRMResult:
    """Container for sample-ratio-mismatch diagnostics."""

    expected_split: float  # expected proportion of users in control
    actual_split: float  # observed proportion of users in control
    chi2_stat: float  # chi-square goodness-of-fit statistic
    p_value: float  # p-value from the chi-square test
    srm_detected: bool  # True if p_value < alpha


@dataclass(frozen=True)
class NoveltyResult:
    """Container for novelty-effect diagnostics."""

    trend_slope: float  # slope of daily conversion rate over time
    p_value: float  # significance of the slope
    novelty_detected: bool  # True if p_value < alpha and trend_slope < 0


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1).")


def check_srm(
    control_users: int,
    treatment_users: int,
    expected_control_proportion: float = 0.50,
    alpha: float = 0.001,
) -> SRMResult:
    """Check for sample ratio mismatch between control and treatment groups.

    Parameters
    ----------
    control_users : int
        Number of observed users assigned to the control group.
    treatment_users : int
        Number of observed users assigned to the treatment group.
    expected_control_proportion : float, default=0.50
        Expected share of total traffic assigned to the control group.
    alpha : float, default=0.001
        Significance level used to flag sample ratio mismatch.

    Returns
    -------
    SRMResult
        Dataclass containing the expected split, actual split, chi-square
        statistic, p-value, and SRM flag.

    Examples
    --------
    >>> result = check_srm(5000, 5000)
    >>> result.srm_detected
    False
    >>> result.actual_split
    0.5
    """
    _validate_alpha(alpha)
    if control_users < 0 or treatment_users < 0:
        raise ValueError("control_users and treatment_users must be non-negative.")
    if not 0.0 < expected_control_proportion < 1.0:
        raise ValueError("expected_control_proportion must be in the open interval (0, 1).")

    total = control_users + treatment_users
    if total == 0:
        raise ValueError("At least one observed user is required to check SRM.")

    expected = np.array(
        [
            total * expected_control_proportion,
            total * (1.0 - expected_control_proportion),
        ],
        dtype=float,
    )
    observed = np.array([control_users, treatment_users], dtype=float)
    chi2_stat, p_value = scipy.stats.chisquare(f_obs=observed, f_exp=expected)
    actual_split = control_users / total

    return SRMResult(
        expected_split=float(expected_control_proportion),
        actual_split=float(actual_split),
        chi2_stat=float(chi2_stat),
        p_value=float(p_value),
        srm_detected=bool(p_value < alpha),
    )


def check_novelty_effect(
    daily_rates: pd.Series,
    alpha: float = 0.05,
) -> NoveltyResult:
    """Check whether conversion rates show a significant decaying time trend.

    Parameters
    ----------
    daily_rates : pandas.Series
        Daily conversion rates ordered over time.
    alpha : float, default=0.05
        Significance level used to flag a novelty effect.

    Returns
    -------
    NoveltyResult
        Dataclass containing the fitted slope, p-value for the slope, and a
        boolean flag indicating whether a significant negative trend was found.

    Examples
    --------
    >>> rates = pd.Series([0.10, 0.09, 0.08, 0.07, 0.06])
    >>> result = check_novelty_effect(rates)
    >>> result.novelty_detected
    True
    """
    _validate_alpha(alpha)

    values = daily_rates.astype(float).to_numpy()
    if values.size < 3:
        raise ValueError("daily_rates must contain at least 3 observations.")
    if not np.isfinite(values).all():
        raise ValueError("daily_rates must contain only finite numeric values.")

    days = np.arange(values.size, dtype=float)
    slope, _intercept, _r_value, p_value, _std_err = scipy.stats.linregress(days, values)
    novelty_detected = bool((p_value < alpha) and (slope < 0.0))

    return NoveltyResult(
        trend_slope=float(slope),
        p_value=float(p_value),
        novelty_detected=novelty_detected,
    )


def holm_bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Apply the Holm-Bonferroni multiple-testing correction.

    Parameters
    ----------
    p_values : list of float
        Raw p-values from multiple hypothesis tests.
    alpha : float, default=0.05
        Family-wise error rate to control.

    Returns
    -------
    list of bool
        Boolean rejection decisions in the same order as the input p-values.
        ``True`` indicates that the null hypothesis is rejected.

    Examples
    --------
    >>> holm_bonferroni_correction([0.01, 0.04, 0.03, 0.001])
    [True, False, False, True]
    """
    _validate_alpha(alpha)
    if not p_values:
        return []
    if any((p_value < 0.0) or (p_value > 1.0) for p_value in p_values):
        raise ValueError("All p-values must be between 0 and 1 inclusive.")

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    decisions = [False] * len(p_values)
    all_previous_rejected = True
    m_tests = len(p_values)

    for rank, (original_index, p_value) in enumerate(indexed, start=1):
        adjusted_alpha = alpha / (m_tests - rank + 1)
        reject = bool(all_previous_rejected and (p_value <= adjusted_alpha))
        decisions[original_index] = reject
        if not reject:
            all_previous_rejected = False

    return decisions
