"""Experiment design utilities."""

import math

from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize


def _validate_rate(name: str, value: float) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1, exclusive.")


def _validate_alpha_power(alpha: float, power: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1, exclusive.")
    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1, exclusive.")


def calculate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate the required sample size per experiment variant.

    Parameters
    ----------
    baseline_rate:
        Baseline conversion rate for the control variant, expressed as a
        proportion between 0 and 1.
    mde:
        Minimum detectable effect as a relative lift over the baseline rate.
        For example, ``0.05`` represents a 5% lift.
    alpha:
        Significance level for the two-sided test.
    power:
        Desired statistical power for detecting the effect.

    Returns
    -------
    int
        Required sample size per variant, rounded up to the nearest whole user.
    """
    _validate_rate("baseline_rate", baseline_rate)
    _validate_alpha_power(alpha, power)
    if mde <= 0:
        raise ValueError("mde must be greater than 0.")

    treatment_rate = baseline_rate * (1 + mde)
    if treatment_rate >= 1:
        raise ValueError("baseline_rate and mde imply a treatment rate of 1 or greater.")

    effect_size = abs(proportion_effectsize(baseline_rate, treatment_rate))
    sample_size = zt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    return math.ceil(sample_size)


def calculate_mde(
    baseline_rate: float,
    sample_size: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Calculate the minimum detectable relative effect for a fixed sample size.

    Parameters
    ----------
    baseline_rate:
        Baseline conversion rate for the control variant, expressed as a
        proportion between 0 and 1.
    sample_size:
        Available sample size per experiment variant.
    alpha:
        Significance level for the two-sided test.
    power:
        Desired statistical power for detecting the effect.

    Returns
    -------
    float
        Minimum detectable effect as a relative lift over the baseline rate.
        For example, ``0.05`` represents a 5% lift.
    """
    _validate_rate("baseline_rate", baseline_rate)
    _validate_alpha_power(alpha, power)
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0.")

    effect_size = zt_ind_solve_power(
        nobs1=sample_size,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    target_effect_size = abs(effect_size)
    maximum_effect_size = abs(proportion_effectsize(baseline_rate, 1.0))
    if target_effect_size > maximum_effect_size:
        raise ValueError("sample_size is too small to detect any achievable lift.")

    low = baseline_rate
    high = 1.0
    for _ in range(100):
        midpoint = (low + high) / 2
        midpoint_effect_size = abs(proportion_effectsize(baseline_rate, midpoint))
        if midpoint_effect_size >= target_effect_size:
            high = midpoint
        else:
            low = midpoint

    detectable_rate = high
    return (detectable_rate - baseline_rate) / baseline_rate


def experiment_duration_days(
    sample_size_per_variant: int,
    daily_traffic: int,
    traffic_split: float = 0.50,
) -> float:
    """Estimate the number of days needed to collect the target sample size.

    Parameters
    ----------
    sample_size_per_variant:
        Required sample size for each experiment variant.
    daily_traffic:
        Total daily traffic available for the experiment.
    traffic_split:
        Share of total daily traffic allocated to each variant. For a balanced
        A/B test, this is typically ``0.50``.

    Returns
    -------
    float
        Number of days required to reach the sample size per variant.
    """
    if sample_size_per_variant <= 0:
        raise ValueError("sample_size_per_variant must be greater than 0.")
    if daily_traffic <= 0:
        raise ValueError("daily_traffic must be greater than 0.")
    if not 0 < traffic_split <= 1:
        raise ValueError("traffic_split must be greater than 0 and no more than 1.")

    return sample_size_per_variant / (daily_traffic * traffic_split)
