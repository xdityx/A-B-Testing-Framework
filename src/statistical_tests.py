"""Statistical testing utilities for A/B experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.stats


@dataclass(frozen=True)
class FrequentistResult:
    """Container for two-proportion z-test results."""

    control_rate: float  # observed conversion rate in control
    treatment_rate: float  # observed conversion rate in treatment
    lift: float  # relative lift = (treatment_rate - control_rate) / control_rate
    cohens_h: float  # Cohen's h effect size = 2*arcsin(sqrt(p_t)) - 2*arcsin(sqrt(p_c))
    z_stat: float  # z-test statistic
    p_value: float  # two-sided p-value
    ci_lower: float  # lower bound of 95% CI on absolute difference
    ci_upper: float  # upper bound of 95% CI on absolute difference
    significant: bool  # True if p_value < alpha
    alpha: float  # significance level used


@dataclass(frozen=True)
class ContinuousResult:
    """Container for continuous outcome test results."""

    control_mean: float  # mean of control group
    treatment_mean: float  # mean of treatment group
    lift: float  # relative lift = (treatment_mean - control_mean) / control_mean
    t_stat: float  # test statistic (t-stat or U-stat depending on test_type)
    p_value: float  # two-sided p-value
    ci_lower: float  # lower bound of 95% CI on difference in means
    ci_upper: float  # upper bound of 95% CI on difference in means
    significant: bool  # True if p_value < alpha
    alpha: float  # significance level used
    test_type: str  # "welch_t" or "mann_whitney"


@dataclass(frozen=True)
class BayesianResult:
    """Container for Bayesian A/B posterior summaries."""

    prob_treatment_wins: float  # P(treatment > control) via Monte Carlo
    expected_loss: float  # E[max(control - treatment, 0)] - risk of choosing treatment
    ci_lower: float  # 2.5th percentile of (treatment - control) posterior
    ci_upper: float  # 97.5th percentile of (treatment - control) posterior
    control_rate: float  # posterior mean of control
    treatment_rate: float  # posterior mean of treatment
    n_samples: int  # number of Monte Carlo samples used


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1).")


def _validate_conversion_inputs(
    control_converted: int,
    control_total: int,
    treatment_converted: int,
    treatment_total: int,
) -> None:
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("control_total and treatment_total must be greater than 0.")
    if control_converted > control_total or treatment_converted > treatment_total:
        raise ValueError("Converted counts cannot exceed total counts.")
    if control_converted < 0 or treatment_converted < 0:
        raise ValueError("Converted counts cannot be negative.")


def _relative_lift(baseline: float, comparison: float) -> float:
    if baseline == 0.0:
        if comparison == 0.0:
            return 0.0
        return math.inf
    return (comparison - baseline) / baseline


def z_test_proportions(
    control_converted: int,
    control_total: int,
    treatment_converted: int,
    treatment_total: int,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Run a two-sided z-test for a difference in conversion proportions.

    Parameters
    ----------
    control_converted : int
        Number of converted users observed in the control group.
    control_total : int
        Total number of users observed in the control group.
    treatment_converted : int
        Number of converted users observed in the treatment group.
    treatment_total : int
        Total number of users observed in the treatment group.
    alpha : float, default=0.05
        Significance level used for the confidence interval and significance flag.

    Returns
    -------
    FrequentistResult
        Dataclass containing observed rates, relative lift, z-statistic, p-value,
        confidence interval on the absolute rate difference, and the significance flag.

    Examples
    --------
    >>> result = z_test_proportions(1000, 10000, 1200, 10000)
    >>> result.significant
    True
    >>> round(result.lift, 2)
    0.2
    """
    _validate_conversion_inputs(
        control_converted=control_converted,
        control_total=control_total,
        treatment_converted=treatment_converted,
        treatment_total=treatment_total,
    )
    _validate_alpha(alpha)

    p_c = control_converted / control_total
    p_t = treatment_converted / treatment_total

    p_pool = (control_converted + treatment_converted) / (control_total + treatment_total)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / control_total + 1.0 / treatment_total))
    diff = p_t - p_c
    if se_pool == 0.0:
        z_stat = 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
    else:
        z_stat = diff / se_pool

    p_value = 2.0 * (1.0 - scipy.stats.norm.cdf(abs(z_stat)))

    se_diff = math.sqrt(p_c * (1.0 - p_c) / control_total + p_t * (1.0 - p_t) / treatment_total)
    z_critical = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
    ci_lower = diff - z_critical * se_diff
    ci_upper = diff + z_critical * se_diff

    cohens_h = float(
        2.0 * math.asin(math.sqrt(p_t)) - 2.0 * math.asin(math.sqrt(p_c))
    )

    return FrequentistResult(
        control_rate=float(p_c),
        treatment_rate=float(p_t),
        lift=float(_relative_lift(p_c, p_t)),
        cohens_h=cohens_h,
        z_stat=float(z_stat),
        p_value=float(p_value),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        significant=bool(p_value < alpha),
        alpha=float(alpha),
    )


def t_test_continuous(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    alpha: float = 0.05,
    use_mann_whitney: bool = False,
) -> ContinuousResult:
    """Run a two-sided test for continuous outcomes.

    Parameters
    ----------
    control_values : numpy.ndarray
        Sample of continuous observations from the control group.
    treatment_values : numpy.ndarray
        Sample of continuous observations from the treatment group.
    alpha : float, default=0.05
        Significance level used for the confidence interval and significance flag.
    use_mann_whitney : bool, default=False
        If True, use the non-parametric Mann-Whitney U test instead of
        Welch's t-test. Useful when data is non-normal or heavily skewed.
        When True, the point estimate reported in ``ci_lower``/``ci_upper``
        is replaced by the Hodges-Lehmann estimator (median of all pairwise
        differences). Exact CIs for Mann-Whitney are not available in scipy;
        both bounds are set to ``float('nan')`` in that case.

    Returns
    -------
    ContinuousResult
        Dataclass containing sample means, relative lift, test statistic,
        p-value, confidence interval on the difference in means, the
        significance flag, and which test was used.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> control = rng.normal(45.0, 10.0, 500)
    >>> treatment = rng.normal(47.0, 10.0, 500)
    >>> result = t_test_continuous(control, treatment)
    >>> result.treatment_mean > result.control_mean
    True
    >>> result.test_type
    'welch_t'
    """
    _validate_alpha(alpha)

    control_array = np.asarray(control_values, dtype=float).ravel()
    treatment_array = np.asarray(treatment_values, dtype=float).ravel()
    if control_array.size < 2 or treatment_array.size < 2:
        raise ValueError("Each input array must contain at least 2 observations.")

    control_mean = float(np.mean(control_array))
    treatment_mean = float(np.mean(treatment_array))
    diff = treatment_mean - control_mean

    if use_mann_whitney:
        mw_result = scipy.stats.mannwhitneyu(
            treatment_array,
            control_array,
            alternative="two-sided",
        )
        stat = float(mw_result.statistic)
        p_value = float(mw_result.pvalue)
        test_type = "mann_whitney"
    else:
        t_test_result = scipy.stats.ttest_ind(
            treatment_array,
            control_array,
            equal_var=False,
        )
        stat = float(t_test_result.statistic)
        p_value = float(t_test_result.pvalue)
        test_type = "welch_t"

    # Confidence interval branch
    if use_mann_whitney:
        # Hodges-Lehmann point estimate: median of all pairwise differences
        # treatment_i - control_j across the full cross-product of samples.
        # Exact Mann-Whitney CIs are not implemented in scipy; ci bounds are NaN.
        pairwise_diffs = (
            treatment_array[:, np.newaxis] - control_array[np.newaxis, :]
        ).ravel()
        hl_estimate = float(np.median(pairwise_diffs))
        ci_lower = float("nan")
        ci_upper = float("nan")
        # Use HL estimate as the primary difference measure for reporting
        diff = hl_estimate
    else:
        # Use exact df from scipy's Welch's t-test result
        treatment_std = float(np.std(treatment_array, ddof=1))
        control_std = float(np.std(control_array, ddof=1))
        n_treatment = treatment_array.size
        n_control = control_array.size
        se = math.sqrt(treatment_std**2 / n_treatment + control_std**2 / n_control)
        df = float(t_test_result.df)
        t_critical = scipy.stats.t.ppf(1.0 - alpha / 2.0, df)
        ci_lower = diff - t_critical * se
        ci_upper = diff + t_critical * se

    return ContinuousResult(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        lift=float(_relative_lift(control_mean, treatment_mean)),
        t_stat=stat,
        p_value=p_value,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        significant=bool(p_value < alpha),
        alpha=float(alpha),
        test_type=test_type,
    )


def bayesian_ab_test(
    control_converted: int,
    control_total: int,
    treatment_converted: int,
    treatment_total: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> BayesianResult:
    """Run a Bayesian A/B test with Beta-Binomial posteriors.

    Parameters
    ----------
    control_converted : int
        Number of converted users observed in the control group.
    control_total : int
        Total number of users observed in the control group.
    treatment_converted : int
        Number of converted users observed in the treatment group.
    treatment_total : int
        Total number of users observed in the treatment group.
    prior_alpha : float, default=1.0
        Alpha parameter of the Beta prior for both variants.
    prior_beta : float, default=1.0
        Beta parameter of the Beta prior for both variants.
    n_samples : int, default=100000
        Number of Monte Carlo posterior draws used for approximation.
    seed : int or None, default=None
        Seed passed to :func:`numpy.random.default_rng`. If None, a random
        seed is used (non-reproducible).

    Returns
    -------
    BayesianResult
        Dataclass containing the posterior probability that treatment beats
        control, expected loss from choosing treatment, a 95% credible interval
        on the posterior rate difference, posterior mean rates, and the Monte
        Carlo sample count.

    Examples
    --------
    >>> result = bayesian_ab_test(800, 10000, 1000, 10000, seed=42)
    >>> result.prob_treatment_wins > 0.95
    True
    """
    _validate_conversion_inputs(
        control_converted=control_converted,
        control_total=control_total,
        treatment_converted=treatment_converted,
        treatment_total=treatment_total,
    )
    if prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("prior_alpha and prior_beta must be greater than 0.")
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0.")

    rng = np.random.default_rng(seed)

    control_alpha_post = prior_alpha + control_converted
    control_beta_post = prior_beta + (control_total - control_converted)
    treatment_alpha_post = prior_alpha + treatment_converted
    treatment_beta_post = prior_beta + (treatment_total - treatment_converted)

    control_samples = rng.beta(control_alpha_post, control_beta_post, size=n_samples)
    treatment_samples = rng.beta(
        treatment_alpha_post,
        treatment_beta_post,
        size=n_samples,
    )

    prob_treatment_wins = float(np.mean(treatment_samples > control_samples))
    expected_loss = float(np.mean(np.maximum(control_samples - treatment_samples, 0.0)))

    diff_samples = treatment_samples - control_samples
    ci_lower, ci_upper = np.percentile(diff_samples, [2.5, 97.5])

    control_rate = control_alpha_post / (control_alpha_post + control_beta_post)
    treatment_rate = treatment_alpha_post / (treatment_alpha_post + treatment_beta_post)

    return BayesianResult(
        prob_treatment_wins=prob_treatment_wins,
        expected_loss=expected_loss,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        control_rate=float(control_rate),
        treatment_rate=float(treatment_rate),
        n_samples=int(n_samples),
    )
