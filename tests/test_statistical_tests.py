"""Tests for statistical testing utilities."""

import numpy as np
import pytest

from src.statistical_tests import (
    bayesian_ab_test,
    t_test_continuous,
    z_test_proportions,
)


def test_z_test_significant_with_large_sample() -> None:
    result = z_test_proportions(1000, 10000, 1200, 10000)

    assert result.significant is True
    assert result.p_value < 0.05
    assert result.lift == pytest.approx(0.20, rel=0.01)
    assert result.cohens_h > 0.0  # positive effect size when treatment > control


def test_z_test_not_significant_identical_rates() -> None:
    result = z_test_proportions(100, 1000, 100, 1000)

    assert result.significant is False
    assert result.z_stat == pytest.approx(0.0, abs=1e-10)
    assert result.p_value == pytest.approx(1.0, rel=0.01)
    assert result.cohens_h == pytest.approx(0.0, abs=1e-10)


def test_z_test_ci_contains_zero_for_null() -> None:
    result = z_test_proportions(100, 1000, 100, 1000)

    assert result.ci_lower < 0.0 < result.ci_upper


def test_z_test_invalid_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError):
        z_test_proportions(101, 100, 100, 1000)

    with pytest.raises(ValueError):
        z_test_proportions(10, 0, 10, 100)


def test_t_test_significant_different_means() -> None:
    rng = np.random.default_rng(42)
    control = rng.normal(45.0, 10.0, 5000)
    treatment = rng.normal(50.0, 10.0, 5000)

    result = t_test_continuous(control, treatment)

    assert result.significant is True
    assert result.p_value < 0.05
    assert result.test_type == "welch_t"


def test_t_test_same_distribution_not_significant() -> None:
    rng = np.random.default_rng(42)
    control = rng.normal(45.0, 10.0, 1000)
    treatment = rng.normal(45.0, 10.0, 1000)

    result = t_test_continuous(control, treatment)

    assert result.significant is False
    assert result.test_type == "welch_t"


def test_t_test_lift_direction() -> None:
    rng = np.random.default_rng(42)
    control = rng.normal(45.0, 10.0, 5000)
    treatment = rng.normal(50.0, 10.0, 5000)

    result = t_test_continuous(control, treatment)

    assert result.lift > 0.0


def test_mann_whitney_significant_different_means() -> None:
    import math

    rng = np.random.default_rng(42)
    control = rng.normal(45.0, 10.0, 5000)
    treatment = rng.normal(50.0, 10.0, 5000)

    result = t_test_continuous(control, treatment, use_mann_whitney=True)

    assert result.significant is True
    assert result.p_value < 0.05
    assert result.test_type == "mann_whitney"
    assert result.lift > 0.0
    # Hodges-Lehmann CI: exact CIs unavailable in scipy → both bounds are NaN
    assert math.isnan(result.ci_lower)
    assert math.isnan(result.ci_upper)


def test_bayesian_high_prob_treatment_wins() -> None:
    result = bayesian_ab_test(800, 10000, 1000, 10000, seed=42)

    assert result.prob_treatment_wins > 0.95


def test_bayesian_aa_scenario_prob_near_half() -> None:
    result = bayesian_ab_test(500, 5000, 500, 5000, seed=42)

    assert 0.40 < result.prob_treatment_wins < 0.60


def test_bayesian_expected_loss_non_negative() -> None:
    result = bayesian_ab_test(500, 5000, 500, 5000, seed=42)

    assert result.expected_loss >= 0.0


def test_bayesian_credible_interval_contains_zero_for_aa() -> None:
    result = bayesian_ab_test(500, 5000, 500, 5000, seed=42)

    assert result.ci_lower < 0.0 < result.ci_upper
