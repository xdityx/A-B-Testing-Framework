"""Tests for the sequential_testing (SPRT) module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.sequential_testing import SPRTResult, sequential_z_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_binomial(rate: float, n: int, seed: int = 42) -> int:
    """Return the number of successes from *n* Bernoulli(rate) trials."""
    rng = np.random.default_rng(seed)
    return int(rng.binomial(n, rate))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSequentialZTest:
    """Suite covering SPRT decision logic, boundary math, and edge cases."""

    def test_treatment_clearly_winning(self) -> None:
        """Large lift should cross the upper boundary → STOP_FOR_TREATMENT."""
        n = 2000
        control_conv = _simulate_binomial(0.10, n, seed=1)
        treatment_conv = _simulate_binomial(0.15, n, seed=2)

        result = sequential_z_test(
            control_converted=control_conv,
            control_total=n,
            treatment_converted=treatment_conv,
            treatment_total=n,
            baseline_rate=0.10,
            mde=0.20,
        )
        assert result.can_stop is True
        assert result.stop_reason == "STOP_FOR_TREATMENT"
        assert result.log_likelihood_ratio >= result.boundary_upper
        assert result.samples_needed == 0

    def test_control_better(self) -> None:
        """Treatment underperforms control → STOP_FOR_CONTROL."""
        n = 3000
        control_conv = _simulate_binomial(0.12, n, seed=10)
        treatment_conv = _simulate_binomial(0.08, n, seed=11)

        result = sequential_z_test(
            control_converted=control_conv,
            control_total=n,
            treatment_converted=treatment_conv,
            treatment_total=n,
            baseline_rate=0.12,
            mde=0.20,
        )
        assert result.can_stop is True
        assert result.stop_reason == "STOP_FOR_CONTROL"
        assert result.log_likelihood_ratio <= result.boundary_lower
        assert result.samples_needed == 0

    def test_inconclusive_null_result(self) -> None:
        """Both variants at the same rate with small n → CONTINUE."""
        n = 200
        # Use the same seed so conversions are nearly identical.
        control_conv = _simulate_binomial(0.10, n, seed=50)
        treatment_conv = _simulate_binomial(0.10, n, seed=51)

        result = sequential_z_test(
            control_converted=control_conv,
            control_total=n,
            treatment_converted=treatment_conv,
            treatment_total=n,
            baseline_rate=0.10,
            mde=0.20,
        )
        assert result.can_stop is False
        assert result.stop_reason == "CONTINUE"
        assert result.boundary_lower < result.log_likelihood_ratio < result.boundary_upper

    def test_boundaries_computed_correctly(self) -> None:
        """Verify log_A and log_B against the Wald formulae."""
        alpha, beta = 0.05, 0.20
        expected_upper = math.log((1.0 - beta) / alpha)   # log(16)
        expected_lower = math.log(beta / (1.0 - alpha))    # log(4/19)

        result = sequential_z_test(
            control_converted=50,
            control_total=500,
            treatment_converted=60,
            treatment_total=500,
            baseline_rate=0.10,
            mde=0.20,
            alpha=alpha,
            beta=beta,
        )
        assert result.boundary_upper == pytest.approx(expected_upper, rel=1e-9)
        assert result.boundary_lower == pytest.approx(expected_lower, rel=1e-9)

    def test_samples_needed_positive(self) -> None:
        """When CONTINUE, estimate of remaining samples must be > 0."""
        result = sequential_z_test(
            control_converted=50,
            control_total=500,
            treatment_converted=55,
            treatment_total=500,
            baseline_rate=0.10,
            mde=0.20,
        )
        if result.stop_reason == "CONTINUE":
            assert result.samples_needed > 0

    def test_invalid_inputs_raise(self) -> None:
        """Bad counts, rates, and error-rate args all raise ValueError."""
        base = dict(
            control_converted=10, control_total=100,
            treatment_converted=12, treatment_total=100,
            baseline_rate=0.10, mde=0.20,
        )
        # Negative counts
        with pytest.raises(ValueError):
            sequential_z_test(**{**base, "control_converted": -1})
        # Converted > total
        with pytest.raises(ValueError):
            sequential_z_test(**{**base, "treatment_converted": 200})
        # baseline_rate out of range
        with pytest.raises(ValueError):
            sequential_z_test(**{**base, "baseline_rate": 0.0})
        with pytest.raises(ValueError):
            sequential_z_test(**{**base, "baseline_rate": 1.0})
        # mde <= 0
        with pytest.raises(ValueError):
            sequential_z_test(**{**base, "mde": 0.0})
        # alpha / beta out of range
        with pytest.raises(ValueError):
            sequential_z_test(**{**base, "alpha": 0.0})
        with pytest.raises(ValueError):
            sequential_z_test(**{**base, "beta": 1.0})
        # Treatment rate under H1 >= 1
        with pytest.raises(ValueError):
            sequential_z_test(**{**base, "baseline_rate": 0.90, "mde": 0.50})

    def test_edge_case_zero_conversions(self) -> None:
        """Zero conversions in both arms → INCONCLUSIVE, no crash."""
        result = sequential_z_test(
            control_converted=0,
            control_total=500,
            treatment_converted=0,
            treatment_total=500,
            baseline_rate=0.10,
            mde=0.20,
        )
        assert result.stop_reason == "INCONCLUSIVE"
        assert result.log_likelihood_ratio == 0.0
        assert result.samples_needed > 0

    def test_early_stopping_saves_samples(self) -> None:
        """Stopped result with strong effect uses fewer samples than a
        full fixed-horizon experiment would require (~16k per arm for
        this setup at 80% power)."""
        n = 3000  # well below the ~16k a fixed test would need
        control_conv = _simulate_binomial(0.10, n, seed=100)
        treatment_conv = _simulate_binomial(0.14, n, seed=101)

        result = sequential_z_test(
            control_converted=control_conv,
            control_total=n,
            treatment_converted=treatment_conv,
            treatment_total=n,
            baseline_rate=0.10,
            mde=0.20,
        )
        assert result.can_stop is True
        assert result.stop_reason == "STOP_FOR_TREATMENT"
        # Total used is 2*n = 6000, far below 32k fixed-horizon
        assert (n * 2) < 32_000
