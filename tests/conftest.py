"""Shared pytest fixtures used across test modules."""

import pytest

from src.data_simulation import simulate_ab_test
from src.diagnostics import SRMResult
from src.statistical_tests import BayesianResult, FrequentistResult


@pytest.fixture
def sample_ab_data():
    """Simulated A/B test DataFrame for use across tests."""
    return simulate_ab_test(
        n_control=1000,
        n_treatment=1000,
        control_rate=0.10,
        treatment_rate=0.12,
        seed=42,
    )


@pytest.fixture
def sample_freq_result():
    """Frequentist test result for use across tests."""
    return FrequentistResult(
        control_rate=0.10,
        treatment_rate=0.12,
        lift=0.20,
        z_stat=1.96,
        p_value=0.04,
        ci_lower=0.001,
        ci_upper=0.039,
        significant=True,
        alpha=0.05,
    )


@pytest.fixture
def sample_bayesian_result():
    """Bayesian test result for use across tests."""
    return BayesianResult(
        prob_treatment_wins=0.95,
        expected_loss=0.001,
        ci_lower=0.001,
        ci_upper=0.039,
        control_rate=0.10,
        treatment_rate=0.12,
        n_samples=100000,
    )


@pytest.fixture
def sample_srm_result():
    """SRM check result for use across tests."""
    return SRMResult(
        expected_split=0.50,
        actual_split=0.50,
        chi2_stat=0.0,
        p_value=1.0,
        srm_detected=False,
    )
