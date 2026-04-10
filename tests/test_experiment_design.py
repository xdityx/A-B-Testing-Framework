"""Tests for experiment design utilities."""

import math

import pytest

from src.experiment_design import (
    calculate_mde,
    calculate_sample_size,
    experiment_duration_days,
)


def test_calculate_sample_size_returns_per_variant_count() -> None:
    assert calculate_sample_size(0.10, 0.05) == 57756


def test_calculate_mde_reverses_sample_size_calculation() -> None:
    sample_size = calculate_sample_size(0.10, 0.05)

    assert math.isclose(calculate_mde(0.10, sample_size), 0.05, rel_tol=0.001)


def test_experiment_duration_days_uses_allocated_variant_traffic() -> None:
    assert experiment_duration_days(5000, daily_traffic=10000, traffic_split=0.50) == 1.0


def test_invalid_sample_size_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError):
        calculate_sample_size(0.10, 0)
