"""Tests for causal_validation module."""

from __future__ import annotations

import pandas as pd

from src.causal_validation import validate_experiment_assumptions


def _make_df(n_control: int, n_treatment: int) -> pd.DataFrame:
    """Build a minimal experiment DataFrame."""
    variants = ["control"] * n_control + ["treatment"] * n_treatment
    return pd.DataFrame({"variant": variants})


def test_sutva_violation() -> None:
    """Randomisation and metric levels differ → sutva_ok=False, HIGH risk."""
    df = _make_df(500, 500)
    result = validate_experiment_assumptions(
        df, randomization_level="session", metric_level="user",
    )
    assert result.sutva_ok is False
    assert result.confounding_risk == "HIGH"
    assert any("SUTVA" in w for w in result.warnings)


def test_balanced_assignment() -> None:
    """Exact 50/50 split → assignment_balanced=True, LOW risk."""
    df = _make_df(500, 500)
    result = validate_experiment_assumptions(
        df, randomization_level="user", metric_level="user",
    )
    assert result.assignment_balanced is True
    assert result.confounding_risk == "LOW"
    assert result.warnings == []


def test_imbalanced_assignment() -> None:
    """60/40 split exceeds tolerance → assignment_balanced=False."""
    df = _make_df(600, 400)
    result = validate_experiment_assumptions(
        df, randomization_level="user", metric_level="user",
    )
    assert result.assignment_balanced is False
    assert result.confounding_risk == "MEDIUM"
    assert any("Split" in w for w in result.warnings)
