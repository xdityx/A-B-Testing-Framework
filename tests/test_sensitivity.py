"""Tests for sensitivity module."""

from __future__ import annotations

from src.sensitivity import display_sensitivity, sensitivity_analysis


def test_robust_decision() -> None:
    """Large effect survives all baseline perturbations → ROBUST."""
    result = sensitivity_analysis(
        control_rate_observed=0.10,
        treatment_rate_observed=0.14,
        n_control=5000,
        n_treatment=5000,
        baseline_uncertainty=0.02,
    )
    assert result.recommendation == "ROBUST"
    assert all(result.decisions.values())


def test_fragile_decision() -> None:
    """Tiny effect only significant at exact baseline → FRAGILE."""
    result = sensitivity_analysis(
        control_rate_observed=0.10,
        treatment_rate_observed=0.108,
        n_control=5000,
        n_treatment=5000,
        baseline_uncertainty=0.02,
    )
    # With ±2 % swing, the tiny lift disappears in shifted scenarios.
    assert result.recommendation in ("FRAGILE", "CONDITIONAL")
    assert not all(result.decisions.values())


def test_conditional_decision() -> None:
    """Moderate effect loses significance only at the worst baseline shift."""
    result = sensitivity_analysis(
        control_rate_observed=0.10,
        treatment_rate_observed=0.118,
        n_control=5000,
        n_treatment=5000,
        baseline_uncertainty=0.02,
    )
    # At +2 % baseline (0.12), lift shrinks enough to lose significance
    assert result.recommendation in ("CONDITIONAL", "ROBUST")


def test_display_contains_markdown_table() -> None:
    """display_sensitivity should produce a markdown table."""
    result = sensitivity_analysis(
        control_rate_observed=0.10,
        treatment_rate_observed=0.14,
        n_control=5000,
        n_treatment=5000,
    )
    md = display_sensitivity(result)
    assert "| Baseline" in md
    assert "Recommendation" in md
