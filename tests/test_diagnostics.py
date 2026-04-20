"""Tests for diagnostic utilities."""

import pandas as pd

from src.diagnostics import (
    check_novelty_effect,
    check_srm,
    holm_bonferroni_correction,
)


def test_srm_not_detected_balanced() -> None:
    result = check_srm(5000, 5000)

    assert result.srm_detected is False
    assert result.actual_split == 0.5


def test_srm_detected_imbalanced() -> None:
    result = check_srm(5100, 4900, alpha=0.05)

    assert result.srm_detected is True


def test_novelty_detected_decaying_trend() -> None:
    daily_rates = pd.Series([0.10, 0.09, 0.08, 0.07, 0.06])

    result = check_novelty_effect(daily_rates)

    assert result.novelty_detected is True
    assert result.kendall_tau < 0.0


def test_novelty_not_detected_flat_trend() -> None:
    daily_rates = pd.Series([0.10, 0.10, 0.11, 0.09, 0.10])

    result = check_novelty_effect(daily_rates)

    assert result.novelty_detected is False


def test_holm_bonferroni_correction() -> None:
    results = holm_bonferroni_correction([0.01, 0.04, 0.03, 0.001])

    assert results == [True, False, False, True]
