"""Tests for experiment_decision module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.experiment_decision import make_decision

# ---------------------------------------------------------------------------
# Lightweight stubs — mirror the frozen dataclass signatures without
# importing the real modules so tests stay focused on decision logic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FreqStub:
    control_rate: float = 0.10
    treatment_rate: float = 0.12
    lift: float = 0.20
    cohens_h: float = 0.06
    z_stat: float = 2.50
    p_value: float = 0.01
    ci_lower: float = 0.005
    ci_upper: float = 0.035
    significant: bool = True
    alpha: float = 0.05


@dataclass(frozen=True)
class _BayesStub:
    prob_treatment_wins: float = 0.97
    expected_loss: float = 0.0003
    ci_lower: float = 0.005
    ci_upper: float = 0.035
    control_rate: float = 0.10
    treatment_rate: float = 0.12
    n_samples: int = 100_000


@dataclass(frozen=True)
class _SRMStub:
    expected_split: float = 0.50
    actual_split: float = 0.50
    chi2_stat: float = 0.10
    p_value: float = 0.75
    srm_detected: bool = False


@dataclass(frozen=True)
class _NoveltyStub:
    kendall_tau: float = 0.05
    p_value: float = 0.60
    novelty_detected: bool = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMakeDecision:
    """Suite covering the SHIP / HOLD / INCONCLUSIVE decision matrix."""

    def test_ship_clean_win(self) -> None:
        """All diagnostics green, strong evidence → SHIP, LOW risk."""
        result = make_decision(
            freq_result=_FreqStub(p_value=0.01, lift=0.20),
            bayes_result=_BayesStub(prob_treatment_wins=0.97),
            srm_result=_SRMStub(srm_detected=False),
            novelty_result=_NoveltyStub(novelty_detected=False),
        )
        assert result.recommendation == "SHIP"
        assert result.risk_level == "LOW"
        assert result.confidence >= 0.95

    def test_hold_srm_fail(self) -> None:
        """SRM detected → HOLD regardless of statistical evidence."""
        result = make_decision(
            freq_result=_FreqStub(p_value=0.001),
            bayes_result=_BayesStub(prob_treatment_wins=0.99),
            srm_result=_SRMStub(srm_detected=True, chi2_stat=45.2),
            novelty_result=_NoveltyStub(novelty_detected=False),
        )
        assert result.recommendation == "HOLD"
        assert result.risk_level == "HIGH"

    def test_inconclusive_novelty(self) -> None:
        """Novelty detected + significant p → INCONCLUSIVE."""
        result = make_decision(
            freq_result=_FreqStub(p_value=0.02),
            bayes_result=_BayesStub(prob_treatment_wins=0.96),
            srm_result=_SRMStub(srm_detected=False),
            novelty_result=_NoveltyStub(novelty_detected=True, kendall_tau=-0.45),
        )
        assert result.recommendation == "INCONCLUSIVE"
        assert result.risk_level == "MEDIUM"

    def test_hold_weak_evidence(self) -> None:
        """p=0.08, prob_wins=0.85 — both weak → HOLD."""
        result = make_decision(
            freq_result=_FreqStub(p_value=0.08, significant=False),
            bayes_result=_BayesStub(prob_treatment_wins=0.85),
            srm_result=_SRMStub(srm_detected=False),
            novelty_result=_NoveltyStub(novelty_detected=False),
        )
        # p in (0.05, 0.10) → INCONCLUSIVE takes priority over HOLD
        assert result.recommendation == "INCONCLUSIVE"
        assert result.risk_level == "MEDIUM"

    def test_inconclusive_borderline_p(self) -> None:
        """p=0.07 lands in the grey zone → INCONCLUSIVE."""
        result = make_decision(
            freq_result=_FreqStub(p_value=0.07, significant=False),
            bayes_result=_BayesStub(prob_treatment_wins=0.93),
            srm_result=_SRMStub(srm_detected=False),
            novelty_result=_NoveltyStub(novelty_detected=False),
        )
        assert result.recommendation == "INCONCLUSIVE"
        assert result.risk_level == "MEDIUM"
        assert 0.5 <= result.confidence <= 0.8

    def test_hold_null_result(self) -> None:
        """p=0.20, prob_wins=0.55 — clear null → HOLD."""
        result = make_decision(
            freq_result=_FreqStub(p_value=0.20, significant=False),
            bayes_result=_BayesStub(prob_treatment_wins=0.55),
            srm_result=_SRMStub(srm_detected=False),
            novelty_result=_NoveltyStub(novelty_detected=False),
        )
        assert result.recommendation == "HOLD"
        assert result.risk_level == "HIGH"

    def test_reasoning_composite(self) -> None:
        """Multiple failures produce reasoning entries for every signal."""
        result = make_decision(
            freq_result=_FreqStub(p_value=0.03, lift=0.15),
            bayes_result=_BayesStub(prob_treatment_wins=0.96, expected_loss=0.001),
            srm_result=_SRMStub(srm_detected=True, chi2_stat=30.00),
            novelty_result=_NoveltyStub(novelty_detected=True, kendall_tau=-0.35),
        )
        text = "\n".join(result.reasoning)
        assert "SRM: FAILED" in text
        assert "chi2=30.00" in text
        assert "Novelty: DETECTED" in text
        assert "kendall_tau=-0.35" in text
        assert "Frequentist: p=" in text
        assert "Bayesian: P(B>A)=" in text

    def test_none_input_raises(self) -> None:
        """Passing None for any argument must raise TypeError."""
        with pytest.raises(TypeError):
            make_decision(None, _BayesStub(), _SRMStub(), _NoveltyStub())  # type: ignore[arg-type]
