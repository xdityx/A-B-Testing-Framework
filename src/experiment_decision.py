"""Decision-rule engine for A/B experiment recommendations.

Integrates frequentist, Bayesian, SRM, and novelty diagnostics into a
single SHIP / HOLD / INCONCLUSIVE recommendation with calibrated
confidence and structured reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.diagnostics import NoveltyResult, SRMResult
from src.statistical_tests import BayesianResult, FrequentistResult


@dataclass(frozen=True)
class ExperimentDecision:
    """Immutable container for an experiment-level shipping recommendation.

    Attributes
    ----------
    recommendation : {"SHIP", "HOLD", "INCONCLUSIVE"}
        The action recommended based on all available evidence.
    confidence : float
        Scalar in [0, 1] expressing how confident the engine is in the
        recommendation.  Higher values indicate stronger evidence.
    reasoning : list[str]
        Human-readable evidence trail, one entry per diagnostic signal.
    risk_level : {"LOW", "MEDIUM", "HIGH"}
        Qualitative risk if the recommendation is followed.
    """

    recommendation: Literal["SHIP", "HOLD", "INCONCLUSIVE"]
    confidence: float
    reasoning: list[str] = field(default_factory=list)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"


def _build_reasoning(
    freq_result: FrequentistResult,
    bayes_result: BayesianResult,
    srm_result: SRMResult,
    novelty_result: NoveltyResult,
) -> list[str]:
    """Assemble a reasoning list from all diagnostic inputs."""
    lines: list[str] = []

    if srm_result.srm_detected:
        lines.append(f"SRM: FAILED - chi2={srm_result.chi2_stat:.2f}")
    else:
        lines.append("SRM: PASSED")

    if novelty_result.novelty_detected:
        lines.append(f"Novelty: DETECTED - kendall_tau={novelty_result.kendall_tau:.2f}")
    else:
        lines.append("Novelty: NOT DETECTED")

    lines.append(f"Frequentist: p={freq_result.p_value:.4f}, lift={freq_result.lift:.2%}")
    lines.append(
        f"Bayesian: P(B>A)={bayes_result.prob_treatment_wins:.4f}, "
        f"EL={bayes_result.expected_loss:.4f}"
    )

    return lines


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def make_decision(
    freq_result: FrequentistResult,
    bayes_result: BayesianResult,
    srm_result: SRMResult,
    novelty_result: NoveltyResult,
) -> ExperimentDecision:
    """Produce a single shipping recommendation from all diagnostic signals.

    Parameters
    ----------
    freq_result : FrequentistResult
        Output of ``z_test_proportions``.
    bayes_result : BayesianResult
        Output of ``bayesian_ab_test``.
    srm_result : SRMResult
        Output of ``check_srm``.
    novelty_result : NoveltyResult
        Output of ``check_novelty_effect``.

    Returns
    -------
    ExperimentDecision
        Frozen dataclass with recommendation, confidence, reasoning, and
        risk level.

    Raises
    ------
    TypeError
        If any input is ``None``.
    """
    if any(arg is None for arg in (freq_result, bayes_result, srm_result, novelty_result)):
        raise TypeError("All four diagnostic inputs are required (got None).")

    reasoning = _build_reasoning(freq_result, bayes_result, srm_result, novelty_result)

    p = freq_result.p_value
    prob_wins = bayes_result.prob_treatment_wins
    srm_fail = srm_result.srm_detected
    novelty = novelty_result.novelty_detected

    evidence_strength = max(1.0 - p, prob_wins)

    # ---- HOLD: hard vetoes first ----
    if srm_fail:
        return ExperimentDecision(
            recommendation="HOLD",
            confidence=_clamp(1.0 - evidence_strength),
            reasoning=reasoning,
            risk_level="HIGH",
        )

    if p > 0.10 and prob_wins < 0.90:
        return ExperimentDecision(
            recommendation="HOLD",
            confidence=_clamp(1.0 - evidence_strength),
            reasoning=reasoning,
            risk_level="HIGH",
        )

    # ---- INCONCLUSIVE: ambiguous signals ----
    if novelty and p < 0.05:
        return ExperimentDecision(
            recommendation="INCONCLUSIVE",
            confidence=_clamp(0.5 + 0.3 * evidence_strength, 0.5, 0.8),
            reasoning=reasoning,
            risk_level="MEDIUM",
        )

    if 0.05 <= p <= 0.10:
        return ExperimentDecision(
            recommendation="INCONCLUSIVE",
            confidence=_clamp(0.5 + 0.3 * evidence_strength, 0.5, 0.8),
            reasoning=reasoning,
            risk_level="MEDIUM",
        )

    if 0.80 <= prob_wins <= 0.95:
        # Only inconclusive on Bayesian alone when frequentist isn't decisive
        if p >= 0.05:
            return ExperimentDecision(
                recommendation="INCONCLUSIVE",
                confidence=_clamp(0.5 + 0.3 * evidence_strength, 0.5, 0.8),
                reasoning=reasoning,
                risk_level="MEDIUM",
            )

    # ---- SHIP: clean win ----
    if not novelty and (p < 0.05 or prob_wins > 0.95):
        return ExperimentDecision(
            recommendation="SHIP",
            confidence=_clamp(evidence_strength),
            reasoning=reasoning,
            risk_level="LOW",
        )

    # ---- Fallback: nothing matched → HOLD ----
    return ExperimentDecision(
        recommendation="HOLD",
        confidence=_clamp(1.0 - evidence_strength),
        reasoning=reasoning,
        risk_level="HIGH",
    )
