"""Sensitivity analysis for A/B experiment decisions.

Tests whether the shipping decision is robust to plausible uncertainty
in the observed baseline conversion rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.statistical_tests import z_test_proportions


@dataclass(frozen=True)
class SensitivityResult:
    """Immutable container for baseline-sensitivity analysis.

    Attributes
    ----------
    baseline_assumptions : dict[str, float]
        Mapping from scenario label to the assumed baseline rate.
    decisions : dict[str, bool]
        Mapping from scenario label to whether the result is significant.
    recommendation : {"ROBUST", "FRAGILE", "CONDITIONAL"}
        Overall robustness of the shipping decision.
    interpretation : str
        Plain-language explanation of the recommendation.
    """

    baseline_assumptions: dict[str, float] = field(default_factory=dict)
    decisions: dict[str, bool] = field(default_factory=dict)
    recommendation: Literal["ROBUST", "FRAGILE", "CONDITIONAL"] = "ROBUST"
    interpretation: str = ""


def sensitivity_analysis(
    control_rate_observed: float,
    treatment_rate_observed: float,
    n_control: int,
    n_treatment: int,
    baseline_uncertainty: float = 0.02,
) -> SensitivityResult:
    """Test decision robustness to baseline rate uncertainty.

    Re-runs the z-test under three scenarios where the true control rate
    is shifted by ``-baseline_uncertainty``, ``0``, and
    ``+baseline_uncertainty`` relative to the observed value.

    Parameters
    ----------
    control_rate_observed : float
        Observed control conversion rate.
    treatment_rate_observed : float
        Observed treatment conversion rate.
    n_control : int
        Number of observations in the control group.
    n_treatment : int
        Number of observations in the treatment group.
    baseline_uncertainty : float, default 0.02
        Half-width of the plausible baseline range (absolute).

    Returns
    -------
    SensitivityResult
        Contains per-scenario significance flags and an overall
        robustness recommendation.
    """
    if not 0.0 < control_rate_observed < 1.0:
        raise ValueError("control_rate_observed must be in (0, 1).")
    if not 0.0 < treatment_rate_observed < 1.0:
        raise ValueError("treatment_rate_observed must be in (0, 1).")
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("n_control and n_treatment must be > 0.")
    if baseline_uncertainty < 0.0:
        raise ValueError("baseline_uncertainty must be >= 0.")

    offsets = {
        f"-{baseline_uncertainty:.0%}": -baseline_uncertainty,
        "±0%": 0.0,
        f"+{baseline_uncertainty:.0%}": +baseline_uncertainty,
    }

    assumptions: dict[str, float] = {}
    decisions: dict[str, bool] = {}

    for label, offset in offsets.items():
        adjusted_baseline = control_rate_observed + offset
        # Clamp to valid probability range
        adjusted_baseline = max(1e-6, min(adjusted_baseline, 1.0 - 1e-6))

        control_conv = int(round(adjusted_baseline * n_control))
        treatment_conv = int(round(treatment_rate_observed * n_treatment))

        # Guard against edge cases from rounding
        control_conv = max(0, min(control_conv, n_control))
        treatment_conv = max(0, min(treatment_conv, n_treatment))

        freq = z_test_proportions(
            control_converted=control_conv,
            control_total=n_control,
            treatment_converted=treatment_conv,
            treatment_total=n_treatment,
        )

        assumptions[label] = adjusted_baseline
        decisions[label] = freq.significant

    n_sig = sum(decisions.values())
    if n_sig == 3:
        rec: Literal["ROBUST", "FRAGILE", "CONDITIONAL"] = "ROBUST"
        interp = "Decision holds across baseline uncertainty"
    elif n_sig == 2:
        rec = "CONDITIONAL"
        interp = "Decision depends on accurate baseline"
    else:
        rec = "FRAGILE"
        interp = "Baseline assumption is critical; validate offline"

    return SensitivityResult(
        baseline_assumptions=assumptions,
        decisions=decisions,
        recommendation=rec,
        interpretation=interp,
    )


def display_sensitivity(result: SensitivityResult) -> str:
    """Render a markdown table summarising the sensitivity analysis.

    Parameters
    ----------
    result : SensitivityResult
        Output of :func:`sensitivity_analysis`.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    lines = [
        "| Baseline | Significant |",
        "|----------|-------------|",
    ]
    for label in result.baseline_assumptions:
        rate = result.baseline_assumptions[label]
        sig = "Yes" if result.decisions[label] else "No"
        lines.append(f"| {label} ({rate:.2%}) | {sig} |")

    lines.append(f"\n**Recommendation:** {result.recommendation}")
    lines.append(f"  \n{result.interpretation}")
    return "\n".join(lines)
