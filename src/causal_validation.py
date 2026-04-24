"""Causal assumption validation for A/B experiments.

Checks SUTVA compliance, assignment balance, and confounding risk before
results are interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class CausalValidationResult:
    """Immutable container for experiment assumption checks.

    Attributes
    ----------
    sutva_ok : bool
        True when the randomization unit matches the metric aggregation unit.
    assignment_balanced : bool
        True when the observed variant split is within *tolerance* of the
        expected proportion.
    confounding_risk : {"LOW", "MEDIUM", "HIGH"}
        Overall risk that causal interpretation is compromised.
    warnings : list[str]
        Specific issues discovered during validation.
    notes : str
        Actionable guidance for the experimenter.
    """

    sutva_ok: bool
    assignment_balanced: bool
    confounding_risk: Literal["LOW", "MEDIUM", "HIGH"]
    warnings: list[str] = field(default_factory=list)
    notes: str = ""


def validate_experiment_assumptions(
    df: pd.DataFrame,
    randomization_level: Literal["user", "session", "device"],
    metric_level: Literal["user", "session", "device"],
    expected_split: float = 0.50,
    tolerance: float = 0.05,
) -> CausalValidationResult:
    """Validate core causal assumptions before interpreting experiment results.

    Parameters
    ----------
    df : pandas.DataFrame
        Experiment data containing at least a ``variant`` column with values
        ``"control"`` and ``"treatment"``.
    randomization_level : {"user", "session", "device"}
        Unit at which traffic was randomly assigned.
    metric_level : {"user", "session", "device"}
        Unit at which the success metric is aggregated.
    expected_split : float, default 0.50
        Expected proportion of observations in the control group.
    tolerance : float, default 0.05
        Maximum acceptable absolute deviation from *expected_split*.

    Returns
    -------
    CausalValidationResult
        Frozen dataclass summarising SUTVA compliance, balance, and risk.
    """
    if "variant" not in df.columns:
        raise ValueError("DataFrame must contain a 'variant' column.")

    warnings: list[str] = []
    notes_parts: list[str] = []

    # -- SUTVA --
    sutva_ok = randomization_level == metric_level
    if not sutva_ok:
        warnings.append(
            f"SUTVA violation: randomized at {randomization_level} "
            f"but measuring at {metric_level}"
        )
        notes_parts.append(
            f"Cluster at {randomization_level} level to fix SUTVA"
        )

    # -- Assignment balance --
    counts = df["variant"].value_counts()
    n_control = int(counts.get("control", 0))
    n_total = len(df)
    actual_split = n_control / n_total if n_total > 0 else 0.0
    assignment_balanced = abs(actual_split - expected_split) <= tolerance
    if not assignment_balanced:
        warnings.append(
            f"Split is {actual_split:.1%}, expected {expected_split:.1%}"
        )
        notes_parts.append("Reweight samples for balance")

    # -- Confounding risk --
    if not sutva_ok:
        confounding_risk: Literal["LOW", "MEDIUM", "HIGH"] = "HIGH"
        notes_parts.append("Metric aggregation may bias results")
    elif not assignment_balanced:
        confounding_risk = "MEDIUM"
        notes_parts.append("Unequal sample sizes inflate variance")
    else:
        confounding_risk = "LOW"

    return CausalValidationResult(
        sutva_ok=sutva_ok,
        assignment_balanced=assignment_balanced,
        confounding_risk=confounding_risk,
        warnings=warnings,
        notes="; ".join(notes_parts) if notes_parts else "No issues detected",
    )
