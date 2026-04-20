"""Data simulation utilities for A/A and A/B test scenarios."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Final

import numpy as np
import pandas as pd

_DAY_OF_WEEK_WEIGHTS: Final[np.ndarray] = np.array(
    [0.12, 0.13, 0.14, 0.13, 0.15, 0.18, 0.15],
    dtype=float,
)


def _validate_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0 inclusive.")


def _resolve_start_date(start_date: str | None) -> pd.Timestamp:
    if start_date is None:
        return pd.Timestamp(date.today() - timedelta(days=30))
    return pd.Timestamp(start_date).normalize()


def _generate_timestamps(
    rng: np.random.Generator,
    n_users: int,
    start_timestamp: pd.Timestamp,
    duration_days: int,
) -> pd.Series:
    date_range = pd.date_range(start=start_timestamp, periods=duration_days, freq="D")
    weekday_weights = _DAY_OF_WEEK_WEIGHTS[date_range.dayofweek.to_numpy()]
    probabilities = weekday_weights / weekday_weights.sum()
    selected_days = rng.choice(
        date_range.to_numpy(dtype="datetime64[ns]"),
        size=n_users,
        replace=True,
        p=probabilities,
    )
    hours = rng.integers(0, 24, size=n_users)
    minutes = rng.integers(0, 60, size=n_users)
    timestamps = (
        pd.to_datetime(selected_days)
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )
    return pd.Series(timestamps, dtype="datetime64[ns]")


def _simulate_revenue(
    rng: np.random.Generator,
    converted: np.ndarray,
    avg_revenue_per_converter: float,
    revenue_variance: float,
) -> np.ndarray:
    revenue = np.zeros(converted.shape[0], dtype=float)
    converter_count = int(converted.sum())
    if converter_count == 0:
        return revenue

    revenue[converted == 1] = rng.lognormal(
        mean=float(np.log(avg_revenue_per_converter)),
        sigma=revenue_variance,
        size=converter_count,
    )
    return revenue


def simulate_ab_test(
    n_control: int,
    n_treatment: int,
    control_rate: float,
    treatment_rate: float,
    avg_revenue_per_converter: float = 45.0,
    revenue_variance: float = 0.8,
    start_date: str | None = None,
    duration_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate user-level data for a two-variant A/B test.

    Parameters
    ----------
    n_control : int
        Number of users assigned to the control variant.
    n_treatment : int
        Number of users assigned to the treatment variant.
    control_rate : float
        Conversion rate used to sample binary outcomes for control users.
    treatment_rate : float
        Conversion rate used to sample binary outcomes for treatment users.
    avg_revenue_per_converter : float, default=45.0
        Scale parameter used in the lognormal revenue model for converted users.
    revenue_variance : float, default=0.8
        Lognormal sigma used when sampling revenue for converted users.
    start_date : str or None, default=None
        Experiment start date in ISO format (``YYYY-MM-DD``). When omitted,
        the start date defaults to 30 days before the current date.
    duration_days : int, default=30
        Number of days in the experiment window. User timestamps are sampled
        from the half-open interval ``[start_date, start_date + duration_days)``.
    seed : int, default=42
        Seed passed to :func:`numpy.random.default_rng`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``user_id``, ``variant``, ``converted``,
        ``revenue``, ``timestamp``, and ``day_of_week``. Rows are sorted by
        ascending timestamp.

    Examples
    --------
    >>> df = simulate_ab_test(1000, 1000, 0.10, 0.12, seed=42)
    >>> list(df.columns)
    ['user_id', 'variant', 'converted', 'revenue', 'timestamp', 'day_of_week']
    >>> set(df['variant'])
    {'control', 'treatment'}
    """
    _validate_non_negative_int("n_control", n_control)
    _validate_non_negative_int("n_treatment", n_treatment)
    _validate_probability("control_rate", control_rate)
    _validate_probability("treatment_rate", treatment_rate)
    _validate_positive_int("duration_days", duration_days)
    if avg_revenue_per_converter <= 0.0:
        raise ValueError("avg_revenue_per_converter must be greater than 0.")
    if revenue_variance < 0.0:
        raise ValueError("revenue_variance must be non-negative.")

    rng = np.random.default_rng(seed)
    start_timestamp = _resolve_start_date(start_date)

    control_converted = rng.binomial(1, control_rate, size=n_control).astype(np.int64)
    treatment_converted = rng.binomial(1, treatment_rate, size=n_treatment).astype(np.int64)

    control_revenue = _simulate_revenue(
        rng,
        control_converted,
        avg_revenue_per_converter,
        revenue_variance,
    )
    treatment_revenue = _simulate_revenue(
        rng,
        treatment_converted,
        avg_revenue_per_converter,
        revenue_variance,
    )

    total_users = n_control + n_treatment
    timestamps = _generate_timestamps(rng, total_users, start_timestamp, duration_days)

    dataframe = pd.DataFrame(
        {
            "user_id": np.arange(1, total_users + 1, dtype=np.int64),
            "variant": np.concatenate(
                (
                    np.full(n_control, "control", dtype=object),
                    np.full(n_treatment, "treatment", dtype=object),
                )
            ),
            "converted": np.concatenate((control_converted, treatment_converted)).astype(np.int64),
            "revenue": np.concatenate((control_revenue, treatment_revenue)).astype(float),
            "timestamp": timestamps.to_numpy(),
        }
    )
    dataframe["day_of_week"] = dataframe["timestamp"].dt.day_name()
    dataframe = dataframe.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return dataframe[["user_id", "variant", "converted", "revenue", "timestamp", "day_of_week"]]


def simulate_aa_test(
    n_per_variant: int,
    rate: float,
    avg_revenue_per_converter: float = 45.0,
    start_date: str | None = None,
    duration_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate an A/A test where both variants share the same conversion rate.

    Parameters
    ----------
    n_per_variant : int
        Number of users assigned to each variant.
    rate : float
        Shared conversion rate used for both control and treatment.
    avg_revenue_per_converter : float, default=45.0
        Scale parameter used in the lognormal revenue model for converted users.
    start_date : str or None, default=None
        Experiment start date in ISO format (``YYYY-MM-DD``). When omitted,
        the start date defaults to 30 days before the current date.
    duration_days : int, default=30
        Number of days in the experiment window.
    seed : int, default=42
        Seed passed to :func:`numpy.random.default_rng`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame following the same schema as :func:`simulate_ab_test`.

    Examples
    --------
    >>> df = simulate_aa_test(500, 0.10, seed=42)
    >>> df['variant'].value_counts().sort_index().to_dict()
    {'control': 500, 'treatment': 500}
    """
    return simulate_ab_test(
        n_control=n_per_variant,
        n_treatment=n_per_variant,
        control_rate=rate,
        treatment_rate=rate,
        avg_revenue_per_converter=avg_revenue_per_converter,
        start_date=start_date,
        duration_days=duration_days,
        seed=seed,
    )


def simulate_srm_scenario(
    total_users: int,
    intended_split: float = 0.50,
    actual_control_split: float = 0.60,
    control_rate: float = 0.10,
    treatment_rate: float = 0.12,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a sample-ratio-mismatch scenario for A/B testing.

    Parameters
    ----------
    total_users : int
        Total number of users in the simulated experiment.
    intended_split : float, default=0.50
        Expected share of traffic that should have gone to control.
    actual_control_split : float, default=0.60
        Observed share of traffic that actually went to control.
    control_rate : float, default=0.10
        Conversion rate used to sample control outcomes.
    treatment_rate : float, default=0.12
        Conversion rate used to sample treatment outcomes.
    seed : int, default=42
        Seed passed to :func:`numpy.random.default_rng`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame following the same schema as :func:`simulate_ab_test`.
        Metadata about the intended and actual splits is attached via
        ``DataFrame.attrs``.

    Examples
    --------
    >>> df = simulate_srm_scenario(1000, intended_split=0.50, actual_control_split=0.60)
    >>> df.attrs['actual_control_split']
    0.6
    """
    _validate_positive_int("total_users", total_users)
    _validate_probability("intended_split", intended_split)
    _validate_probability("actual_control_split", actual_control_split)

    n_control = round(total_users * actual_control_split)
    n_treatment = total_users - n_control
    dataframe = simulate_ab_test(
        n_control=n_control,
        n_treatment=n_treatment,
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        seed=seed,
    )
    dataframe.attrs["intended_split"] = intended_split
    dataframe.attrs["actual_control_split"] = actual_control_split
    return dataframe
