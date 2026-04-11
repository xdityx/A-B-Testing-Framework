"""Tests for data simulation utilities."""

import pandas as pd

from src.data_simulation import (
    simulate_aa_test,
    simulate_ab_test,
    simulate_srm_scenario,
)


def test_schema_and_dtypes() -> None:
    dataframe = simulate_ab_test(1000, 1000, 0.10, 0.12, seed=42)

    assert list(dataframe.columns) == [
        "user_id",
        "variant",
        "converted",
        "revenue",
        "timestamp",
        "day_of_week",
    ]
    assert pd.api.types.is_integer_dtype(dataframe["user_id"])
    assert pd.api.types.is_object_dtype(dataframe["variant"])
    assert pd.api.types.is_integer_dtype(dataframe["converted"])
    assert pd.api.types.is_float_dtype(dataframe["revenue"])
    assert pd.api.types.is_datetime64_ns_dtype(dataframe["timestamp"])
    assert pd.api.types.is_object_dtype(dataframe["day_of_week"])
    assert set(dataframe["converted"].unique()) <= {0, 1}
    assert (dataframe["revenue"] >= 0.0).all()
    assert dataframe["user_id"].is_unique


def test_conversion_rate_approximates_target() -> None:
    dataframe = simulate_ab_test(10000, 10000, 0.10, 0.12, seed=42)
    conversion_rates = dataframe.groupby("variant")["converted"].mean()

    assert abs(conversion_rates["control"] - 0.10) < 0.01
    assert abs(conversion_rates["treatment"] - 0.12) < 0.01


def test_non_converters_have_zero_revenue() -> None:
    dataframe = simulate_ab_test(1000, 1000, 0.10, 0.12, seed=42)

    assert (dataframe.loc[dataframe["converted"] == 0, "revenue"] == 0.0).all()
    assert (dataframe.loc[dataframe["converted"] == 1, "revenue"] > 0.0).all()


def test_user_ids_are_sequential_and_unique() -> None:
    n_control = 1000
    n_treatment = 1000
    dataframe = simulate_ab_test(n_control, n_treatment, 0.10, 0.12, seed=42)

    assert sorted(dataframe["user_id"].tolist()) == list(
        range(1, n_control + n_treatment + 1)
    )
    assert dataframe["user_id"].is_unique


def test_sorted_by_timestamp() -> None:
    dataframe = simulate_ab_test(1000, 1000, 0.10, 0.12, seed=42)

    assert dataframe["timestamp"].is_monotonic_increasing


def test_aa_test_identical_rates() -> None:
    dataframe = simulate_aa_test(5000, 0.10, seed=42)
    conversion_rates = dataframe.groupby("variant")["converted"].mean()

    assert abs(conversion_rates["control"] - conversion_rates["treatment"]) < 0.015


def test_srm_scenario_imbalance() -> None:
    dataframe = simulate_srm_scenario(
        10000,
        intended_split=0.50,
        actual_control_split=0.60,
        seed=42,
    )
    control_count = int((dataframe["variant"] == "control").sum())

    assert abs(control_count - 6000) <= 50
    assert dataframe.attrs["intended_split"] == 0.50
