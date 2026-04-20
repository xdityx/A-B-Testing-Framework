"""Tests for reporting utilities."""

import pathlib

from src.data_simulation import simulate_ab_test
from src.diagnostics import check_novelty_effect, check_srm
from src.reporting import generate_report
from src.statistical_tests import bayesian_ab_test, z_test_proportions


def _build_report(tmp_path: pathlib.Path) -> pathlib.Path:
    dataframe = simulate_ab_test(
        500,
        500,
        0.10,
        0.12,
        start_date="2026-01-01",
        seed=42,
    )
    control = dataframe[dataframe["variant"] == "control"]
    treatment = dataframe[dataframe["variant"] == "treatment"]
    freq_result = z_test_proportions(
        int(control["converted"].sum()),
        len(control),
        int(treatment["converted"].sum()),
        len(treatment),
    )
    bayesian_result = bayesian_ab_test(
        int(control["converted"].sum()),
        len(control),
        int(treatment["converted"].sum()),
        len(treatment),
        n_samples=10_000,
        seed=42,
    )
    srm_result = check_srm(len(control), len(treatment))
    daily_rates = (
        dataframe.assign(date=dataframe["timestamp"].dt.date).groupby("date")["converted"].mean()
    )
    novelty_result = check_novelty_effect(daily_rates)

    return generate_report(
        dataframe,
        freq_result,
        bayesian_result,
        srm_result,
        novelty_result,
        output_path=str(tmp_path / "report.html"),
    )


def test_report_file_is_created(tmp_path: pathlib.Path) -> None:
    result = _build_report(tmp_path)

    assert result.exists()


def test_report_contains_key_sections(tmp_path: pathlib.Path) -> None:
    result = _build_report(tmp_path)
    html = result.read_text(encoding="utf-8")

    assert "A/B Test Report" in html
    assert "P(B > A)" in html
    assert "Conversion Rate by Variant" in html


def test_report_returns_path_object(tmp_path: pathlib.Path) -> None:
    result = _build_report(tmp_path)

    assert isinstance(result, pathlib.Path)
