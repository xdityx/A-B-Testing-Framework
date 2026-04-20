"""Tests for MLflow tracking utilities."""

import pandas as pd

from mlflow_tracking.log_experiment import compare_experiments, log_experiment
from src.diagnostics import SRMResult
from src.statistical_tests import BayesianResult, FrequentistResult


def _frequentist_result(
    baseline_rate: float = 0.10,
    significant: bool = True,
) -> FrequentistResult:
    treatment_rate = baseline_rate * 1.20
    return FrequentistResult(
        control_rate=baseline_rate,
        treatment_rate=treatment_rate,
        lift=(treatment_rate - baseline_rate) / baseline_rate,
        cohens_h=0.064,
        z_stat=3.1,
        p_value=0.002,
        ci_lower=0.01,
        ci_upper=0.03,
        significant=significant,
        alpha=0.05,
    )


def _bayesian_result() -> BayesianResult:
    return BayesianResult(
        prob_treatment_wins=0.98,
        expected_loss=0.001,
        ci_lower=0.005,
        ci_upper=0.035,
        control_rate=0.10,
        treatment_rate=0.12,
        n_samples=100_000,
    )


def _srm_result() -> SRMResult:
    return SRMResult(
        expected_split=0.50,
        actual_split=0.50,
        chi2_stat=0.0,
        p_value=1.0,
        srm_detected=False,
    )


def test_log_experiment_returns_run_id(tmp_path) -> None:
    run_id = log_experiment(
        experiment_name="test_experiment",
        baseline_rate=0.10,
        mde=0.05,
        sample_size_per_variant=5000,
        freq_result=_frequentist_result(),
        bayesian_result=_bayesian_result(),
        srm_result=_srm_result(),
        tracking_uri=str(tmp_path / "mlruns"),
    )

    assert isinstance(run_id, str)
    assert run_id


def test_log_experiment_with_artifact(tmp_path) -> None:
    report_path = tmp_path / "report.html"
    report_path.write_text("<html></html>", encoding="utf-8")

    run_id = log_experiment(
        experiment_name="test_experiment",
        baseline_rate=0.10,
        mde=0.05,
        sample_size_per_variant=5000,
        freq_result=_frequentist_result(),
        bayesian_result=_bayesian_result(),
        srm_result=_srm_result(),
        report_path=str(report_path),
        tracking_uri=str(tmp_path / "mlruns"),
    )

    assert isinstance(run_id, str)
    assert run_id


def test_compare_experiments_returns_dataframe(tmp_path) -> None:
    tracking_uri = str(tmp_path / "mlruns")
    experiment_name = "comparison_experiment"

    log_experiment(
        experiment_name=experiment_name,
        baseline_rate=0.10,
        mde=0.05,
        sample_size_per_variant=5000,
        freq_result=_frequentist_result(0.10),
        bayesian_result=_bayesian_result(),
        srm_result=_srm_result(),
        tracking_uri=tracking_uri,
    )
    log_experiment(
        experiment_name=experiment_name,
        baseline_rate=0.12,
        mde=0.05,
        sample_size_per_variant=5000,
        freq_result=_frequentist_result(0.12),
        bayesian_result=_bayesian_result(),
        srm_result=_srm_result(),
        tracking_uri=tracking_uri,
    )

    result = compare_experiments(experiment_name, tracking_uri=tracking_uri)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
