"""MLflow experiment logging utilities."""

from __future__ import annotations

import pathlib

import mlflow
import pandas as pd

from src.diagnostics import SRMResult
from src.statistical_tests import BayesianResult, FrequentistResult


def _normalize_tracking_uri(tracking_uri: str) -> str:
    if "://" in tracking_uri:
        return tracking_uri

    tracking_path = pathlib.Path(tracking_uri)
    if tracking_path.is_absolute():
        return tracking_path.as_uri()
    return tracking_uri


def log_experiment(
    experiment_name: str,
    baseline_rate: float,
    mde: float,
    sample_size_per_variant: int,
    freq_result: FrequentistResult,
    bayesian_result: BayesianResult,
    srm_result: SRMResult,
    report_path: str | None = None,
    tracking_uri: str = "mlruns",
) -> str:
    """Log an A/B test run to MLflow.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment to create or use.
    baseline_rate : float
        Baseline conversion rate used during experiment planning.
    mde : float
        Minimum detectable relative effect used during experiment planning.
    sample_size_per_variant : int
        Required or observed sample size per experiment variant.
    freq_result : FrequentistResult
        Frequentist statistical test result to log as metrics and tags.
    bayesian_result : BayesianResult
        Bayesian A/B test result to log as metrics.
    srm_result : SRMResult
        Sample-ratio-mismatch diagnostic result to log as metrics and tags.
    report_path : str or None, default=None
        Optional path to an HTML report artifact. The artifact is logged only
        when the path is provided and exists.
    tracking_uri : str, default="mlruns"
        MLflow tracking URI. A local path creates or uses a file-backed store.

    Returns
    -------
    str
        MLflow run ID for the logged run.

    Examples
    --------
    >>> from src.statistical_tests import FrequentistResult, BayesianResult
    >>> from src.diagnostics import SRMResult
    >>> freq = FrequentistResult(0.10, 0.12, 0.20, 4.0, 0.001, 0.01, 0.03, True, 0.05)
    >>> bayes = BayesianResult(0.99, 0.001, 0.01, 0.03, 0.10, 0.12, 100000)
    >>> srm = SRMResult(0.50, 0.50, 0.0, 1.0, False)
    >>> run_id = log_experiment("example", 0.10, 0.05, 5000, freq, bayes, srm)
    >>> isinstance(run_id, str)
    True
    """
    mlflow.set_tracking_uri(_normalize_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "baseline_rate": baseline_rate,
                "mde": mde,
                "sample_size_per_variant": sample_size_per_variant,
                "alpha": freq_result.alpha,
            }
        )
        mlflow.log_metrics(
            {
                "control_rate": freq_result.control_rate,
                "treatment_rate": freq_result.treatment_rate,
                "lift": freq_result.lift,
                "z_stat": freq_result.z_stat,
                "p_value": freq_result.p_value,
                "significant": float(freq_result.significant),
                "prob_treatment_wins": bayesian_result.prob_treatment_wins,
                "expected_loss": bayesian_result.expected_loss,
                "srm_detected": float(srm_result.srm_detected),
                "srm_p_value": srm_result.p_value,
            }
        )
        mlflow.set_tags(
            {
                "decision": "ship" if freq_result.significant else "hold",
                "srm_status": "FAIL" if srm_result.srm_detected else "PASS",
            }
        )

        if report_path is not None:
            artifact_path = pathlib.Path(report_path)
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        return run.info.run_id


def compare_experiments(
    experiment_name: str,
    tracking_uri: str = "mlruns",
) -> pd.DataFrame:
    """Return all MLflow runs for an experiment as a DataFrame.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment whose runs should be retrieved.
    tracking_uri : str, default="mlruns"
        MLflow tracking URI. A local path reads from a file-backed store.

    Returns
    -------
    pandas.DataFrame
        DataFrame returned by :func:`mlflow.search_runs` for the experiment.

    Examples
    --------
    >>> runs = compare_experiments("example")
    >>> isinstance(runs, pd.DataFrame)
    True
    """
    mlflow.set_tracking_uri(_normalize_tracking_uri(tracking_uri))
    return mlflow.search_runs(experiment_names=[experiment_name])
