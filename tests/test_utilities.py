import numpy as np
from eqtmfusion.utilities import (
    classification_metrics, regression_metrics, ordinal_metrics,
    concordance_correlation_coefficient, run_cv, bootstrap_metric,
)
from sklearn.linear_model import LogisticRegression, Ridge


def test_classification_metrics_keys():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.4, 0.2, 0.9])
    metrics = classification_metrics(y_true, y_pred, y_proba)
    for key in ["accuracy", "precision", "recall", "f1", "mcc", "sensitivity", "specificity", "auc"]:
        assert key in metrics


def test_regression_metrics_keys_and_ccc_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["rmse"] == 0.0
    assert abs(metrics["ccc"] - 1.0) < 1e-6


def test_ordinal_metrics_perfect_agreement():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    metrics = ordinal_metrics(y_true, y_pred)
    assert metrics["quadratic_weighted_kappa"] == 1.0
    assert metrics["ordinal_accuracy"] == 1.0


def test_run_cv_classification():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    result = run_cv(lambda: LogisticRegression(), X, y, task="classification", n_splits=3)
    assert len(result["fold_results"]) == 3


def test_bootstrap_metric_ci():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
    y_pred = y_true + np.random.normal(0, 0.1, 50)
    from sklearn.metrics import mean_squared_error
    result = bootstrap_metric(y_true, y_pred, lambda a, b: mean_squared_error(a, b), n_bootstrap=100)
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
