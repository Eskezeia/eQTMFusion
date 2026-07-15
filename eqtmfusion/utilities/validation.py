"""
eqtmfusion.utilities.validation
===================================
Internal validation helpers: k-fold CV, repeated CV, bootstrap resampling,
and external (independent cohort) validation.
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.base import clone


def run_cv(
    model_fn, X: np.ndarray, y: np.ndarray, task: str = "classification",
    n_splits: int = 5, n_repeats: int = 1, random_state: int = 42,
) -> dict:
    """
    Run (repeated) k-fold CV. model_fn: zero-arg callable returning an
    unfitted estimator. Returns out-of-fold predictions and per-fold indices.
    """
    if task == "classification":
        splitter = (RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                    if n_repeats > 1 else
                    StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state))
        split_iter = splitter.split(X, y)
    else:
        splitter = (RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                    if n_repeats > 1 else
                    KFold(n_splits=n_splits, shuffle=True, random_state=random_state))
        split_iter = splitter.split(X)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(split_iter):
        model = clone(model_fn()) if hasattr(model_fn(), "get_params") else model_fn()
        model.fit(X[train_idx], y[train_idx])
        if task == "classification" and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X[test_idx])[:, 1]
        else:
            y_proba = None
        y_pred = model.predict(X[test_idx])
        fold_results.append({
            "fold": fold_i, "train_idx": train_idx, "test_idx": test_idx,
            "y_true": y[test_idx], "y_pred": y_pred, "y_proba": y_proba,
        })
    return {"fold_results": fold_results}


def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap: int = 1000,
                      random_state: int = 42) -> dict:
    """
    Bootstrap confidence interval for any metric_fn(y_true, y_pred) -> float.
    Returns point estimate, 2.5%/97.5% CI bounds.
    """
    rng = np.random.default_rng(random_state)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    return {
        "point_estimate": metric_fn(y_true, y_pred),
        "ci_lower": float(np.percentile(scores, 2.5)),
        "ci_upper": float(np.percentile(scores, 97.5)),
        "bootstrap_scores": scores,
    }


def external_validation(fitted_model, X_external: np.ndarray, y_external: np.ndarray,
                         task: str = "classification") -> dict:
    """
    Apply an already-fitted model to an independent external cohort.
    Returns predictions for downstream metric computation.
    """
    y_pred = fitted_model.predict(X_external)
    y_proba = None
    if task == "classification" and hasattr(fitted_model, "predict_proba"):
        y_proba = fitted_model.predict_proba(X_external)[:, 1]
    return {"y_true": y_external, "y_pred": y_pred, "y_proba": y_proba}
