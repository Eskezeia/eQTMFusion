"""
eqtmfusion.utilities.metrics
================================
Unified metric computation for classification, regression, and ordinal tasks.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, mean_squared_error, mean_absolute_error,
    r2_score, cohen_kappa_score,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    if y_proba is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["auc"] = np.nan
    return metrics


def concordance_correlation_coefficient(y_true, y_pred) -> float:
    """Lin's Concordance Correlation Coefficient (CCC)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "ccc": float(concordance_correlation_coefficient(y_true, y_pred)),
    }


def ordinal_metrics(y_true, y_pred) -> dict:
    return {
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "ordinal_accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_absolute_ordinal_error": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
    }
