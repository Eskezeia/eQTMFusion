"""
eqtmfusion.feature_selection
================================
Three-tier feature selection pipeline:

  Tier 1 (biological prior): features already identified as significant
    cis-/trans-eQTMs (from eqtmfusion.eqtm), included regardless of raw
    variance.

  Tier 2 (unsupervised, data-driven): VarianceThreshold and/or Median
    Absolute Deviation (MAD) ranking, used to backfill remaining feature
    budget with high-information features not already captured by Tier 1.

  Tier 3 (supervised, ML-based): LASSO / ElasticNet coefficient selection,
    Random Forest / XGBoost importance, and SHAP-based selection, applied
    against a specific outcome (severity, FEV1, IgE, etc). Boruta is
    supported optionally if the `boruta` package is installed.

Tiers can be used independently or chained (e.g. Tier 1 union Tier 2, then
Tier 3 applied on top of that reduced set for the final supervised model).
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import median_abs_deviation


# ----------------------------- Tier 1 -----------------------------------

def tier1_eqtm_prior(feature_names: list, significant_eqtm_features: list) -> list:
    """Intersect the modality's feature names with the eQTM-significant set."""
    sig_set = set(significant_eqtm_features)
    return [f for f in feature_names if f in sig_set]


# ----------------------------- Tier 2 -----------------------------------

def tier2_variance_threshold(df: pd.DataFrame, threshold: float = 0.0,
                              top_n: int = None) -> list:
    """
    Apply sklearn's VarianceThreshold, optionally then rank survivors by
    variance and keep only the top_n.
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df.values)
    kept = df.columns[selector.get_support()].tolist()
    if top_n is not None and len(kept) > top_n:
        variances = df[kept].var(axis=0).sort_values(ascending=False)
        kept = variances.index[:top_n].tolist()
    return kept


def tier2_mad_ranking(df: pd.DataFrame, top_n: int) -> list:
    """Rank features by Median Absolute Deviation (robust alternative to variance)."""
    mad_scores = df.apply(lambda col: median_abs_deviation(col.dropna()), axis=0)
    return mad_scores.sort_values(ascending=False).index[:top_n].tolist()


def combine_tier1_tier2(
    df: pd.DataFrame,
    tier1_features: list,
    target_n: int,
    tier2_method: str = "variance",
) -> tuple[list, dict]:
    """
    Union Tier 1 (biological prior) with Tier 2 backfill up to target_n total
    features. This is the "eQTM + variance" strategy used in the asthma
    methylation pipeline.
    """
    tier1_present = [f for f in tier1_features if f in df.columns]
    remaining_budget = max(target_n - len(tier1_present), 0)
    remaining_df = df.drop(columns=tier1_present, errors="ignore")

    if remaining_budget > 0 and remaining_df.shape[1] > 0:
        if tier2_method == "variance":
            tier2_features = tier2_variance_threshold(remaining_df, top_n=remaining_budget)
        elif tier2_method == "mad":
            tier2_features = tier2_mad_ranking(remaining_df, top_n=remaining_budget)
        else:
            raise ValueError(f"Unknown tier2_method: {tier2_method}")
    else:
        tier2_features = []

    final_features = tier1_present + tier2_features
    report = {
        "n_tier1_prioritized": len(tier1_present),
        "n_tier2_backfilled": len(tier2_features),
        "n_total": len(final_features),
    }
    return final_features, report


# ----------------------------- Tier 3 -----------------------------------

def tier3_lasso_selection(X: np.ndarray, y: np.ndarray, feature_names: list,
                           alpha: float = 0.01, task: str = "regression") -> list:
    """LASSO (or L1 logistic regression for classification) coefficient-based selection."""
    if task == "classification":
        model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0 / max(alpha, 1e-6))
    else:
        model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(X, y)
    coefs = model.coef_.ravel() if model.coef_.ndim > 1 else model.coef_
    selected = [feature_names[i] for i in range(len(feature_names)) if abs(coefs[i]) > 1e-8]
    return selected


def tier3_elasticnet_selection(X: np.ndarray, y: np.ndarray, feature_names: list,
                                alpha: float = 0.01, l1_ratio: float = 0.5) -> list:
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    model.fit(X, y)
    selected = [feature_names[i] for i in range(len(feature_names)) if abs(model.coef_[i]) > 1e-8]
    return selected


def tier3_random_forest_importance(X: np.ndarray, y: np.ndarray, feature_names: list,
                                    top_n: int = 100, task: str = "regression",
                                    n_estimators: int = 300) -> list:
    if task == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).index[:top_n].tolist()


def tier3_xgboost_importance(X: np.ndarray, y: np.ndarray, feature_names: list,
                              top_n: int = 100, task: str = "regression") -> list:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        warnings.warn("xgboost not installed; skipping tier3_xgboost_importance.")
        return []
    if task == "classification":
        model = XGBClassifier(n_estimators=300, eval_metric="logloss")
    else:
        model = XGBRegressor(n_estimators=300)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).index[:top_n].tolist()


def tier3_boruta_selection(X: np.ndarray, y: np.ndarray, feature_names: list,
                            task: str = "regression", random_state: int = 42) -> list:
    """
    Optional Boruta all-relevant feature selection. Requires `pip install
    Boruta`. Falls back to a warning + empty list if not installed.
    """
    try:
        from boruta import BorutaPy
    except ImportError:
        warnings.warn("boruta not installed (`pip install Boruta`); skipping tier3_boruta_selection.")
        return []

    if task == "classification":
        estimator = RandomForestClassifier(n_estimators=300, n_jobs=-1, max_depth=5)
    else:
        estimator = RandomForestRegressor(n_estimators=300, n_jobs=-1, max_depth=5)

    boruta_selector = BorutaPy(estimator, n_estimators="auto", random_state=random_state)
    boruta_selector.fit(X, y)
    selected = [feature_names[i] for i in range(len(feature_names)) if boruta_selector.support_[i]]
    return selected


def tier3_shap_selection(X: np.ndarray, y: np.ndarray, feature_names: list,
                          top_n: int = 100, task: str = "regression",
                          n_estimators: int = 200) -> list:
    """
    Fit a tree model then rank features by mean absolute SHAP value.
    Requires `pip install shap`.
    """
    try:
        import shap
    except ImportError:
        warnings.warn("shap not installed (`pip install shap`); skipping tier3_shap_selection.")
        return []

    if task == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):  # legacy multi-class format: list of (n, f) arrays
        shap_values = np.mean(np.abs(np.stack(shap_values)), axis=0)
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:  # modern format: (n_samples, n_features, n_classes)
            shap_values = np.abs(shap_values).mean(axis=2)
        else:
            shap_values = np.abs(shap_values)

    mean_abs_shap = pd.Series(shap_values.mean(axis=0), index=feature_names)
    return mean_abs_shap.sort_values(ascending=False).index[:top_n].tolist()


TIER3_METHODS = {
    "lasso": tier3_lasso_selection,
    "elasticnet": tier3_elasticnet_selection,
    "random_forest": tier3_random_forest_importance,
    "xgboost": tier3_xgboost_importance,
    "boruta": tier3_boruta_selection,
    "shap": tier3_shap_selection,
}
