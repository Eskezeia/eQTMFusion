"""
eqtmfusion.explainability
=============================
Model explanation methods:
  - SHAP (TreeExplainer for tree models, KernelExplainer fallback for others)
  - Permutation importance (model-agnostic, sklearn-based)
  - Integrated Gradients (for PyTorch models, e.g. CoralOrdinalNet)
  - LIME (optional, requires `pip install lime`)

Output helpers aggregate feature importances into "top biomarkers" tables
broken out by modality (CpG / gene / SNP / miRNA / protein / metabolite)
based on a feature-name prefix convention (e.g. "methylation__cg000123").
"""

import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.inspection import permutation_importance


def shap_explain(model, X: np.ndarray, feature_names: list, model_type: str = "tree") -> pd.DataFrame:
    """
    model_type: "tree" uses TreeExplainer (fast, exact for RF/XGBoost/LightGBM);
                "kernel" uses KernelExplainer (model-agnostic, slower -- use
                a background sample subset for large X).
    Returns a dataframe of mean |SHAP value| per feature, sorted descending.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed: pip install shap")

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        background = shap.sample(X, min(100, X.shape[0]))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):  # legacy multi-class format: list of (n, f) arrays
        shap_values = np.mean(np.abs(np.stack(shap_values)), axis=0)
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:  # modern format: (n_samples, n_features, n_classes)
            shap_values = np.abs(shap_values).mean(axis=2)
        else:
            shap_values = np.abs(shap_values)

    mean_abs_shap = pd.Series(shap_values.mean(axis=0), index=feature_names, name="mean_abs_shap")
    return mean_abs_shap.sort_values(ascending=False).to_frame()


def permutation_importance_explain(
    model, X: np.ndarray, y: np.ndarray, feature_names: list,
    n_repeats: int = 10, random_state: int = 42, scoring: str = None,
) -> pd.DataFrame:
    """
    Permutation importance on a FITTED model, evaluated on the SAME data
    passed in. IMPORTANT: if `model` was fit on this same X/y and has
    overfit (common with many features and few samples), permutation
    importance computed this way will be misleadingly close to zero for
    every feature, because permuting a single column barely perturbs a
    decision surface that has already memorized the training labels via
    other feature combinations. Always pass held-out data here, or use
    cv_permutation_importance() below, which handles this correctly by
    construction.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring
    )
    df = pd.DataFrame({
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }, index=feature_names)
    return df.sort_values("importance_mean", ascending=False)


def cv_permutation_importance(
    model_fn, X: np.ndarray, y: np.ndarray, feature_names: list,
    task: str = "classification", n_splits: int = 5, n_repeats: int = 5,
    scoring: str = None, random_state: int = 42,
) -> pd.DataFrame:
    """
    Cross-validated permutation importance: for each of n_splits folds, fit
    a fresh model on the training portion and compute permutation importance
    on the HELD-OUT portion, then average across folds. This is the
    methodologically correct way to rank features by importance -- unlike
    in-sample permutation importance, it does not collapse to near-zero for
    an overfit model, because the held-out fold was never seen during
    training for that fold.

    model_fn: zero-arg callable returning an unfitted estimator with
              .fit(X, y) and .predict(X) (and .predict_proba for
              classification, if `scoring` needs it)
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X)

    fold_importances = []
    for train_idx, test_idx in split_iter:
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        result = permutation_importance(
            model, X[test_idx], y[test_idx], n_repeats=n_repeats,
            random_state=random_state, scoring=scoring,
        )
        fold_importances.append(result.importances_mean)

    stacked = np.stack(fold_importances)  # (n_splits, n_features)
    df = pd.DataFrame({
        "importance_mean": stacked.mean(axis=0),
        "importance_std": stacked.std(axis=0),
    }, index=feature_names)
    return df.sort_values("importance_mean", ascending=False)


def integrated_gradients(
    model: torch.nn.Module, X: np.ndarray, baseline: np.ndarray = None,
    n_steps: int = 50, target_output_index: int = 0, device: str = "cpu",
) -> np.ndarray:
    """
    Integrated Gradients (Sundararajan et al., 2017) for any PyTorch model
    whose forward(x) returns a tensor. Works with CoralOrdinalNet and any
    custom regression/classification torch model in this package.

    baseline: reference input (default: zero vector, standard choice for
              standardized omics data where 0 = cohort mean)
    target_output_index: which output dimension to attribute for
              multi-output models (e.g. which CORAL threshold logit)
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    if baseline is None:
        baseline_t = torch.zeros_like(X_t)
    else:
        baseline_t = torch.tensor(baseline, dtype=torch.float32, device=device)

    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1, 1)
    interpolated = baseline_t.unsqueeze(0) + alphas * (X_t - baseline_t).unsqueeze(0)  # (n_steps, n, d)
    interpolated.requires_grad_(True)

    grads = []
    for step in range(n_steps):
        out = model(interpolated[step])
        target = out[:, target_output_index] if out.ndim > 1 else out
        grad = torch.autograd.grad(target.sum(), interpolated, retain_graph=True, create_graph=False)[0][step]
        grads.append(grad.detach())

    avg_grads = torch.stack(grads).mean(dim=0)
    attributions = (X_t - baseline_t) * avg_grads
    return attributions.cpu().numpy()


def lime_explain(model, X: np.ndarray, feature_names: list, instance_idx: int = 0,
                  mode: str = "classification", num_features: int = 20):
    """
    Optional LIME explanation for a single instance. Requires `pip install lime`.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        warnings.warn("lime not installed (`pip install lime`); skipping lime_explain.")
        return None

    explainer = LimeTabularExplainer(
        X, feature_names=feature_names, mode=mode, discretize_continuous=True
    )
    predict_fn = model.predict_proba if mode == "classification" else model.predict
    explanation = explainer.explain_instance(X[instance_idx], predict_fn, num_features=num_features)
    return explanation


def top_biomarkers_by_modality(importance_df: pd.DataFrame, top_n: int = 20) -> dict:
    """
    Split a feature-importance dataframe (indexed by "modality__feature"
    names, as produced by fusion.early_fusion) into per-modality top-N
    tables: top CpGs, top genes, top SNPs, top miRNAs, top proteins,
    top metabolites.
    """
    results = {}
    modality_prefixes = importance_df.index.to_series().str.extract(r"^([^_]+)__")[0].dropna().unique()
    for prefix in modality_prefixes:
        subset = importance_df[importance_df.index.str.startswith(f"{prefix}__")]
        results[prefix] = subset.head(top_n)
    return results
