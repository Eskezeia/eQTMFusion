"""
eqtmfusion.visualization.publication_figures
================================================
Multi-panel, journal-quality diagnostic figures. Each figure function
returns a matplotlib Figure styled per eqtmfusion.visualization.style and
is meant to be saved via save_publication_figure() as PNG (300 dpi) + PDF.

Panel conventions follow common Nature-family layouts: lettered panels
(A, B, C...), consistent axis styling, colorblind-safe palette.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve

from eqtmfusion.visualization.style import (
    publication_style, add_panel_label, style_axes, PALETTE, MODALITY_COLORS,
)


def fig_classification_diagnostics(
    y_true, y_pred, y_proba, class_names=("Negative", "Positive"),
    title: str = "Model performance: classification",
) -> plt.Figure:
    """
    4-panel classification diagnostic figure:
    (A) ROC curve  (B) Precision-Recall curve
    (C) Confusion matrix  (D) Calibration (reliability) curve
    """
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)

        # (A) ROC
        ax = axes[0, 0]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[0], linewidth=1.6, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right")
        style_axes(ax); add_panel_label(ax, "A")

        # (B) PR
        ax = axes[0, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=PALETTE[1], linewidth=1.6, label=f"AUC = {pr_auc:.3f}")
        baseline = np.mean(y_true)
        ax.axhline(baseline, linestyle="--", color="gray", linewidth=0.8, label=f"Baseline = {baseline:.2f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower left")
        style_axes(ax); add_panel_label(ax, "B")

        # (C) Confusion matrix
        ax = axes[1, 0]
        y_pred_arr = np.asarray(y_pred)
        cm = confusion_matrix(y_true, y_pred_arr)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
        style_axes(ax); add_panel_label(ax, "C")

        # (D) Calibration curve
        ax = axes[1, 1]
        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", markersize=3, color=PALETTE[2], linewidth=1.2,
                    label="Model")
        except ValueError:
            ax.text(0.5, 0.5, "Insufficient data\nfor calibration curve",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8, label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left")
        style_axes(ax); add_panel_label(ax, "D")

        fig.tight_layout()
    return fig


def fig_regression_diagnostics(
    y_true, y_pred, title: str = "Model performance: regression",
) -> plt.Figure:
    """
    4-panel regression diagnostic figure:
    (A) Predicted vs. observed  (B) Residuals vs. fitted
    (C) Residual distribution   (D) Bland-Altman agreement plot
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)

        # (A) Predicted vs observed
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, s=10, alpha=0.6, color=PALETTE[0], edgecolors="none")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, linestyle="--", color="gray", linewidth=0.8, label="Identity")
        r2 = 1 - np.sum(residuals ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        ax.text(0.05, 0.92, f"R\u00b2 = {r2:.3f}", transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
        ax.legend(loc="lower right")
        style_axes(ax); add_panel_label(ax, "A")

        # (B) Residuals vs fitted
        ax = axes[0, 1]
        ax.scatter(y_pred, residuals, s=10, alpha=0.6, color=PALETTE[1], edgecolors="none")
        ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
        ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
        style_axes(ax); add_panel_label(ax, "B")

        # (C) Residual distribution
        ax = axes[1, 0]
        ax.hist(residuals, bins=20, color=PALETTE[2], edgecolor="white", linewidth=0.5)
        ax.axvline(0, linestyle="--", color="gray", linewidth=0.8)
        ax.set_xlabel("Residual"); ax.set_ylabel("Count")
        style_axes(ax); add_panel_label(ax, "C")

        # (D) Bland-Altman
        ax = axes[1, 1]
        mean_vals = (y_true + y_pred) / 2
        diff_vals = y_true - y_pred
        md = diff_vals.mean(); sd = diff_vals.std()
        ax.scatter(mean_vals, diff_vals, s=10, alpha=0.6, color=PALETTE[3], edgecolors="none")
        ax.axhline(md, color="black", linewidth=0.8, label=f"Mean bias = {md:.2f}")
        ax.axhline(md + 1.96 * sd, linestyle="--", color="gray", linewidth=0.8, label="\u00b11.96 SD")
        ax.axhline(md - 1.96 * sd, linestyle="--", color="gray", linewidth=0.8)
        ax.set_xlabel("Mean of observed & predicted"); ax.set_ylabel("Observed \u2212 Predicted")
        ax.legend(loc="upper right", fontsize=6)
        style_axes(ax); add_panel_label(ax, "D")

        fig.tight_layout()
    return fig


def fig_ordinal_diagnostics(
    y_true, y_pred, class_labels=None, title: str = "Model performance: ordinal outcome",
) -> plt.Figure:
    """
    2-panel ordinal diagnostic figure:
    (A) Confusion matrix heatmap (row-normalized)
    (B) Per-class accuracy bar chart
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    classes = sorted(set(y_true) | set(y_pred))
    if class_labels is None:
        class_labels = [str(c) for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.05)

        ax = axes[0]
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(len(classes)):
            for j in range(len(classes)):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized frequency")
        style_axes(ax); add_panel_label(ax, "A")

        ax = axes[1]
        per_class_acc = np.diag(cm_norm)
        ax.bar(range(len(classes)), per_class_acc, color=PALETTE[0])
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_ylabel("Per-class accuracy")
        ax.set_ylim(0, 1.05)
        style_axes(ax); add_panel_label(ax, "B")

        fig.tight_layout()
    return fig


def fig_feature_importance_panel(
    importance_df: pd.DataFrame, value_col: str = None, top_n: int = 20,
    title: str = "Top features by importance",
) -> plt.Figure:
    """
    Horizontal bar chart of top features, colored by omics modality when the
    index follows the "modality__feature" naming convention (as produced by
    fusion.early_fusion). Falls back to a single color otherwise.
    """
    if value_col is None:
        value_col = importance_df.columns[0]
    top = importance_df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]

    modalities = top.index.to_series().str.extract(r"^([^_]+)__")[0]
    colors = [MODALITY_COLORS.get(m, PALETTE[0]) for m in modalities.fillna("other")]
    labels = top.index.to_series().str.replace(r"^[^_]+__", "", regex=True)

    with publication_style():
        fig, ax = plt.subplots(figsize=(6.0, max(3.0, top_n * 0.28)))
        ax.barh(range(len(top)), top[value_col].values, color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel(value_col.replace("_", " ").title())
        ax.set_title(title, fontsize=9, fontweight="bold")
        style_axes(ax)

        if modalities.notna().any():
            handles = [plt.Rectangle((0, 0), 1, 1, color=MODALITY_COLORS.get(m, PALETTE[0]))
                       for m in sorted(modalities.dropna().unique())]
            ax.legend(handles, sorted(modalities.dropna().unique()), loc="lower right", fontsize=6.5)
        fig.tight_layout()
    return fig


def fig_cv_performance_summary(
    cv_metrics_df: pd.DataFrame, title: str = "Cross-validated performance",
) -> plt.Figure:
    """
    Box + strip plot of metric values across CV folds, one panel per metric,
    for reporting model stability rather than a single point estimate.
    """
    metrics = list(cv_metrics_df.columns)
    n = len(metrics)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    with publication_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows), squeeze=False)
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)
        for idx, metric in enumerate(metrics):
            ax = axes[idx // ncols, idx % ncols]
            vals = cv_metrics_df[metric].dropna().values
            bp = ax.boxplot([vals], widths=0.5, patch_artist=True,
                             medianprops=dict(color="black", linewidth=1.2),
                             boxprops=dict(facecolor=PALETTE[idx % len(PALETTE)], alpha=0.5, linewidth=0.8),
                             whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
            jitter = np.random.default_rng(0).normal(1, 0.04, size=len(vals))
            ax.scatter(jitter, vals, s=10, color=PALETTE[idx % len(PALETTE)], alpha=0.8, zorder=3)
            ax.set_xticks([]); ax.set_title(metric, fontsize=7.5)
            style_axes(ax)
        for idx in range(n, nrows * ncols):
            axes[idx // ncols, idx % ncols].axis("off")
        fig.tight_layout()
    return fig


def fig_eqtm_forest_plot(
    eqtm_results: pd.DataFrame, top_n: int = 20,
    title: str = "Top eQTM associations (effect size \u00b1 95% CI)",
) -> plt.Figure:
    """
    Forest plot of the top eQTM associations by significance, showing beta
    coefficient and 95% CI (beta \u00b1 1.96*SE) -- the standard genetics-paper
    presentation for ranked association effect sizes.

    eqtm_results: dataframe with columns cpg, gene, beta, se, pvalue, fdr
    (as returned by eqtm.run_eqtm_analysis(...).results)
    """
    df = eqtm_results.sort_values("pvalue").head(top_n).iloc[::-1].copy()
    df["label"] = df["cpg"] + " \u2192 " + df["gene"]
    df["ci_lower"] = df["beta"] - 1.96 * df["se"]
    df["ci_upper"] = df["beta"] + 1.96 * df["se"]

    with publication_style():
        fig, ax = plt.subplots(figsize=(6.0, max(3.0, top_n * 0.3)))
        y_pos = np.arange(len(df))
        colors = [PALETTE[1] if b < 0 else PALETTE[0] for b in df["beta"]]
        ax.errorbar(df["beta"], y_pos,
                     xerr=[df["beta"] - df["ci_lower"], df["ci_upper"] - df["beta"]],
                     fmt="o", markersize=4, color="black", ecolor="gray",
                     elinewidth=1.0, capsize=2, zorder=2)
        ax.scatter(df["beta"], y_pos, color=colors, s=28, zorder=3)
        ax.axvline(0, linestyle="--", color="gray", linewidth=0.8)
        ax.set_yticks(y_pos); ax.set_yticklabels(df["label"], fontsize=6.5)
        ax.set_xlabel("Effect size (\u03b2, methylation \u2192 expression)")
        ax.set_title(title, fontsize=9, fontweight="bold")
        style_axes(ax)
        fig.tight_layout()
    return fig
