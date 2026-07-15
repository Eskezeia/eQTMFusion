import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eqtmfusion.visualization import (
    fig_classification_diagnostics, fig_regression_diagnostics, fig_ordinal_diagnostics,
    fig_feature_importance_panel, fig_cv_performance_summary, fig_eqtm_forest_plot,
    save_publication_figure,
)


def _check_no_text_overlaps(fig):
    """Programmatic collision check: renders the figure and verifies no two
    text elements (labels, titles, annotations) within the same axes overlap
    in pixel space -- catches real layout bugs, not just 'it didn't crash'."""
    fig.canvas.draw()
    overlaps = []
    for ax in fig.get_axes():
        texts = ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title]
        bboxes = [(t.get_text()[:20], t.get_window_extent()) for t in texts if t.get_text().strip()]
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if bboxes[i][1].overlaps(bboxes[j][1]):
                    overlaps.append((bboxes[i][0], bboxes[j][0]))
    return overlaps


def test_fig_classification_diagnostics_renders_without_overlap():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0.2, 0.2, 100), 0, 1)
    y_pred = (y_proba > 0.5).astype(int)

    fig = fig_classification_diagnostics(y_true, y_pred, y_proba)
    assert len(fig.get_axes()) == 4
    assert _check_no_text_overlaps(fig) == []
    plt.close(fig)


def test_fig_regression_diagnostics_renders_without_overlap():
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 15, 100)
    y_pred = y_true + rng.normal(0, 5, 100)

    fig = fig_regression_diagnostics(y_true, y_pred)
    assert len(fig.get_axes()) == 4
    assert _check_no_text_overlaps(fig) == []
    plt.close(fig)


def test_fig_ordinal_diagnostics_renders_without_overlap():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 100)
    y_pred = np.clip(y_true + rng.choice([-1, 0, 0, 1], 100), 0, 3)

    fig = fig_ordinal_diagnostics(y_true, y_pred, class_labels=["Control", "Mild", "Moderate", "Severe"])
    assert len(fig.get_axes()) >= 2
    assert _check_no_text_overlaps(fig) == []
    plt.close(fig)


def test_fig_feature_importance_panel_modality_coloring():
    idx = [f"methylation__cg{i}" for i in range(10)] + [f"expression__GENE{i}" for i in range(10)]
    imp = pd.DataFrame({"importance_mean": np.linspace(1, 0.1, 20)}, index=idx)
    fig = fig_feature_importance_panel(imp, value_col="importance_mean", top_n=15)
    assert len(fig.get_axes()) == 1
    plt.close(fig)


def test_fig_cv_performance_summary_handles_multiple_metrics():
    cv_df = pd.DataFrame({
        "accuracy": [0.7, 0.72, 0.68, 0.71, 0.69],
        "auc": [0.8, 0.82, 0.79, 0.81, 0.80],
    })
    fig = fig_cv_performance_summary(cv_df)
    assert _check_no_text_overlaps(fig) == []
    plt.close(fig)


def test_fig_eqtm_forest_plot_renders():
    df = pd.DataFrame({
        "cpg": [f"cg{i}" for i in range(10)],
        "gene": [f"GENE{i}" for i in range(10)],
        "beta": np.linspace(-0.5, 0.5, 10),
        "se": np.full(10, 0.1),
        "pvalue": np.linspace(0.0001, 0.05, 10),
        "fdr": np.linspace(0.001, 0.1, 10),
    })
    fig = fig_eqtm_forest_plot(df, top_n=8)
    assert len(fig.get_axes()) == 1
    plt.close(fig)


def test_save_publication_figure_writes_multiple_formats(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    saved = save_publication_figure(fig, str(tmp_path / "test_fig"), formats=("png", "pdf"))
    assert saved["png"].endswith(".png")
    assert saved["pdf"].endswith(".pdf")
    import os
    assert os.path.exists(saved["png"])
    assert os.path.exists(saved["pdf"])
    plt.close(fig)


def test_save_publication_figure_png_is_300dpi(tmp_path):
    from PIL import Image
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    saved = save_publication_figure(fig, str(tmp_path / "test_fig"), formats=("png",), dpi=300)
    img = Image.open(saved["png"])
    dpi = img.info.get("dpi")
    assert dpi is not None
    assert round(dpi[0]) == 300
    plt.close(fig)
