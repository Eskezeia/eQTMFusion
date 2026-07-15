"""
eqtmfusion.reporting.publication_outputs
============================================
Orchestrates the full set of publication-standard outputs (figures + tables)
into a conventionally-named directory structure:

    outdir/
    ├── figures/
    │   ├── Figure1_model_diagnostics.png / .pdf
    │   ├── Figure2_feature_importance.png / .pdf
    │   ├── Figure3_cv_performance_summary.png / .pdf
    │   └── Figure4_eqtm_forest_plot.png / .pdf   (if eqtm_results provided)
    └── tables/
        ├── Table1_cohort_characteristics.csv / .xlsx
        ├── Table2_model_performance_summary.csv / .xlsx
        └── Table3_top_biomarkers.csv / .xlsx

This is the single entry point the CLI (and any Python user) calls to get
a complete, journal-ready output bundle rather than assembling individual
figure/table calls by hand.
"""

import os
import pandas as pd

from eqtmfusion.visualization.style import save_publication_figure
from eqtmfusion.visualization.publication_figures import (
    fig_classification_diagnostics, fig_regression_diagnostics, fig_ordinal_diagnostics,
    fig_feature_importance_panel, fig_cv_performance_summary, fig_eqtm_forest_plot,
)
from eqtmfusion.reporting.tables import (
    table1_cohort_characteristics, table_model_performance_summary,
    table_top_biomarkers, save_table,
)


def generate_publication_outputs(
    outdir: str,
    task: str,
    y_true, y_pred, y_proba=None,
    clinical_df: pd.DataFrame = None,
    group_col: str = None,
    cv_metrics_df: pd.DataFrame = None,
    importance_df: pd.DataFrame = None,
    importance_value_col: str = None,
    eqtm_results: pd.DataFrame = None,
    class_labels: list = None,
    title_prefix: str = "",
    figure_formats=("png", "pdf"),
    table_formats=("csv", "xlsx"),
) -> dict:
    """
    Generate the full standard publication output bundle. Any input left as
    None skips the corresponding figure/table rather than raising an error,
    so this can be called with partial results (e.g. no eQTM analysis run).

    Returns a dict of {name: {format: path}} for everything generated.
    """
    fig_dir = os.path.join(outdir, "figures")
    table_dir = os.path.join(outdir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    outputs = {"figures": {}, "tables": {}}

    # --- Figure 1: model diagnostics (task-dependent panel type) ---
    if task == "classification" and y_proba is not None:
        fig1 = fig_classification_diagnostics(
            y_true, y_pred, y_proba,
            class_names=class_labels or ("Negative", "Positive"),
            title=f"{title_prefix}Model performance: classification",
        )
        outputs["figures"]["Figure1_model_diagnostics"] = save_publication_figure(
            fig1, os.path.join(fig_dir, "Figure1_model_diagnostics"), formats=figure_formats)
    elif task == "regression":
        fig1 = fig_regression_diagnostics(
            y_true, y_pred, title=f"{title_prefix}Model performance: regression")
        outputs["figures"]["Figure1_model_diagnostics"] = save_publication_figure(
            fig1, os.path.join(fig_dir, "Figure1_model_diagnostics"), formats=figure_formats)
    elif task == "ordinal":
        fig1 = fig_ordinal_diagnostics(
            y_true, y_pred, class_labels=class_labels,
            title=f"{title_prefix}Model performance: ordinal outcome")
        outputs["figures"]["Figure1_model_diagnostics"] = save_publication_figure(
            fig1, os.path.join(fig_dir, "Figure1_model_diagnostics"), formats=figure_formats)

    # --- Figure 2: feature importance panel ---
    if importance_df is not None and len(importance_df) > 0:
        fig2 = fig_feature_importance_panel(
            importance_df, value_col=importance_value_col,
            title=f"{title_prefix}Top features by importance")
        outputs["figures"]["Figure2_feature_importance"] = save_publication_figure(
            fig2, os.path.join(fig_dir, "Figure2_feature_importance"), formats=figure_formats)

    # --- Figure 3: CV performance summary ---
    if cv_metrics_df is not None and len(cv_metrics_df) > 0:
        fig3 = fig_cv_performance_summary(
            cv_metrics_df, title=f"{title_prefix}Cross-validated performance")
        outputs["figures"]["Figure3_cv_performance_summary"] = save_publication_figure(
            fig3, os.path.join(fig_dir, "Figure3_cv_performance_summary"), formats=figure_formats)

    # --- Figure 4: eQTM forest plot ---
    if eqtm_results is not None and len(eqtm_results) > 0:
        fig4 = fig_eqtm_forest_plot(eqtm_results, title=f"{title_prefix}Top eQTM associations")
        outputs["figures"]["Figure4_eqtm_forest_plot"] = save_publication_figure(
            fig4, os.path.join(fig_dir, "Figure4_eqtm_forest_plot"), formats=figure_formats)

    # --- Table 1: cohort characteristics ---
    if clinical_df is not None:
        t1 = table1_cohort_characteristics(clinical_df, group_col=group_col)
        outputs["tables"]["Table1_cohort_characteristics"] = save_table(
            t1, os.path.join(table_dir, "Table1_cohort_characteristics"), formats=table_formats)

    # --- Table 2: model performance summary ---
    if cv_metrics_df is not None and len(cv_metrics_df) > 0:
        t2 = table_model_performance_summary(cv_metrics_df)
        outputs["tables"]["Table2_model_performance_summary"] = save_table(
            t2, os.path.join(table_dir, "Table2_model_performance_summary"), formats=table_formats)

    # --- Table 3: top biomarkers ---
    if importance_df is not None and len(importance_df) > 0:
        t3 = table_top_biomarkers(importance_df, value_col=importance_value_col)
        outputs["tables"]["Table3_top_biomarkers"] = save_table(
            t3, os.path.join(table_dir, "Table3_top_biomarkers"), formats=table_formats)

    return outputs
