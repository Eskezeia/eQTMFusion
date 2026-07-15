import numpy as np
import pandas as pd
from eqtmfusion.reporting import generate_publication_outputs


def test_generate_publication_outputs_classification_full_bundle(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    y_true = rng.integers(0, 2, n)
    y_proba = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, n), 0, 1)
    y_pred = (y_proba > 0.5).astype(int)

    clinical_df = pd.DataFrame({
        "age": rng.normal(45, 10, n),
        "asthma_status": np.where(y_true == 1, "Case", "Control"),
    })
    cv_metrics_df = pd.DataFrame({"accuracy": [0.7, 0.72, 0.68], "auc": [0.8, 0.82, 0.79]})
    idx = [f"methylation__cg{i}" for i in range(10)]
    importance_df = pd.DataFrame({"importance_mean": rng.exponential(0.02, 10)}, index=idx)

    outputs = generate_publication_outputs(
        outdir=str(tmp_path), task="classification",
        y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        clinical_df=clinical_df, group_col="asthma_status",
        cv_metrics_df=cv_metrics_df, importance_df=importance_df,
        importance_value_col="importance_mean",
    )

    assert "Figure1_model_diagnostics" in outputs["figures"]
    assert "Figure2_feature_importance" in outputs["figures"]
    assert "Figure3_cv_performance_summary" in outputs["figures"]
    assert "Table1_cohort_characteristics" in outputs["tables"]
    assert "Table2_model_performance_summary" in outputs["tables"]
    assert "Table3_top_biomarkers" in outputs["tables"]

    import os
    for fig_paths in outputs["figures"].values():
        for path in fig_paths.values():
            assert os.path.exists(path) and os.path.getsize(path) > 0
    for table_paths in outputs["tables"].values():
        for path in table_paths.values():
            assert os.path.exists(path) and os.path.getsize(path) > 0


def test_generate_publication_outputs_skips_missing_inputs_gracefully(tmp_path):
    """No exception should be raised when only partial inputs are provided --
    e.g. no CV metrics, no importance, no eQTM results."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 10, 30)
    y_pred = y_true + rng.normal(0, 3, 30)

    outputs = generate_publication_outputs(
        outdir=str(tmp_path), task="regression", y_true=y_true, y_pred=y_pred,
    )
    assert "Figure1_model_diagnostics" in outputs["figures"]
    assert "Figure2_feature_importance" not in outputs["figures"]
    assert "Table1_cohort_characteristics" not in outputs["tables"]
