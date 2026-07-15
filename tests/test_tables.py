import numpy as np
import pandas as pd
from eqtmfusion.reporting import (
    table1_cohort_characteristics, table_model_performance_summary,
    table_top_biomarkers, save_table, dataframe_to_image,
)


def test_table1_cohort_characteristics_basic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.normal(45, 10, 100),
        "sex": rng.choice(["M", "F"], 100),
        "group": rng.choice(["Case", "Control"], 100),
    })
    t1 = table1_cohort_characteristics(df, group_col="group",
                                        continuous_cols=["age"], categorical_cols=["sex"])
    assert "Variable" in t1.columns
    assert "Overall" in t1.columns
    assert "Case" in t1.columns and "Control" in t1.columns
    assert (t1["Variable"] == "age").any()
    assert (t1["Variable"] == "N").any()


def test_table1_detects_significant_difference():
    """A variable with a real group difference should get a small p-value;
    this is a real statistical sanity check, not just a shape check."""
    rng = np.random.default_rng(0)
    n = 200
    group = np.array(["Case"] * (n // 2) + ["Control"] * (n // 2))
    # deliberately different means between groups
    value = np.concatenate([rng.normal(10, 1, n // 2), rng.normal(5, 1, n // 2)])
    df = pd.DataFrame({"biomarker": value, "group": group})
    t1 = table1_cohort_characteristics(df, group_col="group", continuous_cols=["biomarker"],
                                        categorical_cols=[])
    row = t1[t1["Variable"] == "biomarker"].iloc[0]
    assert float(row["p-value"]) < 0.001


def test_table_model_performance_summary_formats_correctly():
    cv_df = pd.DataFrame({
        "accuracy": [0.7, 0.75, 0.72],
        "auc": [0.81, 0.83, 0.80],
    })
    t2 = table_model_performance_summary(cv_df)
    assert set(t2["Metric"]) == {"accuracy", "auc"}
    assert "N folds" in t2.columns
    assert (t2["N folds"] == 3).all()
    assert "\u00b1" in t2["Mean \u00b1 SD"].iloc[0]


def test_table_top_biomarkers_splits_modality_and_feature():
    idx = ["methylation__cg001", "expression__GENE001", "unlabeled_feature"]
    imp = pd.DataFrame({"importance_mean": [0.5, 0.3, 0.1]}, index=idx)
    t3 = table_top_biomarkers(imp, value_col="importance_mean", top_n=3)
    assert list(t3["Rank"]) == [1, 2, 3]
    assert t3.iloc[0]["Modality"] == "methylation"
    assert t3.iloc[0]["Feature"] == "cg001"
    assert t3.iloc[2]["Modality"] == "(unspecified)"


def test_save_table_csv_and_xlsx(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    saved = save_table(df, str(tmp_path / "test_table"), formats=("csv", "xlsx"))
    import os
    assert os.path.exists(saved["csv"])
    assert os.path.exists(saved["xlsx"])
    reloaded = pd.read_csv(saved["csv"])
    assert reloaded.equals(df)


def test_dataframe_to_image_produces_valid_png(tmp_path):
    from PIL import Image
    df = pd.DataFrame({"Feature": ["cg1", "cg2"], "Importance": [0.5, 0.3]})
    path = str(tmp_path / "table_image.png")
    dataframe_to_image(df, path, title="Test table")
    img = Image.open(path)
    assert img.size[0] > 0 and img.size[1] > 0
    dpi = img.info.get("dpi")
    assert round(dpi[0]) == 300
