import numpy as np
import pandas as pd
from eqtmfusion.qc import run_qc
from eqtmfusion.preprocessing import preprocess_omics_matrix, encode_severity_ordinal


def test_run_qc_drops_high_missing_features_and_samples():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(20, 10)), index=[f"S{i}" for i in range(20)],
                       columns=[f"F{i}" for i in range(10)])
    df.iloc[:, 0] = np.nan  # feature entirely missing -> should be dropped
    df.iloc[0, :] = np.nan  # sample entirely missing -> should be dropped

    filtered, report = run_qc(df, "test_modality", drop_outlier_samples=False)
    assert "F0" not in filtered.columns
    assert "S0" not in filtered.index
    assert report.n_features_before == 10
    assert report.n_samples_before == 20


def test_preprocess_omics_matrix_no_nans_after():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(15, 5)), index=[f"S{i}" for i in range(15)],
                       columns=[f"F{i}" for i in range(5)])
    df.iloc[2, 3] = np.nan
    out = preprocess_omics_matrix(df, impute_method="mean")
    assert out.isna().sum().sum() == 0
    assert abs(out.mean().mean()) < 1e-6  # standard-scaled


def test_encode_severity_ordinal():
    clinical = pd.DataFrame({"severity": ["Control", "Mild", "Severe", "Moderate"]})
    codes = encode_severity_ordinal(clinical)
    assert list(codes) == [0, 1, 3, 2]
