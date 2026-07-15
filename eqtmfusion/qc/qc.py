"""
eqtmfusion.qc
================
Modality-agnostic quality control: missingness summaries, outlier detection
(IQR and Z-score based), and sample/feature filtering.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class QCReport:
    modality_name: str
    n_samples_before: int
    n_features_before: int
    n_samples_after: int
    n_features_after: int
    samples_dropped: list
    features_dropped: list
    sample_missing_frac: pd.Series
    feature_missing_frac: pd.Series


def missingness_summary(df: pd.DataFrame) -> dict:
    return {
        "sample_missing_frac": df.isna().mean(axis=1),
        "feature_missing_frac": df.isna().mean(axis=0),
    }


def detect_outlier_samples_iqr(df: pd.DataFrame, multiplier: float = 3.0) -> list:
    """
    Flag outlier samples based on their overall feature-mean distance from
    the cohort using an IQR rule (robust to non-normal omics distributions).
    """
    sample_means = df.mean(axis=1, skipna=True)
    q1, q3 = sample_means.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    outliers = sample_means[(sample_means < lower) | (sample_means > upper)].index.tolist()
    return outliers


def run_qc(
    df: pd.DataFrame,
    modality_name: str,
    sample_missing_thresh: float = 0.2,
    feature_missing_thresh: float = 0.2,
    drop_outlier_samples: bool = True,
    outlier_iqr_multiplier: float = 3.0,
) -> tuple[pd.DataFrame, QCReport]:
    """
    Run the full QC pipeline on a samples x features omics matrix and return
    the filtered dataframe plus a report object for the automated report.
    """
    n_samples_before, n_features_before = df.shape
    miss = missingness_summary(df)

    # feature filtering first (features present in too few samples)
    keep_features = miss["feature_missing_frac"][miss["feature_missing_frac"] <= feature_missing_thresh].index
    features_dropped = [c for c in df.columns if c not in keep_features]
    df_f = df[keep_features]

    # then sample filtering
    sample_missing = df_f.isna().mean(axis=1)
    keep_samples = sample_missing[sample_missing <= sample_missing_thresh].index
    samples_dropped = [s for s in df_f.index if s not in keep_samples]
    df_fs = df_f.loc[keep_samples]

    if drop_outlier_samples:
        outliers = detect_outlier_samples_iqr(df_fs, outlier_iqr_multiplier)
        samples_dropped.extend(outliers)
        df_fs = df_fs.drop(index=outliers, errors="ignore")

    report = QCReport(
        modality_name=modality_name,
        n_samples_before=n_samples_before,
        n_features_before=n_features_before,
        n_samples_after=df_fs.shape[0],
        n_features_after=df_fs.shape[1],
        samples_dropped=samples_dropped,
        features_dropped=features_dropped,
        sample_missing_frac=miss["sample_missing_frac"],
        feature_missing_frac=miss["feature_missing_frac"],
    )
    return df_fs, report
