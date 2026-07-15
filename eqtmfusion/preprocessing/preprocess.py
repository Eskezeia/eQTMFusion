"""
eqtmfusion.preprocessing
============================
Modality-agnostic preprocessing: missing-value imputation, scaling, and
categorical encoding for clinical variables. Works on any omics matrix
(methylation, expression, SNPs, miRNA, metabolomics, proteomics, or a
user-supplied custom modality) as long as it is samples x features.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder


def impute_missing(df: pd.DataFrame, method: str = "knn", n_neighbors: int = 5) -> pd.DataFrame:
    """
    method: "knn" (default, matches published multi-omics convention) or
            "mean" / "median" (faster for very high-dimensional matrices like
            50k-CpG methylation, where KNN imputation can be slow).
    """
    if df.isna().sum().sum() == 0:
        return df

    if method == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
    elif method in ("mean", "median"):
        imputer = SimpleImputer(strategy=method)
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    values = imputer.fit_transform(df.values)
    return pd.DataFrame(values, index=df.index, columns=df.columns)


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standard-scale each feature to mean 0, SD 1."""
    scaler = StandardScaler()
    values = scaler.fit_transform(df.values)
    return pd.DataFrame(values, index=df.index, columns=df.columns)


def preprocess_omics_matrix(
    df: pd.DataFrame,
    impute_method: str = "knn",
    scale: bool = True,
) -> pd.DataFrame:
    """One-call preprocessing for any omics modality: impute then scale."""
    df = impute_missing(df, method=impute_method)
    if scale:
        df = scale_features(df)
    return df


def encode_clinical_categoricals(
    clinical_df: pd.DataFrame,
    categorical_cols: list = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical clinical columns (sex, race, smoking_status,
    asthma_status). Returns encoded dataframe and a dict of fitted encoders
    (needed to decode predictions / apply consistently to new data).
    """
    if categorical_cols is None:
        categorical_cols = [
            c for c in clinical_df.columns
            if clinical_df[c].dtype == object
        ]

    df = clinical_df.copy()
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def encode_severity_ordinal(clinical_df: pd.DataFrame,
                             severity_col: str = "severity",
                             order: list = None) -> pd.Series:
    """
    Map ordered severity labels to integer codes (0=Control, 1=Mild, ...)
    for ordinal regression / CORAL-style modeling.
    """
    if order is None:
        order = ["Control", "Mild", "Moderate", "Severe"]
    mapping = {label: i for i, label in enumerate(order)}
    return clinical_df[severity_col].map(mapping)
