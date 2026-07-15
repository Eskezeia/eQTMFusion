"""
eqtmfusion.eqtm
===================
cis- and trans-eQTM analysis: tests association between each CpG and each
gene's expression, adjusting for covariates (age, sex, BMI, smoking, batch,
cell proportions), using linear models (OLS) or linear mixed models (for
batch/family structure), with Benjamini-Hochberg FDR correction.

Definitions used here:
  - cis-eQTM: CpG-gene pair within a user-specified genomic window (requires
    genomic position annotation for both; if not supplied, all pairs in the
    user-provided `candidate_pairs` list are treated as "cis" candidates)
  - trans-eQTM: CpG-gene pairs with no positional restriction, tested across
    a (typically pre-filtered) candidate set since all-vs-all is O(CpGs x genes)

For genome-wide trans scans, restrict to variance-filtered or otherwise
pre-selected CpGs/genes -- testing all 50,000 x 10,000 pairs (5x10^8 tests)
is not memory/compute-feasible in a single-machine pass; this module batches
the computation and warns if the requested scan is very large.
"""

import itertools
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass


@dataclass
class EQTMResult:
    results: pd.DataFrame  # columns: cpg, gene, beta, se, pvalue, fdr
    eqtm_type: str          # "cis" or "trans"


def _build_covariate_matrix(covariates: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical covariates, leave numeric ones as-is."""
    cov = covariates.copy()
    cat_cols = [c for c in cov.columns if cov[c].dtype == object]
    if cat_cols:
        cov = pd.get_dummies(cov, columns=cat_cols, drop_first=True)
    return cov.astype(float)


def _fit_single_pair_ols(y_expr: np.ndarray, x_meth: np.ndarray,
                          cov_matrix: np.ndarray) -> tuple:
    """
    Fit: expression ~ methylation + covariates
    Returns (beta_methylation, se_methylation, pvalue_methylation)
    """
    X = np.column_stack([x_meth, cov_matrix])
    X = sm.add_constant(X)
    try:
        model = sm.OLS(y_expr, X, missing="drop").fit()
        beta = model.params[1]       # index 0 = const, index 1 = methylation
        se = model.bse[1]
        pval = model.pvalues[1]
    except Exception:
        beta, se, pval = np.nan, np.nan, np.nan
    return beta, se, pval


def _fit_single_pair_mixedlm(y_expr: np.ndarray, x_meth: np.ndarray,
                              cov_matrix: np.ndarray, groups: np.ndarray) -> tuple:
    """
    Fit a linear mixed model with a random intercept for `groups` (e.g.
    batch, family, or site), otherwise identical to the OLS formulation.
    Use when batch/family structure would otherwise inflate false positives.
    """
    X = np.column_stack([x_meth, cov_matrix])
    df = pd.DataFrame(X, columns=["meth"] + [f"cov{i}" for i in range(cov_matrix.shape[1])])
    df["y"] = y_expr
    df["group"] = groups
    formula = "y ~ meth + " + " + ".join([c for c in df.columns if c.startswith("cov")])
    try:
        model = sm.MixedLM.from_formula(formula, groups="group", data=df).fit(reml=False)
        beta = model.params["meth"]
        se = model.bse["meth"]
        pval = model.pvalues["meth"]
    except Exception:
        beta, se, pval = np.nan, np.nan, np.nan
    return beta, se, pval


def run_eqtm_analysis(
    methylation: pd.DataFrame,
    expression: pd.DataFrame,
    covariates: pd.DataFrame,
    candidate_pairs: list = None,
    eqtm_type: str = "cis",
    model: str = "linear",
    batch_col: str = None,
    fdr_alpha: float = 0.05,
    max_pairs_warning: int = 2_000_000,
) -> EQTMResult:
    """
    Run eQTM analysis over `candidate_pairs` (list of (cpg, gene) tuples).
    If candidate_pairs is None, tests all CpG x gene combinations present in
    both dataframes (only advisable for small/pre-filtered matrices).

    covariates: samples x covariate dataframe (age, sex, bmi, smoking, batch,
                cell_proportion_*, etc. -- any subset present will be used)
    model: "linear" (OLS) or "mixed" (requires batch_col for random effect)
    """
    common_samples = methylation.index.intersection(expression.index).intersection(covariates.index)
    if len(common_samples) == 0:
        raise ValueError("No common samples between methylation, expression, and covariates.")

    meth = methylation.loc[common_samples]
    expr = expression.loc[common_samples]
    cov_df = covariates.loc[common_samples]

    if batch_col and batch_col in cov_df.columns:
        groups = cov_df[batch_col].values
        cov_for_model = cov_df.drop(columns=[batch_col])
    else:
        groups = None
        cov_for_model = cov_df

    cov_matrix = _build_covariate_matrix(cov_for_model).values

    if candidate_pairs is None:
        candidate_pairs = list(itertools.product(meth.columns, expr.columns))

    if len(candidate_pairs) > max_pairs_warning:
        warnings.warn(
            f"Requested {len(candidate_pairs):,} CpG-gene pairs, which exceeds "
            f"max_pairs_warning={max_pairs_warning:,}. This may be slow/memory "
            f"heavy. Consider pre-filtering CpGs/genes (e.g. via variance "
            f"threshold) before a trans-eQTM scan."
        )

    records = []
    for cpg, gene in candidate_pairs:
        if cpg not in meth.columns or gene not in expr.columns:
            continue
        x_meth = meth[cpg].values
        y_expr = expr[gene].values

        if model == "mixed" and groups is not None:
            beta, se, pval = _fit_single_pair_mixedlm(y_expr, x_meth, cov_matrix, groups)
        else:
            beta, se, pval = _fit_single_pair_ols(y_expr, x_meth, cov_matrix)

        records.append({"cpg": cpg, "gene": gene, "beta": beta, "se": se, "pvalue": pval})

    result_df = pd.DataFrame(records).dropna(subset=["pvalue"])
    if len(result_df) > 0:
        _, fdr, _, _ = multipletests(result_df["pvalue"].values, alpha=fdr_alpha, method="fdr_bh")
        result_df["fdr"] = fdr
    else:
        result_df["fdr"] = []

    result_df = result_df.sort_values("pvalue").reset_index(drop=True)
    return EQTMResult(results=result_df, eqtm_type=eqtm_type)


def get_significant_eqtm_cpgs(eqtm_result: EQTMResult, fdr_thresh: float = 0.05) -> list:
    """Return the list of unique CpGs with at least one significant eQTM at the given FDR."""
    sig = eqtm_result.results[eqtm_result.results["fdr"] <= fdr_thresh]
    return sorted(sig["cpg"].unique().tolist())


def get_significant_eqtm_genes(eqtm_result: EQTMResult, fdr_thresh: float = 0.05) -> list:
    """Return the list of unique genes with at least one significant eQTM at the given FDR."""
    sig = eqtm_result.results[eqtm_result.results["fdr"] <= fdr_thresh]
    return sorted(sig["gene"].unique().tolist())
