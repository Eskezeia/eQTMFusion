"""
eqtmfusion.cli.main
=======================
Command-line interface for eQTMFusion.

Usage:
    eqtmfusion simulate --out data/ --n-subjects 200 --n-cpgs 2000 --n-genes 1000
    eqtmfusion run-all --data-dir data/ --outcome severity --out results/
    eqtmfusion run-all --data-dir data/ --outcome fev1 --task regression --model xgboost --out results/
    eqtmfusion run-all --data-dir data/ --outcome my_custom_label --task classification --out results/

`run-all` auto-discovers omics modalities: any CSV in --data-dir other than
`clinical.csv` and `ground_truth.json` is treated as a modality, named after
its filename (e.g. `proteomics.csv` -> modality "proteomics"). This means
adding a new omics type is just adding a new CSV -- no code changes needed.

`--outcome` accepts ANY column name present in clinical.csv, not a fixed
list. `--task` can be left as "auto" to infer classification / ordinal /
regression from the column's dtype and cardinality, or set explicitly.
"""

import click
import os
import glob
import json
import numpy as np
import pandas as pd

from eqtmfusion.simulation import simulate_cohort, save_cohort
from eqtmfusion.preprocessing import preprocess_omics_matrix, encode_severity_ordinal
from eqtmfusion.qc import run_qc
from eqtmfusion.feature_selection import combine_tier1_tier2
from eqtmfusion.fusion import early_fusion
from eqtmfusion.models import get_classifier, get_regressor, train_coral_model, predict_coral, CLASSIFIER_NAMES, REGRESSOR_NAMES
from eqtmfusion.explainability import permutation_importance_explain, cv_permutation_importance
from eqtmfusion.eqtm import run_eqtm_analysis
from eqtmfusion.utilities import classification_metrics, ordinal_metrics, regression_metrics, run_cv, get_logger
from eqtmfusion.reporting import generate_html_report, generate_publication_outputs

logger = get_logger()

RESERVED_FILES = {"clinical.csv", "ground_truth.json"}


@click.group()
def cli():
    """eQTMFusion: eQTM-guided multi-omics ML for asthma and beyond."""
    pass


@cli.command()
@click.option("--out", default="data", help="Output directory for simulated data.")
@click.option("--n-subjects", default=200, help="Number of subjects.")
@click.option("--n-cpgs", default=2000, help="Number of CpGs.")
@click.option("--n-genes", default=1000, help="Number of genes.")
@click.option("--n-snps", default=500, help="Number of SNPs.")
@click.option("--n-mirna", default=100, help="Number of miRNAs.")
@click.option("--n-metabolites", default=100, help="Number of metabolites.")
@click.option("--n-proteins", default=200, help="Number of proteins.")
@click.option("--seed", default=42, help="Random seed.")
def simulate(out, n_subjects, n_cpgs, n_genes, n_snps, n_mirna, n_metabolites, n_proteins, seed):
    """Generate a synthetic multi-omics asthma cohort."""
    logger.info(f"Simulating cohort: n_subjects={n_subjects}, n_cpgs={n_cpgs}, n_genes={n_genes}")
    cohort = simulate_cohort(
        n_subjects=n_subjects, n_cpgs=n_cpgs, n_genes=n_genes, n_snps=n_snps,
        n_mirna=n_mirna, n_metabolites=n_metabolites, n_proteins=n_proteins,
        random_state=seed,
    )
    save_cohort(cohort, out)
    logger.info(f"Saved simulated cohort to {out}/")


def discover_omics_modalities(data_dir: str) -> dict:
    """
    Auto-discover omics modalities: every CSV in data_dir except clinical.csv
    is treated as a modality, keyed by its filename stem. This is what makes
    the pipeline accept "any omics modality automatically" -- adding SNPs,
    metabolomics, proteomics, or a completely custom modality is just
    dropping a new CSV into the directory.
    """
    modalities = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        fname = os.path.basename(path)
        if fname in RESERVED_FILES:
            continue
        modality_name = os.path.splitext(fname)[0]
        modalities[modality_name] = pd.read_csv(path, index_col=0)
    return modalities


def infer_task(clinical: pd.DataFrame, outcome: str) -> str:
    """Infer classification / ordinal / regression from the outcome column."""
    col = clinical[outcome]
    is_stringlike = (
        pd.api.types.is_object_dtype(col)
        or pd.api.types.is_string_dtype(col)
        or isinstance(col.dtype, pd.CategoricalDtype)
    )
    if is_stringlike:
        n_unique = col.nunique()
        return "ordinal" if n_unique > 2 else "classification"
    if pd.api.types.is_integer_dtype(col) and col.nunique() <= 10:
        return "ordinal" if col.nunique() > 2 else "classification"
    return "regression"


class _CoralPredictWrapper:
    """Minimal sklearn-compatible wrapper around an ALREADY-FITTED
    CoralOrdinalNet, for permutation_importance_explain (predict-only,
    no retraining)."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, X):
        return predict_coral(self.model, X, device=self.device)


class _CoralEstimator:
    """Full sklearn-style estimator (.fit + .predict) wrapping CORAL
    training, so it can be passed as `model_fn` to cv_permutation_importance
    (which needs to retrain a fresh model per fold, not just predict)."""
    def __init__(self, num_classes, device, n_epochs=200):
        self.num_classes = num_classes
        self.device = device
        self.n_epochs = n_epochs
        self.model_ = None

    def fit(self, X, y):
        self.model_ = train_coral_model(X, y, self.num_classes, n_epochs=self.n_epochs,
                                         verbose_every=0, device=self.device)
        return self

    def predict(self, X):
        return predict_coral(self.model_, X, device=self.device)


@cli.command()
@click.option("--data-dir", required=True, help="Directory containing clinical.csv plus any omics CSVs.")
@click.option("--outcome", default="severity", help="Any column name present in clinical.csv.")
@click.option("--task", default="auto", type=click.Choice(["auto", "classification", "ordinal", "regression"]))
@click.option("--model", default=None, help=f"Classifier: {CLASSIFIER_NAMES} | Regressor: {REGRESSOR_NAMES}. "
                                             f"Defaults to random_forest (classification) / elasticnet (regression) / CORAL (ordinal).")
@click.option("--out", default="results", help="Output directory for the report and artifacts.")
@click.option("--target-n-features", default=500, help="Total features to retain per modality (Tier1+Tier2), where eQTM ground truth is available.")
@click.option("--eqtm-modality", default="methylation", help="Which discovered modality (if any) to apply the eQTM Tier1 prior to.")
@click.option("--n-cv-folds", default=5, help="Number of CV folds for reportable (out-of-fold) metrics. Set to 0 to skip CV and report in-sample metrics only (fast, exploratory).")
@click.option("--device", default="auto", help="Torch device for CORAL ordinal models: 'auto' (use CUDA if available), 'cpu', or 'cuda'.")
def run_all(data_dir, outcome, task, model, out, target_n_features, eqtm_modality, n_cv_folds, device):
    """Run the full pipeline end-to-end on ANY omics modalities found in
    --data-dir and ANY outcome column found in clinical.csv: preprocess ->
    QC -> feature select -> fuse -> train -> cross-validate -> report."""
    os.makedirs(out, exist_ok=True)

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Discovering data...")
    clinical = pd.read_csv(f"{data_dir}/clinical.csv", index_col=0)
    modalities = discover_omics_modalities(data_dir)
    logger.info(f"Found {len(modalities)} omics modalities: {list(modalities.keys())}")
    if outcome not in clinical.columns:
        raise click.ClickException(
            f"Outcome '{outcome}' not found in clinical.csv columns: {list(clinical.columns)}"
        )

    if task == "auto":
        task = infer_task(clinical, outcome)
        logger.info(f"Inferred task type for outcome '{outcome}': {task}")

    eqtm_cpgs = []
    gt_path = f"{data_dir}/ground_truth.json"
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt = json.load(f)
        eqtm_cpgs = gt.get("causal_cpgs", [])

    logger.info("Running QC on each modality...")
    qc_reports = {}
    for name, df in modalities.items():
        modalities[name], qc_reports[name] = run_qc(df, name)

    logger.info("Preprocessing each modality...")
    for name, df in modalities.items():
        modalities[name] = preprocess_omics_matrix(df)
    preprocessed_modalities = {name: df.copy() for name, df in modalities.items()}

    logger.info("Feature selection (Tier1 eQTM prior where applicable + Tier2 variance)...")
    fs_reports = {}
    for name, df in modalities.items():
        prior = eqtm_cpgs if name == eqtm_modality else []
        selected_features, report = combine_tier1_tier2(df, prior, target_n_features)
        modalities[name] = df[selected_features]
        fs_reports[name] = report
        logger.info(f"  [{name}] {report}")

    logger.info(f"Fusing {len(modalities)} modalities (early fusion)...")
    fused = early_fusion(modalities)

    common_idx = fused.index.intersection(clinical.index)
    fused = fused.loc[common_idx]
    clinical = clinical.loc[common_idx]
    y_raw = clinical[outcome]

    logger.info(f"Task: {task} | Outcome: {outcome} | Model: {model or '(default)'}")

    cv_metrics_list = []
    diag_y_true, diag_y_pred, diag_y_proba = None, None, None  # out-of-fold, for the diagnostic figure
    class_labels_for_plot = None

    if task == "classification":
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw.astype(int)
            class_labels_for_plot = ["0", "1"]
        else:
            categories = sorted(y_raw.astype(str).unique())
            if len(categories) != 2:
                raise click.ClickException(
                    f"Outcome '{outcome}' has {len(categories)} categories; "
                    f"use --task ordinal for multi-class ordered outcomes."
                )
            y = (y_raw.astype(str) == categories[-1]).astype(int)
            class_labels_for_plot = categories
        model_name = model or "random_forest"
        model_fn = lambda: get_classifier(model_name)

        if n_cv_folds > 0:
            cv_result = run_cv(model_fn, fused.values, y.values, task="classification", n_splits=n_cv_folds)
            for fold in cv_result["fold_results"]:
                cv_metrics_list.append(classification_metrics(fold["y_true"], fold["y_pred"], fold["y_proba"]))
            # concatenate out-of-fold predictions across all folds for an honest
            # (non-in-sample) diagnostic figure
            diag_y_true = np.concatenate([f["y_true"] for f in cv_result["fold_results"]])
            diag_y_pred = np.concatenate([f["y_pred"] for f in cv_result["fold_results"]])
            diag_y_proba = np.concatenate([f["y_proba"] for f in cv_result["fold_results"]])

        fitted_model = model_fn()
        fitted_model.fit(fused.values, y.values)
        y_pred = fitted_model.predict(fused.values)
        y_proba = fitted_model.predict_proba(fused.values)[:, 1] if hasattr(fitted_model, "predict_proba") else None
        insample_metrics = classification_metrics(y.values, y_pred, y_proba)
        importance_df = cv_permutation_importance(
            model_fn, fused.values, y.values, fused.columns.tolist(),
            task="classification", n_splits=max(n_cv_folds, 3), n_repeats=5)

        if diag_y_true is None:  # n_cv_folds == 0: fall back to in-sample (flagged in the figure title)
            diag_y_true, diag_y_pred, diag_y_proba = y.values, y_pred, y_proba

    elif task == "ordinal":
        is_stringlike = pd.api.types.is_object_dtype(y_raw) or pd.api.types.is_string_dtype(y_raw)
        if is_stringlike:
            order = sorted(y_raw.astype(str).unique())
            y = encode_severity_ordinal(clinical, severity_col=outcome, order=order)
            class_labels_for_plot = order
        else:
            y = y_raw.astype(int)
            class_labels_for_plot = [str(c) for c in sorted(y.unique())]
        num_classes = y.nunique()

        if n_cv_folds > 0:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            oof_true, oof_pred = [], []
            for train_idx, test_idx in skf.split(fused.values, y.values):
                fold_model = train_coral_model(
                    fused.values[train_idx], y.values[train_idx], num_classes,
                    n_epochs=200, verbose_every=0, device=device,
                )
                fold_pred = predict_coral(fold_model, fused.values[test_idx], device=device)
                cv_metrics_list.append(ordinal_metrics(y.values[test_idx], fold_pred))
                oof_true.append(y.values[test_idx]); oof_pred.append(fold_pred)
            diag_y_true = np.concatenate(oof_true)
            diag_y_pred = np.concatenate(oof_pred)

        fitted_model = train_coral_model(fused.values, y.values, num_classes, n_epochs=200, verbose_every=50, device=device)
        y_pred = predict_coral(fitted_model, fused.values, device=device)
        insample_metrics = ordinal_metrics(y.values, y_pred)
        importance_df = cv_permutation_importance(
            lambda: _CoralEstimator(num_classes, device), fused.values, y.values,
            fused.columns.tolist(), task="classification",  # stratified splitter is fine for ordinal too
            n_splits=max(n_cv_folds, 3), n_repeats=5, scoring="accuracy")

        if diag_y_true is None:
            diag_y_true, diag_y_pred = y.values, y_pred

    else:  # regression
        y = y_raw.astype(float)
        model_name = model or "elasticnet"
        model_fn = lambda: get_regressor(model_name)

        if n_cv_folds > 0:
            cv_result = run_cv(model_fn, fused.values, y.values, task="regression", n_splits=n_cv_folds)
            for fold in cv_result["fold_results"]:
                cv_metrics_list.append(regression_metrics(fold["y_true"], fold["y_pred"]))
            diag_y_true = np.concatenate([f["y_true"] for f in cv_result["fold_results"]])
            diag_y_pred = np.concatenate([f["y_pred"] for f in cv_result["fold_results"]])

        fitted_model = model_fn()
        fitted_model.fit(fused.values, y.values)
        y_pred = fitted_model.predict(fused.values)
        insample_metrics = regression_metrics(y.values, y_pred)
        importance_df = cv_permutation_importance(
            model_fn, fused.values, y.values, fused.columns.tolist(),
            task="regression", n_splits=max(n_cv_folds, 3), n_repeats=5)

        if diag_y_true is None:
            diag_y_true, diag_y_pred = y.values, y_pred

    logger.info(f"In-sample metrics: {insample_metrics}")
    metrics_tables = {"In-sample metrics (optimistic; see CV table for reportable results)":
                       pd.DataFrame([insample_metrics])}
    cv_df = None
    if cv_metrics_list:
        cv_df = pd.DataFrame(cv_metrics_list)
        cv_summary = cv_df.agg(["mean", "std"]).transpose()
        cv_summary.columns = ["cv_mean", "cv_std"]
        logger.info(f"{n_cv_folds}-fold CV metrics (mean +/- std):\n{cv_summary}")
        metrics_tables[f"{n_cv_folds}-fold cross-validated metrics"] = cv_summary
    else:
        logger.info("CV skipped (--n-cv-folds 0); in-sample metrics only (optimistic, not reportable).")

    # --- Optional eQTM forest plot: only when the demo simulator's ground_truth.json
    # provides candidate (CpG, gene) pairs and both an eQTM-prior modality and an
    # "expression" modality are available. For real cohorts without this file, run
    # eqtmfusion.eqtm.run_eqtm_analysis yourself and pass the results directly to
    # eqtmfusion.reporting.generate_publication_outputs(eqtm_results=...).
    eqtm_result_df = None
    if eqtm_cpgs and "expression" in preprocessed_modalities and eqtm_modality in preprocessed_modalities:
        try:
            with open(gt_path) as f:
                gt_pairs = json.load(f).get("eqtm_pairs", [])
            if gt_pairs:
                logger.info(f"Running eQTM analysis for the forest plot ({len(gt_pairs)} candidate pairs)...")
                covariate_cols = [c for c in ["age", "bmi"] if c in clinical.columns]
                eqtm_covariates = clinical[covariate_cols] if covariate_cols else pd.DataFrame(index=clinical.index)
                eqtm_out = run_eqtm_analysis(
                    preprocessed_modalities[eqtm_modality], preprocessed_modalities["expression"],
                    eqtm_covariates, candidate_pairs=[tuple(p) for p in gt_pairs],
                )
                eqtm_result_df = eqtm_out.results
        except Exception as e:
            logger.info(f"Skipping eQTM forest plot ({e})")

    logger.info("Generating publication-quality figures and tables...")
    pub_outputs = generate_publication_outputs(
        outdir=out, task=task,
        y_true=diag_y_true, y_pred=diag_y_pred, y_proba=diag_y_proba,
        clinical_df=clinical,
        group_col=outcome if task in ("classification", "ordinal") else None,
        cv_metrics_df=cv_df, importance_df=importance_df, importance_value_col="importance_mean",
        eqtm_results=eqtm_result_df, class_labels=class_labels_for_plot,
        title_prefix=f"{outcome}: ",
    )
    for category, items in pub_outputs.items():
        for name, paths in items.items():
            logger.info(f"  [{category}] {name}: {list(paths.values())}")

    report_path = generate_html_report(
        output_path=f"{out}/report.html",
        title=f"eQTMFusion Report -- outcome: {outcome} ({task})",
        data_summary_tables={"Clinical": clinical.describe(include="all").transpose()},
        qc_tables={f"{name} QC": pd.DataFrame([vars(r)]) for name, r in qc_reports.items()},
        feature_selection_tables={f"{name} Tier1+Tier2": pd.DataFrame([r]) for name, r in fs_reports.items()},
        metrics_tables=metrics_tables,
        biomarker_tables={"Top 25 features (permutation importance)":
                           importance_df.sort_values("importance_mean", ascending=False).head(25)},
    )
    logger.info(f"Report written to {report_path}")
    logger.info(f"Publication-quality figures written to {out}/figures/ (PNG @ 300dpi + PDF)")
    logger.info(f"Publication-quality tables written to {out}/tables/ (CSV + XLSX)")


if __name__ == "__main__":
    cli()
