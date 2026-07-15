"""
Example: end-to-end eQTMFusion pipeline on simulated asthma data.

Run with:
    python examples/run_end_to_end.py
"""
import matplotlib
matplotlib.use("Agg")

from eqtmfusion.simulation import simulate_cohort
from eqtmfusion.qc import run_qc
from eqtmfusion.preprocessing import preprocess_omics_matrix, encode_severity_ordinal
from eqtmfusion.eqtm import run_eqtm_analysis, get_significant_eqtm_cpgs
from eqtmfusion.feature_selection import combine_tier1_tier2
from eqtmfusion.fusion import early_fusion
from eqtmfusion.models import get_classifier, train_coral_model, predict_coral
from eqtmfusion.explainability import permutation_importance_explain, top_biomarkers_by_modality
from eqtmfusion.visualization import plot_roc_curve, plot_pca, plot_feature_importance
from eqtmfusion.utilities import classification_metrics, ordinal_metrics, run_cv, get_logger
from eqtmfusion.reporting import generate_html_report

logger = get_logger()


def main():
    # 1. Simulate data (small size for a fast demo; scale up for real runs)
    logger.info("Simulating cohort...")
    cohort = simulate_cohort(
        n_subjects=300, n_cpgs=2000, n_genes=1000, n_snps=500,
        n_mirna=100, n_metabolites=100, n_proteins=200,
        n_eqtm_pairs=30, random_state=42,
    )

    # 2. QC
    logger.info("Running QC...")
    methylation, meth_qc = run_qc(cohort.methylation, "methylation")
    expression, expr_qc = run_qc(cohort.expression, "expression")

    # 3. Preprocessing
    logger.info("Preprocessing...")
    methylation = preprocess_omics_matrix(methylation, impute_method="mean")  # mean for speed at 2000 CpGs
    expression = preprocess_omics_matrix(expression, impute_method="mean")

    # 4. eQTM analysis (restricted to a candidate pair set for tractability)
    logger.info("Running eQTM analysis on ground-truth candidate pairs + negative controls...")
    covariates = cohort.clinical.loc[methylation.index, ["age", "bmi"]]
    candidate_pairs = cohort.ground_truth["eqtm_pairs"]
    eqtm_result = run_eqtm_analysis(methylation, expression, covariates, candidate_pairs=candidate_pairs)
    sig_cpgs = get_significant_eqtm_cpgs(eqtm_result, fdr_thresh=0.2)  # lenient for demo-size cohort
    logger.info(f"Found {len(sig_cpgs)} significant eQTM CpGs at FDR<=0.2")

    # 5. Feature selection: Tier1 (eQTM) + Tier2 (variance) for methylation
    logger.info("Feature selection...")
    meth_features, fs_report = combine_tier1_tier2(methylation, sig_cpgs, target_n=300)
    methylation_selected = methylation[meth_features]
    logger.info(f"Feature selection report: {fs_report}")

    # 6. Fusion (early fusion for simplicity in this example)
    logger.info("Fusing modalities...")
    fused = early_fusion({"methylation": methylation_selected, "expression": expression})
    clinical = cohort.clinical.loc[fused.index]

    # 7a. Classification: asthma status
    logger.info("Training classifier for asthma_status...")
    y_status = (clinical["asthma_status"] == "Case").astype(int)
    clf = get_classifier("random_forest")
    cv_result = run_cv(lambda: get_classifier("random_forest"), fused.values, y_status.values,
                        task="classification", n_splits=5)
    fold0 = cv_result["fold_results"][0]
    clf_metrics = classification_metrics(fold0["y_true"], fold0["y_pred"], fold0["y_proba"])
    logger.info(f"Classification metrics (fold 0): {clf_metrics}")

    # 7b. Ordinal: severity
    logger.info("Training CORAL ordinal model for severity...")
    y_severity = encode_severity_ordinal(clinical)
    coral_model = train_coral_model(fused.values, y_severity.values, num_classes=y_severity.nunique(),
                                     n_epochs=200, verbose_every=50)
    severity_pred = predict_coral(coral_model, fused.values)
    sev_metrics = ordinal_metrics(y_severity.values, severity_pred)
    logger.info(f"Ordinal severity metrics (in-sample): {sev_metrics}")

    # 8. Explainability
    logger.info("Computing permutation importance...")
    clf.fit(fused.values, y_status.values)
    importance = permutation_importance_explain(clf, fused.values, y_status.values,
                                                  fused.columns.tolist(), n_repeats=5)
    top_biomarkers = top_biomarkers_by_modality(importance, top_n=15)

    # 9. Visualization
    logger.info("Generating figures...")
    fig_roc = plot_roc_curve(fold0["y_true"], fold0["y_proba"], title="Asthma Status ROC (fold 0)")
    fig_pca = plot_pca(fused.values, labels=y_status, title="PCA of Fused Multi-Omics")
    fig_importance = plot_feature_importance(importance, value_col="importance_mean", top_n=20,
                                              title="Top 20 Fused Features by Permutation Importance")

    # 10. Report
    logger.info("Generating HTML report...")
    report_path = generate_html_report(
        output_path="results/example_report.html",
        title="eQTMFusion Example: Asthma Multi-Omics Analysis",
        data_summary_tables={"Clinical": clinical.describe(include="all").transpose()},
        qc_tables={"Methylation QC": __import__("pandas").DataFrame([vars(meth_qc)]),
                   "Expression QC": __import__("pandas").DataFrame([vars(expr_qc)])},
        feature_selection_tables={"Methylation Tier1+Tier2": __import__("pandas").DataFrame([fs_report])},
        metrics_tables={
            "Asthma status classification (fold 0)": __import__("pandas").DataFrame([clf_metrics]),
            "Severity ordinal (in-sample)": __import__("pandas").DataFrame([sev_metrics]),
        },
        figures=[
            {"figure": fig_roc, "caption": "ROC curve, asthma status classification, fold 0"},
            {"figure": fig_pca, "caption": "PCA of fused methylation+expression features"},
            {"figure": fig_importance, "caption": "Top fused features by permutation importance"},
        ],
        biomarker_tables=top_biomarkers,
    )
    logger.info(f"Done. Report at {report_path}")


if __name__ == "__main__":
    main()
