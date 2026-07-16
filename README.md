# eQTMFusion

**An eQTM-Guided Multi-Omics Machine Learning Framework for Predicting Asthma Severity, Lung Function, IgE, and Other Clinical Outcomes**

## What this is

A working, tested Python package for multi-omics integration and prediction, built around an eQTM-informed feature-selection philosophy (biological prior + variance backfill, extending the reasoning from the [IntegrAO](https://doi.org/10.1038/s42256-024-00942-3) framework). It is demonstrated on a synthetic asthma cohort but is disease-agnostic — swap in your own clinical/omics tables via the schema in `eqtmfusion/simulation/simulate.py::SCHEMA`.

## Package structure

```
eqtmfusion/
├── simulation/        Example-synthetic multi-omics asthma cohort 
├── qc/                missingness/outlier QC
├── preprocessing/     imputation, scaling, categorical/ordinal encoding
├── eqtm/               cis-/trans-eQTM analysis (OLS + mixed models, FDR)
├── feature_selection/  Tier1 (eQTM prior) / Tier2 (variance, MAD) / Tier3 (LASSO, ElasticNet, RF, XGBoost, Boruta, SHAP)
├── fusion/             early / intermediate (AE, VAE) / late (ensemble, stacking) / graph-based
├── models/             classification, ordinal (CORAL, ordinal logistic), regression
├── explainability/     SHAP, permutation importance, integrated gradients, LIME (optional)
├── visualization/      ROC/PR, volcano, heatmap, PCA/UMAP/t-SNE, correlation, feature importance, eQTM network
├── reporting/          Jinja2 HTML report (+ optional PDF via weasyprint)
├── utilities/          metrics, CV/bootstrap/external validation, logging
└── cli/                click-based CLI

tests/                  25 unit tests, all passing
examples/                run_end_to_end.py: full working walkthrough
```


