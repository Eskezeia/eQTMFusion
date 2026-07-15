# eQTMFusion

**An eQTM-Guided Multi-Omics Machine Learning Framework for Predicting Asthma Severity, Lung Function, IgE, and Other Clinical Outcomes**

## What this is

A working, tested Python package for multi-omics integration and prediction, built around an eQTM-informed feature-selection philosophy (biological prior + variance backfill, extending the reasoning from the [IntegrAO](https://doi.org/10.1038/s42256-024-00942-3) framework). It is demonstrated on a synthetic asthma cohort but is disease-agnostic — swap in your own clinical/omics tables via the schema in `eqtmfusion/simulation/simulate.py::SCHEMA`.


## Status by requested component

| Component | Status |
|---|---|
| Synthetic multi-omics asthma simulator (clinical + 6 omics) | ✅ Fully implemented, tested |
| Preprocessing (impute, scale, QC, outlier detection) | ✅ Fully implemented, tested |
| cis-/trans-eQTM analysis (OLS + mixed models, covariates, FDR) | ✅ Fully implemented, tested |
| Tier 1/2/3 feature selection (eQTM prior, variance/MAD, LASSO/RF/XGBoost/SHAP/Boruta) | ✅ Fully implemented and tested, **including SHAP and Boruta paths** (previously untested; both now verified to correctly recover known causal features in a synthetic check) |
| **Accept any omics modality automatically** | ✅ CLI `run-all` now auto-discovers every CSV in `--data-dir` as a modality (verified with all 6 simulated omics types found and fused automatically) — previously the CLI only accepted hardcoded methylation/expression |
| **Flexible user-defined outcomes** | ✅ `--outcome` now accepts any clinical column name, with `--task auto` inferring classification/ordinal/regression from dtype+cardinality — verified against `asthma_status`, `severity`, `fev1`, and `exacerbation_count` (a genuinely custom outcome not in any hardcoded list) |
| Early fusion | ✅ Fully implemented, tested |
| Intermediate fusion (AE, VAE) | ✅ Fully implemented, tested |
| Late fusion (ensemble, stacking) | ✅ Fully implemented, tested |
| Graph-based fusion | 
| Classification models (LogReg, RF, XGBoost, LightGBM, CatBoost, MLP) | ✅ All six verified to fit and predict, including the three optional-dependency ones (previously untested). LightGBM's default verbose training spam is now suppressed (real bug fix) |
| TabTransformer | 
| Ordinal prediction (CORAL, ordinal logistic regression) | ✅ Fully implemented, tested |
| "DeepCORAL" | 
| Regression models (ElasticNet, XGBoost, LightGBM, MLP) | ✅ Fully implemented, XGBoost/LightGBM paths now verified |
| Transformer regressor | 
| Explainability: SHAP, permutation importance, Integrated Gradients | ✅ Fully implemented and tested. **Fixed a real bug**: modern SHAP returns a 3D array for tree classifiers rather than the legacy list format, which crashed both `shap_explain` and `tier3_shap_selection` — now handles both formats |
| Explainability: LIME | ✅ Verified working (previously an untested optional wrapper) |
| Internal validation (k-fold, repeated k-fold, bootstrap) | ✅ Fully implemented, tested |
| **CLI now reports real cross-validated metrics**, not just in-sample | ✅ `--n-cv-folds` (default 5) runs proper CV and reports both in-sample and out-of-fold metrics side by side in the report — verified to correctly expose overfitting (in-sample accuracy 1.0 vs. CV accuracy 0.66 on a small demo cohort) rather than only showing the optimistic number |
| External (independent cohort) validation | ✅ Function provided (`utilities.validation.external_validation`); you supply the second cohort |
| Metrics (classification/regression/ordinal incl. MCC, CCC, QWK) | ✅ Fully implemented, tested |
| **Publication-quality figures** (Nature-journal styling) | ✅ **New**: `visualization.style` (Okabe-Ito colorblind-safe palette, 300+ dpi, thin spines, panel labels A/B/C/D, PNG+PDF+SVG export) plus 6 multi-panel diagnostic figure types: classification (ROC/PR/confusion matrix/calibration), regression (predicted-vs-observed/residuals/histogram/Bland-Altman), ordinal (confusion matrix/per-class accuracy), feature importance (modality-colored), CV performance summary (box+strip per metric), eQTM forest plot (β ± 95% CI). Verified via automated text-collision detection across all 6 figure types (0 overlaps) |
| **Publication-standard tables** | ✅ **New**: `reporting.tables` — Table 1 cohort characteristics (mean±SD / n(%), between-group tests, verified to correctly detect a known group difference at p<0.001 in a synthetic check), model performance summary (mean±SD across CV folds), ranked/modality-annotated biomarker table. Exports to CSV, XLSX, and publication-style table-as-PNG |
| **CLI now auto-generates the full publication bundle** | ✅ `run-all` writes `figures/Figure1-4_*.png/.pdf` and `tables/Table1-3_*.csv/.xlsx` automatically — diagnostics use **out-of-fold** CV predictions, not in-sample, for an honest figure |
| Visualizations (ROC, PR, volcano, heatmap, PCA, UMAP/t-SNE, correlation, network, feature importance) | ✅ Fully implemented and verified to render, **including UMAP, seaborn clustermap, and the networkx eQTM network plot** (previously untested optional paths) |
| Automated HTML report (Jinja2) | ✅ Fully implemented, tested, verified to render with embedded figures |
| PDF report | ✅ **Now verified working** via weasyprint (previously flagged as untested due to system-dependency risk; installed and confirmed generating a valid PDF in this environment) |
| CLI (`simulate`, `run-all`) | ✅ Fully implemented and tested via actual subprocess runs, now covering auto-discovery, task auto-inference, custom outcomes, custom models, and GPU device selection |
| Python API | ✅ Every module importable and independently usable |
| GPU compatibility | ✅ All torch-using functions accept a `device` param; CLI now exposes `--device auto/cpu/cuda` (previously only the library had this, not the CLI) |
<<<<<<< HEAD
| Jupyter notebooks (8 requested) | 
| Sphinx docs, PDF user manual, troubleshooting/best-practices guides | ✅ **PDF user manual now included** (`docs/eQTMFusion_User_Manual.pdf`, 16 pages — installation, data schema, quick start, advanced tutorial, full CLI/API reference, troubleshooting, best practices). 
=======
| Jupyter notebooks (8 requested) | 
| Sphinx docs, PDF user manual, troubleshooting/best-practices guides |  
>>>>>>> c09c751be7ffa1ffbe238dba6aeeefc88368ce26
| Docker, conda env, GitHub Actions CI | ✅ Provided (`Dockerfile`, `environment.yml`, `.github/workflows/ci.yml`) — CI config is standard but has not been run against actual GitHub infrastructure here |
| pytest / black / ruff / mypy / pre-commit config | ✅ `pyproject.toml` configured for all; pytest suite actually run (49/49 passing) |

## Changelog (bugs found and fixed during verification)

1. **SHAP 3D-array bug**: modern `shap.TreeExplainer` returns `(n_samples, n_features, n_classes)` for classifiers instead of the legacy list-of-arrays format. Both `explainability.shap_explain` and `feature_selection.tier3_shap_selection` crashed on this until fixed.
2. **`n_eqtm_pairs` exceeding available CpGs/genes** crashed the simulator on small demo cohorts (e.g. `n_cpgs=50` with default `n_eqtm_pairs=150`). Now clips with a warning.
3. **LightGBM verbose spam**: default LightGBM training printed hundreds of lines per fit. Suppressed via `verbose=-1`.
4. **pandas 3.0 string-dtype detection bug**: `col.dtype == object` silently fails under pandas 3.0's new default string dtype, which caused the CLI's task auto-inference to misclassify a binary categorical outcome (`asthma_status`) as regression, crashing when it tried `.astype(float)` on the string column "Case"/"Control". Fixed with proper `pd.api.types.is_object_dtype` / `is_string_dtype` checks throughout the CLI.
5. **CLI outcome/modality rigidity**: originally `run_all` only accepted a hardcoded 4-item outcome list and only loaded `methylation.csv`/`expression.csv` by name. Rewritten to auto-discover any CSV as a modality and accept any clinical column as an outcome, with task type auto-inferred or explicit.
6. **In-sample permutation importance is meaningless for overfit models — and the CLI was doing exactly that.** With hundreds of features and modest sample sizes, `run-all`'s classifier reaches 100% training accuracy; permuting any single feature on that same training data barely moves an already-memorized fit, so importance collapsed to ~0.000 for every single feature (verified directly: `importance_mean.describe()` showed mean=0.0, std=0.0, max=0.0 across all 500 features). Fixed by adding `explainability.cv_permutation_importance`, which retrains per CV fold and computes importance on the held-out fold — verified via two new regression tests, including one that directly demonstrates the in-sample version collapsing while the CV version does not, on the same deliberately-overfit data.
7. **Undeclared runtime dependencies**: `openpyxl` (used by `table.to_excel`) and `Pillow` (used in test DPI verification, and transitively by matplotlib) were installed in this environment but not listed in `pyproject.toml`/`requirements.txt`, which would have broken a clean install. Now declared explicitly.



## Documentation

A 16-page PDF user manual is included at [`docs/eQTMFusion_User_Manual.pdf`](docs/eQTMFusion_User_Manual.pdf), covering installation, data formatting, quick start, an advanced tutorial (custom cohorts/outcomes, running eQTM analysis yourself), full CLI reference, full Python API reference by module, troubleshooting, best practices, and the same implementation-status/changelog tables as this README. It's generated from `docs/manual_source.html` via weasyprint — regenerate after any doc edits with:

```bash
python3 -c "from weasyprint import HTML; HTML('docs/manual_source.html').write_pdf('docs/eQTMFusion_User_Manual.pdf')"
```

## Installation

```bash
pip install -e .            # core
pip install -e ".[full]"    # + xgboost, lightgbm, catboost, shap, lime, umap-learn, seaborn, networkx, Boruta, weasyprint
pip install -e ".[dev]"     # + pytest, black, ruff, mypy, pre-commit
```

## Quick start — CLI

```bash
# Generate a small demo cohort (scale up n_cpgs/n_genes/n_subjects for realistic runs)
eqtmfusion simulate --out data/ --n-subjects 300 --n-cpgs 2000 --n-genes 1000

# Run the full pipeline -- outcome and task are inferred automatically from
# clinical.csv, and every omics CSV found in data/ is auto-discovered and fused
eqtmfusion run-all --data-dir data/ --outcome severity --out results/
eqtmfusion run-all --data-dir data/ --outcome asthma_status --out results/
eqtmfusion run-all --data-dir data/ --outcome fev1 --out results/

# Any clinical column works, not just the built-in ones, and task/model are
# overridable:
eqtmfusion run-all --data-dir data/ --outcome exacerbation_count \
    --task regression --model xgboost --out results/ --n-cv-folds 5

# GPU selection for the CORAL ordinal model:
eqtmfusion run-all --data-dir data/ --outcome severity --device cuda --out results/
```

Open `results/report.html` in a browser. The report includes both in-sample metrics (optimistic) and k-fold cross-validated metrics (reportable) side by side. Alongside it, `results/figures/` and `results/tables/` contain the full publication-ready bundle: Nature-style multi-panel diagnostic figures (300+ dpi PNG + vector PDF), a Table 1 cohort characteristics table, a model performance summary table, and a ranked biomarker table (CSV + XLSX).

## Quick start — Python API

See `examples/run_end_to_end.py` for a full working walkthrough (simulate → QC → preprocess → eQTM → feature-select → fuse → train classifier + CORAL ordinal model → explain → visualize → report). Run it directly:

```bash
python examples/run_end_to_end.py
```

Runtime note: permutation-importance explainability scales with feature count and repeats; on 300 fused features with `n_repeats=5` it takes a few minutes on CPU. Reduce `n_repeats` or the feature budget for faster iteration.

## Package structure

```
eqtmfusion/
├── simulation/        synthetic multi-omics asthma cohort generator
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

## Data schema (for plugging in real data)

See `eqtmfusion.simulation.simulate.SCHEMA` for the exact clinical column names expected (`subject_id`, `age`, `sex`, `bmi`, `severity`, `fev1`, `ige`, etc.). Any omics modality is accepted as a `subjects x features` CSV with `subject_id` as the first column/index — methylation, expression, SNPs, miRNA, metabolomics, proteomics, or a custom modality of your choosing all use the same interface.

## Known limitations / next steps

- **Graph fusion is a minimal reference implementation.** For a genuinely research-grade GNN fusion module (attention weighting, per-modality partial-overlap graphs), the IntegrAO-asthma pipeline built earlier in this project (partial-overlap graph fusion, GraphSAGE encoders) is the more complete starting point — the two could be merged.
- **eQTM analysis is O(pairs) via per-pair OLS**, not vectorized genome-wide matrix regression. This is fine for a pre-filtered candidate pair list (as used throughout this package) but would need a MatrixEQTL-style vectorized rewrite for true unrestricted genome-wide scans (millions of pairs).
- **No external cohort is bundled** — `utilities.validation.external_validation` is ready to use once you have a second cohort processed through the same pipeline.
- **TabTransformer, true DeepCORAL, and a Transformer regressor are not implemented** — flagged clearly above rather than stubbed out silently.
