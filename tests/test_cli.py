import os
import subprocess
import sys
import pandas as pd
from eqtmfusion.cli.main import discover_omics_modalities, infer_task
from eqtmfusion.simulation import simulate_cohort, save_cohort


def test_discover_omics_modalities_finds_all_non_reserved_csvs(tmp_path):
    cohort = simulate_cohort(n_subjects=30, n_cpgs=20, n_genes=15, n_snps=10,
                              n_mirna=5, n_metabolites=5, n_proteins=8, random_state=1)
    save_cohort(cohort, str(tmp_path))
    modalities = discover_omics_modalities(str(tmp_path))
    assert set(modalities.keys()) == {
        "methylation", "expression", "snps", "mirna", "metabolomics", "proteomics"
    }
    for df in modalities.values():
        assert df.shape[0] == 30


def test_infer_task_classification_for_binary_string_column():
    df = pd.DataFrame({"asthma_status": ["Case", "Control", "Case", "Control"]})
    assert infer_task(df, "asthma_status") == "classification"


def test_infer_task_ordinal_for_multicategory_string_column():
    df = pd.DataFrame({"severity": ["Control", "Mild", "Moderate", "Severe"]})
    assert infer_task(df, "severity") == "ordinal"


def test_infer_task_regression_for_continuous_float_column():
    df = pd.DataFrame({"fev1": [88.5, 72.3, 95.1, 60.0]})
    assert infer_task(df, "fev1") == "regression"


def test_run_all_cli_end_to_end_via_subprocess(tmp_path):
    """Full subprocess-level test of the actual CLI entrypoint, matching how
    a real user would invoke it, for a genuinely custom outcome column not
    in any hardcoded list."""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "results"
    cohort = simulate_cohort(n_subjects=60, n_cpgs=50, n_genes=30, n_snps=20,
                              n_mirna=10, n_metabolites=10, n_proteins=15,
                              n_eqtm_pairs=5, random_state=2)
    save_cohort(cohort, str(data_dir))

    result = subprocess.run(
        [sys.executable, "-m", "eqtmfusion.cli.main", "run-all",
         "--data-dir", str(data_dir), "--outcome", "eosinophils",
         "--task", "regression", "--out", str(out_dir),
         "--target-n-features", "30", "--n-cv-folds", "2"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert (out_dir / "report.html").exists()
    report_html = (out_dir / "report.html").read_text()
    assert "eosinophils" in report_html
