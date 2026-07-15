import numpy as np
from eqtmfusion.simulation import simulate_cohort


def test_simulate_cohort_shapes():
    cohort = simulate_cohort(
        n_subjects=50, n_cpgs=200, n_genes=100, n_snps=50, n_mirna=20,
        n_metabolites=20, n_proteins=30, n_eqtm_pairs=10, random_state=1,
    )
    assert cohort.clinical.shape[0] == 50
    assert cohort.methylation.shape == (50, 200)
    assert cohort.expression.shape == (50, 100)
    assert cohort.snps.shape == (50, 50)
    assert cohort.mirna.shape == (50, 20)
    assert cohort.metabolomics.shape == (50, 20)
    assert cohort.proteomics.shape == (50, 30)


def test_simulate_cohort_ground_truth_eqtm_pairs():
    cohort = simulate_cohort(
        n_subjects=50, n_cpgs=200, n_genes=100, n_eqtm_pairs=10, random_state=1,
    )
    assert len(cohort.ground_truth["eqtm_pairs"]) == 10
    for cpg, gene in cohort.ground_truth["eqtm_pairs"]:
        assert cpg in cohort.methylation.columns
        assert gene in cohort.expression.columns


def test_simulate_cohort_severity_categories():
    cohort = simulate_cohort(n_subjects=100, n_cpgs=50, n_genes=50, random_state=1)
    assert set(cohort.clinical["severity"].unique()) <= {"Control", "Mild", "Moderate", "Severe"}
