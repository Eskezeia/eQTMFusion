import numpy as np
import pandas as pd
from eqtmfusion.simulation import simulate_cohort
from eqtmfusion.eqtm import run_eqtm_analysis, get_significant_eqtm_cpgs


def test_eqtm_recovers_simulated_ground_truth_pairs():
    cohort = simulate_cohort(
        n_subjects=300, n_cpgs=100, n_genes=50, n_eqtm_pairs=5, random_state=7,
    )
    covariates = cohort.clinical[["age", "bmi"]]  # numeric-only for a fast/simple test

    true_pairs = cohort.ground_truth["eqtm_pairs"]
    # test true pairs plus a random negative-control sample of pairs
    rng = np.random.default_rng(0)
    neg_cpgs = rng.choice(cohort.methylation.columns, 20, replace=False)
    neg_genes = rng.choice(cohort.expression.columns, 20, replace=False)
    candidate_pairs = true_pairs + list(zip(neg_cpgs, neg_genes))

    result = run_eqtm_analysis(
        cohort.methylation, cohort.expression, covariates,
        candidate_pairs=candidate_pairs, eqtm_type="cis", model="linear",
    )
    assert len(result.results) > 0

    # true pairs should generally show smaller p-values than random pairs
    true_pair_set = set(true_pairs)
    result.results["is_true_pair"] = result.results.apply(
        lambda r: (r["cpg"], r["gene"]) in true_pair_set, axis=1
    )
    mean_p_true = result.results.loc[result.results["is_true_pair"], "pvalue"].mean()
    mean_p_false = result.results.loc[~result.results["is_true_pair"], "pvalue"].mean()
    assert mean_p_true < mean_p_false


def test_get_significant_eqtm_cpgs_returns_list():
    cohort = simulate_cohort(n_subjects=200, n_cpgs=50, n_genes=30, n_eqtm_pairs=5, random_state=3)
    covariates = cohort.clinical[["age", "bmi"]]
    result = run_eqtm_analysis(
        cohort.methylation, cohort.expression, covariates,
        candidate_pairs=cohort.ground_truth["eqtm_pairs"],
    )
    sig_cpgs = get_significant_eqtm_cpgs(result, fdr_thresh=1.0)  # lenient threshold for small-sample test
    assert isinstance(sig_cpgs, list)
