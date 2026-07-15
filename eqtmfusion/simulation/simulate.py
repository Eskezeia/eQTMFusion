"""
eqtmfusion.simulation
========================
Generates a realistic synthetic multi-omics asthma cohort for demonstration,
testing, and tutorials. All effect structure (which CpGs regulate which
genes, which features drive severity/FEV1/IgE) is embedded with known
ground truth so downstream eQTM and feature-selection modules can be
validated against it.

This is a SIMULATOR, not real patient data -- swap in real data via the
same schema documented in `SCHEMA` below.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


SCHEMA = {
    "clinical_columns": [
        "subject_id", "age", "sex", "bmi", "race", "smoking_status",
        "asthma_status", "severity", "gina_step", "fev1", "fvc", "fev1_fvc",
        "ige", "eosinophils", "ics_use", "hospitalization", "exacerbation_count",
    ],
    "severity_levels": ["Control", "Mild", "Moderate", "Severe"],
}


@dataclass
class SimulatedCohort:
    clinical: pd.DataFrame
    methylation: pd.DataFrame       # subjects x CpGs
    expression: pd.DataFrame        # subjects x genes
    snps: pd.DataFrame              # subjects x variants (0/1/2)
    mirna: pd.DataFrame             # subjects x miRNAs
    metabolomics: pd.DataFrame      # subjects x metabolites
    proteomics: pd.DataFrame        # subjects x proteins
    ground_truth: dict = field(default_factory=dict)  # true eQTM pairs, causal features


def _make_subject_ids(n: int) -> np.ndarray:
    return np.array([f"SUBJ_{i:05d}" for i in range(n)])


def simulate_clinical(n_subjects: int, rng: np.random.Generator) -> pd.DataFrame:
    subject_id = _make_subject_ids(n_subjects)
    age = rng.normal(45, 15, n_subjects).clip(5, 90)
    sex = rng.choice(["F", "M"], n_subjects)
    bmi = rng.normal(27, 5, n_subjects).clip(15, 55)
    race = rng.choice(["White", "Black", "Asian", "Hispanic", "Other"], n_subjects,
                       p=[0.55, 0.2, 0.1, 0.1, 0.05])
    smoking_status = rng.choice(["Never", "Former", "Current"], n_subjects, p=[0.6, 0.3, 0.1])

    asthma_status = rng.choice(["Case", "Control"], n_subjects, p=[0.6, 0.4])

    severity = np.array(["Control"] * n_subjects, dtype=object)
    case_mask = asthma_status == "Case"
    severity[case_mask] = rng.choice(
        ["Mild", "Moderate", "Severe"], case_mask.sum(), p=[0.4, 0.4, 0.2]
    )
    gina_step_map = {"Control": 0, "Mild": rng.integers(1, 3), "Moderate": 3, "Severe": rng.integers(4, 6)}
    gina_step = np.array([
        0 if s == "Control" else
        int(rng.integers(1, 3)) if s == "Mild" else
        3 if s == "Moderate" else
        int(rng.integers(4, 6))
        for s in severity
    ])

    severity_penalty = pd.Series(severity).map(
        {"Control": 0, "Mild": 8, "Moderate": 18, "Severe": 32}
    ).values
    fev1_pct = (95 - severity_penalty + rng.normal(0, 8, n_subjects)).clip(25, 130)
    fvc_pct = (100 - severity_penalty * 0.6 + rng.normal(0, 7, n_subjects)).clip(35, 130)
    fev1_fvc = (fev1_pct / fvc_pct).clip(0.3, 1.0)

    log_ige_base = 3.0 + severity_penalty * 0.03
    ige = np.expm1(rng.normal(log_ige_base, 0.6, n_subjects)).clip(2, 5000)

    eosinophils = (100 + severity_penalty * 8 + rng.normal(0, 60, n_subjects)).clip(0, 1500)

    ics_use = np.where(case_mask, rng.choice([0, 1], n_subjects, p=[0.3, 0.7]), 0)
    hospitalization = rng.binomial(1, np.clip(severity_penalty / 100, 0, 0.5))
    exacerbation_count = rng.poisson(np.clip(severity_penalty / 15, 0, 5))

    df = pd.DataFrame({
        "subject_id": subject_id, "age": age.round(1), "sex": sex, "bmi": bmi.round(1),
        "race": race, "smoking_status": smoking_status, "asthma_status": asthma_status,
        "severity": severity, "gina_step": gina_step, "fev1": fev1_pct.round(1),
        "fvc": fvc_pct.round(1), "fev1_fvc": fev1_fvc.round(3), "ige": ige.round(1),
        "eosinophils": eosinophils.round(1), "ics_use": ics_use,
        "hospitalization": hospitalization, "exacerbation_count": exacerbation_count,
    }).set_index("subject_id")

    return df


def _simulate_omic_with_causal_structure(
    n_subjects: int, n_features: int, n_causal: int, severity_penalty: np.ndarray,
    rng: np.random.Generator, feature_prefix: str, causal_effect_scale: float = 0.02,
) -> tuple[pd.DataFrame, list]:
    """
    Generate an omics matrix where `n_causal` features carry a real
    (simulated) association with disease severity; the rest are noise.
    Returns the dataframe and the list of causal feature names (ground truth).
    """
    X = rng.normal(0, 1, (n_subjects, n_features))
    causal_idx = rng.choice(n_features, size=n_causal, replace=False)
    for idx in causal_idx:
        X[:, idx] += causal_effect_scale * severity_penalty * rng.choice([-1, 1])
    feature_names = [f"{feature_prefix}_{i:06d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names,
                       index=[f"SUBJ_{i:05d}" for i in range(n_subjects)])
    causal_names = [feature_names[i] for i in causal_idx]
    return df, causal_names


def simulate_eqtm_linked_omics(
    n_subjects: int, n_cpgs: int, n_genes: int, n_eqtm_pairs: int,
    severity_penalty: np.ndarray, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Jointly simulate methylation and expression with embedded true cis-eQTM
    relationships: for `n_eqtm_pairs` (CpG, gene) pairs, methylation causally
    (negatively, as is typical) influences expression, and both are linked
    to severity through the same causal pathway.

    Returns (methylation_df, expression_df, list of (cpg, gene) ground-truth pairs)
    """
    meth, causal_cpgs = _simulate_omic_with_causal_structure(
        n_subjects, n_cpgs, n_eqtm_pairs, severity_penalty, rng, "cg", causal_effect_scale=0.015
    )
    expr = rng.normal(0, 1, (n_subjects, n_genes))
    gene_names = [f"GENE_{i:06d}" for i in range(n_genes)]
    eqtm_gene_idx = rng.choice(n_genes, size=n_eqtm_pairs, replace=False)

    eqtm_pairs = []
    for k, cpg_name in enumerate(causal_cpgs):
        gene_idx = eqtm_gene_idx[k]
        # canonical negative methylation-expression coupling
        expr[:, gene_idx] += -0.6 * meth[cpg_name].values + rng.normal(0, 0.3, n_subjects)
        eqtm_pairs.append((cpg_name, gene_names[gene_idx]))

    expr_df = pd.DataFrame(expr, columns=gene_names, index=meth.index)
    return meth, expr_df, eqtm_pairs


def simulate_cohort(
    n_subjects: int = 2000,
    n_cpgs: int = 50_000,
    n_genes: int = 10_000,
    n_snps: int = 5_000,
    n_mirna: int = 500,
    n_metabolites: int = 500,
    n_proteins: int = 1_000,
    n_eqtm_pairs: int = 150,
    random_state: int = 42,
) -> SimulatedCohort:
    """
    Full simulated cohort generator matching the requested schema. Default
    sizes match the spec (2000 subjects, 50k CpGs, 10k genes, 5k SNPs,
    500 miRNA, 500 metabolites, 1000 proteins).

    NOTE: at full default size this generates ~2000 x 50000 methylation and
    similarly large matrices (several GB in memory). For interactive/demo
    use, pass smaller sizes, e.g. n_subjects=200, n_cpgs=2000, n_genes=1000.
    """
    rng = np.random.default_rng(random_state)

    if n_eqtm_pairs > min(n_cpgs, n_genes):
        clipped = min(n_eqtm_pairs, n_cpgs, n_genes)
        import warnings
        warnings.warn(
            f"n_eqtm_pairs={n_eqtm_pairs} exceeds available n_cpgs={n_cpgs} or "
            f"n_genes={n_genes}; clipping to {clipped}."
        )
        n_eqtm_pairs = clipped

    clinical = simulate_clinical(n_subjects, rng)
    severity_penalty = clinical["severity"].map(
        {"Control": 0, "Mild": 8, "Moderate": 18, "Severe": 32}
    ).values

    methylation, expression, eqtm_pairs = simulate_eqtm_linked_omics(
        n_subjects, n_cpgs, n_genes, n_eqtm_pairs, severity_penalty, rng
    )

    snps_raw = rng.choice([0, 1, 2], size=(n_subjects, n_snps), p=[0.49, 0.42, 0.09])
    snps = pd.DataFrame(
        snps_raw, columns=[f"rs{1000000 + i}" for i in range(n_snps)], index=clinical.index
    )

    mirna, causal_mirna = _simulate_omic_with_causal_structure(
        n_subjects, n_mirna, max(5, n_mirna // 50), severity_penalty, rng, "miR", 0.02
    )
    metabolomics, causal_metab = _simulate_omic_with_causal_structure(
        n_subjects, n_metabolites, max(5, n_metabolites // 50), severity_penalty, rng, "MET", 0.02
    )
    proteomics, causal_prot = _simulate_omic_with_causal_structure(
        n_subjects, n_proteins, max(5, n_proteins // 50), severity_penalty, rng, "PROT", 0.02
    )

    ground_truth = {
        "eqtm_pairs": eqtm_pairs,
        "causal_cpgs": [p[0] for p in eqtm_pairs],
        "causal_genes": [p[1] for p in eqtm_pairs],
        "causal_mirna": causal_mirna,
        "causal_metabolites": causal_metab,
        "causal_proteins": causal_prot,
    }

    return SimulatedCohort(
        clinical=clinical, methylation=methylation, expression=expression,
        snps=snps, mirna=mirna, metabolomics=metabolomics, proteomics=proteomics,
        ground_truth=ground_truth,
    )


def save_cohort(cohort: SimulatedCohort, out_dir: str) -> None:
    """Write all cohort tables to CSV under out_dir/."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    cohort.clinical.to_csv(f"{out_dir}/clinical.csv")
    cohort.methylation.to_csv(f"{out_dir}/methylation.csv")
    cohort.expression.to_csv(f"{out_dir}/expression.csv")
    cohort.snps.to_csv(f"{out_dir}/snps.csv")
    cohort.mirna.to_csv(f"{out_dir}/mirna.csv")
    cohort.metabolomics.to_csv(f"{out_dir}/metabolomics.csv")
    cohort.proteomics.to_csv(f"{out_dir}/proteomics.csv")

    import json
    with open(f"{out_dir}/ground_truth.json", "w") as f:
        json.dump(cohort.ground_truth, f, indent=2)
