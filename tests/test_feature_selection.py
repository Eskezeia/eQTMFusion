import numpy as np
import pandas as pd
from eqtmfusion.feature_selection import combine_tier1_tier2, tier2_variance_threshold, tier3_random_forest_importance


def test_combine_tier1_tier2_prioritizes_eqtm_features():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 30)), columns=[f"cg{i}" for i in range(30)])
    eqtm_features = ["cg1", "cg5", "cg10"]

    selected, report = combine_tier1_tier2(df, eqtm_features, target_n=10, tier2_method="variance")
    assert set(eqtm_features) <= set(selected)
    assert report["n_tier1_prioritized"] == 3
    assert report["n_total"] == 10


def test_tier2_variance_threshold_ranks_by_variance():
    df = pd.DataFrame({
        "low_var": np.ones(20) + np.random.normal(0, 0.001, 20),
        "high_var": np.random.normal(0, 10, 20),
    })
    selected = tier2_variance_threshold(df, top_n=1)
    assert selected == ["high_var"]


def test_tier3_random_forest_importance_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = X[:, 0] * 2 + rng.normal(0, 0.1, 60)  # feature 0 should dominate importance
    feature_names = [f"f{i}" for i in range(8)]
    top = tier3_random_forest_importance(X, y, feature_names, top_n=3, task="regression")
    assert "f0" in top
