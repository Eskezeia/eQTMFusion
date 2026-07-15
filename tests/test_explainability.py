import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from eqtmfusion.explainability import permutation_importance_explain, cv_permutation_importance


def test_cv_permutation_importance_identifies_informative_feature_classification():
    """Regression test for the in-sample-permutation-importance bug: with a
    heavily overfit model (many features, few samples), in-sample
    permutation importance collapses to ~0 for every feature. Cross-validated
    permutation importance must NOT collapse -- it should still identify the
    one genuinely informative feature among noise."""
    rng = np.random.default_rng(0)
    n, d = 80, 60  # more features than would be safe in-sample
    X = rng.normal(size=(n, d))
    y = (X[:, 0] > 0).astype(int)  # only feature 0 is informative
    feature_names = [f"f{i}" for i in range(d)]

    imp = cv_permutation_importance(
        lambda: RandomForestClassifier(n_estimators=100, max_depth=None),
        X, y, feature_names, task="classification", n_splits=5, n_repeats=5,
    )
    # the informative feature should rank highly and have non-trivial importance
    assert imp.loc["f0", "importance_mean"] > 0.02
    top5 = imp.sort_values("importance_mean", ascending=False).head(5).index.tolist()
    assert "f0" in top5


def test_cv_permutation_importance_not_degenerate_for_overfit_model():
    """Directly demonstrates the bug that was fixed: an overfit in-sample
    permutation importance is near-zero everywhere, while the CV version
    is not."""
    rng = np.random.default_rng(1)
    n, d = 60, 100  # deliberately overfit-prone: more features than samples/2
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    feature_names = [f"f{i}" for i in range(d)]

    # in-sample version (the old, buggy usage pattern)
    model = RandomForestClassifier(n_estimators=200, max_depth=None)
    model.fit(X, y)
    insample_imp = permutation_importance_explain(model, X, y, feature_names, n_repeats=5)

    # cross-validated version (the fix)
    cv_imp = cv_permutation_importance(
        lambda: RandomForestClassifier(n_estimators=200, max_depth=None),
        X, y, feature_names, task="classification", n_splits=5, n_repeats=5,
    )

    # the CV version's max importance should be meaningfully larger than the
    # in-sample version's max importance for this deliberately overfit setup
    assert cv_imp["importance_mean"].max() > insample_imp["importance_mean"].max()


def test_cv_permutation_importance_regression_task():
    rng = np.random.default_rng(0)
    n, d = 80, 20
    X = rng.normal(size=(n, d))
    y = 3 * X[:, 0] + rng.normal(0, 0.5, n)
    feature_names = [f"f{i}" for i in range(d)]

    imp = cv_permutation_importance(
        lambda: RandomForestRegressor(n_estimators=100),
        X, y, feature_names, task="regression", n_splits=4, n_repeats=5,
    )
    top1 = imp.sort_values("importance_mean", ascending=False).index[0]
    assert top1 == "f0"
