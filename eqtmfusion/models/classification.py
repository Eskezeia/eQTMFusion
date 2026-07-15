"""
eqtmfusion.models.classification
====================================
Classification model zoo for binary outcomes (e.g. asthma case vs control).
LightGBM and CatBoost are optional -- wrapped so the package still works
without them installed.
"""

import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_classifier(name: str, **kwargs):
    name = name.lower()
    if name in ("logreg", "logistic_regression"):
        return LogisticRegression(max_iter=2000, **kwargs)
    if name in ("random_forest", "rf"):
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, **kwargs)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed: pip install xgboost")
        return XGBClassifier(n_estimators=300, eval_metric="logloss", **kwargs)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("lightgbm not installed: pip install lightgbm")
        kwargs.setdefault("verbose", -1)
        return LGBMClassifier(n_estimators=300, **kwargs)
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("catboost not installed: pip install catboost")
        return CatBoostClassifier(iterations=300, verbose=False, **kwargs)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, **kwargs)
    if name == "tabtransformer":
        raise NotImplementedError(
            "TabTransformer is not implemented in this release. Use 'mlp', "
            "'xgboost', or 'lightgbm' instead, or contribute a "
            "pytorch-tabnet / tab-transformer-pytorch integration."
        )
    raise ValueError(f"Unknown classifier: {name}")


CLASSIFIER_NAMES = ["logreg", "random_forest", "xgboost", "lightgbm", "catboost", "mlp"]
