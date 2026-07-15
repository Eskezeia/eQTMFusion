"""
eqtmfusion.models.regression
================================
Regression model zoo for continuous outcomes (FEV1, IgE, BMI, eosinophils).
"""

from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor


def get_regressor(name: str, **kwargs):
    name = name.lower()
    if name == "elasticnet":
        return ElasticNet(**kwargs)
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost not installed: pip install xgboost")
        return XGBRegressor(n_estimators=300, **kwargs)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError("lightgbm not installed: pip install lightgbm")
        kwargs.setdefault("verbose", -1)
        return LGBMRegressor(n_estimators=300, **kwargs)
    if name == "mlp":
        return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, **kwargs)
    if name == "transformer":
        raise NotImplementedError(
            "A dedicated tabular Transformer regressor is not implemented in "
            "this release. Use 'mlp', 'xgboost', or 'lightgbm' instead."
        )
    raise ValueError(f"Unknown regressor: {name}")


REGRESSOR_NAMES = ["elasticnet", "xgboost", "lightgbm", "mlp"]
