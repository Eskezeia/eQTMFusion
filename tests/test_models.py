import numpy as np
from eqtmfusion.models import (
    get_classifier, get_regressor, train_coral_model, predict_coral,
    fit_ordinal_logistic, predict_ordinal_logistic,
)


def test_get_classifier_logreg_fits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    y = (X[:, 0] > 0).astype(int)
    model = get_classifier("logreg")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (50,)


def test_get_regressor_elasticnet_fits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    y = X[:, 0] * 2 + rng.normal(0, 0.1, 50)
    model = get_regressor("elasticnet")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (50,)


def test_coral_ordinal_model_trains_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6)).astype(np.float32)
    y = np.digitize(X[:, 0], bins=[-1, 0, 1]).astype(np.int64)  # 4 ordinal classes
    num_classes = len(np.unique(y))
    model = train_coral_model(X, y, num_classes, n_epochs=30, verbose_every=0)
    preds = predict_coral(model, X)
    assert preds.shape == (80,)
    assert preds.min() >= 0 and preds.max() < num_classes


def test_ordinal_logistic_regression_fits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = np.digitize(X[:, 0], bins=[-0.5, 0.5]).astype(int)
    fitted = fit_ordinal_logistic(X, y)
    preds = predict_ordinal_logistic(fitted, X)
    assert preds.shape == (60,)


def test_tabtransformer_raises_not_implemented():
    import pytest
    with pytest.raises(NotImplementedError):
        get_classifier("tabtransformer")
