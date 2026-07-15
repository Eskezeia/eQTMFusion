import numpy as np
import pandas as pd
from eqtmfusion.fusion import early_fusion, intermediate_fusion_autoencoder, late_fusion_ensemble, graph_fusion_embed
from sklearn.linear_model import LogisticRegression


def _make_modality(n, d, seed, index=None):
    rng = np.random.default_rng(seed)
    idx = index if index is not None else [f"S{i}" for i in range(n)]
    return pd.DataFrame(rng.normal(size=(n, d)), index=idx, columns=[f"f{i}" for i in range(d)])


def test_early_fusion_concatenates_and_aligns():
    a = _make_modality(20, 5, 1)
    b = _make_modality(20, 3, 2)
    fused = early_fusion({"a": a, "b": b})
    assert fused.shape == (20, 8)
    assert all(c.startswith("a__") or c.startswith("b__") for c in fused.columns)


def test_intermediate_fusion_autoencoder_shape():
    a = _make_modality(30, 10, 1)
    b = _make_modality(30, 6, 2)
    latent = intermediate_fusion_autoencoder({"a": a, "b": b}, latent_dim=4, n_epochs=10)
    assert latent.shape == (30, 4)


def test_late_fusion_ensemble_runs():
    a = _make_modality(40, 5, 1)
    b = _make_modality(40, 5, 2)
    y = pd.Series((np.arange(40) % 2), index=a.index)
    result = late_fusion_ensemble({"a": a, "b": b}, y, lambda: LogisticRegression(), task="classification", n_splits=3)
    assert "ensembled_oof" in result
    assert len(result["ensembled_oof"]) == 40


def test_graph_fusion_embed_shape():
    a = _make_modality(25, 6, 1)
    b = _make_modality(25, 4, 2)
    embed = graph_fusion_embed({"a": a, "b": b}, k=5, out_dim=8, n_epochs=10)
    assert embed.shape == (25, 8)
