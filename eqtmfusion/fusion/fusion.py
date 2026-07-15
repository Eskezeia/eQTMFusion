"""
eqtmfusion.fusion
=====================
Multi-omics fusion strategies:

  - Early fusion: simple feature concatenation across modalities
  - Intermediate fusion: joint latent space via autoencoder (AE) or
    variational autoencoder (VAE)
  - Late fusion: train per-modality models, combine via averaging (ensemble)
    or a meta-learner (stacking)
  - Graph-based fusion: a lightweight GCN-style patient-similarity fusion
    (single-graph-convolution reference implementation -- NOT a full
    GAT/attention model; see module docstring note below for how to extend)

All fusion functions operate on ALIGNED sample sets. If your cohort has
partial overlap across modalities (common in real data), align/impute to a
common sample index first, or use only the samples present in every
modality you intend to fuse.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold


# ----------------------------- Early fusion -------------------------------

def early_fusion(modality_dfs: dict) -> pd.DataFrame:
    """
    Concatenate features across modalities for samples present in ALL
    modalities (inner join on index). Simplest and most common baseline.
    """
    common_idx = None
    for df in modality_dfs.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    aligned = [df.loc[common_idx].add_prefix(f"{name}__")
               for name, df in modality_dfs.items()]
    return pd.concat(aligned, axis=1)


# ------------------------- Intermediate fusion (AE/VAE) -------------------

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder_hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        h = self.encoder_hidden(x)
        mu, logvar = self.mu_layer(h), self.logvar_layer(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


def intermediate_fusion_autoencoder(
    modality_dfs: dict, latent_dim: int = 32, hidden_dim: int = 128,
    variational: bool = False, n_epochs: int = 200, lr: float = 1e-3,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Concatenate modalities (early-fusion style input), then compress through
    a joint autoencoder/VAE to obtain a shared latent embedding per sample.
    Returns a samples x latent_dim dataframe.
    """
    concat_df = early_fusion(modality_dfs)
    X = torch.tensor(concat_df.values, dtype=torch.float32, device=device)
    input_dim = X.shape[1]

    if variational:
        model = VariationalAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    else:
        model = Autoencoder(input_dim, latent_dim, hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        if variational:
            recon, mu, logvar = model(X)
            loss = vae_loss(recon, X, mu, logvar)
        else:
            recon, z = model(X)
            loss = F.mse_loss(recon, X)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        if variational:
            h = model.encoder_hidden(X)
            z = model.mu_layer(h)  # use mean as the deterministic embedding
        else:
            _, z = model(X)

    latent_cols = [f"latent_{i}" for i in range(latent_dim)]
    return pd.DataFrame(z.cpu().numpy(), index=concat_df.index, columns=latent_cols)


# ----------------------------- Late fusion --------------------------------

def late_fusion_ensemble(
    modality_dfs: dict, y: pd.Series, base_model_fn, task: str = "classification",
    n_splits: int = 5, random_state: int = 42,
) -> dict:
    """
    Train one model per modality (on its own feature set), then average
    predicted probabilities/values across modalities (simple ensemble late
    fusion). Returns out-of-fold predictions per modality and the ensembled
    prediction, via cross-validation.

    base_model_fn: zero-arg callable returning an unfitted sklearn estimator
    """
    common_idx = None
    for df in modality_dfs.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    y_aligned = y.loc[common_idx]

    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(np.zeros(len(common_idx)), y_aligned)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(np.zeros(len(common_idx)))

    oof_preds = {name: np.zeros(len(common_idx)) for name in modality_dfs}
    for train_idx, test_idx in split_iter:
        for name, df in modality_dfs.items():
            X = df.loc[common_idx].values
            model = clone(base_model_fn()) if hasattr(base_model_fn(), "get_params") else base_model_fn()
            model.fit(X[train_idx], y_aligned.values[train_idx])
            if task == "classification" and hasattr(model, "predict_proba"):
                pred = model.predict_proba(X[test_idx])[:, 1]
            else:
                pred = model.predict(X[test_idx])
            oof_preds[name][test_idx] = pred

    ensembled = np.mean(np.stack(list(oof_preds.values())), axis=0)
    return {"per_modality_oof": oof_preds, "ensembled_oof": ensembled, "y_true": y_aligned.values,
            "sample_index": common_idx}


def late_fusion_stacking(
    modality_dfs: dict, y: pd.Series, base_model_fn, meta_model=None,
    task: str = "classification", n_splits: int = 5, random_state: int = 42,
) -> dict:
    """
    Late fusion via stacking: per-modality out-of-fold predictions become
    input features to a meta-learner (default: LogisticRegression for
    classification, Ridge for regression).
    """
    result = late_fusion_ensemble(modality_dfs, y, base_model_fn, task, n_splits, random_state)
    meta_X = np.column_stack(list(result["per_modality_oof"].values()))
    meta_y = result["y_true"]

    if meta_model is None:
        meta_model = LogisticRegression() if task == "classification" else Ridge()
    meta_model.fit(meta_X, meta_y)

    if task == "classification" and hasattr(meta_model, "predict_proba"):
        stacked_pred = meta_model.predict_proba(meta_X)[:, 1]
    else:
        stacked_pred = meta_model.predict(meta_X)

    result["meta_model"] = meta_model
    result["stacked_pred"] = stacked_pred
    return result


# ------------------------- Graph-based fusion (lightweight) ---------------

class SimpleGCNFusion(nn.Module):
    """
    Lightweight single-layer graph convolution fusion, reusing the same
    patient-similarity-graph idea as IntegrAO (see Ma et al. 2025,
    Nature Machine Intelligence, https://doi.org/10.1038/s42256-024-00942-3)
    but simplified to one shared graph built from concatenated features
    rather than per-modality fusion.

    NOTE: This is a minimal educational reference implementation, not a full
    GAT (graph attention) model. For a production GNN-fusion module with
    attention weighting and per-modality graphs, extend this class or adapt
    the partial-overlap fusion pipeline built for the IntegrAO-asthma project.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 32):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, X: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        h = F.relu(self.W1((adj @ X) / deg))
        out = self.W2((adj @ h) / deg)
        return out


def build_knn_adjacency(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Build a simple unweighted K-nearest-neighbor adjacency matrix from
    Euclidean distance, for use as input to SimpleGCNFusion."""
    from scipy.spatial.distance import squareform, pdist
    dist = squareform(pdist(X, metric="euclidean"))
    n = dist.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dist[i])[1:k + 1]  # exclude self
        adj[i, neighbors] = 1.0
    adj = np.maximum(adj, adj.T)  # symmetrize
    return adj


def graph_fusion_embed(
    modality_dfs: dict, k: int = 10, out_dim: int = 32, n_epochs: int = 200,
    lr: float = 1e-3, device: str = "cpu",
) -> pd.DataFrame:
    """
    Early-fuse modalities, build a KNN patient-similarity graph, and train a
    simple GCN autoencoder-style embedding (reconstructing the concatenated
    feature matrix from graph-propagated representations) to obtain a fused
    graph embedding per patient.
    """
    concat_df = early_fusion(modality_dfs)
    X_np = concat_df.values
    adj_np = build_knn_adjacency(X_np, k=k)

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    adj = torch.tensor(adj_np, dtype=torch.float32, device=device)

    encoder = SimpleGCNFusion(X.shape[1], hidden_dim=128, out_dim=out_dim).to(device)
    decoder = nn.Linear(out_dim, X.shape[1]).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        z = encoder(X, adj)
        recon = decoder(z)
        loss = F.mse_loss(recon, X)
        loss.backward()
        optimizer.step()

    encoder.eval()
    with torch.no_grad():
        z_final = encoder(X, adj)

    latent_cols = [f"graph_latent_{i}" for i in range(out_dim)]
    return pd.DataFrame(z_final.cpu().numpy(), index=concat_df.index, columns=latent_cols)
