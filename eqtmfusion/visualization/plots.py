"""
eqtmfusion.visualization
============================
Publication-quality plotting functions. All functions return a matplotlib
Figure and optionally save to PNG/PDF/SVG via `save_path` (extension
determines format).

UMAP is optional (`pip install umap-learn`); falls back to t-SNE if absent.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def _save(fig, save_path: str = None):
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    return fig


def plot_roc_curve(y_true, y_proba, save_path: str = None, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, save_path)


def plot_pr_curve(y_true, y_proba, save_path: str = None, title: str = "Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    return _save(fig, save_path)


def plot_volcano(
    effect_sizes: pd.Series, pvalues: pd.Series, fdr: pd.Series = None,
    fdr_thresh: float = 0.05, effect_thresh: float = 0.5,
    save_path: str = None, title: str = "Volcano Plot",
    xlabel: str = "Effect size (beta)",
):
    neg_log_p = -np.log10(pvalues.clip(lower=1e-300))
    if fdr is not None:
        significant = (fdr <= fdr_thresh) & (effect_sizes.abs() >= effect_thresh)
    else:
        significant = effect_sizes.abs() >= effect_thresh

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(effect_sizes[~significant], neg_log_p[~significant], color="gray", s=8, alpha=0.5)
    ax.scatter(effect_sizes[significant], neg_log_p[significant], color="crimson", s=10)
    ax.axvline(effect_thresh, linestyle="--", color="gray", linewidth=0.8)
    ax.axvline(-effect_thresh, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    return _save(fig, save_path)


def plot_heatmap(matrix: pd.DataFrame, save_path: str = None, title: str = "Heatmap",
                  cmap: str = "vlag", cluster: bool = True):
    try:
        import seaborn as sns
        if cluster:
            g = sns.clustermap(matrix, cmap=cmap, figsize=(8, 8))
            g.fig.suptitle(title, y=1.02)
            if save_path:
                g.savefig(save_path, bbox_inches="tight", dpi=200)
            return g.fig
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(matrix, cmap=cmap, ax=ax)
            ax.set_title(title)
            return _save(fig, save_path)
    except ImportError:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix.values, aspect="auto", cmap=cmap)
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        return _save(fig, save_path)


def plot_pca(X: np.ndarray, labels: pd.Series = None, save_path: str = None,
             title: str = "PCA", n_components: int = 2):
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is not None:
        for lab in pd.Series(labels).unique():
            mask = np.asarray(labels) == lab
            ax.scatter(coords[mask, 0], coords[mask, 1], label=str(lab), s=12, alpha=0.7)
        ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(title)
    return _save(fig, save_path)


def plot_embedding(X: np.ndarray, labels: pd.Series = None, method: str = "umap",
                    save_path: str = None, title: str = None, random_state: int = 42):
    """method: 'umap' (falls back to t-SNE if umap-learn not installed) or 'tsne'."""
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(random_state=random_state)
        except ImportError:
            warnings.warn("umap-learn not installed; falling back to t-SNE.")
            reducer = TSNE(n_components=2, random_state=random_state)
            method = "tsne"
    else:
        reducer = TSNE(n_components=2, random_state=random_state)

    coords = reducer.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is not None:
        for lab in pd.Series(labels).unique():
            mask = np.asarray(labels) == lab
            ax.scatter(coords[mask, 0], coords[mask, 1], label=str(lab), s=12, alpha=0.7)
        ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.7)
    ax.set_title(title or method.upper())
    return _save(fig, save_path)


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None,
                             title: str = "Correlation Matrix", method: str = "pearson"):
    corr = df.corr(method=method)
    return plot_heatmap(corr, save_path=save_path, title=title, cluster=False)


def plot_feature_importance(importance_df: pd.DataFrame, value_col: str = None,
                             top_n: int = 25, save_path: str = None,
                             title: str = "Feature Importance"):
    if value_col is None:
        value_col = importance_df.columns[0]
    top = importance_df.sort_values(value_col, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.3)))
    ax.barh(top.index[::-1], top[value_col][::-1])
    ax.set_xlabel(value_col)
    ax.set_title(title)
    return _save(fig, save_path)


def plot_eqtm_network(eqtm_pairs: list, save_path: str = None, title: str = "eQTM Network",
                       max_edges: int = 200):
    """eqtm_pairs: list of (cpg, gene) tuples, e.g. from eqtm.get_significant_eqtm_cpgs
    combined with the full result table."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx not installed: pip install networkx")

    G = nx.Graph()
    for cpg, gene in eqtm_pairs[:max_edges]:
        G.add_node(cpg, bipartite=0, node_type="cpg")
        G.add_node(gene, bipartite=1, node_type="gene")
        G.add_edge(cpg, gene)

    pos = nx.spring_layout(G, seed=42, k=0.5)
    fig, ax = plt.subplots(figsize=(9, 9))
    cpg_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "cpg"]
    gene_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "gene"]
    nx.draw_networkx_nodes(G, pos, nodelist=cpg_nodes, node_color="steelblue", node_size=40, ax=ax, label="CpG")
    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_color="crimson", node_size=40, ax=ax, label="Gene")
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    ax.set_title(title)
    ax.legend()
    ax.axis("off")
    return _save(fig, save_path)
