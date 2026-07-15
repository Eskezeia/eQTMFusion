"""
eqtmfusion.models.ordinal
=============================
Ordinal prediction for GINA severity steps (1-5) / severity categories.

Implements:
  - CORAL (COnsistent RAnk Logits): rank-consistent ordinal neural network
    (Cao, Mirjalili & Raschka, 2020)
  - Ordinal Logistic Regression (proportional odds model, via statsmodels
    OrderedModel)

NOTE ON NAMING: "DeepCORAL" in the literature most commonly refers to a
*domain-adaptation* method (Sun & Saenko, 2016) unrelated to ordinal
regression, which is a likely source of confusion in the original spec. If
what's needed is a deeper CORAL-style ordinal network (more layers before
the rank-consistent output), that is directly achievable by increasing
`hidden_dims` in `CoralOrdinalNet` below -- no separate implementation is
required. True cross-cohort domain adaptation (matching the actual
DeepCORAL method) is not implemented in this release.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel


class CoralOrdinalNet(nn.Module):
    """
    Feed-forward network with a CORAL rank-consistent output layer.
    `hidden_dims`: list of hidden layer sizes, e.g. [256, 128, 64] for a
    deeper network.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            prev_dim = h
        self.backbone = nn.Sequential(*layers)
        self.num_classes = num_classes
        self.shared_layer = nn.Linear(prev_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        base = self.shared_layer(h)
        return base + self.biases  # (n, num_classes - 1)

    @staticmethod
    def encode_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
        levels = torch.zeros(y.size(0), num_classes - 1, device=y.device)
        for i in range(num_classes - 1):
            levels[:, i] = (y > i).float()
        return levels

    @staticmethod
    def coral_loss(logits: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        return -torch.mean(
            torch.sum(
                F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels),
                dim=1,
            )
        )

    @staticmethod
    def predict_labels(logits: torch.Tensor) -> torch.Tensor:
        probas = torch.sigmoid(logits)
        return (probas > 0.5).sum(dim=1)


def train_coral_model(
    X: np.ndarray, y: np.ndarray, num_classes: int, hidden_dims: list = None,
    n_epochs: int = 300, lr: float = 1e-3, device: str = "cpu", verbose_every: int = 50,
) -> CoralOrdinalNet:
    model = CoralOrdinalNet(X.shape[1], num_classes, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    levels = CoralOrdinalNet.encode_labels(y_t, num_classes)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t)
        loss = CoralOrdinalNet.coral_loss(logits, levels)
        loss.backward()
        optimizer.step()
        if verbose_every and epoch % verbose_every == 0:
            print(f"[CORAL] epoch {epoch:4d} | loss={loss.item():.4f}")

    return model


def predict_coral(model: CoralOrdinalNet, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_t)
        preds = CoralOrdinalNet.predict_labels(logits)
    return preds.cpu().numpy()


def fit_ordinal_logistic(X: np.ndarray, y: np.ndarray, feature_names: list = None):
    """
    Proportional-odds ordinal logistic regression via statsmodels OrderedModel.
    y must be integer-coded ordinal labels (0, 1, 2, ...).
    """
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    model = OrderedModel(y, df, distr="logit")
    fitted = model.fit(method="bfgs", disp=False)
    return fitted


def predict_ordinal_logistic(fitted_model, X: np.ndarray) -> np.ndarray:
    probs = fitted_model.predict(X)
    return np.argmax(probs, axis=1)
