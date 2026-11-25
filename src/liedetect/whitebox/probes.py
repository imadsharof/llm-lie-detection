"""The 8 white-box probes (LR, LR-G, DIM, LDA, PCA, PCA-G, LAT, CCS).

Re-implementation of G. Vasile's wb_probes.py, itself based on mishajw/repeng.
Every probe is linear: score(x) = w . x + b, a higher score meaning "true".
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

METHODS = ["lr", "lr-g", "dim", "lda", "pca", "pca-g", "lat", "ccs"]
UNSUPERVISED = {"dim", "pca", "pca-g", "lat", "ccs"}  # sign must be fixed afterwards


def center_by_group(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract the mean activation of each question (removes the topic)."""
    X = X.copy()
    for g in np.unique(groups):
        mask = groups == g
        X[mask] -= X[mask].mean(axis=0)
    return X


@dataclass
class LinearProbe:
    w: np.ndarray
    b: float = 0.0
    center: np.ndarray | None = None   # vector subtracted before projection (LAT)
    group_center: bool = False         # centre test activations per question (LR-G)
    sign: float = 1.0

    def score(self, X: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        X = X.astype(np.float64)
        if self.group_center and groups is not None:
            X = center_by_group(X, groups)
        if self.center is not None:
            X = X - self.center
        return self.sign * (X @ self.w + self.b)


def _fit_lr(X, y, C=1.0):
    clf = LogisticRegression(C=C, solver="newton-cg", max_iter=10000)
    clf.fit(X, y)
    return clf.coef_[0], float(clf.intercept_[0])


def _first_component(X):
    pca = PCA(n_components=1).fit(X)
    w = pca.components_[0]
    return w / np.linalg.norm(w)


def _fit_ccs(X, y, groups, seed=42, attempts=3):
    """Contrast-Consistent Search on (true, false) pairs of the same question."""
    pos, neg = [], []
    for g in np.unique(groups):
        mask = groups == g
        if y[mask].min() == 0 and y[mask].max() == 1:
            pos.append(X[mask][y[mask] == 1][0])
            neg.append(X[mask][y[mask] == 0][0])
    if not pos:
        raise ValueError("CCS needs questions with both a true and a false answer")
    a = torch.tensor(np.array(pos), dtype=torch.float32)
    b = torch.tensor(np.array(neg), dtype=torch.float32)

    best = None
    for attempt in range(attempts):
        torch.manual_seed(seed + attempt)
        linear = torch.nn.Linear(X.shape[1], 1)
        optimizer = torch.optim.LBFGS(linear.parameters(), lr=0.1, max_iter=100)

        def closure():
            optimizer.zero_grad()
            p_a = torch.sigmoid(linear(a)).squeeze(-1)
            p_b = torch.sigmoid(linear(b)).squeeze(-1)
            consistency = (p_a - (1 - p_b)).pow(2).mean()   # p(true) + p(false) = 1
            confidence = torch.min(p_a, p_b).pow(2).mean()  # avoid 0.5 / 0.5
            l2 = 0.01 * sum(p.norm() ** 2 for p in linear.parameters())
            loss = consistency + confidence + l2
            loss.backward()
            return loss

        optimizer.step(closure)
        loss = closure().item()
        if best is None or loss < best[0]:
            best = (loss, linear)
        if loss < 0.2:
            break
    linear = best[1]
    return linear.weight.detach().numpy()[0].astype(np.float64), float(linear.bias.detach())


def fit_probe(method: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42) -> LinearProbe:
    """Train one probe on activations X (n, d), labels y (1 = true) and question ids."""
    X = X.astype(np.float64)
    y = np.asarray(y).astype(int)

    if method == "lr":
        w, b = _fit_lr(X, y)
        probe = LinearProbe(w, b)
    elif method == "lr-g":
        w, b = _fit_lr(center_by_group(X, groups), y)
        probe = LinearProbe(w, b, group_center=True)
    elif method == "dim":
        probe = LinearProbe(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))
    elif method == "lda":
        lda = LinearDiscriminantAnalysis().fit(X, y)   # SVD solver, stable when d > n
        probe = LinearProbe(lda.coef_[0], float(lda.intercept_[0]))
    elif method == "pca":
        probe = LinearProbe(_first_component(X - X.mean(axis=0)))
    elif method == "pca-g":
        # direction found on group-centred data, test data projected raw
        probe = LinearProbe(_first_component(center_by_group(X, groups)))
    elif method == "lat":
        # PCA on differences between random pairs of activations
        idx = np.random.default_rng(seed).permutation(len(X))
        idx = idx[: len(idx) // 2 * 2].reshape(2, -1)
        diffs = X[idx[0]] - X[idx[1]]
        center = diffs.mean(axis=0)
        probe = LinearProbe(_first_component(diffs - center), center=center)
    elif method == "ccs":
        w, b = _fit_ccs(X, y, groups, seed)
        probe = LinearProbe(w, b)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Unsupervised directions have an arbitrary sign: fix it on the TRAINING data only
    if method in UNSUPERVISED and roc_auc_score(y, probe.score(X, groups)) < 0.5:
        probe.sign = -1.0
    return probe