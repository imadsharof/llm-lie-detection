"""Pick the best layer for a probe with k-fold cross-validation on the training set."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from liedetect.whitebox.probes import fit_probe


def cv_layer_scores(get_layer, layers, method, y, groups, n_splits=4, grouped=True, seed=42) -> dict:
    """Mean validation AUC per layer. get_layer(l) returns the (n, d) training matrix.

    grouped=True keeps all answers of one question in the same fold
    (G. Vasile used a plain StratifiedKFold, i.e. grouped=False).
    """
    if grouped:
        folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(y, y, groups))
    else:
        folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(y, y))

    scores = {}
    for layer in layers:
        X = get_layer(layer)
        aucs = []
        for train_idx, val_idx in folds:
            try:
                probe = fit_probe(method, X[train_idx], y[train_idx], groups[train_idx], seed)
                aucs.append(roc_auc_score(y[val_idx], probe.score(X[val_idx], groups[val_idx])))
            except ValueError:
                continue
        scores[layer] = float(np.mean(aucs)) if aucs else np.nan
    return scores