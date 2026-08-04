"""
Clustering and classification helpers for the mean slow-eigenvalue analysis.

Both clustering methods operate on the 2-D mean slow-eigenvalue feature
(mean Re(omega), mean Im(omega)) after z-scoring.

Convention (matches the manuscript):
  - DBSCAN: noise points (label -1) are the predicted dropouts; any dense
    cluster is "completed".
  - KMeans (k=2): whichever cluster-to-dropout assignment best matches the
    ground truth (highest F1) is reported — the *favourable* mapping. This
    gives KMeans its best honest score; applied identically everywhere KMeans
    is used to predict dropouts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_features_csv(path: "str | Path"):
    """Read a features_*.csv written by compute_features.py -> (ids, X[mean_real,mean_imag])."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/compute_features.py first.")
    df = pd.read_csv(path)
    ids = df["marine_id"].astype(int).to_numpy()
    X = df[["mean_real", "mean_imag"]].to_numpy(dtype=float)
    return ids, X


def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def dbscan_labels(X: np.ndarray, eps: float = 0.75, min_samples: int = 6) -> np.ndarray:
    """DBSCAN labels on z-scored X (-1 = noise). Requires scikit-learn."""
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(standardize(X))


def kmeans2_labels(X: np.ndarray, random_state: int = 0) -> np.ndarray:
    """KMeans (k=2) labels on z-scored X (0/1). Requires scikit-learn."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(
        standardize(X)
    )


def dbscan_predicted_dropout(labels: np.ndarray) -> np.ndarray:
    """Boolean mask: True where DBSCAN predicts a dropout (noise point)."""
    return labels == -1


def kmeans_predicted_dropout(labels: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Boolean mask: True where KMeans predicts a dropout.

    KMeans returns two unlabelled clusters, so one must be mapped to "dropout".
    We use the *favourable* mapping: whichever cluster-to-dropout assignment
    best matches the ground truth (highest F1) is reported. This gives KMeans
    its best honest score, and is applied identically everywhere KMeans is
    used to predict dropouts.
    """
    labels = np.asarray(labels)
    actual = np.asarray(actual, dtype=bool)
    best_pred, best_f1 = None, -1.0
    for dropout_cluster in (0, 1):
        pred = labels == dropout_cluster
        tp = int(np.sum(pred & actual))
        fp = int(np.sum(pred & ~actual))
        fn = int(np.sum(~pred & actual))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_pred = f1, pred
    return best_pred


def confusion_and_metrics(pred_dropout: np.ndarray, actual_dropout: np.ndarray) -> dict:
    """
    Build the 2x2 confusion matrix and precision/recall/F1/accuracy.

    Positive class = dropout.
    Returns dict with tp, fp, fn, tn and precision, recall, f1, accuracy.
    """
    pred = np.asarray(pred_dropout, dtype=bool)
    actual = np.asarray(actual_dropout, dtype=bool)
    tp = int(np.sum(pred & actual))
    fp = int(np.sum(pred & ~actual))
    fn = int(np.sum(~pred & actual))
    tn = int(np.sum(~pred & ~actual))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1, accuracy=accuracy,
    )
