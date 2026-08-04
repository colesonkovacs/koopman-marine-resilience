#!/usr/bin/env python3
"""
Confusion matrices and classification metrics for DBSCAN and KMeans
(reproduces Table 1 = confusion matrix and Table 2 = metrics, for both methods).

Positive class = dropout.  DBSCAN predicts dropouts as noise points; KMeans (k=2)
predicts dropouts using its most favourable cluster-to-dropout mapping.

Reads:  figures/features_full.csv (run compute_features.py first)
Writes: figures/confusion_matrices.csv, figures/classification_metrics.csv
Produces: Table 1 (DBSCAN confusion), Table 2 (DBSCAN metrics), Table S1 (KMeans metrics)
        and prints both tables to stdout.
Run:    python scripts/make_confusion_tables.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from marines import load_config, project_root, ensure_output_dirs
from marines.clustering import (
    dbscan_labels,
    kmeans2_labels,
    dbscan_predicted_dropout,
    kmeans_predicted_dropout,
    confusion_and_metrics,
)


def main() -> None:
    cfg = load_config()
    ccfg = cfg.get("clustering", {})
    dropout_ids = set(ccfg.get("dropout_ids", []))
    eps = float(ccfg.get("dbscan_eps", 0.75))
    min_samples = int(ccfg.get("dbscan_min_samples", 6))

    csv_path = project_root() / cfg["paths"]["figures_dir"] / "features_full.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run scripts/compute_features.py first."
        )
    df = pd.read_csv(csv_path)
    ids = df["marine_id"].astype(int).to_numpy()
    X = df[["mean_real", "mean_imag"]].to_numpy(dtype=float)
    actual = np.array([fid in dropout_ids for fid in ids], dtype=bool)

    results = {
        "DBSCAN": dbscan_predicted_dropout(
            dbscan_labels(X, eps=eps, min_samples=min_samples)
        ),
        "KMeans": kmeans_predicted_dropout(kmeans2_labels(X), actual),
    }

    figures_dir = ensure_output_dirs(cfg)
    conf_rows, metric_rows = [], []

    for method, pred in results.items():
        m = confusion_and_metrics(pred, actual)
        print(f"\n=== {method} ===")
        print("Confusion matrix (positive = dropout):")
        print(f"                 Predicted Dropout   Predicted Completed")
        print(f"  Actual Dropout        {m['tp']:>3d}                 {m['fn']:>3d}")
        print(f"  Actual Completed      {m['fp']:>3d}                 {m['tn']:>3d}")
        print(f"Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  Accuracy={m['accuracy']:.4f}")

        conf_rows.append([method, "Actual Dropout", m["tp"], m["fn"]])
        conf_rows.append([method, "Actual Completed", m["fp"], m["tn"]])
        metric_rows.append([method, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                            f"{m['f1']:.4f}", f"{m['accuracy']:.4f}"])

    with (figures_dir / "confusion_matrices.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "row", "Predicted Dropout", "Predicted Completed"])
        w.writerows(conf_rows)
    with (figures_dir / "classification_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "precision", "recall", "f1", "accuracy"])
        w.writerows(metric_rows)

    print(f"\nSaved: {figures_dir / 'confusion_matrices.csv'}")
    print(f"Saved: {figures_dir / 'classification_metrics.csv'}")


if __name__ == "__main__":
    main()
