#!/usr/bin/env python3
"""
Figure 3  — mean slow eigenvalues with DBSCAN clustering (main text).
Figure S2 — mean slow eigenvalues with KMeans clustering (supplement).

One point per marine at (mean Re(omega), mean Im(omega)) from features_full.csv.
DBSCAN noise points are the predicted dropouts; KMeans (k=2) uses its most
favourable cluster-to-dropout mapping.

Reads:  figures/features_full.csv   (run compute_features.py first)
Output: figures/fig3_mean_eig_dbscan.png, figures/figS2_mean_eig_kmeans.png
Run:    python scripts/fig3_figS2_mean_eig.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from marines import load_config, project_root, ensure_output_dirs
from marines.clustering import (
    read_features_csv, dbscan_labels, kmeans2_labels,
    dbscan_predicted_dropout, kmeans_predicted_dropout,
)
from marines.plotting import draw_mean_eig
import plot_style


def main() -> None:
    plot_style.apply_style()
    cfg = load_config()
    ccfg = cfg.get("clustering", {})
    dropout_ids = set(ccfg.get("dropout_ids", []))
    eps = float(ccfg.get("dbscan_eps", 0.75))
    min_samples = int(ccfg.get("dbscan_min_samples", 6))

    ids, X = read_features_csv(project_root() / cfg["paths"]["figures_dir"] / "features_full.csv")
    actual = np.array([fid in dropout_ids for fid in ids], dtype=bool)
    figures_dir = ensure_output_dirs(cfg)

    # Figure 3 — DBSCAN
    pred = dbscan_predicted_dropout(dbscan_labels(X, eps=eps, min_samples=min_samples))
    fig, ax = plt.subplots(figsize=plot_style.FIGURE_SIZE)
    draw_mean_eig(ax, ids, X, pred, dropout_ids,
                  f"Cluster (DBSCAN, eps={eps}, MinPts={min_samples})")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_mean_eig_dbscan.png", dpi=plot_style.SAVE_DPI,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 3 DBSCAN] predicted dropouts={int(pred.sum())} "
          f"(correct={int((pred & actual).sum())}/{int(actual.sum())})")

    # Figure S2 — KMeans (favourable mapping)
    pred = kmeans_predicted_dropout(kmeans2_labels(X), actual)
    fig, ax = plt.subplots(figsize=plot_style.FIGURE_SIZE)
    draw_mean_eig(ax, ids, X, pred, dropout_ids, "Cluster (KMeans, k=2)")
    fig.tight_layout()
    fig.savefig(figures_dir / "figS2_mean_eig_kmeans.png", dpi=plot_style.SAVE_DPI,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig S2 KMeans] predicted dropouts={int(pred.sum())} "
          f"(correct={int((pred & actual).sum())}/{int(actual.sum())})")
    print(f"Saved to {figures_dir}")


if __name__ == "__main__":
    main()
