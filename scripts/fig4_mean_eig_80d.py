#!/usr/bin/env python3
"""
Figure 4 — 80-day mean slow eigenvalues with DBSCAN clustering.

Same as Figure 3 but computed on only the first 80 days of each recording
(most dropouts leave around day 80).

Reads:  figures/features_80d.csv   (run compute_features.py first)
Output: figures/fig4_mean_eig_dbscan_80d.png
Run:    python scripts/fig4_mean_eig_80d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from marines import load_config, project_root, ensure_output_dirs
from marines.clustering import read_features_csv, dbscan_labels, dbscan_predicted_dropout
from marines.plotting import draw_mean_eig
import plot_style


def main() -> None:
    plot_style.apply_style()
    cfg = load_config()
    ccfg = cfg.get("clustering", {})
    dropout_ids = set(ccfg.get("dropout_ids", []))
    eps = float(ccfg.get("dbscan_eps", 0.75))
    min_samples = int(ccfg.get("dbscan_min_samples", 6))

    ids, X = read_features_csv(project_root() / cfg["paths"]["figures_dir"] / "features_80d.csv")
    actual = np.array([fid in dropout_ids for fid in ids], dtype=bool)
    figures_dir = ensure_output_dirs(cfg)

    pred = dbscan_predicted_dropout(dbscan_labels(X, eps=eps, min_samples=min_samples))
    fig, ax = plt.subplots(figsize=plot_style.FIGURE_SIZE)
    draw_mean_eig(ax, ids, X, pred, dropout_ids,
                  f"Cluster (DBSCAN, eps={eps}, MinPts={min_samples})")
    fig.tight_layout()
    out = figures_dir / "fig4_mean_eig_dbscan_80d.png"
    fig.savefig(out, dpi=plot_style.SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4 80-day DBSCAN] predicted dropouts={int(pred.sum())} "
          f"(correct={int((pred & actual).sum())}/{int(actual.sum())})  ->  {out}")


if __name__ == "__main__":
    main()
