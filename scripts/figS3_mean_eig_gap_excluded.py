#!/usr/bin/env python3
"""
Figure S3 — sensitivity analysis excluding data gaps (supplement).

Two DBSCAN mean-eigenvalue panels, computed with windows that span a data gap
longer than one window (1.5 days) dropped: full recording (left) and first
80 days (right).

Reads:  figures/features_full_gapskip.csv, figures/features_80d_gapskip.csv
        (run compute_features.py first)
Output: figures/figS3_mean_eig_gap_excluded.png
Run:    python scripts/figS3_mean_eig_gap_excluded.py
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
    figs = project_root() / cfg["paths"]["figures_dir"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (csv_name, title) in zip(
        axes,
        [("features_full_gapskip.csv", "Full recording (gaps excluded)"),
         ("features_80d_gapskip.csv", "First 80 days (gaps excluded)")],
    ):
        ids, X = read_features_csv(figs / csv_name)
        actual = np.array([fid in dropout_ids for fid in ids], dtype=bool)
        pred = dbscan_predicted_dropout(dbscan_labels(X, eps=eps, min_samples=min_samples))
        draw_mean_eig(ax, ids, X, pred, dropout_ids,
                      f"Cluster (DBSCAN, eps={eps}, MinPts={min_samples})",
                      annot_size=10)
        ax.text(0.01, 0.98, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=11)
        print(f"[{csv_name}] predicted dropouts={int(pred.sum())} "
              f"(correct={int((pred & actual).sum())}/{int(actual.sum())})")

    fig.tight_layout()
    out = ensure_output_dirs(cfg) / "figS3_mean_eig_gap_excluded.png"
    fig.savefig(out, dpi=plot_style.SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
