#!/usr/bin/env python3
"""
Figure 5 — basic HR-statistic clustering (main text).
Table 4  — DBSCAN and KMeans classification across the basic HR statistics.

Mirrors the eigenvalue cluster figure but uses simple HR statistics as features,
in two feature spaces:
  * Mean HR vs HR Std Dev
  * Successive-Difference RMS vs Coefficient of Variation
Both DBSCAN and KMeans are applied (full sample). The printed metrics are Table 4.

KMeans convention: the cluster containing the higher fraction of true dropouts is
taken as the predicted-dropout cluster (its most favourable mapping).

Output: figures/fig5_hr_scatter_<featurespace>_<method>.png  (4 panels)
        figures/table4_hr_stats_metrics.csv
Run:    python scripts/fig5_table4_hr_stats.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from marines import load_config, project_root, ensure_output_dirs
import plot_style

CLUSTER_COLOR = "mediumpurple"
NOISE_COLOR = "gray"


def _confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1)


def _rmssd(hr):
    return float(np.sqrt(np.mean(np.diff(hr) ** 2)))


def _labels(Xz, y_true, method, eps, min_samples):
    """Return (pred_dropout_bool, has_cluster) for the given method, or (None, ...) if DBSCAN empty."""
    if method == "dbscan":
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xz)
        if not any(l >= 0 for l in lab):
            return None, False
        return (lab == -1), True
    lab = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(Xz)
    rate = {k: y_true[lab == k].mean() if np.any(lab == k) else -1 for k in (0, 1)}
    drop = max(rate, key=rate.get)
    return (lab == drop), True


def _scatter(feat1, feat2, ids, dropout_ids, xlabel, ylabel, pred_dropout):
    fig, ax = plt.subplots(figsize=plot_style.FIGURE_SIZE)
    colors = [NOISE_COLOR if p else CLUSTER_COLOR for p in pred_dropout]
    for x, y, c in zip(feat1, feat2, colors):
        ax.scatter(x, y, s=plot_style.SCATTER_SIZE_MEAN * 12, color=c, alpha=0.25,
                   linewidths=0, zorder=1)
    for x, y, fid in zip(feat1, feat2, ids):
        m = "x" if fid in dropout_ids else "o"
        ax.scatter(x, y, marker=m, color="k", s=plot_style.SCATTER_SIZE_MEAN,
                   linewidths=1.5 if m == "x" else plot_style.SCATTER_LW, zorder=2)
        ax.text(x, y, f" {fid}", fontsize=plot_style.FONT_SIZE_ANNOTATION, zorder=3)
    legend = [
        mlines.Line2D([0], [0], marker="x", color="k", label="Dropout", linestyle="none",
                      markersize=12, markeredgewidth=1.5),
        mlines.Line2D([0], [0], marker="o", color="k", label="Completed", linestyle="none",
                      markersize=12),
        mlines.Line2D([0], [0], marker="o", color="w", label="Cluster 1", linestyle="none",
                      markerfacecolor=CLUSTER_COLOR, markeredgecolor="none", markersize=14),
        mlines.Line2D([0], [0], marker="o", color="w", label="Predicted dropout",
                      linestyle="none", markerfacecolor=NOISE_COLOR, markeredgecolor="none",
                      markersize=14),
    ]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(handles=legend, loc="best")
    ax.grid(True, alpha=plot_style.GRID_ALPHA)
    fig.tight_layout()
    return fig


def main() -> None:
    plot_style.apply_style()
    cfg = load_config()
    data_root = Path(cfg["paths"]["data_root"])
    hr_template = cfg["paths"]["hr_template"]
    ccfg = cfg.get("clustering", {})
    dropout_ids = set(ccfg.get("dropout_ids", []))
    excluded = set(ccfg.get("excluded_files", []))
    eps = float(ccfg.get("dbscan_eps", 0.75))
    min_samples = int(ccfg.get("dbscan_min_samples", 6))
    targets = [i for i in range(1, int(ccfg.get("subject_count", 23)) + 1)
               if i not in excluded]

    rows = []
    for fid in targets:
        p = data_root / hr_template.format(id=fid)
        if not p.exists():
            continue
        raw = pd.read_csv(p, usecols=[0, 1], names=["HR", "Time"], header=0)
        raw["HR"] = pd.to_numeric(raw["HR"], errors="coerce")
        raw = raw.dropna(subset=["HR"])
        hr = raw["HR"].values.astype(float)
        mean_hr, std_hr = float(np.mean(hr)), float(np.std(hr))
        rows.append(dict(id=fid, mean_hr=mean_hr, std_hr=std_hr, rmssd=_rmssd(hr),
                         cv=float(std_hr / mean_hr * 100) if mean_hr > 0 else 0.0))
    df = pd.DataFrame(rows).sort_values("id")
    ids = df["id"].tolist()
    y_true = np.array([1 if f in dropout_ids else 0 for f in ids])

    figures_dir = ensure_output_dirs(cfg)
    feature_spaces = [
        (df["mean_hr"].values, df["std_hr"].values, "Mean HR (bpm)", "HR Std Dev (bpm)",
         "mean_std", "Mean vs. Std Dev"),
        (df["rmssd"].values, df["cv"].values, "Successive-Diff RMS (bpm)",
         "Coefficient of Variation (%)", "rms_cv", "Succ-Diff vs. CV"),
    ]

    metric_rows = []
    for f1, f2, lbl1, lbl2, stem, name in feature_spaces:
        X = np.column_stack([f1, f2])
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
        for method in ("dbscan", "kmeans"):
            pred, has_cluster = _labels(Xz, y_true, method, eps, min_samples)
            if pred is None:
                print(f"[{name} / {method}] DBSCAN found no clusters — skipped")
                metric_rows.append([name, method.upper(), "", "", "", ""])
                continue
            m = _confusion(y_true, pred.astype(int))
            fig = _scatter(f1, f2, ids, dropout_ids, lbl1, lbl2, pred)
            out = figures_dir / f"fig5_hr_scatter_{stem}_{method}.png"
            fig.savefig(out, dpi=plot_style.SAVE_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"[{name} / {method}] acc={m['acc']:.3f} prec={m['prec']:.3f} "
                  f"rec={m['rec']:.3f} f1={m['f1']:.3f}  ->  {out.name}")
            metric_rows.append([name, method.upper(), f"{m['acc']:.3f}",
                                f"{m['prec']:.3f}", f"{m['rec']:.3f}", f"{m['f1']:.3f}"])

    with (figures_dir / "table4_hr_stats_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature_space", "method", "accuracy", "precision", "recall", "f1"])
        w.writerows(metric_rows)
    print(f"\nSaved Table 4 metrics: {figures_dir / 'table4_hr_stats_metrics.csv'}")


if __name__ == "__main__":
    main()
