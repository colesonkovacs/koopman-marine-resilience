"""
Shared drawing helper for the mean slow-eigenvalue cluster figures
(Figures 3, 4, S2, S3).  Each point is one marine at (mean Re(omega), mean Im(omega));
markers show ground truth (x = dropout, o = completed) and the shaded blob shows
the predicted class (grey = predicted dropout, purple = predicted completed).
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

CLUSTER_COLOR = "mediumpurple"
NOISE_COLOR = "gray"


def draw_mean_eig(ax, ids, X, pred_dropout, dropout_ids, cluster_label,
                  scatter_size=120, annot_size=13, grid_alpha=0.3):
    """Draw one mean-eigenvalue cluster panel on `ax`."""
    xr, xi = X[:, 0], X[:, 1]
    colors = [NOISE_COLOR if p else CLUSTER_COLOR for p in pred_dropout]
    for x, y, c in zip(xr, xi, colors):
        ax.scatter(x, y, s=scatter_size * 12, color=c, alpha=0.25, linewidths=0, zorder=1)
    for x, y, fid in zip(xr, xi, ids):
        m = "x" if fid in dropout_ids else "o"
        ax.scatter(x, y, marker=m, color="k", s=scatter_size, zorder=2)
        ax.text(x, y, f" {fid}", fontsize=annot_size, zorder=3)

    legend = [
        Line2D([0], [0], marker="x", color="k", label="Dropout (ground truth)",
               linestyle="none", markersize=12),
        Line2D([0], [0], marker="o", color="k", label="Completed (ground truth)",
               linestyle="none", markersize=12),
        Line2D([0], [0], marker="o", color="w", linestyle="none", label=cluster_label,
               markerfacecolor=CLUSTER_COLOR, markeredgecolor="none", markersize=14),
        Line2D([0], [0], marker="o", color="w", linestyle="none", label="Predicted dropout",
               markerfacecolor=NOISE_COLOR, markeredgecolor="none", markersize=14),
    ]
    ax.set_xlabel(r"Mean (Real($\omega$))")
    ax.set_ylabel(r"Mean (Imag($\omega$))")
    ax.legend(handles=legend, loc="best")
    ax.grid(True, alpha=grid_alpha)
