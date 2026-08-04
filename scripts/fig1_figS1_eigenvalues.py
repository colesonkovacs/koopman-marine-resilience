#!/usr/bin/env python3
"""
Continuous Koopman eigenvalues for one subject, coloured slow vs fast.

Collects every stability-gated eigenvalue omega = log(lambda)/dt across all
sliding windows and splits them by the adaptive |omega| threshold used in the
clustering analysis.  Two views are written:

  * Figure 1  (fig1_eigenvalues_zoom.png) — cropped near the origin, where the
    slow/fast boundary is visible.
  * Figure S1 (figS1_eigenvalues_full.png) — auto-scaled to show ALL eigenvalues,
    including the far fast modes (the "zoomed-out" companion figure).

Output: figures/fig1_eigenvalues_zoom.png, figures/figS1_eigenvalues_full.png
Run:    python scripts/fig1_figS1_eigenvalues.py
        python scripts/fig1_figS1_eigenvalues.py --file 22
"""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, pinv

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from marines import load_config, ensure_output_dirs
from marines.dataio import load_hr
from marines.dmd_utils import embed_delay, stability_gate
from marines.features import slow_threshold_via_ordered_kmeans
import plot_style

# Manuscript Figure 1 crop (continuous eigenvalues near the origin)
ZOOM_RE = (-0.0018, 0.0018)
ZOOM_IM = (-0.0008, 0.0008)


def collect_eigenvalues(hr, dcfg):
    dt, d = dcfg["dt"], dcfg["embed_dim"]
    win_len, win_step = dcfg["window_length"], dcfg["window_step"]
    re_cap = dcfg.get("re_cap", 0.2)
    abs_cap = dcfg.get("abs_lambda_cap", 1.05)
    N = len(hr)
    omegas = []
    for start in range(0, N - win_len + 1, win_step):
        H = embed_delay(hr[start : start + win_len], d)
        X, Y = H[:, :-1], H[:, 1:]
        A = Y @ pinv(X)
        ev, V = eig(A)
        mask = stability_gate(ev, V, dt, re_cap=re_cap, abs_lambda_cap=abs_cap)
        ev = ev[mask]
        if ev.size:
            omegas.extend(np.log(ev) / dt)
    return np.array(omegas)


def _scatter(ax, slow, fast):
    ax.axhline(0, color="k", lw=plot_style.AXIS_LINE_LW)
    ax.axvline(0, color="k", lw=plot_style.AXIS_LINE_LW)
    ax.scatter(fast.real, fast.imag, marker="^", facecolors="none",
               edgecolors=plot_style.FAST_COLOR, s=18, lw=0.6, label="Fast", zorder=2)
    ax.scatter(slow.real, slow.imag, marker="o", facecolors="none",
               edgecolors=plot_style.SLOW_COLOR, s=18, lw=0.6, label="Slow", zorder=3)
    ax.set_xlabel(r"Re($\omega$)")
    ax.set_ylabel(r"Im($\omega$)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=plot_style.GRID_ALPHA)


def main() -> None:
    plot_style.apply_style()
    cfg = load_config()
    dcfg = cfg["dmd"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=int,
                        default=int(cfg["paths"].get("eig_subject_id", 22)))
    args = parser.parse_args()

    hr = load_hr(args.file, cfg)
    omegas = collect_eigenvalues(hr, dcfg)

    clamp = dcfg.get("slow_thresh_clamp")
    thr = slow_threshold_via_ordered_kmeans(
        omegas, base_thresh=dcfg["slow_thresh"],
        k=int(dcfg.get("slow_cluster_count", 1)),
        clamp_range=tuple(clamp) if clamp else None,
    )["threshold"]

    is_slow = np.abs(omegas) < thr
    slow, fast = omegas[is_slow], omegas[~is_slow]
    print(f"Marine {args.file}: {omegas.size} eigenvalues "
          f"({slow.size} slow, {fast.size} fast; threshold={thr:.5g})")

    figures_dir = ensure_output_dirs(cfg)

    # Zoomed (Figure 1 style)
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter(ax, slow, fast)
    ax.set_xlim(*ZOOM_RE)
    ax.set_ylim(*ZOOM_IM)
    fig.tight_layout()
    out_zoom = figures_dir / "fig1_eigenvalues_zoom.png"
    fig.savefig(out_zoom, dpi=plot_style.SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # Zoomed-out (all eigenvalues)
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter(ax, slow, fast)
    fig.tight_layout()
    out_full = figures_dir / "figS1_eigenvalues_full.png"
    fig.savefig(out_full, dpi=plot_style.SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_zoom}")
    print(f"Saved: {out_full}")


if __name__ == "__main__":
    main()
