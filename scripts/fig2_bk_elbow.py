#!/usr/bin/env python3
"""
Mode-contribution (|b_k|) elbow figure.

For each sliding window the DMD modes are ranked by amplitude |b_k| and truncated
at an elbow (perpendicular-distance-to-chord).  This figure shows the mean sorted
|b_k| profile across all windows (with a +/-1 SD band), marks the elbow index, and
shades the region of modes that are discarded, so a reader can see how much
relative contribution is dropped at the cut.

Output: figures/fig2_bk_elbow.png
Run:    python scripts/fig2_bk_elbow.py
        python scripts/plot_bk_elbow.py --file 22
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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from marines import load_config, ensure_output_dirs
from marines.dataio import load_hr
from marines.dmd_utils import (
    embed_delay,
    stability_gate,
    compute_modal_amplitudes,
    elbow_index_from_chord,
)
import plot_style


def sorted_b_profiles(hr, dcfg):
    """Return the pre-elbow sorted |b_k| profile for each window (largest first)."""
    dt, d = dcfg["dt"], dcfg["embed_dim"]
    win_len, win_step = dcfg["window_length"], dcfg["window_step"]
    re_cap = dcfg.get("re_cap", 0.2)
    abs_cap = dcfg.get("abs_lambda_cap", 1.05)
    from numpy.linalg import eig, pinv

    profiles = []
    N = len(hr)
    for start in range(0, N - win_len + 1, win_step):
        H = embed_delay(hr[start : start + win_len], d)
        X, Y = H[:, :-1], H[:, 1:]
        A = Y @ pinv(X)
        ev, V = eig(A)
        mask = stability_gate(ev, V, dt, re_cap=re_cap, abs_lambda_cap=abs_cap)
        ev, V = ev[mask], V[:, mask]
        if V.size == 0:
            continue
        b = compute_modal_amplitudes(V, X)
        profiles.append(np.sort(np.abs(b))[::-1])
    return profiles


def main() -> None:
    plot_style.apply_style()
    cfg = load_config()
    dcfg = cfg["dmd"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=int,
                        default=int(cfg["paths"].get("eig_subject_id", 22)))
    parser.add_argument("--max_k", type=int, default=60,
                        help="truncate the x-axis for readability")
    args = parser.parse_args()

    hr = load_hr(args.file, cfg)
    profiles = sorted_b_profiles(hr, dcfg)
    if not profiles:
        print("No windows produced modes.")
        return

    max_len = max(len(p) for p in profiles)
    padded = np.full((len(profiles), max_len), np.nan)
    for i, p in enumerate(profiles):
        padded[i, : len(p)] = p
    avg = np.nanmean(padded, axis=0)
    sd = np.nanstd(padded, axis=0)
    k_elbow = elbow_index_from_chord(avg)

    kmax = min(args.max_k, len(avg))
    k_axis = np.arange(1, kmax + 1)
    avg_c, sd_c = avg[:kmax], sd[:kmax]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(k_axis, avg_c, color="navy", lw=2, marker="o", ms=3, label="Mean $|b_k|$")
    ax.fill_between(k_axis, np.maximum(avg_c - sd_c, 0.0), avg_c + sd_c,
                    color="navy", alpha=0.2, label="$\\pm$1 SD across windows")
    if k_elbow <= kmax:
        ax.axvline(k_elbow, color="red", ls="--", lw=1.6, label=f"Elbow (k = {k_elbow})")
        ax.axvspan(k_elbow, kmax + 0.5, color="gray", alpha=0.15,
                   label="Discarded modes")
    ax.set_xlim(0.5, kmax + 0.5)
    ax.set_xlabel("Sorted mode index $k$")
    ax.set_ylabel(r"Mode amplitude $|b_k|$")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=plot_style.GRID_ALPHA)
    fig.tight_layout()

    figures_dir = ensure_output_dirs(cfg)
    out = figures_dir / "fig2_bk_elbow.png"
    fig.savefig(out, dpi=plot_style.SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Elbow at k={k_elbow} (mean over {len(profiles)} windows). Saved: {out}")


if __name__ == "__main__":
    main()
