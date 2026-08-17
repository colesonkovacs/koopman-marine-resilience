#!/usr/bin/env python3
"""
Heart Rate Time-Delay DMD with Koopman Eigenvalue Decomposition — real subject.

Adapts the standalone heart_rate_tdmd_koopman.py demo (which ran on synthetic
data) to a real marine's heart-rate series from this repo's dataset: loads one
subject the same way every other script does (marines.dataio.load_hr, driven
by config/default.yaml), takes a single window of length dmd.window_length
starting at --start (default: the beginning of the recording), and runs one
global Time-Delay (Hankel) DMD reconstruction on it via PyDMD — exactly the
algorithm in the reference script: DMD(svd_rank=0) + hankel_preprocessing(d),
slow/fast mode split by a median-based threshold on |omega|, five-panel plot.

This is a standalone diagnostic, NOT one of the manuscript figures (see the
README's "What produces what" table) — it does not use this repo's own
SW-HDMD adaptive threshold or ridge-regularised reconstruction
(marines/recon.py). It is not part of run_all.py.

Requires: pydmd (not in requirements.txt — install separately if missing:
          pip install pydmd)

Run:    python scripts/tdmd_koopman_reconstruction.py --file 22
        python scripts/tdmd_koopman_reconstruction.py --file 22 --start 900
Output: figures/tdmd_koopman_reconstruction_marine<ID>.png
"""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from marines import load_config, ensure_output_dirs
from marines.dataio import load_hr

try:
    from pydmd import DMD
    from pydmd.preprocessing.hankel import hankel_preprocessing
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This script requires pydmd, which is not in requirements.txt. "
        "Install it with: pip install pydmd"
    ) from e


def load_segment(fid: int, cfg: dict, start: int, n_steps: int) -> np.ndarray:
    """Zero-mean HR series for one subject, sliced to [start, start + n_steps)."""
    hr = load_hr(fid, cfg)
    if hr.size == 0:
        raise ValueError(f"Marine {fid}: HR series is empty after loading/cleaning.")
    if start < 0:
        raise ValueError(f"--start must be >= 0, got {start}")
    if start + n_steps > hr.size:
        raise ValueError(
            f"Marine {fid}: requested samples [{start}:{start + n_steps}) but the "
            f"recording only has {hr.size} samples after cleaning. Reduce --start "
            f"or dmd.window_length in config/default.yaml."
        )
    return hr[start : start + n_steps]


def fit_delay_dmd(heart_rate: np.ndarray, delays: int):
    """Fit a single global Time-Delay DMD (PyDMD) over the whole segment."""
    n_steps = heart_rate.size
    if delays < 1:
        raise ValueError(f"delays (embedding dimension) must be >= 1, got {delays}")
    if n_steps - delays < 2:
        raise ValueError(
            f"Segment too short for delays={delays}: need at least {delays + 2} "
            f"samples, got {n_steps}."
        )

    dmd = DMD(svd_rank=0)
    delay_dmd = hankel_preprocessing(dmd, d=delays)
    try:
        delay_dmd.fit(heart_rate.reshape(1, -1))
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            "DMD fit failed (degenerate segment — e.g. a near-constant HR "
            "signal produces a singular SVD). Try a different --start."
        ) from e

    if dmd.eigs is None or dmd.eigs.size == 0:
        raise RuntimeError("DMD fit produced no eigenvalues — check the input segment.")
    if np.any(dmd.eigs == 0):
        raise RuntimeError("DMD produced a zero eigenvalue — log(eig)/dt is undefined.")
    return dmd


def reconstruct_from_modes(dmd, mask: np.ndarray) -> np.ndarray:
    """Reconstructed time series using only the modes selected by `mask`."""
    H_rec = dmd.modes * mask @ dmd.dynamics
    return np.real(H_rec)[0, :]


def main() -> None:
    cfg = load_config()
    dcfg = cfg["dmd"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=int,
                        default=int(cfg["paths"].get("eig_subject_id", 22)),
                        help="marine ID to analyze")
    parser.add_argument("--start", type=int, default=0,
                        help="starting sample index of the analysis window")
    args = parser.parse_args()

    dt = dcfg["dt"]
    delays = dcfg["embed_dim"]
    n_steps = dcfg["window_length"]

    heart_rate = load_segment(args.file, cfg, args.start, n_steps)
    time = np.arange(n_steps) * dt

    print("=" * 60)
    print(f"Heart Rate Data Summary — Marine {args.file} "
          f"(samples [{args.start}:{args.start + n_steps}))")
    print("=" * 60)
    print(f"  Time steps   : {n_steps}")
    print(f"  Interval (s) : {dt}")
    print(f"  Total time   : {time[-1] / 3600:.2f} hours")
    print(f"  HR range     : {heart_rate.min():.1f} to {heart_rate.max():.1f} bpm "
          f"(zero-mean)")
    print(f"  HR std       : {heart_rate.std():.1f} bpm")

    dmd = fit_delay_dmd(heart_rate, delays)
    print(f"\nDMD fitted successfully.")
    print(f"  Number of DMD modes : {dmd.eigs.shape[0]}")

    eigs_discrete = dmd.eigs
    eigs_continuous = np.log(eigs_discrete) / dt
    omega = np.abs(eigs_continuous)

    print("\nKoopman Eigenvalue Summary (continuous):")
    print(f"  |omega_c| range: {omega.min():.6f} to {omega.max():.6f} rad/s")

    threshold = np.median(omega) * 0.5
    print(f"  Threshold   : {threshold:.6f} rad/s (median * 0.5)")

    slow_mask = omega <= threshold
    fast_mask = omega > threshold
    print(f"  Slow modes  : {slow_mask.sum()}")
    print(f"  Fast modes  : {fast_mask.sum()}")

    T_hankel = n_steps - delays + 1
    rec_slow = reconstruct_from_modes(dmd, slow_mask)
    rec_fast = reconstruct_from_modes(dmd, fast_mask)
    rec_all = reconstruct_from_modes(dmd, np.ones(len(slow_mask), dtype=bool))

    for name, rec in (("all", rec_all), ("slow", rec_slow), ("fast", rec_fast)):
        if not np.all(np.isfinite(rec)):
            print(f"  [warning] {name}-mode reconstruction contains non-finite "
                  f"values (unstable growing mode over this window's duration).")

    time_rec = time[:T_hankel]
    hr_orig_trimmed = heart_rate[:T_hankel]

    time_hr = time / 3600
    time_rec_hr = time_rec / 3600

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Heart Rate Time-Delay DMD - Koopman Mode Decomposition "
                 f"(Marine {args.file})",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_hr, heart_rate, color="steelblue", lw=1.5, label="Original HR")
    ax1.set_title(f"Marine {args.file} — Heart Rate (zero-mean)", fontsize=12)
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Heart Rate (bpm, deviation from subject mean)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    theta = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.5, label="Unit circle")
    ax2.scatter(eigs_discrete[slow_mask].real, eigs_discrete[slow_mask].imag,
                c="royalblue", s=60, zorder=5, label=f"Slow ({slow_mask.sum()})")
    ax2.scatter(eigs_discrete[fast_mask].real, eigs_discrete[fast_mask].imag,
                c="tomato", s=60, zorder=5, label=f"Fast ({fast_mask.sum()})")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_title("Koopman Eigenvalues (Discrete)", fontsize=11)
    ax2.set_xlabel("Real part")
    ax2.set_ylabel("Imaginary part")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    colors_bar = ["royalblue" if s else "tomato" for s in slow_mask]
    ax3.bar(range(len(omega)), omega, color=colors_bar, edgecolor="k", linewidth=0.4)
    ax3.axhline(threshold, color="black", lw=1.5, linestyle="--",
                label=f"Threshold = {threshold:.5f} rad/s")
    ax3.set_title("|omega_continuous| - Slow vs Fast Mode Separation", fontsize=11)
    ax3.set_xlabel("Mode index")
    ax3.set_ylabel("|omega_c| (rad/s)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time_rec_hr, hr_orig_trimmed, "steelblue", lw=1.5, label="Original")
    ax4.plot(time_rec_hr, rec_all, "darkorange", lw=1.5, ls="--", label="All modes")
    ax4.set_title("Reconstruction - All Modes", fontsize=11)
    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Heart Rate (bpm)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time_rec_hr, hr_orig_trimmed, "steelblue", lw=1, label="Original")
    ax5.plot(time_rec_hr, rec_slow, "tomato", lw=2, ls="--", label="Slow modes")
    ax5.plot(time_rec_hr, rec_fast, "darkgreen", lw=2, ls=":", label="Fast modes")
    ax5.set_title("Reconstruction - Slow vs Fast Modes", fontsize=11)
    ax5.set_xlabel("Time (hours)")
    ax5.set_ylabel("Heart Rate (bpm)")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    figures_dir = ensure_output_dirs(cfg)
    out_path = figures_dir / f"tdmd_koopman_reconstruction_marine{args.file}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
