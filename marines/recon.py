"""
Sliding-window DMD reconstruction.

Exports:
  sliding_window_dmd_core   — base DMD pass: Gaussian windowing + elbow-based mode filtering;
                              returns eigenvalues, reconstructed windows, and |b_k| profiles
  sliding_window_dmd_thresh — extends core with adaptive slow/fast thresholding and
                              returns separate slow and fast reconstructions
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, pinv

from .dmd_utils import (
    embed_delay,
    filter_modes_by_b_elbow,
    stability_gate,
    elbow_index_from_chord,
)
from .features import slow_threshold_via_ordered_kmeans


def sliding_window_dmd_core(
    x: np.ndarray,
    dt: float,
    d: int,
    win_len: int,
    win_step: int,
    re_cap: float = 0.2,
    abs_lambda_cap: float = 1.05,
    use_gaussian: bool = True,
    gauss_factor: float = 8.0,
    gap_mask: np.ndarray | None = None,
):
    """
    Core sliding-window DMD pass shared by reconstruction and clustering.

    Returns a dict with:
      - t: global time array
      - windows: list of dicts, each containing
          'g0', 'L', 'omega', 'V', 'amps', 'mid', 'gauss'
      - eig_all: np.ndarray of all omegas across windows
      - time_all: np.ndarray of corresponding window-center times
      - sorted_b_profiles: list of sorted |b_k| profiles per window
    """
    N = len(x)
    t = np.arange(N) * dt
    if use_gaussian:
        gauss_w = (win_len * dt) / gauss_factor
    else:
        gauss_w = None

    windows = []
    eig_all: list[np.ndarray] = []
    time_all: list[float] = []
    sorted_b_profiles: list[np.ndarray] = []

    def window_gaussian(idx, mid):
        return np.exp(-((idx * dt - mid) ** 2) / gauss_w**2)

    for start in range(0, N - win_len + 1, win_step):
        if gap_mask is not None and np.any(gap_mask[start + 1 : start + win_len]):
            continue
        H = embed_delay(x[start : start + win_len], d)
        X, Y = H[:, :-1], H[:, 1:]
        A = Y @ pinv(X)
        eigv, V = eig(A)

        mask = stability_gate(eigv, V, dt, re_cap=re_cap, abs_lambda_cap=abs_lambda_cap)
        eigv, V = eigv[mask], V[:, mask]
        if V.size == 0:
            continue

        V_keep, eig_keep, b_keep = filter_modes_by_b_elbow(V, eigv, X)
        if V_keep.size == 0:
            continue

        omega = np.log(eig_keep) / dt
        L = X.shape[1]
        mid = (start + (L - 1) / 2) * dt
        idx = start + np.arange(L)
        if use_gaussian and gauss_w is not None:
            gauss = window_gaussian(idx, mid)
        else:
            gauss = np.ones_like(idx, dtype=float)

        windows.append(
            dict(
                g0=start,
                L=L,
                omega=omega,
                V=V_keep,
                amps=b_keep,
                mid=mid,
                gauss=gauss,
            )
        )
        eig_all.extend(omega)
        time_all.extend([mid] * len(omega))
        sorted_b_profiles.append(np.abs(np.sort(b_keep)[::-1]))

    return dict(
        t=t,
        windows=windows,
        eig_all=np.array(eig_all),
        time_all=np.array(time_all),
        sorted_b_profiles=sorted_b_profiles,
    )


def sliding_window_dmd_thresh(
    x: np.ndarray,
    dt: float,
    d: int,
    win_len: int,
    win_step: int,
    base_thresh: float,
    k_slow: int = 10,
    clamp_range: tuple[float, float] = (0.5, 1.5),
    elbow_summary_path: Path | None = None,
    re_cap: float = 0.2,
    abs_lambda_cap: float = 1.05,
    gauss_factor: float = 8.0,
    gap_mask: np.ndarray | None = None,
):
    """
    Sliding-window DMD with deterministic slow/fast thresholding and reconstruction.
    """
    core = sliding_window_dmd_core(
        x,
        dt,
        d,
        win_len,
        win_step,
        re_cap=re_cap,
        abs_lambda_cap=abs_lambda_cap,
        use_gaussian=True,
        gauss_factor=gauss_factor,
        gap_mask=gap_mask,
    )

    t = core["t"]
    windows = core["windows"]
    eig_all_arr = core["eig_all"]
    time_all = core["time_all"]
    sorted_b_profiles = core["sorted_b_profiles"]

    if elbow_summary_path is not None and len(sorted_b_profiles) > 0:
        max_len = max(len(p) for p in sorted_b_profiles)
        padded = np.full((len(sorted_b_profiles), max_len), np.nan)
        for i, prof in enumerate(sorted_b_profiles):
            padded[i, : len(prof)] = prof
        avg_profile = np.nanmean(padded, axis=0)
        std_profile = np.nanstd(padded, axis=0)
        fig, ax = plt.subplots()
        k_axis = np.arange(1, len(avg_profile) + 1)
        ax.plot(k_axis, avg_profile, lw=2, label="Mean |b_k|")
        ax.fill_between(
            k_axis,
            avg_profile - std_profile,
            avg_profile + std_profile,
            alpha=0.3,
            label="±1 SD",
        )
        # Estimate global elbow index k from the averaged profile and show it on the plot.
        try:
            k_elbow = elbow_index_from_chord(avg_profile)
            ax.axvline(
                k_elbow,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Elbow k={k_elbow}",
            )
        except Exception:
            k_elbow = None

        ax.set_xlabel("sorted mode index k", fontweight="bold")
        ax.set_ylabel(r"$|b_k|$", fontweight="bold")
        ax.set_title("|b_k| magnitude profile across windows", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(elbow_summary_path, dpi=200)
        plt.close(fig)

    if eig_all_arr.size:
        thresh_info = slow_threshold_via_ordered_kmeans(
            eig_all_arr,
            base_thresh=base_thresh,
            k=k_slow,
            clamp_range=clamp_range,
        )
        slow_thresh = thresh_info["threshold"]
    else:
        slow_thresh = base_thresh

    N = len(x)
    slow = np.zeros(N, complex)
    fast = np.zeros(N, complex)
    wsum = np.zeros(N)
    for w in windows:
        idx = w["g0"] + np.arange(w["L"])
        wsum[idx] += w["gauss"]
        t_local = np.arange(w["L"]) * dt
        for j, omg in enumerate(w["omega"]):
            if np.real(omg) > 0:
                continue
            part = slow if np.abs(omg) < slow_thresh else fast
            part[idx] += (
                w["amps"][j] * w["V"][0, j] * np.exp(omg * t_local) * w["gauss"]
            )
    mask = wsum > 0
    slow[mask] /= wsum[mask]
    fast[mask] /= wsum[mask]
    return slow, fast, t, eig_all_arr, time_all, slow_thresh


def single_window_reconstruction(
    seg: np.ndarray,
    dt: float,
    d: int,
    base_thresh: float,
    k_slow: int = 1,
    clamp_range: tuple[float, float] | None = None,
    re_cap: float = 0.2,
    abs_lambda_cap: float = 1.05,
    decay_cap: float = 1.0,
    ridge: float = 1e-2,
    edge_crop: float = 0.03,
):
    """
    Global (single-window) Hankel-DMD reconstruction split into slow and fast parts.

    This is the reconstruction used for the manuscript reconstruction figure.  It
    uses exactly the same DMD eigenvalues, stability gate, and adaptive slow/fast
    threshold as the clustering pipeline, so the slow reconstruction is drawn from
    the same eigenvalue population that feeds the mean-eigenvalue clustering.

    Two changes relative to a naive reconstruction make it faithful instead of
    collapsing to zero (see the README methods notes):
      1. Amplitudes are fit by ridge-regularised least squares over the WHOLE
         window (not just the initial condition), so stable modes are not forced
         to decay across the window.
      2. A decay cap ``|lambda| <= decay_cap`` (default 1.0, i.e. Re(omega) <= 0)
         removes marginally growing modes that otherwise blow up at the window
         edges.  ``edge_crop`` additionally hides a small fraction of each edge.

    Returns a dict with keys:
      t_local, original, slow, fast, threshold, n_modes, n_slow, sl (display slice)
    or None if no stable modes survive the gate.
    """
    H = embed_delay(seg, d)
    X, Y = H[:, :-1], H[:, 1:]
    A = Y @ pinv(X)
    eigv, V = eig(A)

    mask = stability_gate(eigv, V, dt, re_cap=re_cap, abs_lambda_cap=abs_lambda_cap)
    eigv, V = eigv[mask], V[:, mask]
    # decay cap for reconstruction stability (edge blow-up control)
    keep = np.abs(eigv) <= decay_cap
    eigv, V = eigv[keep], V[:, keep]
    if eigv.size == 0:
        return None

    omega = np.log(eigv) / dt
    L = X.shape[1]
    t_local = np.arange(L) * dt
    signal = seg[:L].astype(complex)

    # time dynamics of the first Hankel row: T[t, j] = V[0, j] * exp(omega_j t)
    T = V[0, :][None, :] * np.exp(np.outer(t_local, omega))

    # ridge-regularised whole-window least squares for the amplitudes
    lam = ridge * np.linalg.norm(T)
    T_aug = np.vstack([T, lam * np.eye(T.shape[1])])
    s_aug = np.concatenate([signal, np.zeros(T.shape[1], dtype=complex)])
    b, *_ = np.linalg.lstsq(T_aug, s_aug, rcond=None)

    thresh_info = slow_threshold_via_ordered_kmeans(
        omega, base_thresh=base_thresh, k=k_slow, clamp_range=clamp_range
    )
    thr = thresh_info["threshold"]

    slow = np.zeros(L)
    fast = np.zeros(L)
    for j, omg in enumerate(omega):
        contrib = (b[j] * T[:, j]).real
        if np.abs(omg) < thr:
            slow += contrib
        else:
            fast += contrib

    crop = int(edge_crop * L)
    sl = slice(crop, L - crop) if crop > 0 else slice(0, L)
    return dict(
        t_local=t_local,
        original=seg[:L].real,
        slow=slow,
        fast=fast,
        threshold=thr,
        n_modes=len(omega),
        n_slow=int(np.sum(np.abs(omega) < thr)),
        sl=sl,
    )
