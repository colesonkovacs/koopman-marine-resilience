"""
Data loading and per-subject slow-eigenvalue feature extraction.

load_hr(fid, cfg)                    — zero-mean HR series
load_hr_time(fid, cfg, max_days)     — zero-mean HR + Time column, optional day cap
compute_subject_slow_stats(...)      — SW-HDMD over one subject -> slow-eig stats,
                                       with optional 80-day cap and gap-exclusion
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import compute_gap_mask
from .recon import sliding_window_dmd_thresh
from .features import slow_stats_from_omegas, slow_threshold_via_ordered_kmeans


def _read(fid: int, cfg: dict) -> pd.DataFrame:
    root = Path(cfg["paths"]["data_root"])
    df = pd.read_csv(root / cfg["paths"]["hr_template"].format(id=fid),
                     usecols=[0, 1], names=["HR", "Time"], header=0)
    df["HR"] = pd.to_numeric(df["HR"], errors="coerce")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    return df.dropna()


def load_hr(fid: int, cfg: dict) -> np.ndarray:
    hr = _read(fid, cfg)["HR"].values.astype(float)
    return hr - hr.mean()


def load_hr_time(fid: int, cfg: dict, max_days: float | None = None):
    df = _read(fid, cfg)
    if max_days is not None:
        t0 = df["Time"].iloc[0]
        df = df[df["Time"] <= t0 + max_days]
    hr = df["HR"].values.astype(float)
    return hr - hr.mean(), df["Time"].values.astype(float)


def compute_subject_slow_stats(fid: int, cfg: dict,
                               max_days: float | None = None,
                               skip_gaps: bool = False,
                               params: tuple | None = None) -> tuple[int, dict]:
    """
    Run SW-HDMD on one subject and return slow-eigenvalue statistics.

    max_days   — if set, only the first `max_days` days are used (Figure 4 / S3-right).
    skip_gaps  — if True, windows that span a data gap longer than one window are
                 dropped (supplementary Figure S3); the main analysis leaves this
                 False (gaps ignored).
    params     — optional (window_length, window_step, embed_dim, slow_thresh) that
                 override the config values (used by the Table 3 / S2 sweep). Note
                 the gap mask uses the (possibly overridden) window length.
    """
    dcfg = cfg["dmd"]
    if params is not None:
        window_length, window_step, embed_dim, slow_thresh = (
            int(params[0]), int(params[1]), int(params[2]), float(params[3]))
    else:
        window_length, window_step, embed_dim, slow_thresh = (
            dcfg["window_length"], dcfg["window_step"], dcfg["embed_dim"],
            dcfg["slow_thresh"])

    hr, time_days = load_hr_time(fid, cfg, max_days=max_days)
    clamp = dcfg.get("slow_thresh_clamp")
    clamp = tuple(clamp) if clamp else None
    gap_mask = compute_gap_mask(time_days, window_length, dcfg["dt"]) \
        if skip_gaps else None

    _, _, _, omegas, _, _ = sliding_window_dmd_thresh(
        hr, dcfg["dt"], embed_dim, window_length, window_step,
        slow_thresh, k_slow=int(dcfg.get("slow_cluster_count", 1)),
        clamp_range=clamp, re_cap=dcfg.get("re_cap", 0.2),
        abs_lambda_cap=dcfg.get("abs_lambda_cap", 1.05),
        gauss_factor=dcfg.get("gauss_factor", 8), gap_mask=gap_mask,
    )
    thr = slow_threshold_via_ordered_kmeans(
        omegas, base_thresh=slow_thresh,
        k=int(dcfg.get("slow_cluster_count", 1)), clamp_range=clamp)["threshold"]
    return fid, slow_stats_from_omegas(omegas, thr)
