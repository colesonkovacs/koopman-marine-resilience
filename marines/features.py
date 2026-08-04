"""
Feature extraction utilities for slow-mode eigenvalue statistics.

Exports:
  slow_threshold_via_ordered_kmeans — adaptively compute the slow/fast split threshold
                                      using deterministic 1D k-means on eigenvalue magnitudes
  slow_stats_from_omegas            — compute mean, std, and third moment of slow eigenvalues
"""
from __future__ import annotations
import numpy as np

def _seg_cost(prefix, prefix2, i, j):
    """Variance cost of segment [i, j) for 1D k-means (sorted data)."""
    n = j - i
    if n <= 0:
        return 0.0
    s = prefix[j] - prefix[i]
    s2 = prefix2[j] - prefix2[i]
    return float(s2 - (s * s) / n)

def _optimal_1d_kmeans(sorted_vals: np.ndarray, k: int):
    """
    Deterministic 1D k-means via DP (minimizes total within-cluster variance).
    Returns boundaries as (start, end) pairs on the sorted array.
    """
    n = len(sorted_vals)
    k = min(k, n) if n else 0
    if k <= 1:
        return [(0, n)] if n else []

    prefix = np.zeros(n + 1)
    prefix2 = np.zeros(n + 1)
    prefix[1:] = np.cumsum(sorted_vals)
    prefix2[1:] = np.cumsum(sorted_vals ** 2)

    dp = np.full((k + 1, n + 1), np.inf)
    prev = np.full((k + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for clusters in range(1, k + 1):
        for end in range(clusters, n + 1):
            best_cost = np.inf
            best_split = -1
            # end index is exclusive; try all possible previous splits
            for mid in range(clusters - 1, end):
                cost = dp[clusters - 1, mid] + _seg_cost(prefix, prefix2, mid, end)
                if cost < best_cost:
                    best_cost = cost
                    best_split = mid
            dp[clusters, end] = best_cost
            prev[clusters, end] = best_split

    # backtrack to build boundaries
    boundaries = []
    end = n
    for clusters in range(k, 0, -1):
        start = prev[clusters, end]
        boundaries.append((start, end))
        end = start
    boundaries.reverse()
    return boundaries

def slow_threshold_via_ordered_kmeans(
    omegas: np.ndarray,
    base_thresh: float,
    k: int = 10,
    clamp_range: tuple[float, float] | None = None,
) -> dict:
    """
    Compute a slow/fast cutoff from eigenvalue magnitudes using deterministic 1D k-means.
    - Clusters sorted |omega| into k groups (Ward-like stability in 1D).
    - Picks the leftmost cluster as "slow".
    - Returns the midpoint between slow cluster and its right neighbor as the cutoff.
    - Clamps the cutoff around the legacy base_thresh for cross-file consistency.
    """
    mags = np.array([abs(w) for w in omegas if w.imag >= 0], dtype=float)
    if mags.size == 0:
        return dict(
            threshold=base_thresh,
            centroids=[],
            boundaries=[],
            k_used=0,
            mags=mags,
        )

    mags_sorted = np.sort(mags)
    boundaries = _optimal_1d_kmeans(mags_sorted, k=k)
    centroids = []
    for start, end in boundaries:
        seg = mags_sorted[start:end]
        centroids.append(seg.mean() if seg.size else 0.0)

    k_used = len(boundaries)
    if k_used == 0:
        return dict(threshold=base_thresh, centroids=[], boundaries=[], k_used=0, mags=mags_sorted)

    slow_idx = 0  # smallest centroid because mags_sorted is ascending
    if k_used == 1:
        cutoff = centroids[0]
    else:
        slow_start, slow_end = boundaries[slow_idx]
        next_start, next_end = boundaries[slow_idx + 1]
        max_slow = mags_sorted[slow_end - 1]
        min_next = mags_sorted[next_start]
        cutoff = 0.5 * (max_slow + min_next)

    if clamp_range:
        lo, hi = clamp_range
        lo_bound = base_thresh * lo
        hi_bound = base_thresh * hi
        threshold = float(np.clip(cutoff, lo_bound, hi_bound))
    else:
        threshold = float(cutoff)

    return dict(
        threshold=threshold,
        centroids=centroids,
        boundaries=boundaries,
        k_used=k_used,
        mags=mags_sorted,
    )

def slow_stats_from_omegas(omegas: np.ndarray, slow_thresh: float) -> dict:
    slow = np.array([w for w in omegas if (w.imag >= 0 and np.abs(w) < slow_thresh)])
    if slow.size == 0:
        return dict(mean=0+0j, std=0.0, m3=0.0, mean_r=0.0, mean_i=0.0, std_r=0.0, std_i=0.0, m3_r=0.0, m3_i=0.0)
    mean = slow.mean()
    r, im = slow.real, slow.imag
    return dict(
        mean=mean,
        std=np.std(slow),
        m3=np.mean((slow - mean)**3),
        mean_r=mean.real, mean_i=mean.imag,
        std_r=np.std(r), std_i=np.std(im),
        m3_r=np.mean((r-mean.real)**3), m3_i=np.mean((im-mean.imag)**3)
    )
