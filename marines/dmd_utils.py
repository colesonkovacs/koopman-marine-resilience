"""
Low-level DMD linear-algebra utilities.

Exports:
  embed_delay               — build the Hankel time-delay embedding matrix
  stability_gate            — filter eigenvalues by real-part cap, magnitude cap, and residual
  compute_modal_amplitudes  — least-squares modal amplitude estimation
  elbow_index_from_chord    — detect the elbow in a magnitude curve (perpendicular-distance method)
  filter_modes_by_b_elbow   — sort modes by amplitude and keep up to the elbow index
"""
from __future__ import annotations
import numpy as np

def embed_delay(x: np.ndarray, d: int) -> np.ndarray:
    return np.vstack([x[i:len(x)-d+1+i] for i in range(d)])

def stability_gate(eigvals, V, dt,
                   re_cap: float | None = 0.0,
                   abs_lambda_cap: float | None = 1.05,
                   resid_tol: float = 1e-6):
    omega = np.log(eigvals) / dt
    phi_norms = np.linalg.norm(V, axis=0) + 1e-12
    resids = np.linalg.norm((V * eigvals) - (V @ np.diag(eigvals)), axis=0) / phi_norms
    m1 = np.ones_like(eigvals, dtype=bool) if re_cap is None else (np.real(omega) <= re_cap)
    m2 = np.ones_like(eigvals, dtype=bool) if abs_lambda_cap is None else (np.abs(eigvals) <= abs_lambda_cap)
    m3 = (resids <= resid_tol)
    return m1 & m2 & m3

def compute_modal_amplitudes(Phi: np.ndarray, X: np.ndarray,
                             m_avg: int = 8, alpha: float = 1e-6) -> np.ndarray:
    m = min(m_avg, X.shape[1])
    Xm = X[:, :m]
    n_modes = Phi.shape[1]
    A_aug = np.vstack([Phi, np.sqrt(alpha) * np.eye(n_modes)])
    B_aug = np.vstack([Xm, np.zeros((n_modes, m), dtype=complex)])
    B, *_ = np.linalg.lstsq(A_aug, B_aug, rcond=None)
    return np.mean(B, axis=1)

def elbow_index_from_chord(mags: np.ndarray) -> int:
    if len(mags) <= 2:
        return len(mags)
    x = np.arange(len(mags), dtype=float)
    y = mags.astype(float)
    x = (x - x[0]) / (x[-1] - x[0] + 1e-12)
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    vx, vy = x1 - x0, y1 - y0
    denom = np.hypot(vx, vy) + 1e-12
    d = np.abs(vy * (x - x0) - vx * (y - y0)) / denom
    k_star = int(np.argmax(d))
    return max(1, k_star + 1)

def filter_modes_by_b_elbow(Phi: np.ndarray, eigvals: np.ndarray, X: np.ndarray):
    b = compute_modal_amplitudes(Phi, X)
    order = np.argsort(-np.abs(b))
    Phi_s, eig_s, b_s = Phi[:, order], eigvals[order], b[order]
    k = elbow_index_from_chord(np.abs(b_s))
    return Phi_s[:, :k], eig_s[:k], b_s[:k]
