"""
Package initialisation for the self-contained Koopman / SW-HDMD marines toolbox.

This copy is designed to run from inside the ``psych_methods_code`` folder that
accompanies the manuscript.  ``project_root()`` therefore points at that folder
(the parent of this ``marines`` package), and configuration / output paths are
resolved relative to it.

Note: this reproduction analyses the heart-rate series with data gaps IGNORED
(uniform-timestep assumption), which is a reported limitation of the method.

Exports:
  project_root()       — absolute path to the psych_methods_code folder
  load_config(path)    — load config/default.yaml (or a custom path) as a dict
  ensure_output_dirs() — create the figures output directory if missing
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, MutableMapping
import yaml

__all__ = ["project_root", "load_config", "ensure_output_dirs", "compute_gap_mask"]


def project_root() -> Path:
    # marines/__init__.py -> parents[1] is the repository root folder
    return Path(__file__).resolve().parents[1]


def compute_gap_mask(time_days, win_len: int, dt: float):
    """
    Boolean array of length N; True at index i means the jump from sample i-1 to i
    exceeds one window length in time.  Used ONLY by the supplementary gap-excluded
    analysis (Figure S3): windows whose sample range contains any True entry are
    skipped.  The main analysis ignores gaps and does not call this.
    """
    import numpy as np

    threshold_days = win_len * dt / 86400.0
    time_days = np.asarray(time_days, dtype=float)
    mask = np.zeros(len(time_days), dtype=bool)
    mask[1:] = np.diff(time_days) > threshold_days
    return mask


def load_config(path: str | Path | None = None) -> MutableMapping[str, Any]:
    cfg_path = Path(path) if path else project_root() / "config" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_output_dirs(cfg: MutableMapping[str, Any]) -> Path:
    """Create and return the figures output directory (defaults to ./figures)."""
    root = project_root()
    figures = root / cfg.get("paths", {}).get("figures_dir", "figures")
    figures.mkdir(parents=True, exist_ok=True)
    return figures
