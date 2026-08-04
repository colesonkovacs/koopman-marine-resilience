#!/usr/bin/env python3
"""
STEP 1 — compute per-subject slow-eigenvalue features (run this first).

Runs Sliding-window Hankel DMD on every included marine and writes the mean/std/
third-moment slow-eigenvalue statistics to CSV.  Four configurations are produced
because the manuscript figures/tables draw on different time-frames and gap
handling:

  features_full.csv          full recording, gaps ignored   -> Fig 3, Fig S2, Tables 1/2/S1
  features_80d.csv           first 80 days, gaps ignored     -> Fig 4
  features_full_gapskip.csv  full recording, gaps excluded   -> Fig S3 (left)
  features_80d_gapskip.csv   first 80 days, gaps excluded     -> Fig S3 (right)

Output: figures/<name>.csv
Run:    python scripts/compute_features.py            # all four configs
        python scripts/compute_features.py --only full
"""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from marines import load_config, project_root, ensure_output_dirs
from marines.dataio import compute_subject_slow_stats

# name -> (max_days, skip_gaps)
CONFIGS = {
    "full":         (None, False),
    "80d":          (80,   False),
    "full_gapskip": (None, True),
    "80d_gapskip":  (80,   True),
}


def run_config(name: str, max_days, skip_gaps, cfg, targets, max_workers) -> Path:
    print(f"\n=== features_{name}  (max_days={max_days}, skip_gaps={skip_gaps}) ===")
    stats: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(compute_subject_slow_stats, fid, cfg, max_days, skip_gaps): fid
                for fid in targets}
        for fut in as_completed(futs):
            fid = futs[fut]
            try:
                fid_ret, s = fut.result()
                stats[fid_ret] = s
                print(f"  [done] marine {fid_ret}")
            except Exception as e:  # noqa: BLE001
                print(f"  [error] marine {fid}: {e}")

    out = project_root() / cfg["paths"]["figures_dir"] / f"features_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["marine_id", "mean_real", "mean_imag",
                    "std_real", "std_imag", "third_real", "third_imag"])
        for fid in sorted(stats):
            s = stats[fid]
            w.writerow([fid, s["mean_r"], s["mean_i"], s["std_r"], s["std_i"],
                        s["m3_r"], s["m3_i"]])
    print(f"Saved: {out}")
    return out


def main() -> None:
    cfg = load_config()
    ccfg = cfg.get("clustering", {})
    excluded = set(ccfg.get("excluded_files", []))
    subject_count = int(ccfg.get("subject_count", 23))
    max_workers = int(ccfg.get("max_workers", 6))
    targets = [i for i in range(1, subject_count + 1) if i not in excluded]
    ensure_output_dirs(cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(CONFIGS), default=None,
                        help="compute a single configuration instead of all four")
    args = parser.parse_args()

    names = [args.only] if args.only else list(CONFIGS)
    for name in names:
        max_days, skip_gaps = CONFIGS[name]
        run_config(name, max_days, skip_gaps, cfg, targets, max_workers)


if __name__ == "__main__":
    main()
