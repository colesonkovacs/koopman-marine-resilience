#!/usr/bin/env python3
"""
Reproduce every figure and table in the manuscript + supplement, in order.

compute_features.py must run first (it writes the feature CSVs the cluster
figures and confusion tables read). The sensitivity sweep is long-running; pass
--with-sweep to include it.

Run: python run_all.py               # everything except the sweep
     python run_all.py --with-sweep  # also regenerate Table 3 / S2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"

PIPELINE = [
    "compute_features.py",            # step 1: feature CSVs
    "fig1_figS1_eigenvalues.py",      # Figure 1 + Figure S1
    "fig2_bk_elbow.py",               # Figure 2
    "fig3_figS2_mean_eig.py",         # Figure 3 + Figure S2
    "fig4_mean_eig_80d.py",           # Figure 4
    "figS3_mean_eig_gap_excluded.py", # Figure S3
    "fig5_table4_hr_stats.py",        # Figure 5 + Table 4
    "tables1_2_S1_confusion.py",      # Table 1 + Table 2 + Table S1
]
SWEEP = "tables3_S2_sensitivity.py"   # Table 3 + Table S2 (long-running)


def run(script: str) -> None:
    print(f"\n{'=' * 70}\n>>> {script}\n{'=' * 70}")
    rc = subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=HERE).returncode
    if rc != 0:
        print(f"!! {script} exited with code {rc} — stopping.")
        sys.exit(rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-sweep", action="store_true",
                    help="also run the long sensitivity sweep (Table 3 / S2)")
    args = ap.parse_args()
    for s in PIPELINE:
        run(s)
    if args.with_sweep:
        run(SWEEP)
    print("\nAll requested outputs are in the figures/ folder.")


if __name__ == "__main__":
    main()
