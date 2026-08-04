#!/usr/bin/env python3
"""
Table 3  — DBSCAN parameter-sweep permutations ranked by F1 (main text).
Table S2 — KMeans parameter-sweep permutations ranked by F1 (supplement).

Runs the one-at-a-time sensitivity sweep around the sweep baseline
(window 900, step 36, embedding 300, threshold 0.00075) = 22 permutations.
For each permutation it recomputes the per-subject mean slow-eigenvalue feature
(full recording, gaps ignored) and clusters it:
  * DBSCAN: noise points = predicted dropouts.
  * KMeans (k=2): the favourable cluster-to-dropout mapping (highest F1) =
    predicted dropouts. This is the same convention used for the headline
    KMeans result (Table S1 / Figure S2).

WARNING: long-running (22 permutations x ~20 subjects x full DMD).

Writes: figures/table3_dbscan_sweep.csv, figures/tableS2_kmeans_sweep.csv
Run:    python scripts/tables3_S2_sensitivity.py
"""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from marines import load_config, project_root, ensure_output_dirs
from marines.dataio import compute_subject_slow_stats
from marines.clustering import (
    dbscan_labels, kmeans2_labels, kmeans_predicted_dropout, confusion_and_metrics,
)


def build_permutations(scfg) -> list[tuple[int, int, int, str]]:
    wl0, ws0, ed0, th0 = scfg["baseline"]
    wl0, ws0, ed0 = int(wl0), int(ws0), int(ed0)
    th0 = str(th0)
    combos = [(wl0, ws0, ed0, th0)]
    for wl in scfg["window_lengths"]:
        if int(wl) != wl0:
            combos.append((int(wl), ws0, ed0, th0))
    for ws in scfg["steps"]:
        if int(ws) != ws0:
            combos.append((wl0, int(ws), ed0, th0))
    for ed in scfg["dims"]:
        if int(ed) != ed0:
            combos.append((wl0, ws0, int(ed), th0))
    for th in scfg["threshes"]:
        if str(th) != th0:
            combos.append((wl0, ws0, ed0, str(th)))
    seen, out = set(), []
    for c in combos:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def features_for_combo(combo, cfg, targets, max_workers):
    params = (combo[0], combo[1], combo[2], float(combo[3]))
    stats = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(compute_subject_slow_stats, fid, cfg, None, False, params): fid
                for fid in targets}
        for fut in as_completed(futs):
            fid = futs[fut]
            try:
                fid_ret, s = fut.result()
                stats[fid_ret] = s
            except Exception as e:  # noqa: BLE001
                print(f"    [error] marine {fid}: {e}")
    ids = sorted(stats)
    X = np.array([[stats[f]["mean_r"], stats[f]["mean_i"]] for f in ids])
    return np.array(ids), X


def main() -> None:
    cfg = load_config()
    ccfg = cfg.get("clustering", {})
    scfg = cfg["sensitivity"]
    dropout_ids = set(ccfg.get("dropout_ids", []))
    excluded = set(ccfg.get("excluded_files", []))
    subject_count = int(ccfg.get("subject_count", 23))
    max_workers = int(ccfg.get("max_workers", 6))
    eps = float(ccfg.get("dbscan_eps", 0.75))
    min_samples = int(ccfg.get("dbscan_min_samples", 6))
    targets = [i for i in range(1, subject_count + 1) if i not in excluded]

    combos = build_permutations(scfg)
    print(f"Running {len(combos)} permutations x {len(targets)} subjects ...")

    db_rows, km_rows = [], []
    for i, combo in enumerate(combos, 1):
        wl, ws, ed, th = combo
        print(f"[{i:>2}/{len(combos)}] window={wl} step={ws} dim={ed} thresh={th}")
        ids, X = features_for_combo(combo, cfg, targets, max_workers)
        actual = np.array([fid in dropout_ids for fid in ids], dtype=bool)

        db_pred = dbscan_labels(X, eps=eps, min_samples=min_samples) == -1
        dm = confusion_and_metrics(db_pred, actual)
        db_rows.append([wl, ws, ed, th, dm["precision"], dm["recall"], dm["f1"]])

        km_pred = kmeans_predicted_dropout(kmeans2_labels(X), actual)
        km = confusion_and_metrics(km_pred, actual)
        km_rows.append([wl, ws, ed, th, km["precision"], km["recall"], km["f1"]])

    db_rows.sort(key=lambda r: -r[6])
    km_rows.sort(key=lambda r: -r[6])

    figures_dir = ensure_output_dirs(cfg)
    for rows, name, tag in [(db_rows, "table3_dbscan_sweep.csv", "Table 3 DBSCAN"),
                            (km_rows, "tableS2_kmeans_sweep.csv", "Table S2 KMeans")]:
        with (figures_dir / name).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Window", "Step", "Dimension", "Threshold",
                        "Precision", "Recall", "F1"])
            for r in rows:
                w.writerow([r[0], r[1], r[2], r[3],
                            f"{r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.4f}"])
        print(f"\n{tag}: top rows")
        for r in rows[:3]:
            print(f"  {r[0]} {r[1]} {r[2]} {r[3]}  P={r[4]:.3f} R={r[5]:.3f} F1={r[6]:.3f}")
        print(f"Saved: {figures_dir / name}")


if __name__ == "__main__":
    main()
