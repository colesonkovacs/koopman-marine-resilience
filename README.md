# Koopman Analysis of Resilience — Manuscript Reproduction Code

Code to reproduce **every figure and table** in *"Koopman Analysis of Resilience
in Dutch Marines Undergoing Intensive Military Training"* (submitted to
*Psychological Methods*) and its Supplement. Each script produces one or more
specific, labelled manuscript outputs — nothing else.

Method: Sliding-window Hankel Dynamic Mode Decomposition (SW-HDMD) of each
recruit's heart-rate series, extraction of the slow Koopman eigenvalues, and
clustering (DBSCAN / KMeans) of the per-subject mean slow eigenvalue. 20 recruits
(IDs 7, 10, 13 excluded for device issues); 7 ground-truth dropouts (1, 2, 4, 6,
15, 16, 23).

---

## What produces what

| Manuscript item | Script | Output file(s) |
| --- | --- | --- |
| **Figure 1** — eigenvalues (zoomed-in) | `scripts/fig1_figS1_eigenvalues.py` | `fig1_eigenvalues_zoom.png` |
| **Figure S1** — eigenvalues (zoomed-out) | `scripts/fig1_figS1_eigenvalues.py` | `figS1_eigenvalues_full.png` |
| **Figure 2** — b_k elbow filtering | `scripts/fig2_bk_elbow.py` | `fig2_bk_elbow.png` |
| **Figure 3** — mean slow eig, DBSCAN | `scripts/fig3_figS2_mean_eig.py` | `fig3_mean_eig_dbscan.png` |
| **Figure S2** — mean slow eig, KMeans | `scripts/fig3_figS2_mean_eig.py` | `figS2_mean_eig_kmeans.png` |
| **Figure 4** — 80-day mean slow eig, DBSCAN | `scripts/fig4_mean_eig_80d.py` | `fig4_mean_eig_dbscan_80d.png` |
| **Figure S3** — gap-excluded (full + 80-day) | `scripts/figS3_mean_eig_gap_excluded.py` | `figS3_mean_eig_gap_excluded.png` |
| **Figure 5** — basic HR-stat clustering (4 panels) | `scripts/fig5_table4_hr_stats.py` | `fig5_hr_scatter_*_{dbscan,kmeans}.png` |
| **Table 1** — DBSCAN confusion matrix | `scripts/tables1_2_S1_confusion.py` | `confusion_matrices.csv` |
| **Table 2** — DBSCAN metrics | `scripts/tables1_2_S1_confusion.py` | `classification_metrics.csv` |
| **Table S1** — KMeans metrics | `scripts/tables1_2_S1_confusion.py` | `classification_metrics.csv` |
| **Table 4** — DBSCAN/KMeans on basic stats | `scripts/fig5_table4_hr_stats.py` | `table4_hr_stats_metrics.csv` |
| **Table 3** — DBSCAN parameter sweep | `scripts/tables3_S2_sensitivity.py` | `table3_dbscan_sweep.csv` |
| **Table S2** — KMeans parameter sweep | `scripts/tables3_S2_sensitivity.py` | `tableS2_kmeans_sweep.csv` |

All outputs are written to `figures/`. `scripts/compute_features.py` must run
first — it writes the per-subject feature CSVs that Figures 3, 4, S2, S3 and
Tables 1, 2, S1 read (so the expensive DMD pass happens once).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Edit `config/default.yaml` so `paths.data_root` points at the heart-rate CSVs
(`paths.hr_template` matches their filename pattern).

## Run

```bash
python run_all.py                 # every figure + table except the long sweep
python run_all.py --with-sweep    # also regenerate Table 3 / S2 (slow)
```

or individually, e.g. `python scripts/fig3_figS2_mean_eig.py`. Always run
`python scripts/compute_features.py` first.

## Reproduced numbers (baseline)

| Table | Metric | Value |
| --- | --- | --- |
| Table 1 (DBSCAN confusion) | TP / FN / FP / TN | 5 / 2 / 0 / 13 |
| Table 2 (DBSCAN) | Precision / Recall / F1 | 1.000 / 0.714 / 0.833 |
| Table S1 (KMeans) | Accuracy / Precision / Recall / F1 | 0.650 / 0.500 / 0.857 / 0.632 |
| Table 4 — Mean vs Std (KMeans) | Acc / Prec / Rec / F1 | 0.750 / 0.600 / 0.857 / 0.706 |
| Table 4 — SuccDiff vs CV (DBSCAN) | Acc / Prec / Rec / F1 | 0.550 / 0.400 / 0.571 / 0.471 |
| Table 4 — SuccDiff vs CV (KMeans) | Acc / Prec / Rec / F1 | 0.700 / 0.545 / 0.857 / 0.667 |


The raw heart-rate CSVs are **not** included (they live outside this folder and
may be restricted); `config/default.yaml` points at their location. Generated
outputs in `figures/` are git-ignored — anyone who clones the repo regenerates
them with `python run_all.py`.


## Notes on conventions

- **Main analysis ignores data gaps** (uniform-timestep assumption); Figure S3 is
  the supplementary check that excludes windows spanning gaps > one window (1.5 d).

 
