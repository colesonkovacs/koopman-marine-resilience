#!/usr/bin/env python3
"""Central aesthetics for scripts/final_plots/*.py"""
from __future__ import annotations
import matplotlib as mpl

# ── Palette ───────────────────────────────────────────────────────────────────
SLOW_COLOR      = "blue"
FAST_COLOR      = "red"
HR_COLOR        = "black"
DROPOUT_COLOR   = "orange"
COMPLETED_COLOR = "purple"

# ── Line styles (slow=solid, fast=dashed for B&W differentiation) ─────────────
SLOW_LINESTYLE = "-"
FAST_LINESTYLE = "-"
HR_LINESTYLE   = "-"

# ── Scatter marker shapes ─────────────────────────────────────────────────────
SLOW_MARKER = "o"   # circle
FAST_MARKER = "^"   # triangle-up

# ── Line marker shapes (used on reconstruction line plots) ────────────────────
SLOW_LINE_MARKER  = "o"
FAST_LINE_MARKER  = "v"
LINE_MARKER_EVERY = 100   # plot a marker every N data points
LINE_MARKER_SIZE  = 3.5     # marker size in points

# ── Figure geometry ───────────────────────────────────────────────────────────
FIGURE_SIZE = (9, 6)

# ── Line widths ───────────────────────────────────────────────────────────────
HR_LW        = 0.8
SLOW_LW      = 1
FAST_LW      = 1
AXIS_LINE_LW = 0.5

# ── Scatter sizes / edge widths ───────────────────────────────────────────────
SCATTER_SIZE      = 30
SCATTER_LW        = 0.8
SCATTER_SIZE_MEAN = 120

# ── Typography ────────────────────────────────────────────────────────────────
FONT_FAMILY          = "sans-serif"
FONT_SIZE_AXIS_LABEL = 14
FONT_SIZE_TICK       = 12
FONT_SIZE_LEGEND     = 12
FONT_SIZE_ANNOTATION = 13
FONT_WEIGHT_AXIS     = "bold"

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID_ALPHA = 0.3

# ── Save quality ──────────────────────────────────────────────────────────────
SAVE_DPI = 200


def apply_style() -> None:
    """Push all aesthetics into matplotlib's global rcParams."""
    mpl.rcParams.update({
        "font.family":       FONT_FAMILY,
        "font.size":         FONT_SIZE_TICK,
        "axes.labelsize":    FONT_SIZE_AXIS_LABEL,
        "axes.labelweight":  FONT_WEIGHT_AXIS,
        "xtick.labelsize":   FONT_SIZE_TICK,
        "ytick.labelsize":   FONT_SIZE_TICK,
        "legend.fontsize":   FONT_SIZE_LEGEND,
        "legend.framealpha": 0.8,
        "figure.figsize":    list(FIGURE_SIZE),
        "figure.dpi":        100,
        "savefig.dpi":       SAVE_DPI,
        "savefig.bbox":      "tight",
    })