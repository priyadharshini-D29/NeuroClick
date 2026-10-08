"""
neuroclick_fig_style.py — shared style for all NeuroClick manuscript figures.

Usage inside 06_generate_manuscript_figures.py / 08_generate_ensemble_figure.py:

    from neuroclick_fig_style import apply_style, save_fig
    apply_style()                      # once, at the top, after importing matplotlib

    ... build each figure exactly as before (do NOT change figsize) ...

    save_fig(fig, "outputs/manuscript_figures_v1/fig1_dataset_profile.png")
    # replaces plt.savefig(...); always writes 600 dpi + tight bbox

What changes visually:
  - no top/right spines (open frame)
  - background gridlines: faint, dotted, y-axis only, drawn BEHIND the data
  - slightly lighter remaining spines and ticks
  - same fonts/sizes/colors/figsize as before, so layout in the paper is unchanged

What deliberately does NOT change:
  - figure sizes and aspect ratios (the docx frames depend on them)
  - your color palette, titles, annotations
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

GRID_COLOR = "#c9d2d8"   # faint blue-gray
SPINE_COLOR = "#4a4a4a"


def apply_style():
    """Call once before creating any figure."""
    mpl.rcParams.update({
        # gridlines: faint dotted, behind data
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID_COLOR,
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.6,
        "axes.axisbelow": True,
        # open frame
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 0.8,
        # ticks
        "xtick.color": SPINE_COLOR,
        "ytick.color": SPINE_COLOR,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # export
        "savefig.dpi": 600,
        "figure.dpi": 100,
    })


def restyle_axes(ax, grid="y"):
    """Apply the style to an existing Axes (for axes created before apply_style,
    or to override per-axes). grid: 'y' (default), 'both', or 'off'.
    Use grid='off' for heatmaps (Fig. 5) and diagram-like panels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.set_axisbelow(True)
    if grid == "off":
        ax.grid(False)
    else:
        ax.grid(True, axis=grid, color=GRID_COLOR, linestyle=":",
                linewidth=0.7, alpha=0.6)


def save_fig(fig, path, dpi=600):
    """Uniform export: 600 dpi, tight bounding box, white background.
    Never pass a different figsize here — resolution comes from dpi alone,
    so the docx frame aspect ratios stay valid."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"saved {path} @ {dpi} dpi")


# Integration checklist for script 06 (and 08):
# 1. from neuroclick_fig_style import apply_style, restyle_axes, save_fig
# 2. apply_style() immediately after the matplotlib imports.
# 3. Replace every fig.savefig(...)/plt.savefig(...) with save_fig(fig, path).
# 4. For the Fig. 5 heatmap axes: restyle_axes(ax, grid="off") after creation
#    (dotted gridlines on top of a heatmap look wrong).
# 5. If any axes call ax.grid(...) explicitly with heavier settings, delete
#    those lines — the rcParams now handle it.
# 6. Do not touch any figsize=... argument.
