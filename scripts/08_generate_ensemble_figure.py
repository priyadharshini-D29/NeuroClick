#!/usr/bin/env python3
"""Generate Fig. 7 for §4.5: NeuroClick vs the dwell+propensity baseline vs
the best ensemble strategy, grouped by horizon.

Reads the outputs of 07_ensemble_neuroclick_baseline.py directly
(ensemble_per_subject_metrics.csv, ensemble_pairwise_tests.csv) and produces
a grouped bar chart of participant-mean PR AUC per horizon, matching the
palette and style already used by 06_generate_manuscript_figures.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEAL = "#007C83"
BLUE = "#315F87"
GOLD = "#D89B2B"
TEXT = "#1F3540"
WHITE = "#FFFFFF"
PALE_BLUE = "#DCEAF0"

HORIZON_ORDER = ["first1", "first2", "first3", "full"]
HORIZON_LABELS = {"first1": "First visit", "first2": "First two", "first3": "First three", "full": "Complete history"}

# Best-performing ensemble strategy per horizon, as reported in Table 5.
BEST_STRATEGY = {"first1": "ensemble_average", "first2": "ensemble_average", "first3": "ensemble_stacked", "full": "ensemble_stacked"}


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=PALE_BLUE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-subject-metrics", type=Path, required=True,
                         help="cache/final_run_ensemble/ensemble_per_subject_metrics.csv")
    parser.add_argument("--pairwise-tests", type=Path, required=True,
                         help="cache/final_run_ensemble/ensemble_pairwise_tests.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/manuscript_figures_v1"))
    args = parser.parse_args()

    metrics = pd.read_csv(args.per_subject_metrics)
    tests = pd.read_csv(args.pairwise_tests)

    horizons = [h for h in HORIZON_ORDER if h in metrics["horizon"].unique()]
    if not horizons:
        raise ValueError(f"No expected horizons found in {args.per_subject_metrics}")

    gru_means, baseline_means, ensemble_means, ensemble_labels = [], [], [], []
    for horizon in horizons:
        subset = metrics.loc[metrics["horizon"] == horizon]
        gru_means.append(float(subset.loc[subset["model"] == "gru", "pr_auc"].mean()))
        baseline_means.append(float(subset.loc[subset["model"] == "baseline", "pr_auc"].mean()))
        strategy = BEST_STRATEGY[horizon]
        ensemble_means.append(float(subset.loc[subset["model"] == strategy, "pr_auc"].mean()))
        ensemble_labels.append("avg" if strategy == "ensemble_average" else "stacked")

    fig, ax = plt.subplots(figsize=(6.25, 3.1), dpi=600)
    x = np.arange(len(horizons))
    width = 0.26
    ax.bar(x - width, gru_means, width, color=TEAL, edgecolor=WHITE, linewidth=0.4, label="NeuroClick (GRU)", zorder=3)
    ax.bar(x, baseline_means, width, color=BLUE, edgecolor=WHITE, linewidth=0.4, label="Dwell + propensity", zorder=3)
    ax.bar(x + width, ensemble_means, width, color=GOLD, edgecolor=WHITE, linewidth=0.4, label="Ensemble (best)", zorder=3)

    for i, horizon in enumerate(horizons):
        row = tests.loc[
            (tests["horizon"] == horizon)
            & (tests["metric"] == "pr_auc")
            & (tests["candidate"] == BEST_STRATEGY[horizon])
            & (tests["comparator"] == "gru")
        ]
        if not row.empty:
            p_holm = float(row.iloc[0]["p_holm"])
            sig = "*" if p_holm < 0.05 else "n.s."
            ax.text(
                i + width, ensemble_means[i] + 0.008, f"{ensemble_labels[i]}\nvs GRU: {sig}\n(p={p_holm:.3g})",
                ha="center", va="bottom", fontsize=5.8, color=TEXT,
            )

    ax.set_xticks(x, [HORIZON_LABELS[h] for h in horizons])
    ax.set_ylabel("Participant-mean PR AUC")
    ax.set_title("NeuroClick vs dwell+propensity baseline vs ensemble", weight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax.set_ylim(0, max(gru_means + baseline_means + ensemble_means) * 1.32)
    style_axis(ax)
    fig.tight_layout()

    args.output.mkdir(parents=True, exist_ok=True)
    out_base = args.output / "fig07_ensemble_by_horizon"
    fig.savefig(out_base.with_suffix(".png"), dpi=600)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)
    print(f"Saved {out_base}.png and .svg")


if __name__ == "__main__":
    main()
