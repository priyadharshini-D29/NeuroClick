#!/usr/bin/env python3
"""Regenerate Fig. 7 (average ensemble at every horizon) from outputs/statistics/ensemble_v1/ensemble_pairwise_tests.csv. Run from project root: python regenerate_fig7.py"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TESTS = "outputs/statistics/ensemble_v1/ensemble_pairwise_tests.csv"
OUT = "outputs/statistics/ensemble_v1/fig7_ensemble_average.png"

TEAL, NAVY, GOLD, WHITE = "#1f8a80", "#33516e", "#e0a52e", "#ffffff"
HORIZONS = ["first1", "first2", "first3", "full"]
XLABELS = ["First visit", "First two", "First three", "Complete history"]

t = pd.read_csv(TESTS)
t = t[(t.metric == "pr_auc") & (t.candidate == "ensemble_average")]

gru_means, base_means, ens_means, holm_ps = [], [], [], []
for h in HORIZONS:
    vs_gru = t[(t.horizon == h) & (t.comparator == "gru")].iloc[0]
    vs_base = t[(t.horizon == h) & (t.comparator == "baseline")].iloc[0]
    ens_means.append(vs_gru.candidate_mean)
    gru_means.append(vs_gru.comparator_mean)
    base_means.append(vs_base.comparator_mean)
    holm_ps.append(vs_gru.p_holm)

fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=300)
x = np.arange(len(HORIZONS))
w = 0.26
ax.bar(x - w, gru_means, w, color=TEAL, edgecolor=WHITE, linewidth=0.4,
       label="NeuroClick (GRU)", zorder=3)
ax.bar(x, base_means, w, color=NAVY, edgecolor=WHITE, linewidth=0.4,
       label="Dwell + propensity", zorder=3)
ax.bar(x + w, ens_means, w, color=GOLD, edgecolor=WHITE, linewidth=0.4,
       label="Ensemble (average)", zorder=3)

for xi, (m, p) in enumerate(zip(ens_means, holm_ps)):
    if p < 0.05:
        note = f"average\nvs GRU: *\n(p={p:.2e})"
    else:
        note = f"average\nvs GRU: n.s.\n(p={min(p,1.0):.2g})"
    ax.annotate(note, xy=(xi + w, m), xytext=(xi + w, m + 0.045),
                ha="center", va="bottom", fontsize=6.5, color="#333333")

ax.set_ylabel("Participant-mean PR AUC", fontsize=11)
ax.set_title("NeuroClick vs dwell+propensity baseline vs ensemble",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(XLABELS, fontsize=11)
ax.set_ylim(0, 0.8)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT, transparent=True)
print("saved:", OUT)
print("\nValues plotted (for the Table 6 update):")
for h, g, b, e, p in zip(HORIZONS, gru_means, base_means, ens_means, holm_ps):
    print(f"  {h:6s} GRU={g:.4f} baseline={b:.4f} ensemble_avg={e:.4f} "
          f"delta_vs_GRU={e-g:+.4f} Holm={p:.3g}")
