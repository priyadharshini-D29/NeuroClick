#!/usr/bin/env python3
"""Regenerate manuscript Fig. 2 (NeuroClick architecture)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon

GREEN, GREEN_D = "#eaf3ec", "#2e6b46"
BLUE, BLUE_D = "#eef1fa", "#3a55a4"
GOLD, GOLD_D = "#fdf6e6", "#c8860a"
RED = "#c9463d"
NAVY = "#22314f"
GREY = "#8a8f98"
INK = "#333333"

fig, ax = plt.subplots(figsize=(12.4, 5.813), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 47); ax.axis("off")

def box(x, y, w, h, fc, ec, ls="-", lw=1.4, r=0.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, ls=ls, lw=lw, zorder=3)
    ax.add_patch(p); return p

def arrow(x1, y1, x2, y2, color=INK, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=11, color=color, lw=lw, ls=ls, zorder=4))

def T(x, y, s, size=7.5, color=INK, w="normal", st="normal", ha="center", z=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=w,
            fontstyle=st, ha=ha, va="center", zorder=z)

# ---- column headers ----
for x, s, c in [(9.5, "VISIT INPUTS", NAVY), (25.5, "PROJECTIONS", NAVY),
                (43.5, "GATED FUSION", GREEN_D), (62.5, "TEMPORAL", NAVY),
                (76.5, "HAZARD", RED), (91, "BUY RISK", NAVY)]:
    T(x, 45.3, " ".join(s), 8, c, "bold")
    ax.plot([x-5.5, x+5.5], [44.1, 44.1], color=c, lw=1.2)

# ---- outer dashed frame ----
box(2.2, 11.5, 82.5, 31.3, "none", GREY, ls=(0, (5, 4)), lw=1.2, r=1.0)
T(5.0, 41.6, "per completed visit j = 1 … t   forward-only (causal)", 7.2, "#555555", st="italic", ha="left")

# ---- input boxes ----
box(3.8, 33.2, 14.4, 7.0, GREEN, GREEN_D, lw=1.7)
box(3.8, 23.4, 14.4, 6.2, BLUE, BLUE_D, ls=(0, (4, 3)))
box(3.8, 14.2, 14.4, 6.2, GOLD, GOLD_D, ls=(0, (4, 3)))
# browsing icon
box(4.6, 37.0, 1.7, 1.9, "white", GREEN_D, lw=1.1, r=0.2)
ax.plot([4.6, 6.3], [38.35, 38.35], color=GREEN_D, lw=0.9, zorder=6)
ax.add_patch(Polygon([[5.4, 37.9], [6.0, 37.1], [5.55, 37.25]], closed=True, fc=GREEN_D, zorder=6))
T(12.6, 38.6, "Browsing behaviour", 6.4, GREEN_D, "bold")
T(12.6, 37.1, "6 inputs · incl. propensity", 5.4, GREEN_D, st="italic")
T(12.6, 35.8, "(cross-fitted, Sec. 3.3)", 5.4, RED, st="italic")
# EEG icon
ax.add_patch(Circle((6.0, 27.6), 1.05, fc="white", ec=BLUE_D, lw=1.1, zorder=6))
ax.plot([5.35, 5.7, 5.95, 6.2, 6.65], [27.6, 27.6, 28.35, 26.9, 27.6], color=BLUE_D, lw=1.0, zorder=7)
T(12.6, 28.3, "EEG – optional", 7.3, BLUE_D, "bold")
T(12.6, 26.4, "153 inputs", 6.2, BLUE_D)
# eye icon
ax.add_patch(Ellipse((6.0, 17.3), 2.0, 1.15, fc="white", ec=GOLD_D, lw=1.1, zorder=6))
ax.add_patch(Circle((6.0, 17.3), 0.32, fc=GOLD_D, zorder=7))
T(13.0, 19.1, "Eye tracking – optional", 5.7, GOLD_D, "bold")
T(12.6, 17.2, "70 inputs", 6.2, GOLD_D)

# ---- projection boxes ----
box(21.0, 34.3, 9.6, 5.0, GREEN, GREEN_D, lw=1.7)
T(25.8, 37.7, "Behaviour", 7.3, GREEN_D, "bold"); T(25.8, 35.8, r"64-D $\rightarrow$ $\mathbf{b}$", 6.4, GREEN_D)
box(21.0, 24.0, 9.6, 5.0, BLUE, BLUE_D, ls=(0, (4, 3)))
T(25.8, 27.4, "EEG", 7.3, BLUE_D, "bold"); T(25.8, 25.5, r"64-D $\rightarrow$ $\tilde{\mathbf{e}}$", 6.4, BLUE_D)
box(21.0, 14.8, 9.6, 5.0, GOLD, GOLD_D, ls=(0, (4, 3)))
T(25.8, 18.2, "Eye tracking", 7.3, GOLD_D, "bold"); T(25.8, 16.3, r"64-D $\rightarrow$ $\tilde{\mathbf{o}}$", 6.4, GOLD_D)
arrow(18.2, 36.7, 21.0, 36.8, GREEN_D)
arrow(18.2, 26.5, 21.0, 26.5, BLUE_D)
arrow(18.2, 17.3, 21.0, 17.3, GOLD_D)

# ---- gated fusion panel ----
box(34.5, 12.8, 18.0, 28.6, "#f2f8f3", GREEN_D, lw=1.6, r=1.0)
T(43.5, 40.0, "Residual gated fusion", 7.4, GREEN_D, "bold")
box(36.3, 35.2, 14.4, 3.3, "white", GREEN_D, lw=1.3)
T(43.5, 36.85, r"behaviour base $\mathbf{b}$", 6.8, GREEN_D)
T(36.6, 31.4, "optional\nmodalities", 5.8, "#555555", st="italic", ha="left")
# gates
ax.add_patch(Circle((38.6, 26.6), 0.95, fc="white", ec=BLUE_D, lw=1.3, zorder=6))
T(38.6, 26.6, r"$\times$", 8, BLUE_D, "bold", z=7)
box(40.6, 25.0, 10.6, 3.2, "white", BLUE_D, ls=(0, (4, 3)))
T(45.9, 26.6, r"$g(b,v)\cdot\tilde{e}$", 6.6, BLUE_D)
ax.add_patch(Circle((41.6, 21.6), 0.95, fc="white", ec=GREEN_D, lw=1.3, zorder=6))
T(41.6, 21.6, r"$+$", 8, GREEN_D, "bold", z=7)
T(45.6, 21.6, "residual sum", 6.0, "#555555", st="italic", ha="left")
ax.add_patch(Circle((38.6, 17.4), 0.95, fc="white", ec=GOLD_D, lw=1.3, zorder=6))
T(38.6, 17.4, r"$\times$", 8, GOLD_D, "bold", z=7)
box(40.6, 15.8, 10.6, 3.2, "white", GOLD_D, ls=(0, (4, 3)))
T(45.9, 17.4, r"$g(b,v)\cdot\tilde{o}$", 6.6, GOLD_D)
box(36.3, 13.4, 14.4, 2.4, GREEN, GREEN_D, lw=1.2)
T(43.5, 14.6, r"LayerNorm $\rightarrow$ $\mathbf{z}$", 6.6, GREEN_D)
# dotted droppers from behaviour base to gates
for gx, gy, gc in [(38.6, 26.6, BLUE_D), (38.6, 17.4, GOLD_D)]:
    ax.plot([gx, gx], [35.2, gy + 0.95], color=gc, lw=0.9, ls=(0, (1.5, 2)), zorder=4)
ax.plot([43.5, 43.5], [35.2, 28.2], color=GREEN_D, lw=0.9, ls=(0, (1.5, 2)), zorder=4)
arrow(30.6, 36.8, 36.3, 36.8, GREEN_D)
arrow(30.6, 26.5, 37.65, 26.6, BLUE_D, ls=(0, (4, 3)))
arrow(30.6, 17.3, 37.65, 17.4, GOLD_D, ls=(0, (4, 3)))

# ---- GRU ----
box(56.5, 21.8, 11.5, 9.2, "#f2f3f5", "#4a4f57", lw=1.7)
T(62.25, 28.5, "GRU", 10.5, "#2b2f36", "bold")
T(62.25, 25.9, "unidirectional · 64 units", 5.9, "#4a4f57")
T(62.25, 24.3, "state carried across visits", 5.9, "#4a4f57")
ax.add_patch(FancyArrowPatch((60.0, 31.0), (64.5, 31.0), connectionstyle="arc3,rad=-1.2",
             arrowstyle="-|>", mutation_scale=10, color="#2b2f36", lw=1.4, zorder=4))
T(62.25, 34.6, r"$h(j\!-\!1)$", 6.4, "#2b2f36", st="italic")
arrow(52.5, 26.5, 56.5, 26.5, "#2b2f36", lw=1.8)

# ---- Sigmoid head ----
box(71.0, 22.6, 10.6, 7.6, "#fdeeec", RED, lw=1.7)
T(76.3, 28.3, "Sigmoid head", 7.6, RED, "bold")
T(76.3, 26.4, r"visit hazard $\mathbf{h(j)}$", 6.6, INK)
T(76.3, 24.5, r"$p(buy\ at\ j\ |\ no\ buy < j)$", 5.4, RED, st="italic")
arrow(68.0, 26.5, 71.0, 26.5, "#2b2f36", lw=1.8)

# ---- Cumulative risk ----
box(84.8, 19.8, 12.6, 13.2, "#e9f0f7", NAVY, lw=1.7)
T(91.1, 31.1, "Cumulative risk", 7.4, NAVY, "bold")
T(91.1, 28.2, r"$p(t) = 1-\prod_{j=1}^{t}(1-h(j))$", 6.4, NAVY)
T(91.1, 25.4, "nondecreasing · Eq. (1)", 5.6, "#555577")
xs = [86.6, 88.0, 88.0, 89.6, 89.6, 91.4, 91.4, 93.2, 93.2, 95.2]
ys = [21.2, 21.2, 21.9, 21.9, 22.5, 22.5, 23.3, 23.3, 23.9, 23.9]
ax.plot(xs, ys, color=NAVY, lw=1.3, zorder=5)
ax.plot([86.6, 95.4], [20.8, 20.8], color="#9aa5b5", lw=0.8, zorder=4)
arrow(81.6, 26.5, 84.8, 26.5, "#2b2f36", lw=1.8)

# ---- timeline ----
ty = 7.2
ax.annotate("", xy=(96.5, ty), xytext=(4.5, ty),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
for vx, lab in [(9, "visit 1"), (31, "visit 2"), (53, "visit 3"), (73, r"visit  $t$")]:
    ax.add_patch(Circle((vx, ty), 0.42, fc=INK, zorder=6))
    T(vx, ty + 1.9, lab, 6.6, INK)
ax.plot([80.5, 80.5], [ty - 1.3, ty + 1.3], color=RED, lw=1.3, ls=(0, (3, 2)), zorder=6)
T(83.5, ty + 1.9, "audited cutoff", 6.2, RED)
T(50, 3.4, "Each completed visit updates the risk estimate from its own history only;", 6.0, "#666666")
T(50, 2.0, "risk never decreases, and no information beyond the cutoff is used.", 6.0, "#666666")

fig.savefig("fig02_neuroclick_architecture.png", transparent=True,
            bbox_inches=None, pad_inches=0)
print("saved fig02_neuroclick_architecture.png")
