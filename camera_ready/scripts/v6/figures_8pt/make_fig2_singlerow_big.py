"""Fig 2: the ORIGINAL single-row layout (same columns, icons, fusion panel, timeline, footer sentence),
redrawn at the full text width (6.27 in) with type 6.5-8 pt. Labels that no longer fit are wrapped, not removed."""
import matplotlib; matplotlib.use('Agg')
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib import font_manager
from figlib import *
font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'text.color': INK_HEX, 'mathtext.fontset': 'dejavusans'})
GREEN, GREEN_F = '#2E7D4F', '#EAF4EE'; BLUE, BLUE_F = '#2E5FB5', '#E9EEF9'; ORANGE_E, ORANGE_F = '#B9780F', '#FBF2E0'
GREY, GREY_F = '#3C3C3C', '#F2F2F2'; RED, RED_F = '#C0392B', '#FCEBE9'; NAVY_E, NAVY_F = '#1F3F5E', '#E6EEF5'; MUTED = '#4A5A64'
H, T, S, S2 = 8.0, 7.5, 7.0, 6.5     # header / title / detail / small-detail sizes (pt)
LW = 0.8; DASH = (0, (2.6, 1.4))
W_IN, H_IN = 6.27, 3.35; ASP = W_IN / H_IN
fig = plt.figure(figsize=(W_IN, H_IN), dpi=600); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
def box(x, y, w_, h_, edge, face, ls='-', lw=LW, r=1.2, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w_, h_, boxstyle=f'round,pad=0,rounding_size={r}', ec=edge, fc=face, lw=lw, ls=ls, zorder=z, mutation_aspect=ASP))
def label(x, y, s, size=S, weight='normal', color=INK_HEX, ha='center', va='center', style='normal', z=4):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style, zorder=z, linespacing=1.15)
def arrow(p0, p1, color=INK_HEX, lw=0.8, ms=5.5, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=3))
# ---- geometry (percent) : same column order as the original ----
IX, IW = 2.0, 19.5; PX, PW = 24.0, 11.0; FX, FW = 37.5, 18.0; GX, GW = 58.0, 11.5; SX, SW = 72.0, 11.5; RX, RW = 85.3, 14.5
RH = 16.0; rows = {'beh': 66.5, 'eeg': 47.0, 'et': 27.5}; mid = {k: v + RH / 2 for k, v in rows.items()}; YC = mid['eeg']
FRAME = (0.8, 19.5, 83.5, 71.0)
for cx, s, c in [(IX + IW / 2, 'VISIT INPUTS', GREEN), (PX + PW / 2, 'PROJECTIONS', GREEN), (FX + FW / 2, 'GATED FUSION', GREEN),
                 (GX + GW / 2, 'TEMPORAL', INK_HEX), (SX + SW / 2, 'VISIT SCORE', RED), (RX + RW / 2, 'BUY RISK', NAVY_E)]:
    label(cx, 96.5, s, size=H, weight='bold', color=c); ax.plot([cx - 6.5, cx + 6.5], [93.0, 93.0], color=c, lw=0.9)
box(*FRAME, '#9AA5AD', 'white', ls=(0, (3, 2)), lw=0.7, r=1.6, z=1)
label(FRAME[0] + 1.5, FRAME[1] + FRAME[3] - 3.0, 'per completed visit $j = 1\\ldots t$, forward-only (causal)', size=S, ha='left', style='italic', color=MUTED)
# ---- icons (same as the original, scaled to this aspect) ----
EY = lambda r: r * ASP
def icon_browser(x, y, c):
    ax.add_patch(FancyBboxPatch((x - 1.5, y - EY(1.5)), 3.0, EY(3.0), boxstyle='round,pad=0,rounding_size=0.3', ec=c, fc='white', lw=0.5, zorder=4, mutation_aspect=ASP))
    ax.plot([x - 1.5, x + 1.5], [y + EY(0.7), y + EY(0.7)], color=c, lw=0.5, zorder=5)
    pts = [(1, 0), (1, 0.85), (0.78, 0.65), (0.63, 1), (0.5, 0.94), (0.64, 0.6), (0.38, 0.6)]
    x0, y0, cw = x - 0.8, y - EY(1.3), 1.6; poly = [(x0 + u * cw, y0 + v * EY(cw)) for u, v in pts]
    ax.add_patch(plt.Polygon(poly, closed=True, fc=c, ec='white', lw=0.25, zorder=6))
def icon_eeg(x, y, c):
    ax.add_patch(Ellipse((x, y), 3.2, EY(3.2), ec=c, fc='white', lw=0.5, zorder=4)); t = np.linspace(-1.1, 1.1, 60)
    ax.plot(x + t, y + EY(0.9) * np.sin(t * 8) * np.exp(-3 * t * t), color=c, lw=0.5, zorder=5)
def icon_eye(x, y, c):
    ax.add_patch(Ellipse((x, y), 3.2, EY(3.2), ec=c, fc='white', lw=0.5, zorder=4)); t = np.linspace(-1.1, 1.1, 80)
    ax.plot(x + t, y + EY(0.8) * (1 - (t / 1.1) ** 2), color=c, lw=0.55, zorder=5); ax.plot(x + t, y - EY(0.8) * (1 - (t / 1.1) ** 2), color=c, lw=0.55, zorder=5)
    ax.add_patch(Ellipse((x, y), 1.2, EY(1.2), ec=c, fc='white', lw=0.45, zorder=6)); ax.add_patch(Ellipse((x, y), 0.65, EY(0.65), ec=c, fc=c, lw=0.3, zorder=7))
# ---- input boxes ----
for key, ec, fc, ls, t1, t2, ic in [('beh', GREEN, GREEN_F, '-', 'Browsing\nbehaviour', '6 inputs incl.\npropensity', icon_browser),
                                   ('eeg', BLUE, BLUE_F, DASH, 'EEG', 'optional\n153 inputs', icon_eeg),
                                   ('et', ORANGE_E, ORANGE_F, DASH, 'Eye tracking', 'optional\n70 inputs', icon_eye)]:
    y = rows[key]; box(IX, y, IW, RH, ec, fc, ls=ls); ic(IX + 2.8, mid[key], ec); tx = IX + 5.2 + (IW - 5.2) / 2
    if '\n' in t1: label(tx, y + RH - 4.6, t1, size=T, weight='bold', color=ec); label(tx, y + 4.4, t2, size=S2, style='italic')
    else: label(tx, y + RH - 3.8, t1, size=T, weight='bold', color=ec); label(tx, y + 4.6, t2, size=S2, style='italic')
# ---- projection boxes ----
PH = 11.5
for key, ec, ls, t1, sym in [('beh', GREEN, '-', 'Behaviour', '\\mathbf{b}'), ('eeg', BLUE, DASH, 'EEG', '\\tilde{\\mathbf{e}}'), ('et', ORANGE_E, DASH, 'Eye tracking', '\\tilde{\\mathbf{o}}')]:
    y = mid[key] - PH / 2; box(PX, y, PW, PH, ec, 'white', ls=ls)
    label(PX + PW / 2, y + PH - 3.2, t1, size=T, weight='bold', color=ec); label(PX + PW / 2, y + 3.2, f'64-D → ${sym}$', size=S)
    arrow((IX + IW, mid[key]), (PX, mid[key]))
# ---- gated fusion block ----
FY, FH = 21.5, 66.0; box(FX, FY, FW, FH, GREEN, '#F4F9F5', lw=0.9, r=1.6); fcx = FX + FW / 2
label(fcx, FY + FH - 3.6, 'Residual gated fusion', size=T, weight='bold', color=GREEN)
bb_h = 7.0; box(FX + 1.5, mid['beh'] - bb_h / 2, FW - 3.0, bb_h, GREEN, 'white'); label(fcx, mid['beh'], 'behaviour base $\\mathbf{b}$', size=S)
arrow((PX + PW, mid['beh']), (FX + 1.5, mid['beh']), color=GREEN)
gx = FX + 3.2; label(gx + 0.3, (mid['beh'] + mid['eeg']) / 2 - 0.5, 'optional\nmodalities', size=S2, style='italic', color=MUTED)
GB_H = 7.5
for key, ec, sym in [('eeg', BLUE, '\\tilde{\\mathbf{e}}'), ('et', ORANGE_E, '\\tilde{\\mathbf{o}}')]:
    cy = mid[key]; ax.add_patch(Ellipse((gx, cy), 3.0, EY(3.0), ec=ec, fc='white', lw=LW, zorder=4)); label(gx, cy, '×', size=8, color=ec, z=5)
    bx0 = gx + 3.0; bw = FX + FW - 1.5 - bx0; box(bx0, cy - GB_H / 2, bw, GB_H, ec, 'white', ls=DASH)
    label(bx0 + bw / 2, cy, f'$g(\\mathbf{{b}}, v)\\cdot {sym}$', size=S)
    arrow((PX + PW, cy), (gx - 1.5, cy), color=ec); arrow((gx + 1.5, cy), (bx0, cy), color=ec, ms=3.5, ls=(0, (1.3, 1.1)))
for xoff, key in [(-2.4, 'eeg'), (2.4, 'et')]:
    ax.plot([fcx + xoff, fcx + xoff], [mid['beh'] - bb_h / 2, mid[key] + GB_H / 2], color=INK_HEX, lw=0.4, ls=(0, (1, 1.6)), zorder=3)
sy = (mid['eeg'] + mid['et']) / 2; ax.add_patch(Ellipse((fcx - 2.4, sy), 2.8, EY(2.8), ec=GREEN, fc='white', lw=LW, zorder=4))
label(fcx - 2.4, sy, '+', size=8, color=GREEN, z=5); label(fcx - 0.2, sy, 'residual sum', size=S2, style='italic', color=MUTED, ha='left')
ax.plot([fcx - 2.4, fcx - 2.4], [mid['eeg'] - GB_H / 2, mid['et'] + GB_H / 2], color=INK_HEX, lw=0.4, ls=(0, (1, 1.6)), zorder=3)
box(FX + 1.5, FY + 2.0, FW - 3.0, 6.5, GREEN, '#DDEFE3'); label(fcx, FY + 5.2, 'LayerNorm → $\\mathbf{z}$', size=S)
# ---- GRU ----
GH = 26.0; gy = YC - GH / 2; box(GX, gy, GW, GH, GREY, GREY_F, lw=0.9); gcx = GX + GW / 2
label(gcx, gy + GH - 4.0, 'GRU', size=H, weight='bold', color=GREY)
label(gcx, gy + GH - 11.0, 'unidirectional\n64 units', size=S2); label(gcx, gy + 5.6, 'state carried\nacross visits', size=S2)
arrow((FX + FW, YC), (GX, YC))
ax.add_patch(FancyArrowPatch((GX + 3.0, gy + GH), (GX + GW - 3.0, gy + GH), connectionstyle='arc3,rad=-0.75', arrowstyle='-|>', mutation_scale=5, color=GREY, lw=0.7, zorder=3))
label(gcx, gy + GH + 6.0, '$s(j-1)$', size=S, style='italic')
# ---- sigmoid head ----
SH = 22.0; sy0 = YC - SH / 2; box(SX, sy0, SW, SH, RED, RED_F, lw=0.9); scx = SX + SW / 2
label(scx, sy0 + SH - 4.0, 'Sigmoid\nhead', size=T, weight='bold', color=RED)
label(scx, sy0 + 8.0, 'visit score $h(j)$', size=S2); label(scx, sy0 + 3.2, 'prefix-supervised\nEq. (2)', size=S2, style='italic', color=RED)
arrow((GX + GW, YC), (SX, YC))
# ---- cumulative risk (outside the frame) ----
RBH = 36.0; ry0 = YC - RBH / 2; box(RX, ry0, RW, RBH, NAVY_E, NAVY_F, lw=0.9, r=1.6); rcx = RX + RW / 2
label(rcx, ry0 + RBH - 4.0, 'Cumulative\nrisk', size=T, weight='bold', color=NAVY_E)
label(rcx, ry0 + RBH - 14.5, '$p(t)=1-\\prod_{j\\leq t}(1-h(j))$', size=S2 - 0.6)
label(rcx, ry0 + 11.0, 'nondecreasing\nEq. (1)', size=S2)
sx, sy_ = rcx - 4.0, ry0 + 2.4
xs = [sx, sx + 1.6, sx + 1.6, sx + 3.2, sx + 3.2, sx + 4.8, sx + 4.8, sx + 6.4, sx + 6.4, sx + 8.0]
ys = [sy_, sy_, sy_ + 1.0, sy_ + 1.0, sy_ + 2.2, sy_ + 2.2, sy_ + 2.9, sy_ + 2.9, sy_ + 3.7, sy_ + 3.7]
ax.plot(xs, ys, color=NAVY_E, lw=0.8, zorder=4); ax.plot([sx, sx + 8.0], [sy_ - 0.6, sy_ - 0.6], color='#9AA5AD', lw=0.45, zorder=4)
arrow((SX + SW, YC), (RX, YC))
# ---- timeline + footer ----
ty = 11.5; ax.plot([3.0, 92.0], [ty, ty], color=INK_HEX, lw=0.8, zorder=3); arrow((92.0, ty), (97.5, ty))
for x, s in [(7.0, 'visit 1'), (30.0, 'visit 2'), (53.0, 'visit 3'), (74.0, 'visit $t$')]:
    ax.plot(x, ty, 'o', color=INK_HEX, ms=2.8, zorder=4); label(x, ty + 3.8, s, size=S)
ax.plot([84.5, 84.5], [ty - 2.6, ty + 2.6], color=RED, lw=0.9, ls=(0, (2, 1.2)), zorder=4); label(84.5, ty + 3.8, 'audited cutoff', size=S, color=RED)
label(50.0, 3.5, 'Each completed visit updates the risk estimate from its own history only; risk never decreases, and no information beyond the cutoff is used.', size=S)
fig.savefig(os.path.join(OUT, 'fig2.png'), dpi=600, facecolor='white'); print('fig2 single-row big done')
