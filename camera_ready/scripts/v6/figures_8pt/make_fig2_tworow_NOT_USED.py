"""Fig 2: two-row architecture with every label >= 8 pt, drawn at the placed width (4.8 in)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import font_manager
from figlib import *
font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'text.color': INK_HEX, 'mathtext.fontset': 'dejavusans'})
GREEN, GREEN_F = '#2E7D4F', '#EAF4EE'; BLUE, BLUE_F = '#2E5FB5', '#E9EEF9'; ORANGE_E, ORANGE_F = '#B9780F', '#FBF2E0'
GREY, GREY_F = '#3C3C3C', '#F2F2F2'; RED, RED_F = '#C0392B', '#FCEBE9'; NAVY_E, NAVY_F = '#1F3F5E', '#E6EEF5'; MUTED = '#4A5A64'
W_IN, H_IN = 4.8, 3.3
fig = plt.figure(figsize=(W_IN, H_IN), dpi=600); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
def box(x, y, w, h, edge, face, ls='-', lw=0.9, r=1.2): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', ec=edge, fc=face, lw=lw, ls=ls, zorder=2))
def label(x, y, s, size=8, weight='normal', color=INK_HEX, ha='center', va='center', style='normal'): ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style, zorder=4)
def arrow(p0, p1, color=INK_HEX, lw=0.9, ms=6): ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, zorder=3, shrinkA=0, shrinkB=0))
box(1.0, 17.5, 98.0, 81.5, '#9AA5AD', 'white', ls=(0, (3, 2)), lw=0.8, r=1.5)
label(3.0, 96.0, 'per completed visit $j = 1 \\ldots t$, forward-only (causal)', size=8, ha='left', style='italic', color=MUTED)
for x, s, hw in [(16.0, 'VISIT INPUTS', 10), (41.5, 'PROJECTIONS', 10), (73.5, 'GATED FUSION', 11)]:
    label(x, 91.0, s, size=8, weight='bold', color=GREEN); ax.plot([x - hw, x + hw], [89.2, 89.2], color=GREEN, lw=0.9)
rows = {'beh': 74.0, 'eeg': 61.0, 'et': 48.0}; H = 10.0; IX, IW = 2.5, 27.0; PX, PW = 33.0, 17.0; icx, pcx = IX + IW/2, PX + PW/2
for key, ec, fc, ls, t1, t2 in [('beh', GREEN, GREEN_F, '-', 'Browsing behaviour', '6 inputs, incl. propensity'),
                                ('eeg', BLUE, BLUE_F, (0, (3, 1.5)), 'EEG', 'optional, 153 inputs'),
                                ('et', ORANGE_E, ORANGE_F, (0, (3, 1.5)), 'Eye tracking', 'optional, 70 inputs')]:
    box(IX, rows[key], IW, H, ec, fc, ls=ls); label(icx, rows[key] + 6.8, t1, size=8.5, weight='bold', color=ec); label(icx, rows[key] + 2.8, t2, size=8)
for key, ec, ls, t1, t2 in [('beh', GREEN, '-', 'Behaviour', '64-D → $\\mathbf{b}$'), ('eeg', BLUE, (0, (3, 1.5)), 'EEG', '64-D → $\\tilde{\\mathbf{e}}$'), ('et', ORANGE_E, (0, (3, 1.5)), 'Eye tracking', '64-D → $\\tilde{\\mathbf{o}}$')]:
    box(PX, rows[key], PW, H, ec, 'white', ls=ls); label(pcx, rows[key] + 6.8, t1, size=8.5, weight='bold', color=ec); label(pcx, rows[key] + 2.8, t2, size=8)
    arrow((IX + IW, rows[key] + H/2), (PX, rows[key] + H/2))
FX, FY, FW, FH = 55.5, 42.5, 42.5, 46.5; box(FX, FY, FW, FH, GREEN, '#F4F9F5', lw=1.0, r=1.6); fcx = FX + FW/2
label(fcx, 86.6, 'Residual gated fusion', size=8.5, weight='bold', color=GREEN)
box(FX + 3.0, rows['beh'] + 1.5, FW - 6.0, 7.0, GREEN, 'white'); label(fcx + 1.0, rows['beh'] + 5.0, 'behaviour base $\\mathbf{b}$', size=8)
gx = FX + 6.0
for key, ec in [('eeg', BLUE), ('et', ORANGE_E)]:
    cy = rows[key] + H/2; ax.add_patch(Circle((gx, cy), 2.2, ec=ec, fc='white', lw=0.9, zorder=3)); label(gx, cy, '×', size=9, color=ec)
    bw = FW - 6.0 - 6.0 - 3.0; box(gx + 5.0, cy - 3.5, bw, 7.0, ec, 'white', ls=(0, (3, 1.5)))
    sym = '\\tilde{\\mathbf{e}}' if key == 'eeg' else '\\tilde{\\mathbf{o}}'; label(gx + 5.0 + bw/2, cy, f'$g(\\mathbf{{b}}, v)\\cdot {sym}$', size=8)
    arrow((PX + PW, cy), (gx - 2.2, cy), color=ec); ax.plot([gx, gx], [rows['beh'] + 1.5, cy + 2.2], color=GREEN, lw=0.6, ls=(0, (1, 1.5)), zorder=3)
arrow((PX + PW, rows['beh'] + H/2), (FX + 3.0, rows['beh'] + H/2), color=GREEN)
box(FX + 3.0, FY + 1.8, FW - 6.0, 6.2, GREEN, '#DDEFE3'); label(fcx, FY + 4.9, 'sum → LayerNorm → $\\mathbf{z}(j)$', size=8)
ax.plot([fcx, fcx], [rows['beh'] + 1.5, FY + 8.0], color=GREEN, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
y2, h2 = 20.0, 17.5; gcx = 17.0
ax.plot([fcx, fcx, gcx, gcx], [FY, 40.8, 40.8, y2 + h2 + 0.5], color=INK_HEX, lw=0.9, zorder=3); arrow((gcx, y2 + h2 + 3.0), (gcx, y2 + h2))
box(3.0, y2, 28.0, h2, GREY, GREY_F, lw=1.0); label(gcx, y2 + 13.2, 'GRU', size=8.5, weight='bold', color=GREY)
label(gcx, y2 + 8.5, 'unidirectional, 64 units', size=8); label(gcx, y2 + 4.2, 'state carried across visits', size=8)
ax.add_patch(FancyArrowPatch((22.5, y2 + h2), (27.5, y2 + h2), connectionstyle='arc3,rad=-1.0', arrowstyle='-|>', mutation_scale=5, color=GREY, lw=0.8, zorder=3))
label(9.5, y2 + h2 + 0.8, '$s(j-1)$', size=8, style='italic', color=GREY, va='bottom')
box(38.5, y2, 26.0, h2, RED, RED_F, lw=1.0); label(51.5, y2 + 13.2, 'Sigmoid head', size=8.5, weight='bold', color=RED)
label(51.5, y2 + 8.5, 'visit score $h(j) \\in (0,1)$', size=8); label(51.5, y2 + 4.2, 'prefix-supervised, Eq. (2)', size=8)
arrow((31.0, y2 + h2/2), (38.5, y2 + h2/2))
RX, RW = 71.0, 27.5; box(RX, y2 - 1.0, RW, h2 + 1.5, NAVY_E, NAVY_F, lw=1.0); rcx = RX + RW/2
label(rcx, y2 + 14.0, 'Cumulative risk', size=8.5, weight='bold', color=NAVY_E); label(rcx, y2 + 8.0, '$p(t) = 1 - \\prod_{j=1}^{t}(1 - h(j))$', size=8)
label(rcx, y2 + 2.4, 'nondecreasing, Eq. (1)', size=8); arrow((64.5, y2 + h2/2), (RX, y2 + h2/2)); label(67.7, y2 + h2/2 + 2.4, '$h(j)$', size=8, style='italic')
ty = 5.5; ax.plot([3.0, 92.0], [ty, ty], color=INK_HEX, lw=0.9, zorder=3); arrow((92.0, ty), (97.5, ty))
for x, s in [(8.0, 'visit 1'), (30.0, 'visit 2'), (52.0, 'visit 3'), (72.0, 'visit $t$')]:
    ax.plot(x, ty, 'o', color=INK_HEX, ms=3.2, zorder=4); label(x, ty + 3.4, s, size=8)
ax.plot([86.0, 86.0], [ty - 2.4, ty + 2.4], color=RED, lw=1.0, ls=(0, (2, 1.2)), zorder=4); label(86.0, ty + 3.8, 'audited cutoff', size=8, color=RED)
fig.savefig(os.path.join(OUT, 'fig2.png'), dpi=600, facecolor='white'); print('fig2 done')
