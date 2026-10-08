"""Fig 2, clean two-row version: one box per modality, one fusion box, one temporal row, timeline. All text >= 8 pt."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
from figlib import *
font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'text.color': INK_HEX, 'mathtext.fontset': 'dejavusans'})
GREEN, GREEN_F = '#2E7D4F', '#EAF4EE'; BLUE, BLUE_F = '#2E5FB5', '#E9EEF9'; ORANGE_E, ORANGE_F = '#B9780F', '#FBF2E0'
GREY, GREY_F = '#3C3C3C', '#F2F2F2'; RED, RED_F = '#C0392B', '#FCEBE9'; NAVY_E, NAVY_F = '#1F3F5E', '#E6EEF5'; MUTED = '#4A5A64'
W_IN, H_IN = 4.8, 3.12
fig = plt.figure(figsize=(W_IN, H_IN), dpi=600); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(-13.5, 100); ax.axis('off')
ASP = W_IN / H_IN * (113.5 / 100)
def box(x, y, w, h, edge, face, ls='-', lw=0.9): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.4', ec=edge, fc=face, lw=lw, ls=ls, zorder=2, mutation_aspect=ASP))
def label(x, y, s, size=8, weight='normal', color=INK_HEX, ha='center', va='center', style='normal'): ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style, zorder=4)
def arrow(p0, p1, color=INK_HEX, lw=0.9): ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=7, color=color, lw=lw, zorder=3, shrinkA=0, shrinkB=0))
DASH = (0, (3, 1.6))
label(2.0, 97.0, 'per completed visit $j = 1 \\ldots t$ (forward-only)', size=8, ha='left', va='top', style='italic', color=MUTED)
# row 1: modality boxes
IX, IW, IH = 2.0, 34.0, 15.0; rows = {'beh': 72.0, 'eeg': 53.5, 'et': 35.0}
for key, ec, fc, ls, t1, t2 in [('beh', GREEN, GREEN_F, '-', 'Browsing behaviour', '6 inputs → 64-D $\\mathbf{b}$'),
                                ('eeg', BLUE, BLUE_F, DASH, 'EEG (optional)', '153 inputs → 64-D $\\tilde{\\mathbf{e}}$'),
                                ('et', ORANGE_E, ORANGE_F, DASH, 'Eye tracking (optional)', '70 inputs → 64-D $\\tilde{\\mathbf{o}}$')]:
    y = rows[key]; box(IX, y, IW, IH, ec, fc, ls=ls)
    label(IX + IW/2, y + IH - 4.6, t1, size=8.5, weight='bold', color=ec); label(IX + IW/2, y + 4.4, t2, size=8)
# fusion box
FX, FW, FY, FH = 48.0, 50.0, 36.0, 51.0
box(FX, FY, FW, FH, GREEN, '#F4F9F5', lw=1.0); fcx = FX + FW/2
label(fcx, FY + FH - 5.0, 'Gated residual fusion', size=8.5, weight='bold', color=GREEN)
label(fcx, FY + FH - 15.0, 'base: $\\mathbf{b}$', size=8)
label(fcx, FY + FH - 25.0, '+ $g(\\mathbf{b}, v)\\cdot\\tilde{\\mathbf{e}}$  (EEG, validity-gated)', size=8, color=BLUE)
label(fcx, FY + FH - 35.0, '+ $g(\\mathbf{b}, v)\\cdot\\tilde{\\mathbf{o}}$  (eye tracking, validity-gated)', size=8, color=ORANGE_E)
label(fcx, FY + 5.5, 'LayerNorm → $\\mathbf{z}(j)$', size=8)
for key, ec in [('beh', GREEN), ('eeg', BLUE), ('et', ORANGE_E)]:
    arrow((IX + IW, rows[key] + IH/2), (FX, rows[key] + IH/2), color=ec)
# connector down to GRU
y2, h2 = 6.0, 19.0; gcx = 19.0
ax.plot([fcx, fcx, gcx, gcx], [FY, 31.5, 31.5, y2 + h2 + 0.5], color=INK_HEX, lw=0.9, zorder=3); arrow((gcx, y2 + h2 + 2.5), (gcx, y2 + h2))
label(fcx + 1.5, 32.0, '$\\mathbf{z}(j)$', size=8, style='italic', ha='left', va='bottom', color=MUTED)
# row 2
box(2.0, y2, 34.0, h2, GREY, GREY_F, lw=1.0); label(gcx, y2 + h2 - 4.6, 'GRU, 64 units', size=8.5, weight='bold', color=GREY); label(gcx, y2 + 5.0, 'state carried across visits', size=8)
ax.add_patch(FancyArrowPatch((27.0, y2 + h2), (33.0, y2 + h2), connectionstyle='arc3,rad=-0.6', arrowstyle='-|>', mutation_scale=6, color=GREY, lw=0.8, zorder=3))
label(34.0, y2 + h2 + 1.0, '$s(j-1)$', size=8, style='italic', color=GREY, ha='left', va='bottom')
box(41.0, y2, 26.0, h2, RED, RED_F, lw=1.0); label(54.0, y2 + h2 - 4.6, 'Sigmoid head', size=8.5, weight='bold', color=RED); label(54.0, y2 + 5.0, 'visit score $h(j)$, Eq. (2)', size=8)
arrow((36.0, y2 + h2/2), (41.0, y2 + h2/2))
box(72.0, y2 - 2.0, 26.0, h2 + 4.0, NAVY_E, NAVY_F, lw=1.0); label(85.0, y2 + h2 - 2.4, 'Cumulative risk', size=8.5, weight='bold', color=NAVY_E)
label(85.0, y2 + 9.0, '$p(t) = 1 - \\prod_{j \\leq t}(1 - h(j))$', size=8); label(85.0, y2 + 1.6, 'nondecreasing, Eq. (1)', size=8)
arrow((67.0, y2 + h2/2), (72.0, y2 + h2/2)); label(69.5, y2 + h2/2 + 2.0, '$h(j)$', size=8, style='italic', va='bottom')
# timeline
ty = -9.0; ax.plot([2.0, 92.0], [ty, ty], color=INK_HEX, lw=0.9, zorder=3); arrow((92.0, ty), (98.0, ty))
for x, s in [(8.0, 'visit 1'), (30.0, 'visit 2'), (52.0, 'visit 3'), (72.0, 'visit $t$')]:
    ax.plot(x, ty, 'o', color=INK_HEX, ms=3.2, zorder=4); label(x, ty + 2.2, s, size=8, va='bottom')
ax.plot([86.0, 86.0], [ty - 2.2, ty + 2.2], color=RED, lw=1.0, ls=(0, (2, 1.2)), zorder=4); label(86.0, ty + 2.4, 'audited cutoff', size=8, color=RED, va='bottom')
fig.savefig(os.path.join(OUT, 'fig2.png'), dpi=600, facecolor='white'); print('fig2 clean done')
