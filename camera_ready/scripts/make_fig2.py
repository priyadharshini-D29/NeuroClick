"""Fig 2: NeuroClick architecture, redrawn in a two-row layout with legible type."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import font_manager
from figlib import *

font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'text.color': INK_HEX, 'mathtext.fontset': 'dejavusans'})

GREEN, GREEN_F = '#2E7D4F', '#EAF4EE'
BLUE, BLUE_F = '#2E5FB5', '#E9EEF9'
ORANGE_E, ORANGE_F = '#B9780F', '#FBF2E0'
GREY, GREY_F = '#3C3C3C', '#F2F2F2'
RED, RED_F = '#C0392B', '#FCEBE9'
NAVY_E, NAVY_F = '#1F3F5E', '#E6EEF5'
MUTED = '#4A5A64'

fig = plt.figure(figsize=(4.8, 3.75), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')


def box(x, y, w, h, edge, face, ls='-', lw=0.9, r=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', ec=edge, fc=face, lw=lw, ls=ls, zorder=2)
    ax.add_patch(p)
    return p


def label(x, y, s, size=7, weight='normal', color=INK_HEX, ha='center', va='center', style='normal'):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style, zorder=4)


def arrow(p0, p1, color=INK_HEX, lw=0.9, style='-|>', ls='-', ms=6):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=3,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)


# ---------------- outer frame for the per-visit block ----------------
box(1.0, 16.0, 98.0, 83.0, '#9AA5AD', 'white', ls=(0, (3, 2)), lw=0.8, r=1.5)
label(3.0, 96.9, 'per completed visit  $j = 1 \\ldots t$   ·   forward-only (causal)', size=6.4, ha='left', style='italic', color=MUTED)

# column headers (row 1)
for x, s in [(13.5, 'VISIT INPUTS'), (37.5, 'PROJECTIONS'), (66.0, 'GATED FUSION')]:
    label(x, 93.0, s, size=6.4, weight='bold', color=GREEN)
    ax.plot([x - 7.5, x + 7.5], [91.5, 91.5], color=GREEN, lw=0.9)

# ---------------- row 1: inputs ----------------
rows = {'beh': 79.5, 'eeg': 68.0, 'et': 56.5}
H = 8.2
IX, IW = 2.5, 22.0      # input boxes
PX, PW = 29.5, 16.0     # projection boxes
icx, pcx = IX + IW / 2, PX + PW / 2

box(IX, rows['beh'], IW, H, GREEN, GREEN_F)
label(icx, rows['beh'] + 5.9, 'Browsing behaviour', size=6.6, weight='bold', color=GREEN)
label(icx, rows['beh'] + 3.3, '6 inputs · incl. propensity logit', size=5.4)
label(icx, rows['beh'] + 1.2, '(cross-fitted, Sect. 3.3)', size=5.2, style='italic', color=GREEN)
box(IX, rows['eeg'], IW, H, BLUE, BLUE_F, ls=(0, (3, 1.5)))
label(icx, rows['eeg'] + 5.4, 'EEG (optional)', size=6.6, weight='bold', color=BLUE)
label(icx, rows['eeg'] + 2.4, '153 inputs · validity gated', size=5.4)
box(IX, rows['et'], IW, H, ORANGE_E, ORANGE_F, ls=(0, (3, 1.5)))
label(icx, rows['et'] + 5.4, 'Eye tracking (optional)', size=6.6, weight='bold', color=ORANGE_E)
label(icx, rows['et'] + 2.4, '70 inputs · validity gated', size=5.4)

for key, ec, ls, t1, t2 in [
        ('beh', GREEN, '-', 'Behaviour', '64-D → $\\mathbf{b}$'),
        ('eeg', BLUE, (0, (3, 1.5)), 'EEG', '64-D → $\\tilde{\\mathbf{e}}$'),
        ('et', ORANGE_E, (0, (3, 1.5)), 'Eye tracking', '64-D → $\\tilde{\\mathbf{o}}$')]:
    box(PX, rows[key], PW, H, ec, 'white', ls=ls)
    label(pcx, rows[key] + 5.4, t1, size=6.6, weight='bold', color=ec)
    label(pcx, rows[key] + 2.3, t2, size=6.2)
    arrow((IX + IW, rows[key] + H / 2), (PX, rows[key] + H / 2))

# fusion block
FX, FY, FW, FH = 51.0, 49.5, 30.5, 40.5
box(FX, FY, FW, FH, GREEN, '#F4F9F5', lw=1.0, r=1.6)
fcx = FX + FW / 2
label(fcx, FY + FH - 1.9, 'Residual gated fusion', size=6.8, weight='bold', color=GREEN)
box(FX + 2.5, rows['beh'] + 1.0, FW - 5.0, 6.0, GREEN, 'white')
label(fcx, rows['beh'] + 4.0, 'behaviour base  $\\mathbf{b}$', size=6.2)
gx = FX + 4.6
for key, ec in [('eeg', BLUE), ('et', ORANGE_E)]:
    cy = rows[key] + H / 2
    c = Circle((gx, cy), 1.8, ec=ec, fc='white', lw=0.9, zorder=3); ax.add_patch(c)
    label(gx, cy, '×', size=8, color=ec)
    box(gx + 4.0, cy - 3.0, FW - 4.0 - 4.6 - 2.5, 6.0, ec, 'white', ls=(0, (3, 1.5)))
    sym = '\\tilde{\\mathbf{e}}' if key == 'eeg' else '\\tilde{\\mathbf{o}}'
    label(gx + 4.0 + (FW - 4.0 - 4.6 - 2.5) / 2, cy, f'$g(\\mathbf{{b}}, v)\\cdot {sym}$', size=6.2)
    arrow((PX + PW, cy), (gx - 1.8, cy), color=ec)
    ax.plot([gx, gx], [rows['beh'] + 1.0, cy + 1.8], color=GREEN, lw=0.6, ls=(0, (1, 1.5)), zorder=3)
arrow((PX + PW, rows['beh'] + H / 2), (FX + 2.5, rows['beh'] + H / 2), color=GREEN)
label(48.3, 66.3, 'optional\nmodalities', size=5.2, style='italic', color=MUTED)
# residual sum + layer norm strip
box(FX + 2.5, FY + 1.8, FW - 5.0, 5.2, GREEN, '#DDEFE3')
label(fcx, FY + 4.4, 'residual sum  →  LayerNorm  →  $\\mathbf{z}(j)$', size=6.0)
ax.plot([fcx, fcx], [rows['beh'] + 1.0, FY + 7.0], color=GREEN, lw=0.6, ls=(0, (1, 1.5)), zorder=1)

# fused vector path down to GRU
y2, h2 = 21.5, 16.5
gcx = 18.0
ax.plot([fcx, fcx, gcx, gcx], [FY, 45.0, 45.0, y2 + h2 + 0.5], color=INK_HEX, lw=0.9, zorder=3)
arrow((gcx, y2 + h2 + 3.0), (gcx, y2 + h2))
label(44.5, 46.6, 'fused visit representation $\\mathbf{z}(j)$', size=6.0, style='italic', color=MUTED)

# ---------------- row 2 ----------------
box(5.0, y2, 26.0, h2, GREY, GREY_F, lw=1.0)
label(gcx, y2 + 13.0, 'GRU', size=7.4, weight='bold', color=GREY)
label(gcx, y2 + 9.4, 'unidirectional · 64 units', size=5.8)
label(gcx, y2 + 6.7, 'hidden state carried', size=5.8)
label(gcx, y2 + 4.2, 'across completed visits', size=5.8)
label(gcx, y2 + 1.5, 'TEMPORAL', size=5.2, weight='bold', color='#6B7378')
loop = FancyArrowPatch((23.5, y2 + h2), (28.5, y2 + h2), connectionstyle='arc3,rad=-1.4', arrowstyle='-|>',
                       mutation_scale=5, color=GREY, lw=0.8, zorder=3)
ax.add_patch(loop)
label(31.5, y2 + h2 + 4.6, 'state $s(j-1)$', size=5.4, style='italic', color=GREY, ha='left')

box(39.0, y2, 24.0, h2, RED, RED_F, lw=1.0)
label(51.0, y2 + 13.0, 'Sigmoid head', size=7.4, weight='bold', color=RED)
label(51.0, y2 + 9.3, 'visit score  $h(j) \\in (0, 1)$', size=6.2)
label(51.0, y2 + 6.2, 'prefix-supervised with the', size=5.6)
label(51.0, y2 + 4.0, 'final Buy/NoBuy label, Eq. (2)', size=5.6)
label(51.0, y2 + 1.5, 'VISIT SCORE', size=5.2, weight='bold', color='#6B7378')
arrow((31.0, y2 + h2 / 2), (39.0, y2 + h2 / 2))

RX, RW = 71.0, 27.5
box(RX, y2 - 1.0, RW, h2 + 2.0, NAVY_E, NAVY_F, lw=1.0)
rcx = RX + RW / 2
label(rcx, y2 + 14.6, 'Cumulative purchase risk', size=7.0, weight='bold', color=NAVY_E)
label(rcx, y2 + 8.6, '$p(t) = 1 - \\prod_{j=1}^{t}\\,(1 - h(j))$', size=6.6)
label(rcx, y2 + 3.4, 'nondecreasing in $t$ · Eq. (1)', size=5.8)
label(rcx, y2 + 0.9, 'BUY RISK', size=5.2, weight='bold', color='#6B7378')
arrow((63.0, y2 + h2 / 2), (RX, y2 + h2 / 2))
label(67.0, y2 + h2 / 2 + 1.7, '$h(j)$', size=6.0, style='italic')

# ---------------- timeline ----------------
ty = 4.8
ax.plot([3.0, 92.0], [ty, ty], color=INK_HEX, lw=0.9, zorder=3)
arrow((92.0, ty), (97.5, ty))
for x, s in [(8.0, 'visit 1'), (30.0, 'visit 2'), (52.0, 'visit 3'), (74.0, 'visit $t$')]:
    ax.plot(x, ty, 'o', color=INK_HEX, ms=3.2, zorder=4)
    label(x, ty + 2.6, s, size=6.0)
ax.plot([86.0, 86.0], [ty - 2.2, ty + 2.2], color=RED, lw=1.0, ls=(0, (2, 1.2)), zorder=4)
label(86.0, ty + 3.0, 'audited cutoff', size=5.8, color=RED)

fig.savefig(os.path.join(OUT, 'fig2.png'), dpi=600, facecolor='white')
print('fig2 done')
