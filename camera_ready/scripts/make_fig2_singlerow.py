"""Fig 2 reproduced in the original wide single-row geometry, with text-consistent labels."""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
from matplotlib import font_manager
from figlib import *

font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 5, 'text.color': INK_HEX, 'mathtext.fontset': 'dejavusans'})

GREEN, GREEN_F = '#2E7D4F', '#EAF4EE'
BLUE, BLUE_F = '#2E5FB5', '#E9EEF9'
ORANGE_E, ORANGE_F = '#B9780F', '#FBF2E0'
GREY, GREY_F = '#3C3C3C', '#F2F2F2'
RED, RED_F = '#C0392B', '#FCEBE9'
NAVY_E, NAVY_F = '#1F3F5E', '#E6EEF5'
MUTED = '#4A5A64'
H, T, S = 5.0, 4.4, 3.9          # header / title / detail sizes (pt)
LW = 0.7                          # box stroke
DASH = (0, (2.6, 1.4))

fig = plt.figure(figsize=(4.8, 2.23), dpi=600)   # aspect ~2.15, close to the original
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')


def box(x, y, w_, h_, edge, face, ls='-', lw=LW, r=1.2, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w_, h_, boxstyle=f'round,pad=0,rounding_size={r}', ec=edge, fc=face,
                                lw=lw, ls=ls, zorder=z, mutation_aspect=2.15))


def label(x, y, s, size=S, weight='normal', color=INK_HEX, ha='center', va='center', style='normal', z=4):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style, zorder=z)


def arrow(p0, p1, color=INK_HEX, lw=0.7, ms=4.5, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=3))


# ---- geometry (percent of width / height) ----
IX, IW = 2.5, 19.0
PX, PW = 24.5, 9.0
FX, FW = 36.5, 17.5
GX, GW = 56.5, 11.5
SX, SW = 70.5, 10.0
RX, RW = 84.0, 14.5
RH = 13.0
rows = {'beh': 68.3, 'eeg': 49.3, 'et': 30.3}
mid = {k: v + RH / 2 for k, v in rows.items()}
YC = mid['eeg']                       # main flow line
FRAME = (1.2, 22.0, 80.8, 67.0)

# headers
for cx, s, c, ww in [(IX + IW / 2, 'VISIT INPUTS', GREEN, 6.0), (PX + PW / 2, 'PROJECTIONS', GREEN, 5.7),
                     (FX + FW / 2, 'GATED FUSION', GREEN, 6.2), (GX + GW / 2, 'TEMPORAL', INK_HEX, 4.6),
                     (SX + SW / 2, 'VISIT SCORE', RED, 5.6), (RX + RW / 2, 'BUY RISK', NAVY_E, 4.2)]:
    label(cx, 96.0, s, size=H, weight='bold', color=c)
    ax.plot([cx - ww, cx + ww], [92.4, 92.4], color=c, lw=0.8)

# frame
box(*FRAME, '#9AA5AD', 'white', ls=(0, (3, 2)), lw=0.6, r=1.6, z=1)
label(FRAME[0] + 1.8, FRAME[1] + FRAME[3] - 3.6, 'per completed visit $j = 1\\ldots t$   forward-only (causal)',
      size=S, ha='left', style='italic', color=MUTED)


# ---- icons, drawn left of the text like the original ----
def icon_browser(x, y, c):
    ax.add_patch(FancyBboxPatch((x - 1.4, y - 3.0), 2.8, 6.0, boxstyle='round,pad=0,rounding_size=0.3', ec=c, fc='white',
                                lw=0.5, zorder=4, mutation_aspect=2.15))
    ax.plot([x - 1.4, x + 1.4], [y + 1.3, y + 1.3], color=c, lw=0.5, zorder=5)
    # filled mouse-pointer arrow pointing to the lower right (unit cursor shape, mirrored)
    pts = [(1, 0), (1, 0.85), (0.78, 0.65), (0.63, 1), (0.5, 0.94), (0.64, 0.6), (0.38, 0.6)]
    x0, y0, cw, chh = x - 0.75, y - 2.55, 1.55, 1.55 * 2.15
    poly = [(x0 + u * cw, y0 + v * chh) for u, v in pts]
    ax.add_patch(plt.Polygon(poly, closed=True, fc=c, ec='white', lw=0.25, zorder=6))

def icon_eeg(x, y, c):
    ax.add_patch(Ellipse((x, y), 3.0, 6.4, ec=c, fc='white', lw=0.5, zorder=4))
    t = np.linspace(-1.0, 1.0, 60)
    ax.plot(x + t, y + 1.6 * np.sin(t * 8) * np.exp(-3 * t * t), color=c, lw=0.5, zorder=5)

def icon_eye(x, y, c):
    # circle (as for EEG) containing an almond-shaped eye with a filled pupil
    ax.add_patch(Ellipse((x, y), 3.0, 6.4, ec=c, fc='white', lw=0.5, zorder=4))
    t = np.linspace(-1.0, 1.0, 80)
    ax.plot(x + t, y + 1.5 * (1 - t * t), color=c, lw=0.55, zorder=5)
    ax.plot(x + t, y - 1.5 * (1 - t * t), color=c, lw=0.55, zorder=5)
    ax.add_patch(Ellipse((x, y), 1.15, 2.45, ec=c, fc='white', lw=0.45, zorder=6))   # iris ring
    ax.add_patch(Ellipse((x, y), 0.62, 1.32, ec=c, fc=c, lw=0.3, zorder=7))          # pupil
    ax.add_patch(Ellipse((x + 0.17, y + 0.42), 0.2, 0.42, ec='none', fc='white', zorder=8))  # highlight


# ---- input boxes ----
specs = [('beh', GREEN, GREEN_F, '-', 'Browsing behaviour', '6 inputs · incl. propensity', '(cross-fitted, Sect. 3.3)', icon_browser),
         ('eeg', BLUE, BLUE_F, DASH, 'EEG', 'optional · 153 inputs', None, icon_eeg),
         ('et', ORANGE_E, ORANGE_F, DASH, 'Eye tracking', 'optional · 70 inputs', None, icon_eye)]
for key, ec, fc, ls, t1, t2, t3, ic in specs:
    y = rows[key]; box(IX, y, IW, RH, ec, fc, ls=ls)
    ic(IX + 2.6, mid[key], ec)
    tx = IX + 4.8 + (IW - 4.8) / 2
    if t3:
        label(tx, y + 9.9, t1, size=T, weight='bold', color=ec)
        label(tx, y + 6.3, t2, size=S - 0.3, style='italic')
        label(tx, y + 2.9, t3, size=S - 0.4, style='italic', color=ec)
    else:
        label(tx, y + 8.6, t1, size=T, weight='bold', color=ec)
        label(tx, y + 4.4, t2, size=S - 0.2, style='italic')

# ---- projection boxes ----
PH = 10.5
for key, ec, ls, t1, sym in [('beh', GREEN, '-', 'Behaviour', '\\mathbf{b}'), ('eeg', BLUE, DASH, 'EEG', '\\tilde{\\mathbf{e}}'),
                             ('et', ORANGE_E, DASH, 'Eye tracking', '\\tilde{\\mathbf{o}}')]:
    y = mid[key] - PH / 2; box(PX, y, PW, PH, ec, 'white', ls=ls)
    label(PX + PW / 2, y + 7.2, t1, size=(T - 0.5 if len(t1) > 9 else T), weight='bold', color=ec)
    label(PX + PW / 2, y + 3.2, f'64-D → ${sym}$', size=S)
    arrow((IX + IW, mid[key]), (PX, mid[key]))

# ---- gated fusion block ----
FY, FH = 24.5, 62.5
box(FX, FY, FW, FH, GREEN, '#F4F9F5', lw=0.8, r=1.6)
fcx = FX + FW / 2
label(fcx, FY + FH - 3.6, 'Residual gated fusion', size=T, weight='bold', color=GREEN)
bb_h = 6.0
box(FX + 1.6, mid['beh'] - bb_h / 2, FW - 3.2, bb_h, GREEN, 'white')
label(fcx, mid['beh'], 'behaviour base  $\\mathbf{b}$', size=S)
arrow((PX + PW, mid['beh']), (FX + 1.6, mid['beh']), color=GREEN)
gx = FX + 3.0
label(gx + 0.4, (mid['beh'] + mid['eeg']) / 2 - 0.5, 'optional\nmodalities', size=S - 0.6, style='italic', color=MUTED)
GB_H = 6.5
for key, ec, sym in [('eeg', BLUE, '\\tilde{\\mathbf{e}}'), ('et', ORANGE_E, '\\tilde{\\mathbf{o}}')]:
    cy = mid[key]
    ax.add_patch(Ellipse((gx, cy), 2.6, 5.6, ec=ec, fc='white', lw=LW, zorder=4))
    label(gx, cy, '×', size=6, color=ec, z=5)
    bx0 = gx + 2.8; bw = FX + FW - 1.6 - bx0
    box(bx0, cy - GB_H / 2, bw, GB_H, ec, 'white', ls=DASH)
    label(bx0 + bw / 2, cy, f'$g(\\mathbf{{b}}, v)\\cdot {sym}$', size=S)
    arrow((PX + PW, cy), (gx - 1.3, cy), color=ec)
    arrow((gx + 1.3, cy), (bx0, cy), color=ec, ms=3.0, ls=(0, (1.3, 1.1)))
for xoff, key in [(-2.2, 'eeg'), (2.2, 'et')]:
    ax.plot([fcx + xoff, fcx + xoff], [mid['beh'] - bb_h / 2, mid[key] + GB_H / 2], color=INK_HEX, lw=0.4, ls=(0, (1, 1.6)), zorder=3)
sy = (mid['eeg'] + mid['et']) / 2
ax.add_patch(Ellipse((fcx - 2.2, sy), 2.4, 5.2, ec=GREEN, fc='white', lw=LW, zorder=4))
label(fcx - 2.2, sy, '+', size=6, color=GREEN, z=5)
label(fcx - 0.2, sy, 'residual sum', size=S - 0.5, style='italic', color=MUTED, ha='left')
ax.plot([fcx - 2.2, fcx - 2.2], [mid['eeg'] - GB_H / 2, mid['et'] + GB_H / 2], color=INK_HEX, lw=0.4, ls=(0, (1, 1.6)), zorder=3)
box(FX + 1.6, FY + 2.2, FW - 3.2, 5.6, GREEN, '#DDEFE3')
label(fcx, FY + 5.0, 'LayerNorm → $\\mathbf{z}$', size=S)

# ---- GRU ----
GH = 19.0; gy = YC - GH / 2
box(GX, gy, GW, GH, GREY, GREY_F, lw=0.8)
gcx = GX + GW / 2
label(gcx, gy + GH - 4.0, 'GRU', size=T + 1.0, weight='bold', color=GREY)
for k, s in enumerate(['unidirectional · 64 units', 'state carried across visits']):
    label(gcx, gy + GH - 8.4 - 3.9 * k, s, size=S - 0.6)
arrow((FX + FW, YC), (GX, YC))
loop = FancyArrowPatch((GX + 3.0, gy + GH), (GX + GW - 3.0, gy + GH), connectionstyle='arc3,rad=-0.75',
                       arrowstyle='-|>', mutation_scale=4, color=GREY, lw=0.6, zorder=3)
ax.add_patch(loop)
label(gcx, gy + GH + 8.0, '$s(j-1)$', size=S, style='italic')

# ---- sigmoid head ----
SH = 15.5; sy0 = YC - SH / 2
box(SX, sy0, SW, SH, RED, RED_F, lw=0.8)
scx = SX + SW / 2
label(scx, sy0 + SH - 3.4, 'Sigmoid head', size=T, weight='bold', color=RED)
label(scx, sy0 + 8.0, 'visit score  $h(j)$', size=S)
label(scx, sy0 + 4.7, 'prefix-supervised', size=S - 0.6, style='italic', color=RED)
label(scx, sy0 + 1.9, 'Eq. (2)', size=S - 0.6, style='italic', color=RED)
arrow((GX + GW, YC), (SX, YC))

# ---- cumulative risk (outside frame) ----
RBH = 29.0; ry0 = YC - RBH / 2
box(RX, ry0, RW, RBH, NAVY_E, NAVY_F, lw=0.8, r=1.6)
rcx = RX + RW / 2
label(rcx, ry0 + RBH - 3.6, 'Cumulative risk', size=T, weight='bold', color=NAVY_E)
label(rcx, ry0 + RBH - 12.0, '$p(t)=1-\\prod_{j=1}^{t}(1-h(j))$', size=S + 0.1)
label(rcx, ry0 + 9.0, 'nondecreasing · Eq. (1)', size=S - 0.4)
sx, sy_ = rcx - 4.0, ry0 + 2.6
xs = [sx, sx + 1.6, sx + 1.6, sx + 3.2, sx + 3.2, sx + 4.8, sx + 4.8, sx + 6.4, sx + 6.4, sx + 8.0]
ys = [sy_, sy_, sy_ + 1.1, sy_ + 1.1, sy_ + 2.4, sy_ + 2.4, sy_ + 3.1, sy_ + 3.1, sy_ + 4.0, sy_ + 4.0]
ax.plot(xs, ys, color=NAVY_E, lw=0.8, zorder=4)
ax.plot([sx, sx + 8.0], [sy_ - 0.6, sy_ - 0.6], color='#9AA5AD', lw=0.45, zorder=4)
arrow((SX + SW, YC), (RX, YC))

# ---- timeline ----
ty = 12.0
ax.plot([3.0, 92.0], [ty, ty], color=INK_HEX, lw=0.7, zorder=3)
arrow((92.0, ty), (97.0, ty))
for x, s in [(7.0, 'visit 1'), (30.0, 'visit 2'), (53.0, 'visit 3'), (74.0, 'visit $t$')]:
    ax.plot(x, ty, 'o', color=INK_HEX, ms=2.4, zorder=4)
    label(x, ty + 4.6, s, size=S)
ax.plot([84.0, 84.0], [ty - 3.4, ty + 3.4], color=RED, lw=0.8, ls=(0, (2, 1.2)), zorder=4)
label(84.0, ty + 4.8, 'audited cutoff', size=S - 0.3, color=RED)
label(50.0, 4.0, 'Each completed visit updates the risk estimate from its own history only; risk never decreases, '
      'and no information beyond the cutoff is used.', size=S + 0.2, color=INK_HEX)

fig.savefig(os.path.join(OUT, 'fig2_singlerow.png'), dpi=600, facecolor='white')
print('fig2 single-row done')
