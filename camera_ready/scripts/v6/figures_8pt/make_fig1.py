"""Fig 1 at placed width (5.15 in) with >= 8 pt type, from results/audit_full_42_v2/subject_product_audit.csv."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib import font_manager
from figlib import *
font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'axes.titlesize': 9, 'text.color': INK_HEX, 'axes.labelcolor': INK_HEX,
                     'xtick.color': INK_HEX, 'ytick.color': INK_HEX, 'axes.edgecolor': INK_HEX})
a = pd.read_csv(r'D:\PRIYA\ICAIN\neuroclick_repo\results\audit_full_42_v2\subject_product_audit.csv'); c = a[a.considered == 1]
bins = pd.cut(c.n_gaze_visits, [0, 1, 2, 3, 4, 5, 10, 10**6], labels=['1', '2', '3', '4', '5', '6-10', '11+'])
ct = pd.crosstab(bins, c.label_buy, normalize='columns') * 100
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.15, 2.1), dpi=600, gridspec_kw={'width_ratios': [1.15, 1]})
x = np.arange(len(ct)); w = 0.38
ax.bar(x - w/2, ct[0].values, w, color=NAVY_HEX, label='NoBuy', zorder=3)
ax.bar(x + w/2, ct[1].values, w, color='#D9534F', label='Buy', zorder=3)
ax.set_xticks(x); ax.set_xticklabels(ct.index); ax.set_xlabel('Observed visits per product sequence'); ax.set_ylabel('Within-class percentage')
ax.set_title('Sequence-length distribution', fontweight='bold'); ax.legend(frameon=False, handlelength=1.2)
ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#DDE6EA', lw=0.6, zorder=0); ax.set_axisbelow(True)
hz = ['First\nvisit', 'First\ntwo', 'First\nthree', 'Complete\nhistory']; alln = [5715, 4332, 2835, 5715]; buy = [746, 653, 487, 746]
x = np.arange(4)
bx.bar(x - w/2, alln, w, color=TEAL_HEX, label='All instances', zorder=3); bx.bar(x + w/2, buy, w, color=PINK_HEX, label='Buy instances', zorder=3)
for i in range(4):
    bx.text(x[i] - w/2, alln[i] + 120, f'{alln[i]:,}', ha='center', va='bottom', fontsize=8, color=TEAL_HEX, bbox=dict(fc='white', ec='none', pad=0.4))
    bx.text(x[i] + w/2, buy[i] + 120, f'{buy[i]:,}', ha='center', va='bottom', fontsize=8, color=PINK_HEX, bbox=dict(fc='white', ec='none', pad=0.4))
bx.set_xticks(x); bx.set_xticklabels(hz); bx.set_ylabel('Participant-product instances'); bx.set_ylim(0, 7000)
bx.set_title('Available instances by horizon', fontweight='bold'); bx.legend(frameon=False, handlelength=1.2, loc='upper center')
bx.spines[['top', 'right']].set_visible(False); bx.grid(axis='y', color='#DDE6EA', lw=0.6, zorder=0); bx.set_axisbelow(True)
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(os.path.join(OUT, 'fig1.png'), dpi=600); print('fig1 done', ct.round(1).to_dict())
