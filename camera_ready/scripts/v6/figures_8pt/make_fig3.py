"""Fig 3 at placed width (5.15 in) with >= 8 pt type, from the saved first1 out-of-fold predictions."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib import font_manager
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
from figlib import *
font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'axes.titlesize': 9, 'text.color': INK_HEX, 'axes.labelcolor': INK_HEX,
                     'xtick.color': INK_HEX, 'ytick.color': INK_HEX, 'axes.edgecolor': INK_HEX})
R = r'D:\PRIYA\ICAIN\neuroclick_repo\results'
n4 = pd.read_csv(fr'{R}\neuroclick_losocv_final_run_04\losocv_predictions.csv'); n4 = n4[(n4.condition == 'behavior') & (n4.horizon == 'first1')]
c3 = pd.read_csv(fr'{R}\classical_losocv_final_run_03\losocv_predictions.csv'); c3 = c3[c3.horizon == 'first1']
series = [('NeuroClick', n4.label.values, n4.probability.values, TEAL_HEX, '-', 'o'),
          ('Logistic', *[c3[c3.model == 'logreg_behavior'][k].values for k in ('label_buy', 'probability_buy')], NAVY_HEX, '--', 's'),
          ('Total dwell', *[c3[c3.model == 'total_dwell'][k].values for k in ('label_buy', 'probability_buy')], PINK_HEX, '-.', 'D'),
          ('Dwell + propensity', *[c3[c3.model == 'logreg_dwell_propensity'][k].values for k in ('label_buy', 'probability_buy')], ORANGE_HEX, ':', '^')]
prev = float(n4.label.mean())
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.15, 2.2), dpi=600)
for name, y, p, col, ls, mk in series:
    pr, rc, _ = precision_recall_curve(y, p); fp, tp, _ = roc_curve(y, p)
    ap, auc = average_precision_score(y, p), roc_auc_score(y, p); print(f'{name:22s} AP {ap:.4f} ROC {auc:.4f}')
    ax.plot(rc, pr, color=col, ls=ls, lw=1.1, label=f'{name} ({ap:.4f})', marker=mk, markevery=max(1, len(rc)//6), ms=3)
    bx.plot(fp, tp, color=col, ls=ls, lw=1.1, label=f'{name} ({auc:.4f})', marker=mk, markevery=max(1, len(fp)//6), ms=3)
ax.axhline(prev, color=ORANGE_HEX, ls=(0, (1, 1.5)), lw=0.9, label=f'Prevalence ({prev:.4f})')
bx.plot([0, 1], [0, 1], color=ORANGE_HEX, ls=(0, (1, 1.5)), lw=0.9, label='Chance')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-recall (PR AUC)', fontweight='bold'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
bx.set_xlabel('False-positive rate'); bx.set_ylabel('True-positive rate'); bx.set_title('ROC (ROC AUC)', fontweight='bold'); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
ax.legend(frameon=True, framealpha=0.95, edgecolor='none', loc='upper right', handlelength=1.8, borderaxespad=0.2); bx.legend(frameon=True, framealpha=0.95, edgecolor='none', loc='lower right', handlelength=1.8, borderaxespad=0.2)
for a in (ax, bx):
    a.spines[['top', 'right']].set_visible(False); a.grid(color='#DDE6EA', lw=0.6, zorder=0); a.set_axisbelow(True)
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig(os.path.join(OUT, 'fig3.png'), dpi=600); print('fig3 done')
