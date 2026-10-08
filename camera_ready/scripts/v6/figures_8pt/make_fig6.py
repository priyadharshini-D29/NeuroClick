"""Fig 6 at placed width (4.57 in) with >= 8 pt type, from per_subject_metrics.csv and primary_pairwise_tests.csv."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib import font_manager
from figlib import *
font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.titlesize': 9, 'text.color': INK_HEX, 'axes.labelcolor': INK_HEX, 'xtick.color': INK_HEX,
                     'ytick.color': INK_HEX, 'axes.edgecolor': INK_HEX})
R = r'D:\PRIYA\ICAIN\neuroclick_repo\results\neuroclick_primary_v1'
d = pd.read_csv(fr'{R}\per_subject_metrics.csv'); d = d[d.horizon == 'first1']
t = pd.read_csv(fr'{R}\primary_pairwise_tests.csv'); t = t[t.horizon == 'first1']
models = [('behavior', 'Neuro-\nClick', TEAL_HEX), ('logreg_behavior', 'Logistic', NAVY_HEX), ('total_dwell', 'Total\ndwell', PINK_HEX), ('logreg_dwell_propensity', 'Dwell +\npropensity', ORANGE_HEX)]
fig, axes = plt.subplots(1, 2, figsize=(4.57, 2.45), dpi=600)
rng = np.random.default_rng(0)
for ax, metric, title in [(axes[0], 'pr_auc', 'Participant PR AUC'), (axes[1], 'mcc', 'Participant MCC')]:
    piv = d.pivot(index='subject', columns='model', values=metric)[[m for m, _, _ in models]]
    xs = np.arange(len(models)); jit = rng.uniform(-0.14, 0.14, size=len(piv))
    for row, j in zip(piv.values, jit): ax.plot(xs + j, row, color='#9FB6C2', lw=0.35, alpha=0.6, zorder=1)
    for i, (m, lab, col) in enumerate(models):
        ax.scatter(np.full(len(piv), i) + jit, piv[m].values, s=7, color=col, zorder=3, linewidths=0)
        mu = piv[m].mean(); ax.plot([i - 0.28, i + 0.28], [mu, mu], color='black', lw=1.3, zorder=4)
        ax.text(i, mu + 0.012, f'{mu:.4f}', fontsize=8, fontweight='bold', va='bottom', ha='center', zorder=6, bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.9))
    ax.set_xticks(xs); ax.set_xticklabels([lab for _, lab, _ in models]); ax.set_title(title, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#DDE6EA', lw=0.6, zorder=0); ax.set_axisbelow(True)
    ps = {r.comparator: r.p_holm for r in t[t.metric == metric].itertuples()}
    fmt = lambda v: f'{v:.4f}' if v < 0.01 else f'{v:.2g}'
    txt = 'Holm p vs Logistic {}, Dwell {},\nDwell + propensity {}'.format(fmt(ps['logreg_behavior']), fmt(ps['total_dwell']), fmt(ps['logreg_dwell_propensity']))
    ax.set_xlabel(txt, fontsize=8, labelpad=4)
axes[0].set_ylabel('PR AUC'); axes[1].set_ylabel('MCC')
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig(os.path.join(OUT, 'fig6.png'), dpi=600); print('fig6 done')
