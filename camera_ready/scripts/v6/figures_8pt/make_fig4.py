import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from figlib import *

font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 9,
                     'text.color': INK_HEX, 'axes.labelcolor': INK_HEX, 'xtick.color': INK_HEX,
                     'ytick.color': INK_HEX, 'axes.edgecolor': INK_HEX})

labels = ['First visit', 'First two', 'First three', 'Complete history']
nc = [.2736, .3742, .4132, .4429]
lg = [.2450, .3190, .3790, .4652]
pv = [.1305, .1507, .1718, .1305]
deltas = ['+0.0286', '+0.0551', '+0.0343', '−0.0222']

fig, ax = plt.subplots(figsize=(4.53, 2.3), dpi=600)
x = range(4)
ax.plot(x, nc, '-o', color=TEAL_HEX, lw=1.6, ms=5, label='NeuroClick behaviour', zorder=3)
ax.plot(x, lg, '--s', color=NAVY_HEX, lw=1.6, ms=5, label='Logistic behaviour', zorder=3)
ax.plot(x, pv, ':D', color=ORANGE_HEX, lw=1.4, ms=4.5, label='Observed prevalence', zorder=3)
for i in range(4):
    if i < 3:
        ax.annotate(deltas[i], (i, nc[i]), xytext=(0, 7), textcoords='offset points', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=TEAL_HEX)
    else:
        ax.annotate(deltas[i], (i, nc[i]), xytext=(0, -8), textcoords='offset points', ha='center', va='top',
                    fontsize=8, fontweight='bold', color=PINK_HEX)
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylim(0.09, 0.52); ax.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
ax.set_ylabel('Pooled PR AUC')
ax.set_xlim(-0.25, 3.25)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#DDE6EA', lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.33), frameon=True, framealpha=0.95, edgecolor='none', handlelength=2.4)
ax.set_title('PR AUC across observation horizons', fontweight='bold')
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, 'fig4.png'), dpi=600)
print('fig4 done')


