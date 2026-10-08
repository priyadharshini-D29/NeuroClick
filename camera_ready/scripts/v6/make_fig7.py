import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from figlib import *

font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'axes.titlesize': 9,
                     'text.color': INK_HEX, 'axes.labelcolor': INK_HEX, 'xtick.color': INK_HEX,
                     'ytick.color': INK_HEX, 'axes.edgecolor': INK_HEX})

labels = ['First visit', 'First two', 'First three', 'Complete history']
# Participant-mean PR AUC. Values printed in the paper text are used where available; the
# remaining bars are read from the original figure raster (tick-label calibrated, +/-0.001).
nc = [.3532, .4703, .5283, .5558]
base = [.3557, .4527, .5133, .6176]
ens = [.3612, .4713, .5303, .6195]
ann = ['n.s. (Holm p = 1.00)', 'n.s. (Holm p = 1.00)', 'n.s. (Holm p = 1.00)', 'Holm p < 0.0001']

fig, ax = plt.subplots(figsize=(4.8, 2.9), dpi=600)
x = np.arange(4); w = 0.26
b1 = ax.bar(x - w, nc, w, color=TEAL_HEX, label='NeuroClick (GRU)', zorder=3)
b2 = ax.bar(x, base, w, color=NAVY_HEX, label='Dwell + propensity', zorder=3)
b3 = ax.bar(x + w, ens, w, color=ORANGE_HEX, label='Ensemble (average)', zorder=3)
for i in range(4):
    top = max(nc[i], base[i], ens[i])
    ax.text(x[i], top + 0.012, 'Ensemble vs NeuroClick\n' + ann[i], ha='center', va='bottom', fontsize=6.8,
            linespacing=1.25, color=INK_HEX)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 0.78); ax.set_yticks(np.arange(0, 0.71, 0.1))
ax.set_ylabel('Participant-mean PR AUC')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#DDE6EA', lw=0.6, zorder=0); ax.set_axisbelow(True)
ax.legend(loc='upper left', frameon=False, ncol=1, handlelength=1.4, handleheight=1.0, borderaxespad=0.2)
ax.set_title('NeuroClick vs dwell + propensity baseline vs ensemble', fontweight='bold')
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, 'fig7.png'), dpi=600)
print('fig7 done')

