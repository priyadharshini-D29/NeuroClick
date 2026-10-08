import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from figlib import *

font_manager.fontManager.addfont(ARIAL); font_manager.fontManager.addfont(ARIAL_BOLD)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.titlesize': 9, 'text.color': INK_HEX, 'axes.labelcolor': INK_HEX,
                     'xtick.color': INK_HEX, 'ytick.color': INK_HEX, 'axes.edgecolor': INK_HEX})

vals = np.array([[.2736, .3742, .4132, .4429],
                 [.3003, .3832, .3983, .4286],
                 [.2726, .3666, .3962, .4269],
                 [.2651, .3545, .3946, .4142]])
rows = ['Behaviour', 'Behaviour + ET', 'Behaviour + EEG', 'Behaviour + EEG + ET']
cols = ['First visit', 'First two', 'First three', 'Complete\nhistory']

cmap = LinearSegmentedColormap.from_list('teal_seq', ['#E3F1F1', '#7FBDB8', '#1F8A80', '#0E4F4C'])
fig, ax = plt.subplots(figsize=(4.65, 2.2), dpi=600)
im = ax.imshow(vals, cmap=cmap, vmin=0.26, vmax=0.45, aspect='auto')
for i in range(4):
    for j in range(4):
        v = vals[i, j]
        ax.text(j, i, f'{v:.4f}', ha='center', va='center', fontsize=8.5, fontweight='bold',
                color='white' if v > 0.345 else INK_HEX)
ax.set_xticks(range(4)); ax.set_xticklabels(cols)
ax.set_yticks(range(4)); ax.set_yticklabels(rows)
ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
ax.grid(which='minor', color='white', lw=1.5)
ax.tick_params(which='both', length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.set_label('Pooled PR AUC', fontsize=8)
cb.ax.tick_params(labelsize=7.5)
cb.outline.set_visible(False)
ax.set_title('Modality ablation: pooled PR AUC', fontweight='bold')
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, 'fig5.png'), dpi=600)
print('fig5 done')


