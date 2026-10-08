import numpy as np
from figlib import *

img = load('image7.png')
arr = np.array(img)
dm = dark_mask(arr)
H, W = dm.shape
# left spine: column with most dark pixels in x<700
colsum = dm[:, :700].sum(axis=0)
sx = int(np.argmax(colsum))
print('spine x', sx, 'dark count', colsum[sx])
# tick labels: dark text left of the spine; use the vertical centre of each label
band = dm[:, max(0, sx - 130):sx - 20]
rows = np.where(band.sum(axis=1) > 3)[0]
print('label rows span', rows.min(), rows.max(), 'n', len(rows))
ticks = []
start = rows[0]
for a, b in zip(rows[:-1], rows[1:]):
    if b - a > 8:
        ticks.append((start + a) / 2); start = b
ticks.append((start + rows[-1]) / 2)
print('tick rows', ticks)
# ticks are 0.0 .. 0.8 step 0.1 from bottom to top
ticks = sorted(ticks, reverse=True)
vals = [0.1 * i for i in range(len(ticks))]
p = np.polyfit(ticks, vals, 1)
print('fit slope/intercept', p, 'px per 0.1:', 0.1 / abs(p[0]))
tops = {'NeuroClick': [1019, 788, 674, 619], 'Dwell + propensity': [1014, 823, 703, 500], 'Ensemble (average)': [1003, 786, 670, 500]}
for k, v in tops.items():
    print(k, [round(float(np.polyval(p, y)), 4) for y in v])
print('bottom row 1711 ->', np.polyval(p, 1711))
