import numpy as np
from figlib import *

img = load('image7.png')
arr = np.array(img).astype(int)
H, W = arr.shape[:2]


def colour_mask(c, tol=12):
    return (np.abs(arr - np.array(c)).sum(axis=2) < tol)


def bars(c):
    m = colour_mask(c)
    m[:500, :] = False  # drop legend swatches
    cols = m.sum(axis=0) > 20
    # find contiguous column runs wider than 40px
    runs, start = [], None
    for x in range(W):
        if cols[x] and start is None:
            start = x
        elif not cols[x] and start is not None:
            if x - start > 40:
                runs.append((start, x))
            start = None
    out = []
    for a, b in runs:
        sub = m[:, a:b]
        rows = np.where(sub.mean(axis=1) > 0.6)[0]
        if len(rows):
            out.append(((a + b) / 2, rows.min(), rows.max()))
    return out


res = {}
for name, c in [('teal', TEAL), ('navy', NAVY), ('orange', ORANGE)]:
    res[name] = bars(c)
    print(name, res[name])

# baseline (axis zero) = common bottom row
bottom = max(r[2] for v in res.values() for r in v)
print('bottom row', bottom)
# calibrate on NeuroClick first1 = .3532 and full = .5558
t = res['teal']
y1, y4 = t[0][1], t[3][1]
scale = (0.5558 - 0.3532) / (y1 - y4)  # value per pixel (y decreases upward)
print('px per 0.1:', 0.1 / scale)
def val(y):
    return 0.3532 + (y1 - y) * scale
# zero check
print('implied zero value at bottom row:', val(bottom))
for name in ['teal', 'navy', 'orange']:
    print(name, [round(val(r[1]), 4) for r in res[name]])
