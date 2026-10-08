"""Fig 6: move the three Holm p-value lines out of the plotting area into a band under the titles."""
import numpy as np
from PIL import Image, ImageFont
from figlib import *

img = load('image6.png')
arr0 = np.array(img)
H, W = arr0.shape[:2]
dm = dark_mask(arr0)
GRID = np.array([203, 227, 235])

panels = [
    ((232, 150, 1700, 352), ['NeuroClick vs Logistic: Holm p = 0.0006', 'NeuroClick vs Dwell: Holm p = 0.0005', 'NeuroClick vs Dwell+Propensity: Holm p = 0.7570']),
    ((2100, 150, 3560, 352), ['NeuroClick vs Logistic: Holm p = 0.6407', 'NeuroClick vs Dwell: Holm p = 0.0720', 'NeuroClick vs Dwell+Propensity: Holm p = 0.7171']),
]
info = []
for box, lines in panels:
    x0, y0, x1, y1 = box
    frac = dm[400:1500, x0:x1].mean(axis=0)
    spine_cols = [x0 + i for i, f in enumerate(frac) if f > 0.9]
    bb = text_bbox(arr0, box)
    bb = (bb[0] - 4, bb[1] - 4, bb[2] + 4, bb[3] + 10)
    print('text bbox', bb, 'spine', spine_cols[0], spine_cols[-1])
    # gridline rows inside bbox: rows that are grid-coloured over a clean stretch left of the text box
    probe = arr0[bb[1]:bb[3], spine_cols[-1] + 5:spine_cols[-1] + 60]
    def is_grid(px):
        return (px.min(axis=1) > 170) & (px[:, 2] - px[:, 0] > 12)  # light blue-ish, not white
    grid_rows = [bb[1] + i for i in range(probe.shape[0]) if is_grid(probe[i]).mean() > 0.6]
    print('grid rows', grid_rows)
    # stronger mask: threshold 150, 2 dilations
    a = np.array(img)
    mask = np.zeros(a.shape[:2], np.uint8)
    mask[bb[1]:bb[3], bb[0]:bb[2]] = dark_mask(a[bb[1]:bb[3], bb[0]:bb[2]], 150).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    bgr = cv2.inpaint(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), mask, 9, cv2.INPAINT_TELEA)
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # repaint gridline where it was masked and came back near-white
    for r in grid_rows:
        row = a[r, bb[0]:bb[2]]
        m = (mask[r, bb[0]:bb[2]] > 0) & (row.min(axis=1) > 215)
        row[m] = arr0[r, spine_cols[-1] + 20]
    # restore spine columns
    ref = a[bb[3] + 40]
    for x in spine_cols:
        a[bb[1]:bb[3], x] = ref[x]
    img = Image.fromarray(a)
    right_spine = None
    # right edge of plotting area: last dark column in the row band below the box
    rowdark = dm[800, x0:x1 + 200]
    xs = np.where(rowdark)[0]
    info.append((bb, lines))

# font: fit the longest line into the panel width with margin
bb0, lines0 = info[0]
longest = max(lines0, key=len)
avail = bb0[2] - 60
size = fit_font(DEJAVU, longest, avail, 30, 90)
font = ImageFont.truetype(DEJAVU, size)
print('font size', size, 'longest width', font.getlength(longest), 'avail', avail)
line_h = int(size * 1.3)
band_h = 3 * line_h + 26
cut = 122
new = Image.new('RGB', (W, H + band_h), 'white')
new.paste(img.crop((0, 0, W, cut)), (0, 0))
new.paste(img.crop((0, cut, W, H)), (0, cut + band_h))
for bb, lines in info:
    right = bb[2]
    for k, t in enumerate(lines):
        tw = font.getlength(t)
        draw_text_at(new, (int(right - tw), cut + 14 + k * line_h), t, font)
save(new, 'fig6.png')
