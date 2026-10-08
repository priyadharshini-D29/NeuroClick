"""Fig 3: change 'behavior' -> 'behaviour' in both legends (raster edit)."""
import numpy as np
from PIL import ImageFont
from figlib import *

img = load('image3.png')
arr = np.array(img)
# search boxes: (x0,y0,x1,y1) just around each legend line's text (markers excluded)
jobs = [
    ((700, 160, 1750, 262), 'NeuroClick behavior (AUC=0.2736)', 'NeuroClick behaviour (AUC=0.2736)'),
    ((700, 268, 1750, 355), 'Logistic behavior (AUC=0.2450)', 'Logistic behaviour (AUC=0.2450)'),
    ((2540, 845, 3600, 935), 'NeuroClick behavior (AUC=0.7020)', 'NeuroClick behaviour (AUC=0.7020)'),
    ((2540, 942, 3600, 1030), 'Logistic behavior (AUC=0.6363)', 'Logistic behaviour (AUC=0.6363)'),
]
for box, old, new in jobs:
    bb = text_bbox(np.array(img), box)
    print(old, '-> bbox', bb)
    img, _ = erase_ink(img, bb, inpaint=True)
    size = fit_font(ARIAL, old, bb[2] - bb[0], 40, 120)
    font = ImageFont.truetype(ARIAL, size)
    print('  font size', size, 'old width', bb[2] - bb[0], 'rendered', font.getlength(old))
    draw_text_at(img, (bb[0], bb[1]), new, font)
save(img, 'fig3.png')
