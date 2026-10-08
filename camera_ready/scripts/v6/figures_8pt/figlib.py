"""Shared helpers for camera-ready figure fixes."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

BASE = r'C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin\1282218c-8e1c-4ad5-a29d-8076480f7dbc\scratchpad\v6\fig7'
MEDIA = os.path.join(BASE, 'unpacked', 'word', 'media')
OUT = os.path.join(BASE, 'newfigs')
os.makedirs(OUT, exist_ok=True)

ARIAL = r'C:\Windows\Fonts\arial.ttf'
ARIAL_BOLD = r'C:\Windows\Fonts\arialbd.ttf'
import matplotlib
_MPLF = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
DEJAVU = os.path.join(_MPLF, 'DejaVuSans.ttf')
DEJAVU_BOLD = os.path.join(_MPLF, 'DejaVuSans-Bold.ttf')

# Palette sampled from the existing figures
TEAL = (31, 138, 128)
NAVY = (51, 81, 110)
ORANGE = (224, 165, 46)
PINK = (169, 77, 149)
INK = (31, 53, 64)
TEAL_HEX, NAVY_HEX, ORANGE_HEX, PINK_HEX, INK_HEX = '#1F8A80', '#33516E', '#E0A52E', '#A94D95', '#1F3540'


def load(name):
    return Image.open(os.path.join(MEDIA, name)).convert('RGB')


def dark_mask(arr, thresh=110):
    """Boolean mask of 'ink' pixels: dark and low-saturation."""
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return (mx < thresh) & ((mx - mn) < 60)


def text_bbox(arr, box, thresh=110):
    """Bounding box (x0,y0,x1,y1) of ink pixels inside box=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    sub = dark_mask(arr[y0:y1, x0:x1], thresh)
    ys, xs = np.where(sub)
    if len(xs) == 0:
        return None
    return (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)


def erase_ink(img, box, thresh=110, radius=4, inpaint=True, keep_cols=()):
    """Remove ink pixels inside box, inpainting what was underneath.
    keep_cols: list of (x0,x1) column ranges to restore afterwards from a reference row."""
    arr = np.array(img)
    x0, y0, x1, y1 = box
    mask = np.zeros(arr.shape[:2], np.uint8)
    sub = dark_mask(arr[y0:y1, x0:x1], thresh)
    mask[y0:y1, x0:x1] = sub.astype(np.uint8) * 255
    # dilate a little so anti-aliased fringes go too
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    if inpaint:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        bgr = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
        arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        arr[mask > 0] = 255
    return Image.fromarray(arr), mask


def fit_font(path, text, target_width, lo=10, hi=200):
    """Find the font size whose rendered width of `text` is closest to target_width."""
    best = None
    for size in range(lo, hi):
        f = ImageFont.truetype(path, size)
        w = f.getlength(text)
        d = abs(w - target_width)
        if best is None or d < best[0]:
            best = (d, size)
    return best[1]


def draw_text_at(img, xy_top_left, text, font, fill=INK):
    """Draw text so that its ink bbox top-left lands on xy_top_left."""
    d = ImageDraw.Draw(img)
    l, t, r, b = font.getbbox(text)
    x, y = xy_top_left
    d.text((x - l, y - t), text, font=font, fill=fill)
    return (x, y, x + (r - l), y + (b - t))


def save(img, name):
    p = os.path.join(OUT, name)
    img.save(p, dpi=(600, 600))
    print('saved', p, img.size)
    return p

