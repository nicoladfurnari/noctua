"""Automatic crop of the ragged borders left by registration/integration."""
import numpy as np

from .stats import luminance


def detect_borders(img, low=0.8, high=1.5, max_frac=0.06, pad=12):
    """Return (left, right, top, bottom) margins in pixels.

    Every channel is checked on its own: after combining filters, one channel alone can have a
    bad edge (a colored line) while the luminance looks fine.
    """
    limit = round(max_frac * min(img.shape[:2]))

    def walk(profile, gmed):
        n = 0
        while n < limit and not (low * gmed <= profile[n] <= high * gmed):
            n += 1
        return n

    planes = [luminance(img)] + ([img[..., c] for c in range(img.shape[-1])] if img.shape[-1] > 1 else [])
    margins = np.zeros(4, int)
    for p in planes:
        gmed = float(np.median(p[::8, ::8]))
        col = np.median(p[::4, :], axis=0)
        row = np.median(p[:, ::4], axis=1)
        m = (walk(col, gmed), walk(col[::-1], gmed), walk(row, gmed), walk(row[::-1], gmed))
        margins = np.maximum(margins, m)
    return tuple(int(v) + pad for v in margins)


def auto_crop(img, **kw):
    left, right, top, bottom = detect_borders(img, **kw)
    h, w = img.shape[:2]
    return img[top:h - bottom, left:w - right].copy(), (left, right, top, bottom)
