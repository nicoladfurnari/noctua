"""Automatic crop of the ragged borders left by registration/integration."""
import numpy as np

from .stats import luminance


def detect_borders(img, low=0.8, high=1.5, max_frac=0.06, pad=12):
    """Return (left, right, top, bottom) margins in pixels.

    Every channel is checked on its own: after combining filters, one channel alone can have a
    bad edge (a colored line) while the luminance looks fine.
    """
    limit = round(max_frac * min(img.shape[:2]))

    def walk(profile, gmed, pixel_noise=0.0):
        lo, hi = low * gmed, high * gmed
        # noise of the profile, from the differences of neighbours (the structure of the field does not count)
        mid = profile[len(profile) // 4: -len(profile) // 4 or None]
        sig = max(1.4826 * float(np.median(np.abs(np.diff(mid)))) / 2 ** 0.5, 1e-7) if len(mid) > 2 else 1e-7
        if gmed < 3 * max(sig, pixel_noise):
            # background at or below zero, or within the pixel noise of it (a background-subtracted stack, also
            # with its negative half clipped to 0 by the loader): a band proportional to the median is
            # empty or narrower than the noise, and every side was cut to the maximum. Each side is compared with
            # the field just inside it (a gradient moves the level far from the global median), within the noise
            # or the large-scale spread of the profile, whichever is wider.
            near = profile[limit:2 * limit] if limit > 0 and len(profile) >= 2 * limit else mid
            ref = float(np.median(near)) if len(near) else gmed
            spread = 1.4826 * float(np.median(np.abs(mid - np.median(mid)))) if len(mid) else 0.0
            # (the noise next to this side too: where the loader clipped the negative half to 0 the middle of
            # the profile is flat zeros and says nothing about the noise of the side that is above zero)
            wide = profile[limit:3 * limit] if limit > 0 and len(profile) >= 3 * limit else near
            sn = 1.4826 * float(np.median(np.abs(np.diff(wide)))) / 2 ** 0.5 if len(wide) > 2 else 0.0
            tol = max(6 * max(sig, sn), 3 * spread)
            lo, hi = ref - tol, ref + tol
        n = 0
        while n < limit and not (lo <= profile[n] <= hi):
            n += 1
        return n

    planes = [luminance(img)] + ([img[..., c] for c in range(img.shape[-1])] if img.shape[-1] > 1 else [])
    margins = np.zeros(4, int)
    for p in planes:
        gmed = float(np.median(p[::8, ::8]))
        col = np.median(p[::4, :], axis=0)
        row = np.median(p[:, ::4], axis=1)
        sub = p[::8, ::8]
        pn = 1.4826 * float(np.median(np.abs(np.diff(sub, axis=1)))) / 2 ** 0.5 if sub.shape[1] > 1 else 0.0
        m = (walk(col, gmed, pn), walk(col[::-1], gmed, pn), walk(row, gmed, pn), walk(row[::-1], gmed, pn))
        margins = np.maximum(margins, m)
    return tuple(int(v) + pad for v in margins)


def auto_crop(img, **kw):
    left, right, top, bottom = detect_borders(img, **kw)
    h, w = img.shape[:2]
    return img[top:h - bottom, left:w - right].copy(), (left, right, top, bottom)
