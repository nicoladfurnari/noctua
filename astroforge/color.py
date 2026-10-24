"""Background neutralization and star-based white balance."""
import cv2
import numpy as np

from .stats import luminance


def background_levels(img, quantile=0.5):
    """Per-channel median of the darkest pixels (by luminance)."""
    lum = luminance(img)[::4, ::4]
    sel = lum <= np.quantile(lum, quantile)
    sub = img[::4, ::4]
    return np.array([np.median(sub[..., c][sel]) for c in range(img.shape[-1])], dtype=np.float32)


def neutralize_background(img):
    bg = background_levels(img)
    target = bg.mean()
    out = img + (target - bg)
    np.clip(out, 0, 1, out=out)
    return out, bg


def detect_stars(lum, max_stars=3000, saturation=0.9, box=7):
    """Local maxima well above the noise, unsaturated. Returns (ys, xs)."""
    smooth = cv2.GaussianBlur(lum, (0, 0), 1.0)
    dil = cv2.dilate(smooth, np.ones((box, box), np.uint8))
    med = float(np.median(smooth[::4, ::4]))
    mad = float(np.median(np.abs(smooth[::4, ::4] - med))) * 1.4826
    peaks = (smooth == dil) & (smooth > med + 15 * mad) & (lum < saturation)
    ys, xs = np.nonzero(peaks)
    h, w = lum.shape
    r = box
    ok = (ys > r) & (ys < h - r) & (xs > r) & (xs < w - r)
    ys, xs = ys[ok], xs[ok]
    order = np.argsort(-smooth[ys, xs])[:max_stars]
    return ys[order], xs[order]


def white_balance_stars(img, max_stars=3000, max_correction=2.0):
    """Scale channels so the median star is white. Returns (image, factors, n_stars).

    A large aperture makes the measure insensitive to per-color focus differences (refractors);
    `max_correction` limits the change between channels, because stars seen through dust or at
    low altitude are genuinely red and a full correction would turn everything blue.
    """
    lum = luminance(img)
    ys, xs = detect_stars(lum, max_stars)
    if len(ys) < 30:
        return img, np.ones(img.shape[-1], np.float32), len(ys)
    bg = background_levels(img)
    r, ring = 5, 10
    h, w = lum.shape
    ok = (ys >= ring) & (ys < h - ring) & (xs >= ring) & (xs < w - ring)
    ys, xs = ys[ok], xs[ok]
    # local background per star and channel: median of a square ring at radius `ring`
    # (stars in nebulae / the Milky Way sit on colored signal that must not count as star color)
    t = np.arange(-ring, ring + 1)
    ry = np.concatenate([np.full_like(t, -ring), np.full_like(t, ring), t, t])
    rx = np.concatenate([t, t, np.full_like(t, -ring), np.full_like(t, ring)])
    local = np.median(img[ys[:, None] + ry, xs[:, None] + rx, :], axis=1)
    fluxes = np.zeros((len(ys), img.shape[-1]), np.float64)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            fluxes += img[ys + dy, xs + dx, :] - local
    # discard stars clipped in any channel
    peak_ok = (img[ys, xs, :] < 0.9).all(axis=1) & (fluxes > 0).all(axis=1)
    fluxes = fluxes[peak_ok]
    ratios = np.median(fluxes / fluxes[:, 1:2], axis=0)  # relative to green
    k = np.clip(1.0 / ratios, 1 / max_correction, max_correction)
    k = (k / k.max()).astype(np.float32)
    out = (img - bg) * k + bg
    np.clip(out, 0, 1, out=out)
    return out, k, int(peak_ok.sum())
