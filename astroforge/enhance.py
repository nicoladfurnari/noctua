"""Color and tone finishing: SCNR, curves, saturation."""
import numpy as np
from scipy.interpolate import PchipInterpolator

from .stats import luminance


SCNR_METHODS = ("average_neutral", "maximum_neutral", "maximum_mask", "additive_mask")


def scnr(img, amount=0.8, method="average_neutral", color="green", preserve_lightness=True):
    """Subtractive Chromatic Noise Reduction as in PixInsight: removes a colour cast (green for OSC/RGB,
    also red/blue, e.g. magenta-free stars in SHO by inverting first). Neutral methods cap the channel at
    the mean / max of the other two; mask methods attenuate it proportionally to a mask of the others."""
    k = {"red": 0, "green": 1, "blue": 2}[color]
    a, b = [img[..., i] for i in range(3) if i != k]
    c = img[..., k]
    if method == "average_neutral":
        new = c * (1 - amount) + np.minimum(c, 0.5 * (a + b)) * amount
    elif method == "maximum_neutral":
        new = c * (1 - amount) + np.minimum(c, np.maximum(a, b)) * amount
    elif method == "maximum_mask":
        m = np.maximum(a, b)
        new = c * (1 - amount) * (1 - m) + m * c
    elif method == "additive_mask":
        m = np.minimum(1.0, a + b)
        new = c * (1 - amount) * (1 - m) + m * c
    else:
        raise ValueError(f"unknown SCNR method {method}")
    out = img.copy()
    out[..., k] = new
    if preserve_lightness:
        # removing green darkens: restore the original luminance without bringing the cast back
        l0, l1 = luminance(img), luminance(out)
        out *= (l0 / np.maximum(l1, 1e-6))[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


def scnr_green(img, amount=0.8):
    return scnr(img, amount, "average_neutral", "green", preserve_lightness=False)


def curve_lut(points, size=65536):
    xs, ys = zip(*points)
    return np.clip(PchipInterpolator(xs, ys)(np.linspace(0, 1, size)), 0, 1).astype(np.float32)


def apply_curve(img, points):
    lut = curve_lut(points)
    idx = (np.clip(img, 0, 1) * (len(lut) - 1)).astype(np.uint16)
    return lut[idx]


def saturation(img, amount=0.3, mask=None):
    """Boost chroma around luminance. `mask` (H,W) limits where it is applied."""
    lum = luminance(img)[..., None]
    f = 1 + amount * (mask[..., None] if mask is not None else 1)
    out = lum + (img - lum) * f
    return np.clip(out, 0, 1).astype(np.float32)
