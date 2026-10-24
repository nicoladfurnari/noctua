"""Robust statistics helpers. Medians are computed on a subsample for speed."""
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(img):
    if img.shape[-1] == 1:
        return img[..., 0]
    return img @ LUMA


def median_mad(x, step=4):
    s = x[::step, ::step] if x.ndim >= 2 else x
    s = s.ravel()
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    return med, mad


def channel_stats(img, step=4):
    return [median_mad(img[..., c], step) for c in range(img.shape[-1])]


def describe(img):
    parts = []
    for c, (med, mad) in enumerate(channel_stats(img)):
        parts.append(f"c{c} med={med:.5f} mad={mad:.5f} max={img[..., c].max():.4f}")
    return "  ".join(parts)
