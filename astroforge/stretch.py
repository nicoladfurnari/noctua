"""Nonlinear stretch: STF-style auto stretch with the midtones transfer function."""
import numpy as np

from .stats import channel_stats


def mtf(m, x):
    x = np.clip(x, 0, 1)
    # m = 0 or 1 (e.g. auto params on an already stretched image) would give 0/0 at the ends
    m = np.clip(m, 1e-6, 1 - 1e-6)
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def auto_params(img, target_bg=0.2, shadows_clip=-2.8):
    """Linked STF parameters (shadows c0, midtones m)."""
    st = channel_stats(img)
    c0 = np.mean([max(0.0, med + shadows_clip * mad * 1.4826) for med, mad in st])
    med = np.mean([med for med, _ in st])
    m = float(mtf(target_bg, med - c0))
    return float(c0), m


def histogram_transform(img, c0, m, c1=1.0):
    x = (img - c0) / (c1 - c0)
    return mtf(m, x).astype(np.float32)


def auto_stretch(img, target_bg=0.2, shadows_clip=-2.8):
    c0, m = auto_params(img, target_bg, shadows_clip)
    return histogram_transform(img, c0, m), (c0, m)


def _ght(d, D, b):
    """Generalized hyperbolic family (Payne & Cranfield) for d = x - SP >= 0: value and slope, slope(0) = 1."""
    if b == -1:                       # logarithmic
        return np.log1p(D * d) / D, 1 / (1 + D * d)
    if b < 0:                         # integral
        q = 1 - b * D * d
        return (1 - q ** ((b + 1) / b)) / (D * (b + 1)), q ** (1 / b)
    if b == 0:                        # exponential
        e = np.exp(-D * d)
        return (1 - e) / D, e
    q = 1 + b * D * d                 # harmonic (b = 1) and hyperbolic (b > 0)
    return (1 - q ** (-1 / b)) / D, q ** (-(1 + b) / b)


def ghs(x, D, b, SP, LP=0.0, HP=1.0, inverse=False):
    """Generalized Hyperbolic Stretch (ghsastro.co.uk): hyperbolic around SP, mirrored below it, linear below LP
    and above HP with the slope of the curve there (continuous and C1), normalised so 0 -> 0 and 1 -> 1.
    D is the internal stretch factor (the GUI value is ln(D + 1)); D = 0 is the identity. inverse=True
    applies the inverse transformation (from a dense LUT, the curve is strictly increasing)."""
    x = np.clip(x, 0, 1)
    if D <= 0:
        return x.astype(np.float32)
    D, b = float(D), float(b)
    SP = float(np.clip(SP, 0, 1))
    LP, HP = min(float(LP), SP), max(float(HP), SP)
    tl, sl = _ght(SP - LP, D, b)
    th, sh = _ght(HP - SP, D, b)

    def T(v):
        d = v - SP
        y = np.sign(d) * _ght(np.abs(d), D, b)[0]
        y = np.where(v < LP, -tl + sl * (v - LP), y)
        return np.where(v > HP, th + sh * (v - HP), y)
    t0, t1 = float(T(np.float64(0))), float(T(np.float64(1)))
    if inverse:
        g = np.linspace(0, 1, 1 << 17)
        return np.interp(x, (T(g) - t0) / (t1 - t0), g).astype(np.float32)
    return ((T(x) - t0) / (t1 - t0)).astype(np.float32)
