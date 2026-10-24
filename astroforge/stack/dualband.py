"""Ha / OIII extraction from a colour (Bayer) frame shot through a dual-band filter.

Filters such as Optolong L-eXtreme / L-eNhance, Antlia ALP-T, IDAS NBZ pass two narrow bands:
Ha 656.3 nm (+ often SII/NII) and OIII 500.7 nm (+ Hb for the L-eNhance).

Assumptions (per 2x2 Bayer cell, calibrated linear data):
- R sites see Ha (and a small OIII leak `oiii_leak_r`, ~0 for most OSC sensors);
- G sites see OIII, plus a fraction `ha_leak_g` of Ha (the green dye transmits a little at 656 nm,
  typically 0.02-0.10 for Sony IMX sensors);
- B sites see OIII with relative response `b_gain` compared with G (500.7 nm is near the crossover of
  the G and B dyes, ~0.7-1.0), plus a fraction `ha_leak_b` of Ha (usually ~0).

The linear system per pixel
    R = Ha + lr*O ;  G = O + lg*Ha ;  B = gb*O + lb*Ha
is solved exactly. OIII is a weighted mix of the G and B estimates: `b_weight` is the weight of the
B estimate (default 1/3: one B site vs two G sites with the same noise).

`extract` works at superpixel resolution (H/2, W/2): no interpolation, best SNR per pixel.
`extract_full` interpolates the R, G, B sites bilinearly to (H, W) and solves the same system:
use it when the full sensor resolution is wanted (e.g. before registration with other data).
"""
import cv2
import numpy as np

_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32)


def _sites(pattern):
    pattern = pattern.upper()
    if sorted(pattern) != sorted("RGGB"):
        raise ValueError(f"unsupported Bayer pattern {pattern}")
    out = {"R": [], "G": [], "B": []}
    for i, ch in enumerate(pattern):
        out[ch].append(divmod(i, 2))
    return out


def _solve(R, G, B, ha_leak_g, ha_leak_b, oiii_leak_r, b_gain, b_weight):
    bw = float(b_weight)
    mix = (1 - bw) * G + bw * B / b_gain          # = O + L*Ha
    L = (1 - bw) * ha_leak_g + bw * ha_leak_b / b_gain
    det = 1.0 - oiii_leak_r * L
    if abs(det) < 1e-6:
        raise ValueError("leakage coefficients make Ha and OIII indistinguishable")
    ha = (R - oiii_leak_r * mix) / det
    o = mix - L * ha
    return ha.astype(np.float32), o.astype(np.float32)


def _pack(ha, o, mode):
    mode = mode.lower()
    if mode in ("ha_oiii", "hoo", "both"):
        return {"Ha": ha, "OIII": o}
    if mode == "ha":
        return {"Ha": ha}
    if mode == "oiii":
        return {"OIII": o}
    raise ValueError(f"unknown mode {mode}")


def extract(cfa, pattern, mode="ha_oiii", ha_leak_g=0.0, ha_leak_b=0.0, oiii_leak_r=0.0,
            b_gain=1.0, b_weight=1 / 3):
    """CFA (H, W) float -> {"Ha": (H//2, W//2), "OIII": (H//2, W//2)} float32 (superpixel).

    mode: "ha_oiii" (both), "ha" or "oiii".
    """
    cfa = np.asarray(cfa, np.float32)
    h, w = cfa.shape[0] // 2 * 2, cfa.shape[1] // 2 * 2
    s = _sites(pattern)
    plane = lambda yx: cfa[yx[0]:h:2, yx[1]:w:2]
    R = plane(s["R"][0])
    G = 0.5 * (plane(s["G"][0]) + plane(s["G"][1]))
    B = plane(s["B"][0])
    return _pack(*_solve(R, G, B, ha_leak_g, ha_leak_b, oiii_leak_r, b_gain, b_weight), mode)


def _interp(cfa, sites, kernel):
    """Bilinear interpolation of the given sites to full resolution (normalized convolution)."""
    m = np.zeros(cfa.shape, np.float32)
    for y, x in sites:
        m[y::2, x::2] = 1
    num = cv2.filter2D(cfa * m, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    den = cv2.filter2D(m, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    return np.where(m > 0, cfa, num / np.maximum(den, 1e-6))


def extract_full(cfa, pattern, mode="ha_oiii", ha_leak_g=0.0, ha_leak_b=0.0, oiii_leak_r=0.0,
                 b_gain=1.0, b_weight=1 / 3):
    """CFA (H, W) -> {"Ha": (H, W), "OIII": (H, W)} float32 by bilinear interpolation of each site set."""
    cfa = np.ascontiguousarray(cfa, np.float32)
    h, w = cfa.shape[0] // 2 * 2, cfa.shape[1] // 2 * 2
    cfa = np.ascontiguousarray(cfa[:h, :w])
    s = _sites(pattern)
    R = _interp(cfa, s["R"], _K_RB)
    G = _interp(cfa, s["G"], _K_G)
    B = _interp(cfa, s["B"], _K_RB)
    return _pack(*_solve(R, G, B, ha_leak_g, ha_leak_b, oiii_leak_r, b_gain, b_weight), mode)


def estimate_ha_leak_g(cfa, pattern, percentile=99.0):
    """Rough Ha->G leak from pixels dominated by Ha (brightest R relative to B): median of (G-B)/R.

    Assumes b_gain ~ 1. Only meaningful on frames with bright Ha-only regions; returns 0 otherwise.
    """
    s = _sites(pattern)
    cfa = np.asarray(cfa, np.float32)
    h, w = cfa.shape[0] // 2 * 2, cfa.shape[1] // 2 * 2
    p = lambda yx: cfa[yx[0]:h:2, yx[1]:w:2]
    R, B = p(s["R"][0]), p(s["B"][0])
    G = 0.5 * (p(s["G"][0]) + p(s["G"][1]))
    bg = [float(np.median(x)) for x in (R, G, B)]
    R, G, B = R - bg[0], G - bg[1], B - bg[2]
    sel = R > np.percentile(R, percentile)
    if sel.sum() < 20:
        return 0.0
    return float(np.clip(np.median((G[sel] - B[sel]) / np.maximum(R[sel], 1e-9)), 0.0, 0.5))
