"""Dark frame optimization (like PixInsight's dark scaling).

The master dark is scaled by k before subtraction so that the noise of the calibrated light is
minimal:  calibrated = light - bias - k * (dark - bias).  The best k compensates exposure/temperature
differences between darks and lights (the dark current pattern is per pixel, so a wrong k leaves
(1-k) * pattern in the light and raises its high-frequency noise).

Noise estimator: first à-trous wavelet layer (I - B3 spline smoothing) computed on each 2x2 CFA plane
separately (works for mono too). Stars are masked out (extended structures of the light, found after a
morphological opening that ignores single hot pixels) and the standard deviation is taken WITHOUT tight
clipping: hot/warm-pixel residuals are exactly the signal that tells a wrong k, clipping them away biases
k low. The wavelet layer is linear, so it is computed once for light and dark and the golden-section
search over k is cheap.

Amp glow (QHY163, ZWO 1600...): the glow does not always scale like the dark current. `glow_mask`
finds the glow region from the dark itself (smooth excess over the median level); with glow=True a
separate k_glow is fitted for the smooth glow component by least squares on the low-pass residual
(plus a plane for the sky gradient). The calibration is then

    calibrated = light - bias - k * (dark - bias) - (k_glow - k) * glow_component(dark)

see `subtract`. Without a bias the dark is taken as dark current only (it should be bias-subtracted
or the bias pattern negligible); the constant level does not affect k.
"""
import cv2
import numpy as np
from scipy import ndimage

_B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16


def _planes_apply(img, fn, period=2):
    out = np.empty(img.shape, np.float32)
    for dy in range(period):
        for dx in range(period):
            out[dy::period, dx::period] = fn(np.ascontiguousarray(img[dy::period, dx::period], np.float32))
    return out


def wavelet_hp(img, period=2):
    """First wavelet layer per CFA plane (linear high-pass)."""
    return _planes_apply(img, lambda p: p - cv2.sepFilter2D(p, -1, _B3, _B3, borderType=cv2.BORDER_REFLECT_101),
                         period)


def _clipped_std(v, kappa=4.0, iters=5):
    m = float(np.median(v))
    s = float(np.median(np.abs(v - m))) * 1.4826 + 1e-15
    for _ in range(iters):
        sel = np.abs(v - m) < kappa * s
        if sel.sum() < 10:
            break
        m, s = float(v[sel].mean()), float(v[sel].std()) + 1e-15
    return s


def noise(img, period=2):
    """Robust noise estimate of an image (CFA or mono)."""
    hp = wavelet_hp(np.asarray(img, np.float32), period)
    return _clipped_std(hp[::2, ::2].ravel() if hp.size > 4_000_000 else hp.ravel())


def star_mask(light, period=2, k=5.0, grow=3):
    """Stars and other extended bright structures of a raw light: single hot pixels are not included."""
    b = np.ascontiguousarray(_bin2(np.asarray(light, np.float32)) if period == 2 else light, np.float32)
    bg = cv2.resize(cv2.medianBlur(cv2.resize(b, (max(1, b.shape[1] // 8), max(1, b.shape[0] // 8)),
                                              interpolation=cv2.INTER_AREA), 5), (b.shape[1], b.shape[0]))
    d = cv2.morphologyEx(b - bg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # removes 1-2 px hot pixels
    s = float(np.median(np.abs(d - np.median(d)))) * 1.4826 + 1e-12
    m = cv2.dilate((d > k * s).astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    if period == 2:
        m = np.repeat(np.repeat(m, 2, 0), 2, 1)
        full = np.zeros(np.shape(light), bool)
        full[:m.shape[0], :m.shape[1]] = m[:full.shape[0], :full.shape[1]]
        m = full
    return m


def _golden(fn, a, b, tol=1e-4):
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = fn(c), fn(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = fn(d)
    return 0.5 * (a + b)


def _bin2(img):
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    c = img[:h, :w]
    return 0.25 * (c[0::2, 0::2] + c[0::2, 1::2] + c[1::2, 0::2] + c[1::2, 1::2])


def _glow_surface(d, smooth):
    """Smooth version of a (bias-subtracted) dark at half resolution: no hot pixels, no CFA."""
    b = np.ascontiguousarray(_bin2(d), np.float32)
    b = cv2.medianBlur(b, 5)
    return cv2.GaussianBlur(b, (0, 0), max(smooth / 2, 1.0))


def glow_mask(dark, bias=None, nsigma=5.0, frac=0.1, smooth=16, grow=16, max_area=0.4):
    """Boolean (H, W) mask of the amp-glow region found in the master dark (empty if no glow)."""
    d = np.asarray(dark, np.float32) - (bias if bias is not None else 0)
    L = _glow_surface(d, smooth)
    base = float(np.median(L))
    mad = float(np.median(np.abs(L - base))) * 1.4826
    top = float(np.percentile(L, 99.9))
    thr = base + max(nsigma * mad, frac * (top - base), 1e-7)
    m = L > thr
    H, W = d.shape
    if m.mean() > max_area or m.sum() < 16:
        return np.zeros((H, W), bool)
    lab, n = ndimage.label(m)
    if n > 1:   # drop specks
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sizes >= max(16, 0.05 * sizes.max())))
    g = max(1, grow // 2)
    m = cv2.dilate(m.astype(np.uint8), np.ones((2 * g + 1, 2 * g + 1), np.uint8))
    full = np.zeros((H, W), bool)
    up = np.repeat(np.repeat(m, 2, 0), 2, 1).astype(bool)
    full[:up.shape[0], :up.shape[1]] = up
    return full


def glow_component(dark, bias=None, mask=None, smooth=16, feather=16):
    """Smooth amp-glow surface (H, W) float32: excess of the dark over its level outside the mask."""
    d = np.asarray(dark, np.float32) - (bias if bias is not None else 0)
    H, W = d.shape
    if mask is None:
        mask = glow_mask(dark, bias, smooth=smooth)
    if not mask.any():
        return np.zeros((H, W), np.float32)
    L = _glow_surface(d, smooth)
    mb = _bin2(mask.astype(np.float32)) > 0.5
    base = float(np.median(L[~mb])) if (~mb).any() else float(np.median(L))
    g = cv2.resize(np.maximum(L - base, 0), (W // 2 * 2, H // 2 * 2), interpolation=cv2.INTER_LINEAR)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(feather / 2, 1.0))
    out = np.zeros((H, W), np.float32)
    out[:g.shape[0], :g.shape[1]] = g
    return out * np.maximum(soft, mask)


def _crop(a, roi):
    if roi is None:
        return a
    y0, y1, x0, x1 = roi
    return a[y0:y1, x0:x1]


def optimal_scale(light_cal_input, master_dark, bias=None, roi=None, glow=False, k_range=(0.0, 2.0),
                  mask=None, period=2, max_pixels=2_000_000):
    """Dark scale k minimising the noise of light - bias - k*(dark - bias).

    light_cal_input: raw light (H, W) float (before calibration, same area as the dark).
    roi: (y0, y1, x0, x1) where the noise is measured (None = whole frame). Keep y0/x0 even on CFA.
    glow: also fit the amp-glow scale; returns (k, k_glow, mask) instead of k.
    mask: precomputed glow mask (else `glow_mask(master_dark, bias)`).
    """
    L = np.asarray(light_cal_input, np.float32) - (bias if bias is not None else 0)
    D = np.asarray(master_dark, np.float32) - (bias if bias is not None else 0)
    if glow and mask is None:
        mask = glow_mask(master_dark, bias)
    hL, hD = wavelet_hp(L, period), wavelet_hp(D, period)
    sel = ~star_mask(L, period)
    if glow and mask is not None and mask.any():
        sel &= ~mask
    sel = _crop(sel, roi)
    vL, vD = _crop(hL, roi)[sel], _crop(hD, roi)[sel]
    if vL.size > max_pixels:
        st = int(np.ceil(vL.size / max_pixels))
        vL, vD = vL[::st], vD[::st]
    vL, vD = vL.astype(np.float64), vD.astype(np.float64)
    # loose clipping only (cosmic rays, satellite pixels): residual hot pixels must count
    k = float(_golden(lambda k: _clipped_std(vL - k * vD, kappa=25.0), *k_range))
    if not glow:
        return k
    if mask is None or not mask.any():
        return k, k, np.zeros(L.shape, bool)

    G = glow_component(master_dark, bias, mask)
    R = L - k * D
    # low-pass residual, stars removed: R ~ sky plane + (k_glow - k) * G inside/near the glow
    Rb = ndimage.median_filter(_bin2(R), size=5)
    Rb = cv2.GaussianBlur(Rb.astype(np.float32), (0, 0), 2)
    Gb = _bin2(G)
    region = cv2.dilate(_bin2(mask.astype(np.float32)).astype(np.uint8), np.ones((33, 33), np.uint8)) > 0
    if roi is not None:
        rr = np.zeros_like(region)
        y0, y1, x0, x1 = roi
        rr[y0 // 2:y1 // 2, x0 // 2:x1 // 2] = True
        region &= rr
    yy, xx = np.nonzero(region)
    if yy.size < 50:
        return k, k, mask
    A = np.stack([np.ones(yy.size), yy / Rb.shape[0], xx / Rb.shape[1], Gb[yy, xx]], 1).astype(np.float64)
    y = Rb[yy, xx].astype(np.float64)
    use = np.ones(y.size, bool)
    for _ in range(5):
        sol, *_ = np.linalg.lstsq(A[use], y[use], rcond=None)
        res = y - A @ sol
        s = float(np.median(np.abs(res[use]))) * 1.4826 + 1e-12
        new = np.abs(res) < 3 * s
        if new.sum() < 50 or (new == use).all():
            break
        use = new
    k_glow = float(k + sol[3])
    return k, k_glow, mask


def subtract(light, master_dark, bias=None, k=1.0, k_glow=None, mask=None):
    """Calibrated light (before flat): light - bias - k*(dark-bias) - (k_glow-k)*glow_component."""
    L = np.asarray(light, np.float32) - (bias if bias is not None else 0)
    D = np.asarray(master_dark, np.float32) - (bias if bias is not None else 0)
    out = L - np.float32(k) * D
    if k_glow is not None and mask is not None and mask.any() and k_glow != k:
        out -= np.float32(k_glow - k) * glow_component(master_dark, bias, mask)
    return out.astype(np.float32)
