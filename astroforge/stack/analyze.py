"""Star detection with sub-pixel centroids, Moffat PSF fit (FWHM, eccentricity), noise and per-frame quality."""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Quality:
    stars: int
    fwhm: float          # pixels (full-resolution scale)
    eccentricity: float
    background: float
    noise: float
    snr: float
    weight: float = 1.0
    score: float = 0.0
    accepted: bool = True
    reason: str = ""
    psf_signal: float = 0.0   # PSF flux of the brightest fitted stars / MRS noise (PixInsight PSFSW-like)

    def to_dict(self):
        return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in asdict(self).items()}


def detect(lum, max_stars=600, k=8.0, box=7, saturation=0.95):
    """Return dict of arrays: x, y (sub-pixel), flux, fwhm, ecc. lum: float32 (H, W)."""
    smooth = cv2.GaussianBlur(lum, (0, 0), 1.0)
    # local background (stars removed by the downsampled median)
    h, w = lum.shape
    small = cv2.resize(smooth, (max(8, w // 16), max(8, h // 16)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 5), (w, h), interpolation=cv2.INTER_LINEAR)
    det = smooth - bg
    d = det[::4, ::4]
    sigma = float(np.median(np.abs(d - np.median(d)))) * 1.4826
    dil = cv2.dilate(det, np.ones((box, box), np.uint8))
    peaks = (det == dil) & (det > k * sigma) & (lum < saturation)
    ys, xs = np.nonzero(peaks)
    r = 4
    ok = (ys >= r) & (ys < h - r) & (xs >= r) & (xs < w - r)
    ys, xs = ys[ok], xs[ok]
    order = np.argsort(-det[ys, xs])[:max_stars * 2]
    ys, xs = ys[order], xs[order]
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    patch = lum[ys[:, None, None] + dy, xs[:, None, None] + dx] - bg[ys, xs][:, None, None]
    patch = np.clip(patch, 0, None)
    flux = patch.sum((1, 2))
    good = flux > 0
    patch, flux, ys, xs = patch[good], flux[good], ys[good], xs[good]
    cx = (patch * dx).sum((1, 2)) / flux
    cy = (patch * dy).sum((1, 2)) / flux
    mxx = (patch * (dx - cx[:, None, None]) ** 2).sum((1, 2)) / flux
    myy = (patch * (dy - cy[:, None, None]) ** 2).sum((1, 2)) / flux
    mxy = (patch * (dx - cx[:, None, None]) * (dy - cy[:, None, None])).sum((1, 2)) / flux
    tr, det2 = mxx + myy, mxx * myy - mxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det2, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-6)
    fwhm = 2.355 * np.sqrt(tr / 2)
    ecc = np.sqrt(np.clip(1 - l2 / l1, 0, 1))
    # discard hot pixels / cosmic rays (too sharp) and blends (too elongated)
    keep = (fwhm > 1.0) & (ecc < 0.9)
    sel = np.argsort(-flux[keep])[:max_stars]
    return {"x": (xs + cx)[keep][sel], "y": (ys + cy)[keep][sel], "flux": flux[keep][sel],
            "fwhm": fwhm[keep][sel], "ecc": ecc[keep][sel], "sigma": sigma, "bg": float(np.median(bg))}


def mrs_noise(img, k=3.0, iters=10):
    """Gaussian noise sigma from the first a-trous (B3 spline) wavelet layer with k-sigma clipping (MRS-like)."""
    img = np.ascontiguousarray(img, np.float32)
    b3 = np.array([1, 4, 6, 4, 1], np.float32) / 16
    w1 = (img - cv2.sepFilter2D(img, -1, b3, b3, borderType=cv2.BORDER_REFLECT)).ravel()
    w1 = w1[np.isfinite(w1)]
    if w1.size > 4_000_000:
        w1 = w1[::w1.size // 4_000_000]
    sig = float(np.median(np.abs(w1))) * 1.4826
    for _ in range(iters):
        s2 = float(w1[np.abs(w1) < k * sig].std()) if sig > 0 else 0.0
        if s2 <= 0 or abs(s2 - sig) < 1e-4 * sig:
            sig = s2 or sig
            break
        sig = s2
    return sig / 0.8908   # std of the B3 first layer for unit white noise


def _cutouts(img, xs, ys, R):
    """Square cutouts of radius R around integer centres; out of image pixels are NaN."""
    h, w = img.shape
    d = np.arange(-R, R + 1)
    yi, xi = ys[:, None] + d, xs[:, None] + d
    ok = ((yi >= 0) & (yi < h))[:, :, None] & ((xi >= 0) & (xi < w))[:, None, :]
    c = img[np.clip(yi, 0, h - 1)[:, :, None], np.clip(xi, 0, w - 1)[:, None, :]].astype(np.float64)
    c[~ok] = np.nan
    return c


def _moffat(p, dx, dy, jac=False):
    """Elliptical Moffat B + A (1 + u^2/sx^2 + v^2/sy^2)^-beta; p = A, x0, y0, ln sx, ln sy, theta, ln(beta-1), B.
    With jac=True also the (N, 8, ...) derivatives."""
    A, x0, y0, lsx, lsy, th, lb, B = [p[:, i, None, None] for i in range(8)]
    c, s = np.cos(th), np.sin(th)
    ex, ey = dx - x0, dy - y0
    u, v = ex * c + ey * s, ey * c - ex * s
    ax, ay = np.exp(-2 * lsx), np.exp(-2 * lsy)
    t = 1 + u * u * ax + v * v * ay
    eb = np.exp(lb)
    P = t ** -(1 + eb)
    m = B + A * P
    if not jac:
        return m
    dq = -(1 + eb) * A * P / t                    # d m / d q
    ua, va = 2 * u * ax, 2 * v * ay
    J = np.stack(np.broadcast_arrays(P, dq * (-ua * c + va * s), dq * (-ua * s - va * c), dq * (-ua * u),
                                     dq * (-va * v), dq * (ua * v - va * u), -A * P * np.log(t) * eb,
                                     np.ones_like(P)), 1)
    return m, J


_EMPTY = ("x", "y", "fwhm", "fwhm_x", "fwhm_y", "ecc", "beta", "amp", "flux", "bg")


def fit_psf(lum, stars, n=100, saturation=0.95, iters=30):
    """Elliptical Moffat least-squares fit (vectorised Levenberg-Marquardt) of the ~n brightest isolated,
    unsaturated stars, on cutouts sized from an iterated moment estimate.
    Returns dict of arrays for the good fits: x, y, fwhm, fwhm_x (major), fwhm_y (minor), ecc, beta, amp,
    flux, bg, idx (index into `stars`)."""
    lum = np.asarray(lum, np.float32)
    m = len(stars["x"])
    empty = {k: np.zeros(0) for k in _EMPTY} | {"idx": np.zeros(0, int)}
    if m == 0:
        return empty
    xy = np.stack([stars["x"], stars["y"]], 1)
    cand = np.arange(min(m, 3 * n))
    xi, yi = np.round(xy[cand, 0]).astype(int), np.round(xy[cand, 1]).astype(int)
    # first FWHM from the area above half maximum, in windows grown until they contain the star
    # (the 9x9 detection window truncates big stars; plain moments in big windows are noise dominated)
    f = np.maximum(np.asarray(stars["fwhm"], float)[cand], 1.5)
    Rmax = 40
    for _ in range(3):
        R = np.clip(np.ceil(1.6 * f) + 2, 4, Rmax).astype(int)
        Rc = int(R.max())
        cut = _cutouts(lum, xi, yi, Rc + 2)
        dy, dx = np.mgrid[-Rc - 2:Rc + 3, -Rc - 2:Rc + 3]
        rr = np.hypot(dx, dy)[None]
        ring = (rr >= R[:, None, None]) & (rr < R[:, None, None] + 2)
        bg = np.sort(np.where(ring, cut, np.nan).reshape(len(cand), -1), 1)     # NaN sorted last
        bg = bg[np.arange(len(cand)), np.maximum(np.isfinite(bg).sum(1) // 2, 0)]
        inside = rr <= R[:, None, None]
        pa = np.where(inside & np.isfinite(cut), cut - bg[:, None, None], 0)
        half = pa.max((1, 2)) / 2
        hm = pa > half[:, None, None]
        fl = (pa * hm).sum((1, 2)) + 1e-12
        cx, cy = (pa * hm * dx).sum((1, 2)) / fl, (pa * hm * dy).sum((1, 2)) / fl
        f = np.clip(2 * np.sqrt(hm.sum((1, 2)) / np.pi), 1.0, Rmax / 1.6)
    peak = np.nanmax(np.where(inside, cut, np.nan), axis=(1, 2))
    # isolation: no comparable star inside 1.5 R
    tree = cKDTree(xy)
    iso = np.array([not any(j != i and stars["flux"][j] > 0.05 * stars["flux"][i]
                            for j in tree.query_ball_point(xy[i], 1.5 * r)) for i, r in zip(cand, R)])
    good = iso & (peak < saturation) & np.isfinite(bg)
    if good.any():   # blends / galaxies / moment blow-ups would pad every cutout of the batch
        good &= f < 2 * np.median(f[good]) + 1
    sel = np.nonzero(good)[0][:n]
    if len(sel) == 0:
        return empty
    cand, xi, yi, f, bg, cx, cy = cand[sel], xi[sel], yi[sel], f[sel], bg[sel], cx[sel], cy[sel]
    Rf = np.clip(np.ceil(1.5 * f) + 2, 4, Rmax).astype(int)
    Rc = int(Rf.max())
    st = max(1, int(np.median(Rf)) // 10)         # big stars: every 2nd/3rd pixel is plenty
    cut = _cutouts(lum, xi, yi, Rc)[:, ::st, ::st]
    dy, dx = [a[None].astype(np.float64) for a in np.mgrid[-Rc:Rc + 1:st, -Rc:Rc + 1:st]]
    mask = ((dx ** 2 + dy ** 2) <= Rf[:, None, None] ** 2) & np.isfinite(cut)
    A0 = np.maximum(np.nanmax(np.where(mask, cut, np.nan), axis=(1, 2)) - bg, 1e-9)
    data = np.where(mask, (cut - bg[:, None, None]) / A0[:, None, None], 0)   # normalised: A ~ 1
    N = len(cand)
    s0 = f / (2 * np.sqrt(2 ** (1 / 3) - 1))      # beta = 3 start
    p = np.stack([np.ones(N), cx, cy, np.log(s0), np.log(s0), np.zeros(N), np.full(N, np.log(2.0)), np.zeros(N)], 1)
    wm = mask.astype(np.float64)

    def resid(p):
        return (data - _moffat(p, dx, dy)) * wm

    r = resid(p)
    cost = (r ** 2).sum((1, 2))
    lam = np.full(N, 1e-2)
    stall = np.zeros(N, int)
    for _ in range(iters):
        J = (_moffat(p, dx, dy, jac=True)[1] * wm[:, None]).reshape(N, 8, -1)
        Hm = J @ J.transpose(0, 2, 1)
        g = J @ r.reshape(N, -1, 1)
        D = np.einsum("nii->ni", Hm) + 1e-12
        step = np.linalg.solve(Hm + (lam[:, None] * D)[:, :, None] * np.eye(8), g)[..., 0]
        pn = p + step
        pn[:, 3:5] = np.clip(pn[:, 3:5], np.log(0.2), np.log(4.0 * Rmax))
        pn[:, 6] = np.clip(pn[:, 6], np.log(0.1), np.log(12.0))
        rn = resid(pn)
        cn = (rn ** 2).sum((1, 2))
        better = cn < cost
        conv = ~better | (cost - cn < 1e-6 * cost)
        p[better], r[better], cost[better] = pn[better], rn[better], cn[better]
        lam = np.where(better, lam / 3, lam * 4)
        stall = np.where(conv, stall + 1, 0)
        if np.all((lam > 1e6) | (stall >= 3)):
            break
    A, x0, y0 = p[:, 0], p[:, 1], p[:, 2]
    sx, sy, beta = np.exp(p[:, 3]), np.exp(p[:, 4]), 1 + np.exp(p[:, 6])
    k2 = 2 * np.sqrt(2 ** (1 / beta) - 1)
    fx, fy = sx * k2, sy * k2
    rel = np.sqrt(cost / np.maximum(wm.sum((1, 2)), 1)) / np.maximum(np.abs(A), 1e-9)
    ok = (A > 0) & (np.hypot(x0, y0) < 0.5 * Rf) & (np.minimum(fx, fy) > 0.8) & (np.maximum(fx, fy) < 2 * Rf) \
        & (rel < 0.3)
    A = A * A0
    out = {"x": (xi + x0)[ok], "y": (yi + y0)[ok], "fwhm": ((fx + fy) / 2)[ok], "fwhm_x": np.maximum(fx, fy)[ok],
           "fwhm_y": np.minimum(fx, fy)[ok], "beta": beta[ok], "amp": A[ok],
           "flux": (A * np.pi * sx * sy / (beta - 1))[ok], "bg": (bg + p[:, 7] * A0)[ok], "idx": cand[ok]}
    out["ecc"] = np.sqrt(np.clip(1 - (out["fwhm_y"] / np.maximum(out["fwhm_x"], 1e-9)) ** 2, 0, 1))
    return out


def quality(lum, stars, scale=1.0, psf=True, psf_stars=100, saturation=0.95):
    n = len(stars["x"])
    noise = stars["sigma"]
    if n == 0:
        return Quality(0, 0, 0, stars["bg"], noise, 0, 0, 0, False, "no stars")
    bright = slice(0, max(5, n // 2))
    fwhm = float(np.median(stars["fwhm"][bright])) * scale
    ecc = float(np.median(stars["ecc"][bright]))
    snr = float(np.median(stars["flux"][bright])) / max(noise, 1e-9)
    psf_signal = 0.0
    if psf:
        fit = fit_psf(lum, stars, psf_stars, saturation)
        stars["psf"] = fit
        if len(fit["fwhm"]) >= 5:
            fwhm = float(np.median(fit["fwhm"])) * scale
            ecc = float(np.median(fit["ecc"]))
            # mean PSF flux of the brightest fitted stars over the pixel noise (squared = inverse variance weight)
            top = np.sort(fit["flux"])[::-1][:max(5, len(fit["flux"]) // 2)]
            psf_signal = float(np.mean(top)) / max(mrs_noise(lum), 1e-12)
    return Quality(n, fwhm, ecc, stars["bg"], noise, snr, psf_signal=psf_signal)


def score_frames(qs, fwhm_sigma=2.0, ecc_max=0.75, stars_min_frac=0.4):
    """Weights (PSF signal^2 when every frame has it, else 1/noise^2) and automatic rejection, in place."""
    ok = [q for q in qs if q.stars > 0]
    if not ok:
        return qs
    f = np.array([q.fwhm for q in ok])
    fmed = np.median(f)
    fmad = np.median(np.abs(f - fmed)) * 1.4826 + 1e-6
    smax = max(q.stars for q in ok)
    use_psf = all(q.psf_signal > 0 for q in ok)
    # a whole session with slightly elongated stars (tracking, optics, wind) keeps its frames: only frames clearly
    # worse than the session are rejected (IC444 SII: 7 of 12 good frames at 0.75-0.77 were dropped by 0.75 alone)
    ecc_lim = max(ecc_max, float(np.median([q.eccentricity for q in ok])) + 0.05)
    for q in qs:
        if q.stars == 0:
            continue
        q.weight = q.psf_signal ** 2 if use_psf else 1.0 / max(q.noise, 1e-9) ** 2
        q.score = (q.snr / max(q.fwhm, 0.1)) * (1 - q.eccentricity)
        reasons = []
        if q.fwhm > fmed + fwhm_sigma * fmad and q.fwhm > 1.15 * fmed:
            reasons.append(f"FWHM {q.fwhm:.2f} > {fmed:.2f}")
        if q.eccentricity > ecc_lim:
            reasons.append(f"eccentricity {q.eccentricity:.2f}")
        if q.stars < stars_min_frac * smax:
            reasons.append(f"few stars ({q.stars})")
        q.accepted = not reasons
        q.reason = ", ".join(reasons)
    wmax = max(q.weight for q in ok)
    smax = max(q.score for q in ok) or 1
    for q in qs:
        q.weight /= wmax
        q.score = 100 * q.score / smax
    return qs
