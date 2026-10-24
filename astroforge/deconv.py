"""Deconvolution: Moffat PSF measured on the stars, Richardson-Lucy with total variation (RL-TV, Dey et al. 2006),
deringing by local clamping and optional spatially varying PSF by overlapped tiles."""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from . import gpu as _gpu
from .color import detect_stars
from .stats import luminance


def _fwhm_moments(lum, ys, xs, r=6):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    sig = []
    for y, x in zip(ys[:200], xs[:200]):
        if y < r or x < r or y >= lum.shape[0] - r or x >= lum.shape[1] - r:
            continue
        p = lum[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64)
        p = np.clip(p - np.median(p), 0, None)
        if p.sum() > 0:
            sig.append(np.sqrt((p * (xx ** 2 + yy ** 2)).sum() / p.sum() / 2))
    return float(np.median(sig)) * 2.355 if sig else 3.0


def _moffat(p, xx, yy):
    B, A, x0, y0, a1, a2, th, beta = p
    dx, dy = xx - x0, yy - y0
    c, s = np.cos(th), np.sin(th)
    q = ((dx * c + dy * s) / a1) ** 2 + ((-dx * s + dy * c) / a2) ** 2
    return B + A * (1 + q) ** -beta


def _alpha2fwhm(a, beta):
    return 2 * a * np.sqrt(2 ** (1 / beta) - 1)


def estimate_psf(img, max_stars=60, saturation=0.9):
    """Elliptical Moffat fit on unsaturated, isolated stars. `img` (H, W) or (H, W, C).

    Returns dict: fwhm (equivalent circular, px), fwhm_major, fwhm_minor, beta, ell (1 - minor/major),
    angle (radians, major axis from +x), n (stars fitted; 0 = fallback guess).
    """
    lum = luminance(img) if img.ndim == 3 else img
    lum = np.ascontiguousarray(lum, np.float32)
    h, w = lum.shape
    small = cv2.resize(lum, (max(1, w // 16), max(1, h // 16)), interpolation=cv2.INTER_AREA)
    det = lum - cv2.resize(cv2.medianBlur(small, 5), (w, h), interpolation=cv2.INTER_LINEAR)   # gradients out
    ys, xs = detect_stars(det, max_stars=1500, saturation=np.inf)
    keep = lum[ys, xs] < saturation
    ys, xs = ys[keep], xs[keep]
    f0 = _fwhm_moments(lum, ys, xs)
    out = {"fwhm": f0, "fwhm_major": f0, "fwhm_minor": f0, "beta": 4.0, "ell": 0.0, "angle": 0.0, "n": 0}
    if len(ys) < 3:
        return out
    r = int(np.clip(round(2.5 * f0) + 2, 5, 30))
    d, _ = cKDTree(np.c_[xs, ys]).query(np.c_[xs, ys], k=2)
    iso = d[:, 1] > 1.5 * r
    ok = iso & (ys >= r) & (xs >= r) & (ys < h - r) & (xs < w - r)
    ys, xs = ys[ok][:max_stars], xs[ok][:max_stars]
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    a0 = f0 / _alpha2fwhm(1.0, 3.0)
    lo = [-np.inf, 0, -2, -2, 0.3, 0.3, -2 * np.pi, 1.0]
    hi = [np.inf, np.inf, 2, 2, 3 * r, 3 * r, 2 * np.pi, 15.0]
    fits = []
    for y, x in zip(ys, xs):
        cut = lum[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64)
        if cut.max() >= saturation:
            continue
        edge = np.r_[cut[0], cut[-1], cut[:, 0], cut[:, -1]]
        B = np.median(edge)
        p0 = [B, cut[r, r] - B, 0, 0, a0, a0 * 0.95, 0.1, 3.0]
        try:
            res = least_squares(lambda p: (_moffat(p, xx, yy) - cut).ravel(), p0, bounds=(lo, hi),
                                x_scale="jac", max_nfev=200)
        except ValueError:
            continue
        if not res.success:
            continue
        B, A, x0, y0, a1, a2, th, beta = res.x
        f1, f2 = _alpha2fwhm(a1, beta), _alpha2fwhm(a2, beta)
        if f2 > f1:
            f1, f2, th = f2, f1, th + np.pi / 2
        if A <= 0 or not 0.5 < f1 < 2.5 * r:
            continue
        fits.append((f1, f2, beta, th % np.pi))
    if len(fits) < 3:
        return out
    f = np.array(fits)
    ell = 1 - f[:, 1] / f[:, 0]
    ang = 0.5 * np.angle(np.sum(ell * np.exp(2j * f[:, 3])))
    fmaj, fmin = float(np.median(f[:, 0])), float(np.median(f[:, 1]))
    return {"fwhm": float(np.sqrt(fmaj * fmin)), "fwhm_major": fmaj, "fwhm_minor": fmin,
            "beta": float(np.median(f[:, 2])), "ell": float(np.median(ell)), "angle": float(ang % np.pi),
            "n": len(fits)}


def _grid(fwhm, ell, angle, extent):
    major, minor = fwhm / np.sqrt(1 - ell), fwhm * np.sqrt(1 - ell)
    r = int(np.clip(np.ceil(extent * major), 2, 30))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    c, s = np.cos(angle), np.sin(angle)
    return (xx * c + yy * s) / major, (-xx * s + yy * c) / minor


def moffat_kernel(fwhm, beta=4.0, ell=0.0, angle=0.0):
    """Normalized Moffat kernel; `fwhm` is the equivalent circular FWHM in pixels."""
    c = _alpha2fwhm(1.0, beta)                                    # FWHM / alpha
    u, v = _grid(fwhm, ell, angle, np.sqrt(1e-3 ** (-1 / beta) - 1) / c)   # out to 1e-3 of the peak
    k = (1 + c * c * (u * u + v * v)) ** -beta
    return (k / k.sum()).astype(np.float32)


def gaussian_kernel(fwhm, ell=0.0, angle=0.0):
    u, v = _grid(fwhm, ell, angle, 1.6)
    k = np.exp(-4 * np.log(2) * (u * u + v * v))
    return (k / k.sum()).astype(np.float32)


def _tv_div(u, eps=1e-5):
    """Divergence of the normalized gradient (forward differences); u numpy or cupy, 2-D."""
    xp = _gpu.array_module(u)
    gx = xp.zeros_like(u)
    gy = xp.zeros_like(u)
    xp.subtract(u[:, 1:], u[:, :-1], out=gx[:, :-1])
    xp.subtract(u[1:], u[:-1], out=gy[:-1])
    n = xp.sqrt(gx * gx + gy * gy + eps * eps)
    gx /= n
    gy /= n
    div = gx.copy()
    div[:, 1:] -= gx[:, :-1]
    div += gy
    div[1:] -= gy[:-1]
    return div


_EPS = 1e-7
GPU_BYTES = 112           # device bytes per pixel of a band (arrays, FFT buffers and plans, TV temporaries)
MIN_GPU_PIXELS = 1 << 18  # smaller images stay on the CPU (transfers cost more than they save)


def _dering_lower(obs, psf):
    """Local minimum of the observed image (eroded over the PSF core, smoothed): the deringing floor."""
    rc = max(2, int((psf >= psf.max() * 0.05).sum(axis=1).max()))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rc + 1, 2 * rc + 1))
    return cv2.GaussianBlur(cv2.erode(obs, ker), (0, 0), rc / 2)


def _halo(psf, iterations):
    """Rows of context around a band. RL spreads information by up to 2 PSF radii per iteration, but the effect
    decays fast: 6 PSF widths (>= 48 px) keep the banded result within ~1e-4 of the global one."""
    k = max(psf.shape)
    return int(min(max(48, 6 * k), 2 * k * max(1, int(iterations)) + 8))


class _FFTCorr:
    """cv2.filter2D(a, -1, kernel, borderType=BORDER_REFLECT) for a fixed array shape, by FFT (numpy or cupy)."""

    def __init__(self, xp, shape, kernel):
        from scipy.fft import next_fast_len
        self.xp, self.fft = xp, _gpu.fft_module(xp)
        kh, kw = kernel.shape
        self.r = (kh // 2, kw // 2)
        self.k = (kh - 1, kw - 1)
        self.shape = tuple(shape)
        self.s = tuple(next_fast_len(n + k, real=True) for n, k in zip(shape, self.k))
        self.K = self._rfft(xp.asarray(np.ascontiguousarray(kernel[::-1, ::-1], np.float32)))

    def _rfft(self, a):
        if self.xp is np:
            return self.fft.rfft2(a, self.s, workers=-1)
        return self.fft.rfft2(a, self.s)

    def __call__(self, a):
        xp, (ry, rx), (ky, kx), (h, w) = self.xp, self.r, self.k, self.shape
        # BORDER_REFLECT = numpy "symmetric"; the anchor is the kernel centre (k // 2) as in OpenCV
        p = xp.pad(a, ((ry, ky - ry), (rx, kx - rx)), mode="symmetric")
        F = self._rfft(p)
        F *= self.K
        out = self.fft.irfft2(F, self.s, workers=-1) if xp is np else self.fft.irfft2(F, self.s)
        return xp.ascontiguousarray(out[ky:ky + h, kx:kx + w], dtype=xp.float32)


def _rl_core(xp, obs, lower, psf, iterations, tv_weight, dering):
    """RL-TV iterations on one 2-D band, written against xp (numpy or cupy); obs already >= eps."""
    psf = np.asarray(psf, np.float32)
    conv = _FFTCorr(xp, obs.shape, psf)
    conv_t = _FFTCorr(xp, obs.shape, np.ascontiguousarray(psf[::-1, ::-1]))
    est = obs.copy()
    tw, dr = np.float32(tv_weight), np.float32(dering)
    for _ in range(int(iterations)):
        c = conv(est)
        xp.maximum(c, _EPS, out=c)
        xp.divide(obs, c, out=c)
        corr = conv_t(c)
        if tv_weight > 0:
            d = _tv_div(est)
            d *= -tw
            d += 1
            xp.maximum(d, 0.5, out=d)
            corr /= d
        est *= corr
        if lower is not None:
            t = lower - est
            xp.maximum(t, 0, out=t)
            t *= dr
            est += t
    return est


def _row_chunks(h, n):
    step = max(16, -(-h // n))
    return [(a, min(h, a + step)) for a in range(0, h, step)]


def _rl_cpu(obs, lower, psf, iterations, tv_weight, dering, threads=None):
    """CPU RL-TV on row chunks in a thread pool (OpenCV and numpy release the GIL): OpenCV convolutions with a
    halo of PSF radius (exact, and 2-3x faster than FFT on the CPU and than one whole-image filter2D), then the
    per-pixel steps and the TV term. Same arithmetic as _rl_core."""
    psf = np.asarray(psf, np.float32)
    psf_t = np.ascontiguousarray(psf[::-1, ::-1])
    h = obs.shape[0]
    r = psf.shape[0] // 2 + 1
    est = obs.copy()
    c = np.empty_like(obs)
    corr = np.empty_like(obs)
    tw, dr = np.float32(tv_weight), np.float32(dering)
    n = threads or os.cpu_count() or 4
    chunks = _row_chunks(h, 4 * n)

    def filt(src, k, a, b):
        e0, e1 = max(0, a - r), min(h, b + r)
        return cv2.filter2D(src[e0:e1], -1, k, borderType=cv2.BORDER_REFLECT)[a - e0:b - e0]

    def ratio(ab):
        a, b = ab
        c[a:b] = filt(est, psf, a, b)
        np.maximum(c[a:b], _EPS, out=c[a:b])
        np.divide(obs[a:b], c[a:b], out=c[a:b])

    def correction(ab):
        a, b = ab
        corr[a:b] = filt(c, psf_t, a, b)
        if tv_weight > 0:
            e0 = max(0, a - 1)
            d = _tv_div(est[e0:min(h, b + 1)])[a - e0:b - e0]
            d *= -tw
            d += 1
            np.maximum(d, 0.5, out=d)
            corr[a:b] /= d

    def update(ab):
        s = slice(*ab)
        est[s] *= corr[s]
        if lower is not None:
            t = lower[s] - est[s]
            np.maximum(t, 0, out=t)
            t *= dr
            est[s] += t

    with ThreadPoolExecutor(n) as ex:
        for _ in range(int(iterations)):
            for step in (ratio, correction, update):     # each pass completes before the next one reads
                list(ex.map(step, chunks))
    return est


def _rl_bands(obs, lower, psf, iterations, tv_weight, dering, devs=(), rows=None, xp=None):
    """_rl_core on row bands with a halo of context (exact when one band covers the image). devs: GPU indices
    (bands spread with gpu.run_tasks); no devs = xp (numpy) on this thread. rows: band height (None = fit)."""
    h, w = obs.shape
    halo = _halo(psf, iterations)
    if rows is None:
        rows = h
        if devs:
            fit = _gpu.fit_rows(GPU_BYTES * (w + 2 * max(psf.shape)), h + 2 * halo, free=_gpu.min_free(devs))
            rows = h if fit >= h else max(16, fit - 2 * halo)
            if len(devs) > 1 and h >= 512 * len(devs):
                rows = min(rows, -(-h // (2 * len(devs))))
    rows = max(1, min(int(rows), h))
    bands = [(y, min(h, y + rows)) for y in range(0, h, rows)]
    out = np.empty_like(obs)

    def run(band, mod):
        y0, y1 = band
        a0, a1 = max(0, y0 - halo), min(h, y1 + halo)
        lw = None if lower is None else mod.asarray(lower[a0:a1])
        e = _rl_core(mod, mod.asarray(obs[a0:a1]), lw, psf, iterations, tv_weight, dering)
        out[y0:y1] = _gpu.to_host(e[y0 - a0:y1 - a0])

    if devs:
        cp = _gpu.cupy()
        _gpu.run_tasks(bands, lambda b, d: run(b, cp))
        _gpu.free_pool()
    else:
        for b in bands:
            run(b, xp or np)
    return out


def richardson_lucy(img, psf, iterations=20, tv_weight=0.002, dering=0.0, protect=None, device="auto", gpu=True,
                    log=None):
    """RL-TV on a 2D image. `dering` 0..1: fraction of the undershoot below the local minimum of the observed
    image removed at each iteration (1 = no dark rings around stars); `protect` optional mask (1 = original).
    device: "auto" (every GPU when CuPy works, else the CPU), a GPU index or list, "cpu"; gpu=False forces the
    CPU; "numpy" runs the GPU code (FFT convolutions) with numpy, for tests. On the GPU, images larger than the
    free memory are split in row bands with a halo of context."""
    obs = np.maximum(np.ascontiguousarray(img, np.float32), 0) + _EPS
    psf = np.asarray(psf, np.float32)
    lower = _dering_lower(obs, psf) if dering > 0 else None
    devs = _gpu.select(device, gpu, "deconvolution", log) if obs.size >= MIN_GPU_PIXELS else []
    est = None
    if device == "numpy":
        est = _rl_bands(obs, lower, psf, iterations, tv_weight, dering, xp=np)
    elif devs:
        try:
            est = _rl_bands(obs, lower, psf, iterations, tv_weight, dering, devs)
        except Exception as e:      # out of memory, driver...: the CPU gives the same result
            _gpu.free_pool()
            _gpu.log_once(f"deconvolution: GPU failed ({type(e).__name__}: {e}), using the CPU", log)
    if est is None:
        est = _rl_cpu(obs, lower, psf, iterations, tv_weight, dering)
    est -= _EPS
    if protect is not None:
        est = obs * protect + est * (1 - protect)
    return est.astype(np.float32)


def _kernel_for(img, psf, fwhm, ref=None):
    p = estimate_psf(img)
    if p["n"] < 5 and ref is not None:
        p = ref
    p = dict(p)
    if fwhm:
        p["fwhm"] = float(fwhm)
    if psf == "gaussian":
        return gaussian_kernel(p["fwhm"], p["ell"], p["angle"]), p
    return moffat_kernel(p["fwhm"], p["beta"], p["ell"], p["angle"]), p


def deconvolve(img, iterations=20, psf="auto", fwhm=None, tv_weight=0.002, dering=0.7, tiles=1, protect=None,
               log=None, device="auto", gpu=True):
    """Deconvolve a 2D image. psf "auto" (Moffat fit) or "gaussian"; `tiles` > 1 = PSF measured per tile on a
    tiles x tiles grid, tiles deconvolved with overlap and blended with cosine weights (spread over the GPUs
    when there are several). device / gpu: see richardson_lucy."""
    k, p = _kernel_for(img, psf, fwhm)
    if log:
        log(f"PSF {psf}: FWHM {p['fwhm']:.2f}px beta {p['beta']:.2f} ell {p['ell']:.2f} "
            f"angle {np.degrees(p['angle']):.0f} deg from {p['n']} stars")
    if tiles <= 1:
        return richardson_lucy(img, k, iterations, tv_weight, dering, protect, device, gpu, log)
    h, w = img.shape
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    ty, tx = np.linspace(0, h, tiles + 1).astype(int), np.linspace(0, w, tiles + 1).astype(int)
    ov = int(max(32, 0.15 * min(h, w) / tiles)) + k.shape[0]
    lock = threading.Lock()

    def tile(ij, dev):
        i, j = ij
        y0, y1 = max(0, ty[i] - ov), min(h, ty[i + 1] + ov)
        x0, x1 = max(0, tx[j] - ov), min(w, tx[j + 1] + ov)
        sub = img[y0:y1, x0:x1]
        kt, pt = _kernel_for(sub, psf, fwhm, ref=p)
        pr = None if protect is None else protect[y0:y1, x0:x1]
        res = richardson_lucy(sub, kt, iterations, tv_weight, dering, pr, dev, gpu, log)
        wt = np.outer(_taper(y0, y1, ty[i], ty[i + 1], ov, h), _taper(x0, x1, tx[j], tx[j + 1], ov, w))
        wt = wt.astype(np.float32)
        with lock:
            acc[y0:y1, x0:x1] += res * wt
            wsum[y0:y1, x0:x1] += wt
            if log:
                log(f"  tile {i},{j}: FWHM {pt['fwhm']:.2f}px ({pt['n']} stars)")

    cells = [(i, j) for i in range(tiles) for j in range(tiles)]
    devs = _gpu.select(device, gpu, "deconvolution", log) if device != "numpy" else []
    if len(devs) > 1:
        _gpu.run_tasks(cells, tile)              # one tile per GPU at a time, faster cards take more tiles
    else:
        for c in cells:
            tile(c, device)
    return acc / np.maximum(wsum, 1e-6)


def _taper(a0, a1, c0, c1, ov, n):
    """1 inside the core [c0, c1), cosine falloff over the overlap (not at the image borders)."""
    x = np.arange(a0, a1) + 0.5
    t = np.ones(len(x))
    if c0 > 0:
        t = np.where(x < c0, 0.5 + 0.5 * np.cos(np.pi * np.clip((c0 - x) / ov, 0, 1)), t)
    if c1 < n:
        t = np.where(x >= c1, 0.5 + 0.5 * np.cos(np.pi * np.clip((x - c1) / ov, 0, 1)), t)
    return np.maximum(t, 1e-3)
