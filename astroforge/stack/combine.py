"""Pixel rejection and combination of a stack of frames (masters and final integration).

All functions work on a chunk shaped (N, ...) and return (combined, rejected_low, rejected_high).
Rejection works on the per-pixel sorted stack (NaN = not covered, sorted last): every method keeps a
contiguous rank range [klo, khi), so the centre and sigma are re-estimated cheaply at each iteration.
low/high are in sigma units for every method (linear fit: mean abs deviation scaled to sigma).
"""
import ctypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .. import gpu

METHODS = ("auto", "none", "median", "percentile", "sigma", "winsorized", "linear_fit", "esd")
GPU_METHODS = METHODS[1:]   # every method runs on the device (combine_xp); grow_px dilation on the host
TEMP_FACTOR = 8         # bytes of temporaries per band ~ 8x the stacked band (sort, masks, loader copies)
_logged = set()


def auto_method(n):
    # plain sigma clipping cannot reject a single outlier below ~11 frames ((n-1)/sqrt(n) < 3): winsorized
    # (Huber sigma) works from 6 frames; linear fit is the most robust with many frames and residual gradients
    if n < 3:
        return "none"
    if n < 6:
        return "percentile"
    if n < 20:
        return "winsorized"
    return "linear_fit"


def available_memory():
    """Available physical RAM in bytes (psutil, else GlobalMemoryStatusEx / sysconf); 4 GB if unknown."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                           [(k, ctypes.c_ulonglong) for k in ("total", "avail", "tpage", "apage", "tvirt",
                                                              "avirt", "aext")]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.avail)
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return 4 << 30


def plan_bands(n_frames, row_elems, height, max_rows=256, workers=None, avail=None, fraction=0.5):
    """(rows, workers) so that workers * band temporaries stay within fraction of the available RAM."""
    budget = fraction * (available_memory() if avail is None else avail)
    per_row = max(1, n_frames * row_elems * 4 * TEMP_FACTOR)
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    # fewer threads before bands get too thin to be efficient
    workers = int(max(1, min(workers, budget // (32 * per_row))))
    rows = int(budget // (workers * per_row))
    rows = min(rows, max_rows, max(8, -(-height // (2 * workers))))     # >= 2 bands per worker
    return int(max(1, min(height, rows))), workers


def nanmedian(a, axis=0):
    """NaN-aware median via one sort (NaN sorted last), much faster than np.nanmedian on stacks."""
    a = np.moveaxis(np.asarray(a, np.float32), axis, -1)
    s = np.sort(a, axis=-1)
    cnt = (~np.isnan(s)).sum(-1)
    lo = np.take_along_axis(s, np.maximum((cnt - 1) // 2, 0)[..., None], -1)[..., 0]
    hi = np.take_along_axis(s, np.minimum(cnt // 2, s.shape[-1] - 1)[..., None], -1)[..., 0]
    return np.where(cnt > 0, 0.5 * (lo + hi), np.nan).astype(np.float32)


# ---- statistics on the sorted (P, N) stack over the rank range [klo, khi)
# Every helper is written against the array module of its input (numpy or cupy, gpu.array_module), so
# the CPU path and the GPU path run the very same code and give the same rejection.

def _xp(a):
    return np if isinstance(a, np.ndarray) else gpu.array_module(a)


def _take(s, k):
    xp = _xp(s)
    return s[xp.arange(s.shape[0]), xp.clip(k, 0, s.shape[1] - 1)]


def _rmedian(s, klo, khi):
    return 0.5 * (_take(s, (klo + khi - 1) // 2) + _take(s, (klo + khi) // 2))


def _rmask(n, klo, khi):
    r = _xp(klo).arange(n)[None]
    return (r >= klo[:, None]) & (r < khi[:, None])


def _rstd(s, m, k, c=None):
    xp = _xp(s)
    k = xp.maximum(k, 1).astype(xp.float32)
    if c is None:
        c = xp.where(m, s, 0).sum(1) / k
    d = xp.where(m, s - c[:, None], 0)
    return xp.sqrt(xp.einsum("ij,ij->i", d, d) / xp.maximum(k - 1, 1)), c


def _clip_range(s, klo, khi, lo_thr, hi_thr):
    xp = _xp(s)
    return xp.maximum(klo, (s < lo_thr[:, None]).sum(1)), xp.minimum(khi, (s <= hi_thr[:, None]).sum(1))


def _need(cnt):
    xp = _xp(cnt)
    return xp.maximum(xp.minimum(cnt, 2), cnt // 2)


def _iterate(step, s, cnt, iterations, *args):
    """Run step(s, cnt, klo, khi, *args) -> new range, only on the pixels still changing."""
    xp = _xp(s)
    klo, khi, need = xp.zeros_like(cnt), cnt.copy(), _need(cnt)
    act = xp.arange(len(cnt))
    for _ in range(iterations):
        sub = s if act.size == len(cnt) else s[act]
        lo, hi = klo[act], khi[act]
        nlo, nhi = step(sub, cnt[act], lo, hi, *args)
        ok = (nhi - nlo) >= need[act]       # never reject more than half of the frames
        nlo, nhi = xp.where(ok, nlo, lo), xp.where(ok, nhi, hi)
        ch = (nlo != lo) | (nhi != hi)
        klo[act], khi[act] = nlo, nhi
        act = act[ch]
        if not act.size:
            break
    return klo, khi


def _rmad(s, m, k, c):
    xp = _xp(s)
    a = xp.sort(xp.where(m, xp.abs(s - c[:, None]), xp.inf), axis=1)
    k = xp.maximum(k, 1)
    return 1.4826 * 0.5 * (_take(a, (k - 1) // 2) + _take(a, k // 2))


def _robust_sd(s, m, k, c):
    # with few frames an outlier inflates the std so much it can never be rejected: cap it with the MAD
    xp = _xp(s)
    sd, _ = _rstd(s, m, k)
    mad = _rmad(s, m, k, c)
    return xp.where((k >= 4) & (mad > 0), xp.minimum(sd, 1.5 * mad), sd)


def _sigma_step(s, cnt, klo, khi, low, high):
    c = _rmedian(s, klo, khi)                   # centre re-estimated on the surviving pixels
    sd = _robust_sd(s, _rmask(s.shape[1], klo, khi), khi - klo, c)
    return _clip_range(s, klo, khi, c - low * sd, c + high * sd)


def _winsor_step(s, cnt, klo, khi, low, high):
    xp = _xp(s)
    k = khi - klo
    m = _rmask(s.shape[1], klo, khi)
    c = _rmedian(s, klo, khi)
    sd = _robust_sd(s, m, k, c)
    act = xp.arange(len(k))
    for _ in range(10):                 # Huber: winsorize at c +- 1.5 sigma until sigma converges
        full = act.size == len(k)
        ss, ms, cs, sds = (s, m, c, sd.copy()) if full else (s[act], m[act], c[act], sd[act])
        w = xp.clip(ss, (cs - 1.5 * sds)[:, None], (cs + 1.5 * sds)[:, None])
        nsd, nc = _rstd(w, ms, k[act])
        nsd *= 1.134
        sd[act], c[act] = nsd, nc
        act = act[xp.abs(nsd - sds) > 1e-3 * xp.maximum(sds, 1e-12)]
        if not act.size:
            break
    return _clip_range(s, klo, khi, c - low * sd, c + high * sd)


def _linfit_step(s, cnt, klo, khi, low, high):
    """Line fitted to the central 60% of the sorted surviving pixels vs rank (robust to the ends).

    The slope of sorted normal data is ~2.6 sigma / n, which gives sigma; pixels farther than low/high
    sigma from the line are rejected from the ends inward.
    """
    xp = _xp(s)
    n = s.shape[1]
    r = xp.arange(n, dtype=xp.float32)[None]
    ri = xp.arange(n)[None]
    k = khi - klo
    cut = k // 5
    m = _rmask(n, klo + cut, khi - cut)
    kf = xp.maximum(m.sum(1), 1).astype(xp.float32)
    rm = xp.where(m, r, 0).sum(1) / kf
    ym = xp.where(m, s, 0).sum(1) / kf
    dr = xp.where(m, r - rm[:, None], 0)
    b = (dr * xp.where(m, s - ym[:, None], 0)).sum(1) / xp.maximum((dr * dr).sum(1), 1e-12)
    mk = _rmask(n, klo, khi)
    sd = (b * cnt / 2.6).astype(xp.float32)
    sd = xp.where(sd > 0, sd, _rstd(s, mk, k)[0])
    dev = s - (ym[:, None] + b[:, None] * (r - rm[:, None]))
    mid = ((klo + khi) // 2)[:, None]
    nlo = xp.where(mk & (dev < -low * sd[:, None]) & (ri < mid), ri, -1).max(1) + 1
    nhi = xp.where(mk & (dev > high * sd[:, None]) & (ri >= mid), ri, n).min(1)
    return xp.maximum(klo, nlo), xp.minimum(khi, nhi)


_ESD = {}


def _esd_table(n, rmax, alpha):
    key = (n, rmax, alpha)
    if key not in _ESD:
        from scipy import stats
        lam = np.full((n + 1, rmax + 1), np.inf, np.float64)
        for nn in range(3, n + 1):
            for i in range(1, rmax + 1):
                df = nn - i - 1
                if df < 1:
                    break
                t = stats.t.ppf(1 - alpha / (2 * (nn - i + 1)), df)
                lam[nn, i] = (nn - i) * t / np.sqrt((df + t * t) * (nn - i + 1))
        _ESD[key] = lam
    return _ESD[key]


def _esd(s, cnt, outliers=0.3, alpha=0.05):
    """Generalized extreme Studentized deviate test (Rosner): up to outliers*n removed, significance alpha.

    Vectorised: a fixed number of rounds (int(outliers * n)), each removing the most extreme end of the
    sorted surviving range; the result is the range after the last significant round.
    """
    xp = _xp(s)
    n = s.shape[1]
    rmax = int(outliers * n)
    klo, khi = xp.zeros_like(cnt), cnt.copy()
    if rmax < 1:
        return klo, khi
    lam = _esd_table(n, rmax, alpha)
    lam = lam if xp is np else xp.asarray(lam)
    c = _rmedian(s, klo, khi).astype(xp.float64)
    d = xp.where(_rmask(n, klo, khi), s - c[:, None], 0).astype(xp.float64)
    s1, s2 = d.sum(1), (d * d).sum(1)
    lim = xp.floor(outliers * cnt).astype(xp.int64)
    hist = [(klo, khi)]
    nout = xp.zeros_like(cnt)
    rows = xp.arange(len(cnt))
    for i in range(1, rmax + 1):
        k = (khi - klo).astype(xp.float64)
        mean = s1 / xp.maximum(k, 1)
        sd = xp.sqrt(xp.maximum(s2 - k * mean * mean, 0) / xp.maximum(k - 1, 1))
        vlo, vhi = d[rows, xp.clip(klo, 0, n - 1)], d[rows, xp.clip(khi - 1, 0, n - 1)]
        hi_side = (vhi - mean) >= (mean - vlo)
        v = xp.where(hi_side, vhi, vlo)
        R = xp.where(sd > 0, xp.abs(v - mean) / xp.maximum(sd, 1e-300), 0)
        live = i <= lim
        nout = xp.where(live & (R > lam[cnt, i]), i, nout)
        s1, s2 = xp.where(live, s1 - v, s1), xp.where(live, s2 - v * v, s2)
        klo = xp.where(live & ~hi_side, klo + 1, klo)
        khi = xp.where(live & hi_side, khi - 1, khi)
        hist.append((klo, khi))
    los = xp.stack([h[0] for h in hist])
    his = xp.stack([h[1] for h in hist])
    return los[nout, rows], his[nout, rows]


def _grow(hmask, grow_px, min_area=8):
    """Grow high rejections forming connected structures (trails) by grow_px in their frame (CPU, cv2)."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1,) * 2)
    out = np.zeros_like(hmask)
    for f in range(hmask.shape[0]):
        m = hmask[f].astype(np.uint8)
        if m.sum() < min_area:
            continue
        nl, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        big = np.zeros(nl, np.uint8)
        big[1:] = (st[1:, cv2.CC_STAT_AREA] >= min_area) & \
                  (np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= max(8, 2 * grow_px + 3))
        if big.any():
            out[f] = cv2.dilate(big[lab], ker).astype(bool)
    return out


def _large_scale(mask, layers, growth):
    """PixInsight's large-scale pixel rejection: rejected pixels that form structures at the scale of
    `layers` (2^layers px: planes, wide trails, glints) are rejected with their whole structure, grown by
    `growth` px. mask (N, H, W) bool -> (N, H, W) bool."""
    k = 2 ** (int(layers) + 1) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(growth) + 1,) * 2) if growth else None
    out = np.zeros_like(mask)
    for f in range(mask.shape[0]):
        m = mask[f]
        if m.sum() < k:
            continue
        d = cv2.boxFilter(m.astype(np.float32), -1, (k, k))
        big = (d > 0.25).astype(np.uint8)
        if not big.any():
            continue
        big = cv2.dilate(big, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))   # the whole structure
        if ker is not None:
            big = cv2.dilate(big, ker)
        out[f] = big.astype(bool)
    return out


def _resolve(method, n):
    if method == "auto":
        method = auto_method(n)
    if method not in METHODS:
        raise ValueError(f"unknown rejection method {method!r}")
    return method


def _reject(flat, method, low, high, iterations, esd_outliers, esd_alpha):
    """flat (N, P) with NaN = missing -> per pixel (lo_v, hi_v, med, cnt): the kept value range
    [lo_v, hi_v] (None for median), the median of the valid values (inf if none) and the valid count."""
    xp = _xp(flat)
    valid = ~xp.isnan(flat)
    cnt = valid.sum(0)
    # NaN -> inf before sorting: sorted last on every backend, never inside a kept range
    s = xp.sort(xp.where(valid, flat, xp.inf).T, axis=1)          # (P, N)
    zero = xp.zeros_like(cnt)
    med = _rmedian(s, zero, cnt)
    if method == "median":
        return None, None, med, cnt
    if method == "none":
        klo, khi = zero, cnt
    elif method == "percentile":
        # PixInsight bounds (-20 % / +10 % of the median), but never narrower than the stack noise: on a dark,
        # calibrated sky (median ~0.003) the frame-to-frame noise alone is ~10 % and would be rejected
        dev = xp.abs(flat - med)
        sig = 1.4826 * float(xp.nanmedian(dev)) if bool(valid.any()) else 0.0
        lo_b = med - xp.maximum(0.2 * xp.abs(med), low * sig)
        hi_b = med + xp.maximum(0.1 * xp.abs(med), high * sig)
        klo, khi = _iterate(lambda s, cnt, lo, hi: _clip_range(s, lo, hi, lo_b, hi_b), s, cnt, 1)
    elif method in ("sigma", "winsorized"):
        klo, khi = _iterate(_sigma_step if method == "sigma" else _winsor_step, s, cnt, iterations, low, high)
    elif method == "linear_fit":
        klo, khi = _iterate(_linfit_step, s, cnt, iterations, low, high)
        few = xp.flatnonzero(cnt < 8)           # a line through < 8 pixels is not robust: winsorized there
        if few.size:
            klo[few], khi[few] = _iterate(_winsor_step, s[few], cnt[few], iterations, low, high)
    else:
        klo, khi = _esd(s, cnt, esd_outliers, esd_alpha)
    return _take(s, klo), _take(s, khi - 1), med, cnt


def _finish(flat, lo_v, hi_v, med, cnt, weights, grow_px, shape, large_scale=None):
    """Masks of the rejected values, optional grow and large-scale rejection (numpy only), weighted mean of the
    kept ones. large_scale: {"high": layers, "low": layers, "growth": px} (0 / missing = off)."""
    xp = _xp(flat)
    n = flat.shape[0]
    if lo_v is None:                            # median
        out = xp.where(cnt > 0, med, xp.nan).astype(xp.float32).reshape(shape)
        z = xp.zeros(shape, xp.uint16)
        return out, z, z
    valid = ~xp.isnan(flat)
    rej_lo = valid & (flat < lo_v)
    rej_hi = valid & (flat > hi_v)
    if grow_px and len(shape) >= 2:
        if xp is not np:
            raise ValueError("grow_px runs on the CPU: call _finish with numpy arrays")
        hm = rej_hi.reshape((n,) + tuple(shape))
        hm2 = hm if len(shape) == 2 else hm.any(-1)
        g = _grow(hm2, int(grow_px))
        if len(shape) > 2:
            g = np.broadcast_to(g[..., None], hm.shape)
        rej_hi = (rej_hi | g.reshape(n, -1)) & valid & ~rej_lo
    if large_scale and len(shape) >= 2:
        if xp is not np:
            raise ValueError("large_scale runs on the CPU: call _finish with numpy arrays")
        for side in ("high", "low"):
            layers = int(large_scale.get(side) or 0)
            if not layers:
                continue
            m = (rej_hi if side == "high" else rej_lo).reshape((n,) + tuple(shape))
            m2 = m if len(shape) == 2 else m.any(-1)
            g = _large_scale(m2, layers, int(large_scale.get("growth", 2)))
            if len(shape) > 2:
                g = np.broadcast_to(g[..., None], m.shape)
            g = g.reshape(n, -1) & valid
            if side == "high":
                rej_hi = (rej_hi | g) & ~rej_lo
            else:
                rej_lo = (rej_lo | g) & ~rej_hi
    keep = valid & ~rej_lo & ~rej_hi
    w = keep.astype(xp.float32)
    if weights is not None:
        w *= xp.asarray(np.asarray(weights, np.float32).reshape(n, 1))
    den = w.sum(0)
    num = (xp.where(keep, flat, 0) * w).sum(0)
    out = xp.where(den > 0, num / xp.maximum(den, 1e-12), xp.where(cnt > 0, med, xp.nan)).astype(xp.float32)
    return (out.reshape(shape), rej_lo.sum(0).astype(xp.uint16).reshape(shape),
            rej_hi.sum(0).astype(xp.uint16).reshape(shape))


def combine(stack, method="auto", low=4.0, high=3.0, weights=None, iterations=5, grow_px=0,
            esd_outliers=0.3, esd_alpha=0.05, large_scale=None):
    """stack: float32 (N, ...); NaN marks missing pixels. weights: (N,) or None.

    grow_px > 0 (needs (N, H, W[, C]) stacks): high rejections forming structures (satellite / plane trails)
    are grown by grow_px in their frame, so the faint wings of the trail are rejected too.
    """
    stack = np.asarray(stack, np.float32)
    n, shp = stack.shape[0], stack.shape[1:]
    method = _resolve(method, n)
    with np.errstate(invalid="ignore", over="ignore"):    # inf - inf on pixels nobody covers
        flat = stack.reshape(n, -1)
        lo_v, hi_v, med, cnt = _reject(flat, method, low, high, iterations, esd_outliers, esd_alpha)
        return _finish(flat, lo_v, hi_v, med, cnt, weights, grow_px, shp, large_scale)


def _log_once(log, msg):
    if msg not in _logged:
        _logged.add(msg)
        (log or print)(msg)


GPU_BYTES = 48          # device bytes per stack element (sorted copy, sort scratch, masks, float temporaries)
GPU_BYTES_ESD = 80      # esd works in float64 and keeps the range history


def gpu_bytes_per_row(n_frames, row_elems, method):
    return n_frames * row_elems * (GPU_BYTES_ESD if method == "esd" else GPU_BYTES)


def _to_host(a, xp):
    return a if xp is np else xp.asnumpy(a)


def combine_xp(stack, method="auto", low=4.0, high=3.0, weights=None, iterations=5, grow_px=0,
               esd_outliers=0.3, esd_alpha=0.05, xp=None, rows=None, large_scale=None):
    """The GPU implementation of combine(), on any array module xp (cupy on the GPU, numpy in tests).

    stack: host float32 (N, H, ...). It is uploaded `rows` rows at a time along axis 1 (None = all), so
    a band bigger than the device memory is processed in pieces. Every method runs on the device with
    the same estimators as combine() (re-centred capped sigma, Huber winsorized, linear fit, vectorised
    ESD, percentile, median; NaN-aware, weighted mean). With grow_px the device returns the kept value
    range per pixel and the masks, the dilation of trail rejections (cv2) and the weighted mean are done
    on the host for the whole band, so trails crossing the pieces grow seamlessly. An out of memory on
    the device halves the piece and retries. Returns numpy (combined, rejected_low, rejected_high).
    """
    xp = xp or gpu.cupy() or np
    stack = np.asarray(stack, np.float32)
    n, shp = stack.shape[0], stack.shape[1:]
    method = _resolve(method, n)
    grow = (bool(grow_px) or bool(large_scale)) and stack.ndim >= 3 and method != "median"
    h = shp[0] if shp else 1
    rows = h if (rows is None or not shp) else max(1, min(int(rows), h))
    parts = []
    y0 = 0
    with np.errstate(invalid="ignore", over="ignore"):
        while y0 < h:
            y1 = min(h, y0 + rows)
            sub = stack[:, y0:y1] if shp else stack
            try:
                flat = xp.asarray(sub).reshape(n, -1)
                r = _reject(flat, method, low, high, iterations, esd_outliers, esd_alpha)
                if grow:
                    parts.append(tuple(None if a is None else _to_host(a, xp) for a in r))
                else:
                    parts.append(tuple(_to_host(a, xp) for a in _finish(flat, *r, weights, 0, sub.shape[1:])))
                del flat, r
            except Exception as e:
                if xp is np or "OutOfMemory" not in type(e).__name__ or rows == 1:
                    raise
                gpu.free_pool()
                rows = max(1, rows // 2)
                continue
            y0 = y1
        if not grow:
            if not shp:
                return parts[0]
            return tuple(np.concatenate([p[i] for p in parts], 0) for i in range(3))
        lo_v, hi_v, med, cnt = (np.concatenate([p[i] for p in parts]) for i in range(4))
        return _finish(stack.reshape(n, -1), lo_v, hi_v, med, cnt, weights, grow_px, shp, large_scale)


def combine_gpu(stack, method="auto", low=4.0, high=3.0, weights=None, iterations=5, grow_px=0, log=None, **kw):
    """combine() on the current CuPy device (every method; grow_px dilation on the host); CPU without CuPy.

    The band is split into pieces that fit in the free memory of this device (gpu.fit_rows), so cards
    with different memory work on the same bands.
    """
    cp = gpu.cupy()
    if cp is None:
        _log_once(log, "  CuPy not available, rejection on the CPU")
        return combine(stack, method, low, high, weights, iterations, grow_px, **kw)
    n = stack.shape[0]
    method = _resolve(method, n)
    row_elems = int(np.prod(stack.shape[2:])) if stack.ndim > 2 else 1
    rows = gpu.fit_rows(gpu_bytes_per_row(n, row_elems, method), stack.shape[1] if stack.ndim > 1 else 1)
    return combine_xp(stack, method, low, high, weights, iterations, grow_px, xp=cp, rows=rows, **kw)


def combine_chunked(frames, loader, method="auto", low=4.0, high=3.0, weights=None, rows=256, log=None,
                    workers=None, grow_px=0, mem_fraction=0.5, **kw):
    """Combine images too big to keep in memory together, row bands spread over GPUs and CPU threads.

    frames: list of items; loader(item, y0, y1) -> float32 rows of that frame (must be thread safe
    after the first call for each item). rows is the maximum band height: bands and CPU threads are
    sized from the available RAM. Returns (combined, rejected_low, rejected_high) per-pixel counts.
    """
    for f in frames:            # warm up loaders that cache whole frames, before going parallel
        first = loader(f, 0, 1)
    h = loader.height
    out = np.empty((h,) + first.shape[1:], np.float32)
    rl = np.empty(out.shape, np.uint16)
    rh = np.empty(out.shape, np.uint16)
    devs = gpu.devices()
    if devs:
        # bands limited by host RAM (one band in flight per GPU) and >= 4 bands per GPU, so a slower card
        # does not leave the faster one waiting at the end; each GPU splits a band further when its own
        # free memory is smaller (combine_gpu -> gpu.fit_rows), so the smallest card sets no limit
        rows = plan_bands(len(frames), first[0].size, h, 512, len(devs), fraction=mem_fraction)[0]
        rows = max(1, min(rows, -(-h // (4 * len(devs)))))
    else:
        rows, workers = plan_bands(len(frames), first[0].size, h, rows, workers, fraction=mem_fraction)
    bands = [(y0, min(h, y0 + rows)) for y0 in range(0, h, rows)]
    pad = 3 * int(grow_px) + 8 if grow_px else 0     # overlap: trails crossing band edges grow seamlessly
    ls = kw.get("large_scale")
    if ls:
        pad = max(pad, 2 ** (max(int(ls.get("high") or 0), int(ls.get("low") or 0)) + 2) + 3 * int(ls.get("growth", 2)) + 8)
    done = [0]
    args = dict(grow_px=grow_px, **kw)

    def load(band):
        y0, y1 = band
        return np.stack([loader(f, max(0, y0 - pad), min(h, y1 + pad)) for f in frames])

    def store(band, res):
        y0, y1 = band
        a = y0 - max(0, y0 - pad)
        for dst, src in zip((out, rl, rh), res):
            dst[y0:y1] = src[a:a + y1 - y0]
        done[0] += 1
        if log and done[0] % 8 == 0:
            log(f"  combining {done[0]}/{len(bands)} bands")

    def cpu_work(band):
        store(band, combine(load(band), method, low, high, weights, **args))

    def gpu_work(band, dev):
        store(band, combine_gpu(load(band), method, low, high, weights, log=log, **args))

    if devs:
        gpu.run_tasks(bands, gpu_work, cpu_work, cpu_workers=0)
    else:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(cpu_work, bands))
    return out, rl, rh
