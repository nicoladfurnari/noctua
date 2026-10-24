"""Total Generalized Variation (2nd order) denoising, PixInsight TGVDenoise-like.

min_u,w  1/2 |u - f|^2 + a1 * g |grad u - w| + a0 * g |E w|      (Chambolle-Pock primal-dual, Bredies 2010)

g = edge protection (1 on flat areas, -> 0 on gradients above the noise) x local support (less smoothing on
bright structures). Strengths are in units of the noise sigma, so the same values work on linear and stretched
data. Tiled with overlap, early stop on convergence per tile. CPU: tiles on a thread pool (numpy releases the
GIL). GPU (CuPy): the same tiles solved in batches of equal shape (one array (N, h, w) per batch, each tile with
its own convergence test), batches spread over the GPUs with gpu.run_tasks; the solver is the same xp code.
"""
import functools
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import gpu as _gpu

TILE, OVERLAP = 256, 16
_B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16
GPU_BYTES = 160           # device bytes per tile pixel (state, temporaries, coarse level, batch compaction)
MIN_GPU_PIXELS = 1 << 18  # smaller channels stay on the CPU


def noise_sigma(ch):
    """Gaussian noise sigma from the MAD of the first starlet layer."""
    w1 = ch - cv2.sepFilter2D(ch, -1, _B3, _B3, borderType=cv2.BORDER_REFLECT)
    d = w1[::2, ::2]
    return float(np.median(np.abs(d - np.median(d)))) / 0.6745 / 0.890


def edge_weight(ch, sigma, k=3.0):
    """1 / (1 + (|grad f_s| / (k sigma))^2), f_s = f smoothed by 1 px (noise gradients ~0.4 sigma)."""
    if k <= 0:
        return np.ones_like(ch)
    s = cv2.GaussianBlur(ch, (0, 0), 1.0)
    gx = cv2.Sobel(s, -1, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(s, -1, 0, 1, ksize=3, scale=0.125)
    return (1 / (1 + (gx * gx + gy * gy) / (k * sigma) ** 2)).astype(np.float32)


# ------------------------------------------------------------------ finite differences on the last two axes
def _dx(xp, u, out):
    xp.subtract(u[..., 1:], u[..., :-1], out=out[..., :-1])
    out[..., -1] = 0


def _dy(xp, u, out):
    xp.subtract(u[..., 1:, :], u[..., :-1, :], out=out[..., :-1, :])
    out[..., -1, :] = 0


def _divx(xp, p, out):        # out = -dx^T p
    xp.copyto(out[..., :-1], p[..., :-1])
    out[..., -1] = 0
    out[..., 1:] -= p[..., :-1]


def _divy_add(xp, p, out):    # out += -dy^T p
    out[..., :-1, :] += p[..., :-1, :]
    out[..., 1:, :] -= p[..., :-1, :]


def _cp(f, a1, a0, iterations, tol=0.0, state=None, full=True):
    """Chambolle-Pock iterations on f (H, W) or a batch (N, H, W) of independent problems, numpy or cupy.

    state = (u, w1, w2, p1, p2, q1, q2, q3) warm start. Every problem stops on its own convergence (mean change
    per iteration < tol, tested every 10 iterations). Returns (state, its): its an int for (H, W), an array of
    N ints for a batch; full=False returns only (u,) as state (less memory)."""
    xp = _gpu.array_module(f)
    single = f.ndim == 2
    if single:
        f, a1, a0 = f[None], a1[None], a0[None]
        state = None if state is None else [x[None] for x in state]
    ia1, ia0 = 1 / xp.maximum(a1, 1e-12), 1 / xp.maximum(a0, 1e-12)
    tau = sig = np.float32(1 / np.sqrt(12))
    tf, k = tau * f, np.float32(1 / (1 + tau))
    z = lambda: xp.zeros_like(tf)  # noqa: E731
    if state is None:
        u, w1, w2, p1, p2, q1, q2, q3 = f.copy(), z(), z(), z(), z(), z(), z(), z()
    else:
        u, w1, w2, p1, p2, q1, q2, q3 = (xp.ascontiguousarray(x, dtype=xp.float32) for x in state)
    ub, wb1, wb2 = u.copy(), w1.copy(), w2.copy()
    t1, t2, t3, n = z(), z(), z(), z()
    check, it = u.copy(), 0
    nall = f.shape[0]
    its = np.full(nall, int(iterations))
    live = np.arange(nall)            # problem index of each row of the working arrays
    res = None                        # results of the problems that stopped early

    def keep_rows(rows):
        nonlocal res
        if res is None:
            res = [xp.empty_like(tf) for _ in range(8 if full else 1)]
        src = (u, w1, w2, p1, p2, q1, q2, q3) if full else (u,)
        dst = xp.asarray(live[rows])
        for r, s in zip(res, src):
            r[dst] = s[xp.asarray(rows)]

    for it in range(1, iterations + 1):
        # dual p <- proj_a1(p + sig (grad ub - wb))
        _dx(xp, ub, t1); t1 -= wb1; t1 *= sig; p1 += t1                     # noqa: E702
        _dy(xp, ub, t2); t2 -= wb2; t2 *= sig; p2 += t2                     # noqa: E702
        xp.hypot(p1, p2, out=n); n *= ia1; xp.maximum(n, 1, out=n)          # noqa: E702
        p1 /= n; p2 /= n                                                    # noqa: E702
        # dual q <- proj_a0(q + sig E wb), |q| = sqrt(q11^2 + q22^2 + 2 q12^2)
        _dx(xp, wb1, t1); t1 *= sig; q1 += t1                               # noqa: E702
        _dy(xp, wb2, t2); t2 *= sig; q2 += t2                               # noqa: E702
        _dy(xp, wb1, t1); _dx(xp, wb2, t3); t1 += t3; t1 *= sig / 2; q3 += t1  # noqa: E702
        xp.multiply(q1, q1, out=n); xp.multiply(q2, q2, out=t1); n += t1    # noqa: E702
        xp.multiply(q3, q3, out=t1); t1 *= 2; n += t1                       # noqa: E702
        xp.sqrt(n, out=n); n *= ia0; xp.maximum(n, 1, out=n)                # noqa: E702
        q1 /= n; q2 /= n; q3 /= n                                           # noqa: E702
        # primal u <- (u + tau div p + tau f) / (1 + tau);  ub = 2 u_new - u_old
        _divx(xp, p1, t1); _divy_add(xp, p2, t1)                            # noqa: E702
        xp.copyto(ub, u)
        t1 *= tau; t1 += tf; u += t1; u *= k                                # noqa: E702
        xp.subtract(u, ub, out=ub); ub += u                                 # noqa: E702
        # primal w <- w + tau (p + div2 q);  wb = 2 w_new - w_old = w_new + step
        _divx(xp, q1, t1); _divy_add(xp, q3, t1); t1 += p1; t1 *= tau       # noqa: E702
        _divx(xp, q3, t2); _divy_add(xp, q2, t2); t2 += p2; t2 *= tau       # noqa: E702
        w1 += t1; w2 += t2                                                  # noqa: E702
        xp.add(w1, t1, out=wb1); xp.add(w2, t2, out=wb2)                    # noqa: E702
        if tol > 0 and it % 10 == 0:
            xp.subtract(u, check, out=t1)
            change = np.asarray(_gpu.to_host(xp.abs(t1).mean(axis=(1, 2))), np.float64)
            done = change / 10 < tol
            if done.any():
                its[live[done]] = it
                if done.all():
                    if res is not None:
                        keep_rows(np.arange(len(live)))
                        live = live[:0]
                    break
                keep_rows(np.nonzero(done)[0])
                rows = xp.asarray(np.nonzero(~done)[0])
                live = live[~done]
                tf, ia1, ia0, u, w1, w2, p1, p2, q1, q2, q3, ub, wb1, wb2, check = (
                    x[rows] for x in (tf, ia1, ia0, u, w1, w2, p1, p2, q1, q2, q3, ub, wb1, wb2, check))
                t1, t2, t3, n = z(), z(), z(), z()
            xp.copyto(check, u)
    if res is None:
        out = (u, w1, w2, p1, p2, q1, q2, q3) if full else (u,)
    else:
        if len(live):
            keep_rows(np.arange(len(live)))
        out = tuple(res)
    if single:
        return tuple(x[0] for x in out), int(its[0])
    return out, its


# ------------------------------------------------------------------ OpenCV-equivalent resampling, xp
@functools.lru_cache(maxsize=256)
def _resample_tab(n_src, n_dst, kind):
    """(index (n_dst, T), weight (n_dst, T)) of cv2.resize along one axis: INTER_AREA (downscale) or
    INTER_LINEAR (half-pixel centres, edge replicated), float32 weights like OpenCV."""
    taps = []
    scale = 1.0 / (n_dst / n_src)
    for d in range(n_dst):
        if kind == "area":
            f1 = d * scale
            f2 = f1 + scale
            cell = min(scale, n_src - f1)
            s1, s2 = int(np.ceil(f1)), int(np.floor(f2))
            s2 = min(s2, n_src - 1)
            s1 = min(s1, s2)
            t = []
            if s1 - f1 > 1e-3:
                t.append((s1 - 1, (s1 - f1) / cell))
            t += [(s, 1.0 / cell) for s in range(s1, s2)]
            if f2 - s2 > 1e-3:
                t.append((s2, min(min(f2 - s2, 1.0), cell) / cell))
        else:
            fx = np.float32((d + 0.5) * scale - 0.5)
            sx = int(np.floor(fx))
            fx = float(np.float32(fx - sx))
            if sx < 0:
                sx, fx = 0, 0.0
            if sx >= n_src - 1:
                sx, fx = n_src - 1, 0.0
            t = [(sx, 1.0 - fx), (min(sx + 1, n_src - 1), fx)]
        taps.append(t)
    T = max(len(t) for t in taps)
    idx = np.zeros((n_dst, T), np.int64)
    wt = np.zeros((n_dst, T), np.float32)
    for d, t in enumerate(taps):
        for j, (s, a) in enumerate(t):
            idx[d, j], wt[d, j] = s, a
    return idx, wt


def _resample(a, shape, kind):
    """cv2.resize(a, (w, h), INTER_AREA / INTER_LINEAR) on the last two axes of a numpy or cupy array."""
    xp = _gpu.array_module(a)
    for axis, n_dst in ((-1, shape[1]), (-2, shape[0])):     # horizontal pass first, as OpenCV
        n_src = a.shape[axis]
        if n_src == n_dst:
            continue
        idx, wt = _resample_tab(n_src, n_dst, kind)
        out = None
        for j in range(idx.shape[1]):
            i, w = xp.asarray(idx[:, j]), xp.asarray(wt[:, j])
            if axis == -1:
                term = a[..., i] * w
            else:
                term = a[..., i, :] * w[:, None]
            out = term if out is None else out + term
        a = out
    return a


def _solve(f, a1, a0, iterations, tol, coarse, ds, us):
    """TGV with the optional half-resolution warm start; f (H, W) or a batch (N, H, W)."""
    state = None
    h, w = f.shape[-2:]
    if coarse and min(h, w) >= 32:
        st, _ = _cp(ds(f), ds(a1) / 2, ds(a0) / 4, iterations, tol * 2)
        state = [us(x) * k for x, k in zip(st, (1, 0.5, 0.5, 2, 2, 4, 4, 4))]
    state, its = _cp(f, a1, a0, iterations, tol, state, full=False)
    return state[0], its


def tgv_tile(f, a1, a0, iterations=100, tol=0.0, coarse=True):
    """TGV2-L2 on one 2-D tile; a1, a0 = per-pixel weights (arrays) or scalars. Returns (u, iterations run).
    coarse: warm start from the same problem solved at half resolution (equivalent weights a1/2, a0/4, duals
    x2/x4, w /2), which converges ~2x faster than a cold start."""
    f = np.ascontiguousarray(f, np.float32)
    if f.shape[0] < 3 or f.shape[1] < 3:
        return f.copy(), 0
    a1 = np.full(f.shape, a1, np.float32) if np.isscalar(a1) else np.ascontiguousarray(a1, np.float32)
    a0 = np.full(f.shape, a0, np.float32) if np.isscalar(a0) else np.ascontiguousarray(a0, np.float32)
    h, w = f.shape
    ds = lambda a: cv2.resize(a, (w // 2, h // 2), interpolation=cv2.INTER_AREA)   # noqa: E731
    us = lambda a: cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)           # noqa: E731
    return _solve(f, a1, a0, iterations, tol, coarse, ds, us)


def tgv_batch(f, a1, a0, iterations=100, tol=0.0, coarse=True):
    """tgv_tile on a batch (N, h, w) of equal-size tiles (numpy or cupy, e.g. on the GPU): the same solver,
    every tile with its own convergence. Returns (u (N, h, w), iterations per tile)."""
    xp = _gpu.array_module(f)
    if f.shape[-2] < 3 or f.shape[-1] < 3:
        return f.copy(), np.zeros(f.shape[0], int)
    h, w = f.shape[-2:]
    ds = lambda a: _resample(a, (h // 2, w // 2), "area")     # noqa: E731
    us = lambda a: _resample(a, (h, w), "linear")             # noqa: E731
    f, a1, a0 = (xp.ascontiguousarray(x, dtype=xp.float32) for x in (f, a1, a0))
    return _solve(f, a1, a0, iterations, tol, coarse, ds, us)


def _windows(h, w):
    out = []
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            y0, x0 = max(0, y - OVERLAP), max(0, x - OVERLAP)
            out.append((y, x, y0, min(h, y + TILE + OVERLAP), x0, min(w, x + TILE + OVERLAP)))
    return out


def _channel_batched(f, a1, aspect, iterations, tol, out, devs, xp=None):
    """Tiles grouped by shape, solved in batches (cupy on devs via gpu.run_tasks, else xp = numpy)."""
    groups = {}
    for t in _windows(*f.shape):
        groups.setdefault((t[3] - t[2], t[5] - t[4]), []).append(t)
    free = _gpu.min_free(devs) if devs else 2 << 30
    tasks = []
    ntiles = sum(len(g) for g in groups.values())
    for (th, tw), g in groups.items():
        per = _gpu.fit_rows(GPU_BYTES * th * tw, len(g), free=free)
        if len(devs) > 1:
            per = min(per, max(1, -(-ntiles // (4 * len(devs)))))
        tasks += [g[i:i + per] for i in range(0, len(g), per)]
    its = []

    def run(task, mod):
        sl = [(slice(y0, y1), slice(x0, x1)) for _, _, y0, y1, x0, x1 in task]
        F = mod.asarray(np.stack([f[s] for s in sl]))
        A = mod.asarray(np.stack([a1[s] for s in sl]))
        u, n = tgv_batch(F, A, A * aspect, iterations, tol)
        u = _gpu.to_host(u)
        for k, (y, x, y0, _, x0, _) in enumerate(task):
            out[y:y + TILE, x:x + TILE] = u[k, y - y0:y - y0 + TILE, x - x0:x - x0 + TILE]
        its.extend(int(v) for v in n)

    if devs:
        cp = _gpu.cupy()
        _gpu.run_tasks(tasks, lambda t, d: run(t, cp))
        _gpu.free_pool()
    else:
        for t in tasks:
            run(t, xp or np)
    return max(its) if its else 0


def tgv_channel(f, strength, aspect=2.0, weight=None, iterations=100, tol=4e-3, sigma=None, threads=None,
                device="auto", gpu=True, log=None):
    """Denoise one channel. strength = a1 in noise sigmas; aspect = a0 / a1; weight (H, W) multiplies both;
    tol = convergence (mean change per iteration, in noise sigmas; 0 = run all iterations).
    device: "auto" (every GPU when CuPy works, else the CPU), GPU index or list, "cpu"; gpu=False forces the CPU;
    "numpy" runs the batched GPU code with numpy (tests)."""
    f = np.ascontiguousarray(f, np.float32)
    sigma = noise_sigma(f) if sigma is None else sigma
    if sigma <= 0 or strength <= 0:
        return f.copy(), 0
    a1 = np.float32(strength * sigma) * (weight if weight is not None else np.ones_like(f))
    a1 = np.ascontiguousarray(a1, np.float32)
    h, w = f.shape
    out = np.empty_like(f)
    devs = _gpu.select(device, gpu, "TGV", log) if f.size >= MIN_GPU_PIXELS else []
    if device == "numpy":
        return out, _channel_batched(f, a1, aspect, iterations, tol * sigma, out, [], np)
    if devs:
        try:
            return out, _channel_batched(f, a1, aspect, iterations, tol * sigma, out, devs)
        except Exception as e:      # out of memory, driver...: the CPU gives the same result
            _gpu.free_pool()
            _gpu.log_once(f"TGV: GPU failed ({type(e).__name__}: {e}), using the CPU", log)

    def job(t):
        y, x, y0, y1, x0, x1 = t
        aa = a1[y0:y1, x0:x1]
        u, n = tgv_tile(f[y0:y1, x0:x1], aa, aa * aspect, iterations, tol * sigma)
        out[y:y + TILE, x:x + TILE] = u[y - y0:y - y0 + TILE, x - x0:x - x0 + TILE]
        return n

    with ThreadPoolExecutor(threads or os.cpu_count() or 4) as ex:
        its = list(ex.map(job, _windows(h, w)))
    return out, max(its) if its else 0


# ------------------------------------------------------------------ colour: linear luma / chroma (Y, B-Y, R-Y)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def to_ycc(img):
    y = img @ LUMA
    return y, img[..., 2] - y, img[..., 0] - y


def from_ycc(y, cb, cr):
    r, b = y + cr, y + cb
    g = (y - LUMA[0] * r - LUMA[2] * b) / LUMA[1]
    return np.stack([r, g, b], -1).astype(np.float32)


def denoise(img, strength_l=1.5, strength_c=3.0, aspect=2.0, edge=3.0, support=None, iterations=100, tol=4e-3,
            chroma_scale=2, log=None, device="auto", gpu=True):
    """TGV denoise of a (H, W, C) image. RGB: luminance and chrominance (B-Y, R-Y) separately, chrominance
    optionally at 1/chroma_scale resolution. support (H, W) in [0, 1]: regularisation weight (1 = full).
    device / gpu: see tgv_channel (the iterations run on the GPU, the masks and resampling on the CPU)."""
    run = functools.partial(tgv_channel, device=device, gpu=gpu, log=log)
    if img.shape[-1] != 3:
        chans = [np.ascontiguousarray(img[..., c]) for c in range(img.shape[-1])]
        out = []
        for ch in chans:
            s = noise_sigma(ch)
            wgt = edge_weight(ch, s, edge) * (support if support is not None else 1)
            u, n = run(ch, strength_l, aspect, wgt, iterations, tol, s)
            out.append(u)
            if log:
                log(f"TGV: sigma {s:.2e}, {n} iterations")
        return np.clip(np.stack(out, -1), 0, 1).astype(np.float32)
    y, cb, cr = (np.ascontiguousarray(a, np.float32) for a in to_ycc(img))
    sy = noise_sigma(y)
    wy = edge_weight(y, sy, edge) * (support if support is not None else 1)
    yd, ny = run(y, strength_l, aspect, wy, iterations, tol, sy) if strength_l > 0 else (y, 0)
    h, w = y.shape
    k = max(1, int(chroma_scale))
    if k > 1:
        size = (max(3, w // k), max(3, h // k))
        small = lambda a: cv2.resize(a, size, interpolation=cv2.INTER_AREA)  # noqa: E731
        ys = small(y)
        wc = edge_weight(ys, noise_sigma(ys), edge) * (small(support) if support is not None else 1)
    else:
        small, wc = (lambda a: a), wy
    cc = []
    for c in (cb, cr):
        cs = small(c)
        if strength_c > 0:
            d, _ = run(cs, strength_c, aspect, wc, iterations, tol)
            # keep full-resolution chroma detail where nothing was removed (stars): add back the residual of the
            # downscale only through the edge weight complement
            if k > 1:
                up = lambda a: cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)  # noqa: E731
                d = up(d) + (c - up(cs)) * (1 - wy)
            cc.append(d)
        else:
            cc.append(c)
    if log:
        log(f"TGV: luminance sigma {sy:.2e}, {ny} iterations; chrominance at 1/{k}")
    return np.clip(from_ycc(yd, *cc), 0, 1)
