"""Drizzle integration (1x, 2x, 3x...) with optional Bayer drizzle and outlier rejection.

Each input pixel is shrunk by `pixfrac`, mapped onto a finer output grid with the registration
transform, and its value is spread over the output pixels it covers in proportion to the overlap area.

CPU: the mapped footprint (a parallelogram, the same for every pixel of an affine frame) has its exact
overlap with the output cells computed (Green's theorem) on a 128x128 grid of sub-cell positions, then
interpolated bilinearly per pixel: share of a pixel given to a cell off by <= 1e-3 (axis-aligned, worst
case; ~1e-5 with rotation), flux conserved exactly (the old 5x5 sub-points: steps of 4%). Tiles of the
frame run on a thread pool into one shared accumulator (no per-thread copy of the big output arrays).
GPU (CuPy): the footprint is sampled with k x k sub-points.

Distortion (register.Distortion): pixel centres go through the affine and the inverse distortion (evaluated
on a 32 px node grid of the frame, bilinear in between); the footprint of a tile is the parallelogram of the
Jacobian at the tile centre (LUTs cached by quantised Jacobian). Every pixel still deposits exactly its value.
Local normalization (localnorm model, reference coordinates): a/b are evaluated at the mapped position of each
input pixel (and CFA site colour) and value*a+b replaces scale_offset.

Kernels: "square" (exact overlap, default); "gaussian" (FWHM = pixfrac input pixels, cut at 3 sigma);
"lanczos3" (L3(u/pixfrac), support 3*pixfrac: interpolating only at pixfrac 1, sensible at scale ~1 with
many frames; negative lobes, the weight map can get close to 0 with sparse coverage). Smooth kernels use a
17x17-position LUT integrated with 8x8 points per output cell, rows normalised so flux is conserved.

Bayer drizzle: CFA pixels go only to their own color plane, so no demosaicing (no interpolation)
is ever done; with enough dithered frames every output pixel receives real R, G and B samples.
"""
import math
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

_CH = {"R": 0, "G": 1, "B": 2}
_LUT_N = 128
_TILE = 256
_RADIUS = {"square": 0.5, "gaussian": 3 / 2.3548, "lanczos3": 3.0}   # kernel half-size / pixfrac (input px)


def available_ram():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("tot", "avail", "totpf", "availpf", "totv", "availv", "ext")]
        m = MS()
        m.len = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 8 << 30


def _edge_area(px, py, qx, qy, cx, cy):
    """-integral of clamp(y - cy, 0, 1) dx along edge p->q with x clipped to [cx, cx+1] (broadcast)."""
    dx, dy = qx - px, qy - py
    with np.errstate(divide="ignore", invalid="ignore"):
        ta, tb = (cx - px) / dx, (cx + 1 - px) / dx
        lo = np.clip(np.minimum(ta, tb), 0, 1)
        hi = np.maximum(np.clip(np.maximum(ta, tb), 0, 1), lo)
        t = [lo, hi]
        for yy in (cy, cy + 1):
            tc = (yy - py) / dy
            t.append(np.clip(np.where(np.isfinite(tc), tc, lo), lo, hi))
    t = np.sort(np.stack(np.broadcast_arrays(*t)), axis=0)
    g = np.clip(py + t * dy - cy, 0, 1)
    out = -(np.diff(t, axis=0) * (g[1:] + g[:-1]) / 2).sum(0) * dx
    return np.where(dx == 0, 0.0, out)


def footprint_lut(A, pixfrac, n=_LUT_N):
    """Exact overlap fractions of the mapped (pixfrac-shrunk) pixel with the output cells around it.

    A: 2x2 linear part input -> output pixels. Returns (mn, (ny, nx), lut): mn = offset of the footprint
    bounding box from the pixel centre; lut[gy*(n+1)+gx, j*nx+i] = fraction of the footprint on window
    cell (j, i) when the bounding-box corner sits at (gx/n, gy/n) inside its cell.
    """
    h = pixfrac / 2
    cor = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ np.asarray(A, np.float64).T
    area = 0.5 * np.sum(cor[:, 0] * np.roll(cor[:, 1], -1) - np.roll(cor[:, 0], -1) * cor[:, 1])
    if area < 0:
        cor = cor[::-1]
    mn = cor.min(0)
    cor = cor - mn
    nx, ny = (int(math.ceil(v - 1e-9)) + 1 for v in cor.max(0))
    f = np.arange(n + 1) / n
    fy, fx, j, i = np.meshgrid(f, f, np.arange(ny), np.arange(nx), indexing="ij")
    lut = np.zeros(fy.shape)
    for k in range(4):
        p, q = cor[k], cor[(k + 1) % 4]
        lut += _edge_area(p[0] + fx, p[1] + fy, q[0] + fx, q[1] + fy, i, j)
    return mn, (ny, nx), lut.reshape((n + 1) ** 2, ny * nx) / abs(area)


def _lanczos3(x):
    return np.where(np.abs(x) < 3, np.sinc(x) * np.sinc(x / 3), 0.0)


def kernel_lut(A, pixfrac, kernel="gaussian", n=16, q=8):
    """footprint_lut for the smooth kernels: midpoint integration with q x q points per output cell, rows
    normalised to 1 (flux conserved). Same layout and meaning of mn / window / rows as footprint_lut."""
    A = np.asarray(A, np.float64)
    R = _RADIUS[kernel] * pixfrac
    cor = np.array([[-R, -R], [R, -R], [R, R], [-R, R]]) @ A.T
    mn = cor.min(0)
    nx, ny = (int(math.ceil(v - 1e-9)) + 1 for v in cor.max(0) - mn)
    f = np.arange(n + 1) / n
    sub = (np.arange(q) + 0.5) / q
    dx = (np.arange(nx)[:, None] + sub).ravel()[None, :] - (f[:, None] - mn[0])     # (gx, X)
    dy = (np.arange(ny)[:, None] + sub).ravel()[None, :] - (f[:, None] - mn[1])     # (gy, Y)
    Ai = np.linalg.inv(A)
    u = Ai[0, 0] * dx[None, :, None, :] + Ai[0, 1] * dy[:, None, :, None]          # (gy, gx, Y, X)
    v = Ai[1, 0] * dx[None, :, None, :] + Ai[1, 1] * dy[:, None, :, None]
    if kernel == "gaussian":
        k = np.exp(-(u * u + v * v) / (2 * (pixfrac / 2.3548) ** 2))
    else:
        k = _lanczos3(u / pixfrac) * _lanczos3(v / pixfrac)
    k = np.where((np.abs(u) <= R) & (np.abs(v) <= R), k, 0.0)
    k = k.reshape(n + 1, n + 1, ny, q, nx, q).sum((3, 5)).reshape((n + 1) ** 2, ny * nx)
    return mn, (ny, nx), k / k.sum(1, keepdims=True)


class _Mapper:
    """Input pixel -> reference (1x) coordinates: affine M, then the inverse of a register.Distortion (whose
    model maps reference p -> affine-mapped source p + E(p)), solved by fixed-point iteration on a node grid
    of the input frame and interpolated bilinearly."""

    def __init__(self, M, distortion, shape, grid=32):
        self.M, self.g, self.d = np.asarray(M, np.float64), grid, None
        if distortion is None:
            return
        H, W = shape
        xs = np.arange(0, max(W - 1, 1) + grid, grid, dtype=np.float64)
        ys = np.arange(0, max(H - 1, 1) + grid, grid, dtype=np.float64)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        q = np.stack(self._aff(gx.ravel(), gy.ravel()), 1)
        p = q.copy()
        for _ in range(4):
            p = q - distortion.residual(p)
        self.d = (p - q).reshape(len(ys), len(xs), 2)

    def _aff(self, x, y):
        M = self.M
        return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]

    def __call__(self, x, y):
        rx, ry = self._aff(x, y)
        if self.d is None:
            return rx, ry
        ny, nx = self.d.shape[:2]
        u, v = np.clip(x / self.g, 0, nx - 1), np.clip(y / self.g, 0, ny - 1)
        i, j = np.minimum(u.astype(np.int64), nx - 2), np.minimum(v.astype(np.int64), ny - 2)
        tx, ty = (u - i)[..., None], (v - j)[..., None]
        d = self.d
        e = (d[j, i] * (1 - tx) + d[j, i + 1] * tx) * (1 - ty) + (d[j + 1, i] * (1 - tx) + d[j + 1, i + 1] * tx) * ty
        return rx + e[..., 0], ry + e[..., 1]

    def jac(self, x, y):
        if self.d is None:
            return self.M[:, :2]
        h = self.g / 2
        rx, ry = self(np.array([x - h, x + h, x, x]), np.array([y, y, y - h, y + h]))
        return np.array([[rx[1] - rx[0], rx[3] - rx[2]], [ry[1] - ry[0], ry[3] - ry[2]]]) / (2 * h)


def _ln_ab(model, rx, ry):
    """a, b (N, C) of a localnorm model at reference coordinates: its cubic maps sampled every tile/16 px over
    the bounding box, then bilinear."""
    from .localnorm import _eval
    st = max(1, int(model["tile"]) // 16)
    x0, y0 = int(np.floor(rx.min())), int(np.floor(ry.min()))
    x1 = x0 + (int((rx.max() - x0) // st) + 2) * st
    y1 = y0 + (int((ry.max() - y0) // st) + 2) * st
    a = _eval(np.asarray(model["a"]), model["tile"], y0, y1, x0, x1, st)
    b = _eval(np.asarray(model["b"]), model["tile"], y0, y1, x0, x1, st)
    fx, fy = (rx - x0) / st, (ry - y0) / st
    ix, iy = np.minimum(fx.astype(np.int64), a.shape[1] - 2), np.minimum(fy.astype(np.int64), a.shape[0] - 2)
    tx, ty = (fx - ix)[:, None], (fy - iy)[:, None]

    def s(m):
        return (m[iy, ix] * (1 - tx) + m[iy, ix + 1] * tx) * (1 - ty) + (m[iy + 1, ix] * (1 - tx) + m[iy + 1, ix + 1] * tx) * ty
    return s(a), s(b)


def _normalize(vals, ch, ab, so):
    """vals (N, 1) CFA (ch = colour of each site) or (N, C); ab from _ln_ab (wins over so)."""
    if ab is not None:
        a, b = ab
        if ch is not None:
            r, k = np.arange(len(ch)), np.minimum(ch, a.shape[1] - 1)
            return vals * a[r, k][:, None] + b[r, k][:, None]
        return vals * a + b
    if so is not None:
        if ch is not None:
            return vals * so[ch, :1] + so[ch, 1:2]
        return vals * so[:, 0] + so[:, 1]
    return vals


class Drizzle:
    """Accumulator for one device. device=None -> CPU (numpy), else a CuPy GPU index.

    Several Drizzle objects (one per GPU) can share the frames of a stack; merge() sums them.
    kernel: "square" | "gaussian" | "lanczos3" (see the module doc).
    """

    def __init__(self, ref_shape, scale=2, pixfrac=0.9, channels=3, device=None, kernel="square"):
        if kernel not in _RADIUS:
            raise ValueError(f"unknown drizzle kernel {kernel!r}")
        self.scale = scale
        self.pixfrac = pixfrac
        self.kernel = kernel
        self.h = ref_shape[0] * scale
        self.w = ref_shape[1] * scale
        self.c = channels
        self.device = device
        self._luts = {}
        self._warned = set()
        n = self.h * self.w * channels
        if device is None:
            self.xp = np
            self.flux = np.zeros(n, np.float64)
            self.weight = np.zeros(n, np.float64)
        else:
            from .. import gpu
            cp = gpu.cupy()
            self.xp = cp
            with cp.cuda.Device(device):
                self.flux = cp.zeros(n, cp.float32)
                self.weight = cp.zeros(n, cp.float32)
        # sub-points per input pixel side: dense enough that the footprint covers output pixels evenly
        self.k = max(3, math.ceil(2 * scale * pixfrac) + 1)

    def add(self, img, M, weight=1.0, cfa_pattern=None, scale_offset=None, reject=None, distortion=None,
            local_norm=None, rows=256, workers=None):
        """img: (H, W) CFA or (H, W, C). M: 2x3 affine frame -> reference (1x pixels).

        scale_offset: per-channel (scale, offset) normalization applied to the values.
        reject: optional (H, W) bool mask of input pixels to ignore (outliers).
        distortion: register.Distortion fitted with this M (as given to register.warp), or None.
        local_norm: localnorm.fit model of the registered frame (reference coordinates); replaces scale_offset.
        workers: CPU threads (None = all cores, capped by free RAM); ignored on GPU.
        """
        if self.xp is np:
            return self._add_exact(img, M, weight, cfa_pattern, scale_offset, reject, workers, distortion, local_norm)
        return self._add_points(img, M, weight, cfa_pattern, scale_offset, reject, rows, distortion, local_norm)

    def _warn(self, msg):
        if msg not in self._warned:
            self._warned.add(msg)
            warnings.warn(msg)

    def _add_points(self, img, M, weight=1.0, cfa_pattern=None, scale_offset=None, reject=None, rows=256,
                    distortion=None, local_norm=None):
        """k x k sub-point sampling of the footprint (GPU path; works on CPU too, single-threaded).
        Distortion moves the pixel centres; the sub-points keep the affine shape. Kernels other than square
        are not supported here (square used)."""
        if self.kernel != "square":
            self._warn(f"drizzle: kernel {self.kernel!r} not available on the GPU path, square used")
        xp = self.xp
        H, W = img.shape[:2]
        k, s, pf = self.k, self.scale, self.pixfrac
        offs = ((np.arange(k) + 0.5) / k - 0.5) * pf
        oy, ox = np.meshgrid(offs, offs, indexing="ij")
        oy, ox = oy.ravel(), ox.ravel()
        sub_w = weight / (k * k)
        so = None if scale_offset is None else np.asarray(scale_offset, np.float64)
        chan_map = None
        if cfa_pattern:
            chan_map = np.empty((2, 2), np.int64)
            for i, ch in enumerate(cfa_pattern):
                chan_map[i // 2, i % 2] = _CH[ch]
        a, b, c = (float(v) for v in M[0])
        d, e, f = (float(v) for v in M[1])
        mapper = _Mapper(M, distortion, (H, W))
        off = (s - 1) / 2
        fdt = xp.float32 if xp is not np else np.float64
        for y0 in range(0, H, rows):
            y1 = min(H, y0 + rows)
            hy, hx = np.mgrid[y0:y1, 0:W]
            hy, hx = hy.ravel().astype(np.float64), hx.ravel().astype(np.float64)
            rx, ry = mapper(hx, hy)
            ab = None if local_norm is None else _ln_ab(local_norm, rx, ry)
            ch = None if chan_map is None else chan_map[hy.astype(np.int64) % 2, hx.astype(np.int64) % 2]
            vals = np.asarray(img[y0:y1], np.float64).reshape(-1, 1 if cfa_pattern else self.c)
            vals = xp.asarray(_normalize(vals, ch, ab, so).astype(np.float32 if xp is not np else np.float64))
            valid = xp.isfinite(vals).all(1)
            if reject is not None:
                valid &= ~xp.asarray(reject[y0:y1]).ravel()
            px, py = xp.asarray(rx, fdt)[valid], xp.asarray(ry, fdt)[valid]
            vals = vals[valid]
            if cfa_pattern:
                ch = xp.asarray(ch)[valid]
                vals = vals[:, 0]
            for j in range(k * k):
                ix = xp.floor(s * (px + (a * ox[j] + b * oy[j])) + off + 0.5).astype(xp.int64)
                iy = xp.floor(s * (py + (d * ox[j] + e * oy[j])) + off + 0.5).astype(xp.int64)
                inside = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
                base = (iy[inside] * self.w + ix[inside]) * self.c
                if cfa_pattern:
                    self._acc(base + ch[inside], vals[inside], sub_w)
                else:
                    vin = vals[inside]
                    for cc in range(self.c):
                        self._acc(base + cc, vin[:, cc], sub_w)

    def add_many(self, frames, workers=None, callback=None):
        """frames: iterable of add() argument tuples or dicts (a lazy generator is fine).

        The next frame is fetched in the background while the current one is drizzled;
        callback(n_done) after each frame. Returns the number of frames added.
        """
        it = iter(frames)
        n = 0
        with ThreadPoolExecutor(1) as pre:
            nxt = pre.submit(next, it, None)
            while (args := nxt.result()) is not None:
                nxt = pre.submit(next, it, None)
                if isinstance(args, dict):
                    self.add(**args, workers=workers)
                else:
                    self.add(*args, workers=workers)
                n += 1
                if callback:
                    callback(n)
        return n

    def workers_for(self, workers=None):
        """CPU threads: all cores (or `workers`), capped so the per-tile scratch buffers fit in free RAM."""
        n = workers or os.cpu_count() or 1
        cells = (math.ceil(3 * _RADIUS[self.kernel] * self.scale * self.pixfrac) + 1) ** 2
        per = _TILE * _TILE * cells * 8 * 12 + (_TILE * self.scale * 1.5) ** 2 * self.c * 8 * 3
        return int(max(1, min(n, available_ram() * 0.5 // per)))

    def _lut(self, A, cache=False):
        """Footprint LUT for the 2x2 linear part A (input -> output pixels); cache=True: shared by every
        A equal within 1/500 of the scale (distorted frames: one LUT per tile Jacobian)."""
        key = tuple(np.round(np.asarray(A).ravel() * (500 / self.scale)).astype(np.int64)) if cache else None
        hit = self._luts.get(key) if key is not None else None   # .get: a tile thread may clear the cache meanwhile
        if hit is not None:
            return hit
        if self.kernel == "square":
            (mn, (ny, nx), lut), n = footprint_lut(A, self.pixfrac), _LUT_N
        else:
            (mn, (ny, nx), lut), n = kernel_lut(A, self.pixfrac, self.kernel), 16
        keep = np.abs(lut).max(0) > 1e-12
        cy, cx = np.nonzero(keep.reshape(ny, nx))
        out = (mn, ny, nx, cy, cx, np.ascontiguousarray(lut[:, keep], np.float32), n)
        if key is not None:
            if len(self._luts) > 2048:
                self._luts.clear()
            self._luts[key] = out
        return out

    def _add_exact(self, img, M, weight, cfa_pattern, scale_offset, reject, workers, distortion=None, local_norm=None):
        H, W = img.shape[:2]
        s, c = self.scale, self.c
        M = np.asarray(M, np.float64)
        mapper = _Mapper(M, distortion, (H, W))
        fixed = None if distortion is not None else self._lut(s * M[:, :2])
        so = None if scale_offset is None else np.asarray(scale_offset, np.float64)
        if cfa_pattern:
            chan_map = np.empty((2, 2), np.int64)
            for i, ch in enumerate(cfa_pattern):
                chan_map[i // 2, i % 2] = _CH[ch]
        off = (s - 1) / 2 + 0.5       # reference pixel x -> output cell units (cell i = [i, i+1))
        flux3 = self.flux.reshape(self.h, self.w, c)
        wt3 = self.weight.reshape(self.h, self.w, c)
        lock = threading.Lock()

        def tile(y0, x0):
            y1, x1 = min(H, y0 + _TILE), min(W, x0 + _TILE)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            ch = None
            if cfa_pattern:
                vals = img[y0:y1, x0:x1].reshape(-1, 1).astype(np.float64)
                ch = chan_map[yy % 2, xx % 2].ravel()
            else:
                vals = img[y0:y1, x0:x1].reshape(-1, c).astype(np.float64)
            valid = np.isfinite(vals).all(1)
            if reject is not None:
                valid &= ~np.asarray(reject[y0:y1, x0:x1]).ravel()
            if not valid.any():
                return
            xx, yy, vals = xx.ravel()[valid], yy.ravel()[valid], vals[valid]
            if ch is not None:
                ch = ch[valid]
            rx, ry = mapper(xx.astype(np.float64), yy.astype(np.float64))
            vals = _normalize(vals, ch, None if local_norm is None else _ln_ab(local_norm, rx, ry), so)
            if fixed is None:
                mn, ny, nx, cy, cx, lut, N = self._lut(s * mapper.jac((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2), True)
            else:
                mn, ny, nx, cy, cx, lut, N = fixed
            # input pixel centre -> footprint bounding-box corner, in output cell units
            ux, uy = s * rx + (off + mn[0]), s * ry + (off + mn[1])
            bx, by = np.floor(ux), np.floor(uy)
            gx, gy = (ux - bx) * N, (uy - by) * N
            ix, iy = np.minimum(gx.astype(np.int64), N - 1), np.minimum(gy.astype(np.int64), N - 1)
            tx, ty = (gx - ix)[:, None], (gy - iy)[:, None]
            i0 = iy * (N + 1) + ix
            fr = np.take(lut, i0, axis=0)
            fr *= ((1 - tx) * (1 - ty) * weight).astype(np.float32)
            for o, a in ((1, tx * (1 - ty)), (N + 1, (1 - tx) * ty), (N + 2, tx * ty)):
                t = np.take(lut, i0 + o, axis=0)
                t *= (a * weight).astype(np.float32)
                fr += t
            bx, by = bx.astype(np.int64), by.astype(np.int64)
            bx0, by0, bx1, by1 = int(bx.min()), int(by.min()), int(bx.max()) + nx, int(by.max()) + ny
            ox0, oy0, ox1, oy1 = max(0, bx0), max(0, by0), min(self.w, bx1), min(self.h, by1)
            if ox1 <= ox0 or oy1 <= oy0:
                return
            bw, bh = ox1 - ox0, oy1 - oy0
            loc = ((by - oy0) * bw + (bx - ox0))[:, None] + (cy * bw + cx)
            if cfa_pattern:
                loc = loc * c + ch[:, None]
            if (ox0, oy0, ox1, oy1) == (bx0, by0, bx1, by1):    # whole tile lands inside the output
                pick = lambda a: a.ravel()
            else:
                ox, oy = bx[:, None] + cx, by[:, None] + cy
                inside = (ox >= ox0) & (ox < ox1) & (oy >= oy0) & (oy < oy1)
                pick = lambda a: a[inside]
            loc = pick(loc)
            if cfa_pattern:
                n = bh * bw * c
                fl = np.bincount(loc, pick(fr * vals), n).reshape(bh, bw, c)
                wt = np.bincount(loc, pick(fr), n).reshape(bh, bw, c)
            else:
                n = bh * bw
                fl = np.stack([np.bincount(loc, pick(fr * vals[:, k:k + 1]), n) for k in range(c)], -1)
                fl = fl.reshape(bh, bw, c)
                wt = np.bincount(loc, pick(fr), n).reshape(bh, bw, 1)
            with lock:
                flux3[oy0:oy1, ox0:ox1] += fl
                wt3[oy0:oy1, ox0:ox1] += wt

        jobs = [(y0, x0) for y0 in range(0, H, _TILE) for x0 in range(0, W, _TILE)]
        nw = min(self.workers_for(workers), len(jobs))
        if nw == 1:
            for j in jobs:
                tile(*j)
            return
        with ThreadPoolExecutor(nw) as ex:
            for f in [ex.submit(tile, *j) for j in jobs]:
                f.result()

    def _acc(self, idx, vals, w):
        if idx.size == 0:
            return
        if self.xp is not np:
            import cupyx
            cupyx.scatter_add(self.flux, idx, vals * w)
            cupyx.scatter_add(self.weight, idx, self.xp.full(idx.shape, w, self.xp.float32))
            return
        lo, hi = int(idx.min()), int(idx.max()) + 1
        rel = idx - lo
        self.flux[lo:hi] += np.bincount(rel, weights=vals * w, minlength=hi - lo)
        self.weight[lo:hi] += np.bincount(rel, minlength=hi - lo) * w

    def host_arrays(self):
        if self.xp is np:
            return self.flux, self.weight
        return self.xp.asnumpy(self.flux).astype(np.float64), self.xp.asnumpy(self.weight).astype(np.float64)

    @staticmethod
    def merge(parts):
        """Sum several accumulators (one per GPU) into a CPU one."""
        first = parts[0]
        out = Drizzle((first.h // first.scale, first.w // first.scale), first.scale, first.pixfrac, first.c,
                      kernel=first.kernel)
        for p in parts:
            fl, wt = p.host_arrays()
            out.flux += fl
            out.weight += wt
        return out

    def result(self, fill_holes=True, smooth=None):
        """smooth: gaussian sigma (output pixels) applied to flux and weight before dividing.

        With few frames each output pixel gets samples from few input positions and a fine mesh
        appears (worst with Bayer drizzle, where a color covers 1/4 or 1/2 of the sensor). Smoothing
        numerator and denominator together evens the coverage while keeping the drizzle sharpness.
        Default: 0.25 * scale.
        """
        flux, weight = self.host_arrays()
        flux = flux.reshape(self.h, self.w, self.c).astype(np.float32)
        weight = weight.reshape(self.h, self.w, self.c).astype(np.float32)
        if smooth is None:
            smooth = 0.25 * self.scale
        sig = list(smooth) if isinstance(smooth, (list, tuple)) else [smooth] * self.c
        for cc in range(self.c):
            if sig[cc] > 0:
                flux[..., cc] = cv2.GaussianBlur(flux[..., cc], (0, 0), sig[cc])
                weight[..., cc] = cv2.GaussianBlur(weight[..., cc], (0, 0), sig[cc])
        img = flux / np.maximum(weight, 1e-12)
        wmap = weight
        img[wmap <= 1e-9] = 0
        holes = wmap <= 1e-9
        if fill_holes and holes.any():
            # holes (too few frames / too much dither for the scale): fill from a blurred version
            for cc in range(self.c):
                ch = img[..., cc]
                m = holes[..., cc]
                if not m.any():
                    continue
                ch[m] = 0
                num = cv2.GaussianBlur(ch, (0, 0), 1.5)
                den = cv2.GaussianBlur((~m).astype(np.float32), (0, 0), 1.5)
                ch[m] = (num / np.maximum(den, 1e-6))[m]
        return img, wmap


def outlier_mask(cfa_or_rgb, M, reference, noise, background, scale_offset, cfa_pattern=None, k=4.0, rel=0.5,
                 distortion=None, local_norm=None):
    """Input pixels deviating from the integrated reference (satellites, planes, cosmics, hot pixels).

    Tolerance = k*noise + rel*(signal above background): stars change with seeing from frame to
    frame and must not be rejected, a satellite trail on dark sky must.
    distortion / local_norm: as in Drizzle.add (local_norm replaces scale_offset).
    """
    H, W = cfa_or_rgb.shape[:2]
    ab = None
    if distortion is None and local_norm is None:
        Minv = cv2.invertAffineTransform(M)
        # sample the reference at every input pixel position
        ref_in = cv2.warpAffine(reference, Minv, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    else:
        mp = _Mapper(M, distortion, (H, W))
        ref_in, ab = [], ([], [])
        for y0 in range(0, H, 256):
            y1 = min(H, y0 + 256)
            yy, xx = np.mgrid[y0:y1, 0:W]
            rx, ry = mp(xx.ravel().astype(np.float64), yy.ravel().astype(np.float64))
            r = cv2.remap(reference, rx.reshape(y1 - y0, W).astype(np.float32), ry.reshape(y1 - y0, W).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            ref_in.append(r.reshape(y1 - y0, W, -1))
            if local_norm is not None:
                a, b = _ln_ab(local_norm, rx, ry)
                ab[0].append(a.reshape(y1 - y0, W, -1))
                ab[1].append(b.reshape(y1 - y0, W, -1))
        ref_in = np.concatenate(ref_in)
        ab = (np.concatenate(ab[0]), np.concatenate(ab[1])) if local_norm is not None else None
    if ref_in.ndim == 2:
        ref_in = ref_in[..., None]
    so = np.asarray(scale_offset)
    if cfa_pattern:
        ch = np.empty((H, W), np.int64)
        for i, c in enumerate(cfa_pattern):
            ch[i // 2::2, i % 2::2] = _CH[c]
        if ab is not None:
            kk = np.minimum(ch, ab[0].shape[2] - 1)[..., None]
            val = cfa_or_rgb * np.take_along_axis(ab[0], kk, 2)[..., 0] + np.take_along_axis(ab[1], kk, 2)[..., 0]
        else:
            val = cfa_or_rgb * so[ch, 0] + so[ch, 1]
        ref = np.take_along_axis(ref_in, ch[..., None], axis=2)[..., 0]
        nz = np.asarray(noise)[ch]
        tol = k * nz + rel * np.abs(ref - np.asarray(background)[ch])
        return np.abs(val - ref) > tol
    val = cfa_or_rgb * ab[0] + ab[1] if ab is not None else cfa_or_rgb * so[:, 0] + so[:, 1]
    tol = k * np.asarray(noise) + rel * np.abs(ref_in - np.asarray(background))
    return (np.abs(val - ref_in) > tol).any(-1)
