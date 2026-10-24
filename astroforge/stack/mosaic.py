"""Mosaics of partially overlapping panels: pairwise star registration (or WCS), union canvas, global
background matching (scale + offset per panel) and feathered blending."""
from collections import deque

import cv2
import numpy as np

from . import analyze, register

IDENT = np.array([[1, 0, 0], [0, 1, 0]], np.float64)


def _h(M):
    return np.vstack([M, [0, 0, 1]])


def _lum(img):
    a = np.asarray(img, np.float32)
    a = a.mean(-1) if a.ndim == 3 else a
    return np.ascontiguousarray(np.nan_to_num(a, nan=float(np.nanmedian(a))))


def _corners(shape):
    h, w = shape[:2]
    return np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])


def _wcs_affine(wcs, ref_wcs, shape, n=6):
    """Affine panel pixel -> reference panel pixel from two WCS (least squares on an n x n grid)."""
    h, w = shape[:2]
    gx, gy = np.meshgrid(np.linspace(0, w - 1, n), np.linspace(0, h - 1, n))
    ra, dec = wcs.pixel_to_world_values(gx.ravel(), gy.ravel())
    rx, ry = ref_wcs.world_to_pixel_values(ra, dec)
    A = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 1)
    return np.linalg.lstsq(A, np.stack([rx, ry], 1), rcond=None)[0].T


def plan(transforms, shapes):
    """Union canvas. transforms: per panel a 2x3 affine (panel pixel -> common frame, e.g. from register_panels)
    or an astropy WCS (the first panel's WCS is the frame). Returns ((H, W), [2x3 panel -> canvas])."""
    if transforms and hasattr(transforms[0], "pixel_to_world_values"):
        transforms = [_wcs_affine(wc, transforms[0], sh) for wc, sh in zip(transforms, shapes)]
    pts = np.vstack([register._affine(_corners(sh), np.asarray(M, np.float64)) for M, sh in zip(transforms, shapes)])
    x0, y0 = np.floor(pts.min(0) + 0.5)
    x1, y1 = np.ceil(pts.max(0) - 0.5)
    T = np.array([[1, 0, -x0], [0, 1, -y0]], np.float64)
    return (int(y1 - y0 + 1), int(x1 - x0 + 1)), [(_h(T) @ _h(np.asarray(M, np.float64)))[:2] for M in transforms]


def _match_stars(si, sj, min_matches):
    a = np.stack([si["x"], si["y"]], 1)
    b = np.stack([sj["x"], sj["y"]], 1)
    M, n, rms = register.match(a, b, max_points=min(len(a), len(b), 60))
    if n < min_matches:
        raise ValueError(f"only {n} matches")
    return M, n


def _match_ecc(li, lj, down=4):
    """Fallback: phase correlation for the shift, then ECC affine on the overlap (downsampled)."""
    h, w = max(li.shape[0], lj.shape[0]), max(li.shape[1], lj.shape[1])
    pad = [np.pad(x, ((0, h - x.shape[0]), (0, w - x.shape[1])), constant_values=float(np.median(x))) for x in (li, lj)]
    sm = [cv2.resize(x, (w // down, h // down), interpolation=cv2.INTER_AREA) for x in pad]
    sm = [np.log1p(np.maximum(x - np.median(x), 0) / (np.std(x) + 1e-9)).astype(np.float32) for x in sm]
    win = cv2.createHanningWindow(sm[0].shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(sm[1], sm[0], win)   # shift of panel i content relative to j
    W0 = np.array([[1, 0, -dx], [0, 1, -dy]], np.float32)
    mask = cv2.warpAffine(np.ones_like(sm[0], np.uint8), W0, sm[0].shape[::-1], flags=cv2.INTER_NEAREST)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    _, W = cv2.findTransformECC(sm[1], sm[0], W0, cv2.MOTION_AFFINE, crit, mask, 5)
    # W maps j (template) coords -> i coords at the small scale; invert, rescale to full resolution
    Mi = cv2.invertAffineTransform(W.astype(np.float64))
    S, Si = np.diag([down, down, 1.0]), np.diag([1 / down, 1 / down, 1.0])
    return (S @ _h(Mi) @ Si)[:2], 0


def register_panels(panels, min_matches=8, stars=None):
    """2x3 transforms panel -> panel 0 pixel frame, from pairwise matches of the overlapping stars
    (astroalign, cv2 phase correlation + ECC as fallback) chained along the best-matched pairs."""
    lums = [_lum(p) for p in panels]
    stars = stars or [analyze.detect(l, max_stars=800) for l in lums]
    n = len(panels)
    edges = {}
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            try:
                M, k = _match_stars(stars[i], stars[j], min_matches)
            except Exception:
                continue
            edges[(i, j)] = (M, k)
            edges[(j, i)] = (cv2.invertAffineTransform(M), k)
    T = {0: IDENT.copy()}
    q = deque([0])
    while q or len(T) < n:
        if not q:   # disconnected panels: image based fallback against any placed panel
            rest = [i for i in range(n) if i not in T]
            for i in rest:
                for j in list(T):
                    try:
                        M, _ = _match_ecc(lums[i], lums[j])
                    except cv2.error:
                        continue
                    T[i] = (_h(T[j]) @ _h(M))[:2]
                    q.append(i)
                    break
            if not q:
                raise ValueError(f"panels {[i for i in range(n) if i not in T]} could not be registered")
        j = q.popleft()
        cand = sorted(((k, i) for (i, jj), (M, k) in edges.items() if jj == j and i not in T), reverse=True)
        for k, i in cand:
            if i in T:
                continue
            T[i] = (_h(T[j]) @ _h(edges[(i, j)][0]))[:2]
            q.append(i)
    return [T[i] for i in range(n)]


def _blocks(a, b, k=8):
    """Block means of two aligned arrays over pixels valid in both (noise down by ~k)."""
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    if h == 0 or w == 0:
        return None
    a, b = a[:h, :w], b[:h, :w]
    v = np.isfinite(a) & np.isfinite(b)
    cnt = v.reshape(h // k, k, w // k, k).sum((1, 3))
    sa = np.where(v, a, 0).reshape(h // k, k, w // k, k).sum((1, 3))
    sb = np.where(v, b, 0).reshape(h // k, k, w // k, k).sum((1, 3))
    good = cnt >= k * k * 0.75
    return (sa[good] / cnt[good], sb[good] / cnt[good]) if good.sum() >= 4 else None


def _solve(pairs, n, scale, levels):
    """Global least squares for a_k I_k + b_k equal across overlaps; panel 0 fixed (a=1, b=0)."""
    nv = 2 * (n - 1)
    keep = [np.ones(len(x), bool) for _, _, x, _ in pairs]
    for it in range(3):
        rows, rhs = [], []
        for (i, j, x, y), kp in zip(pairs, keep):
            x, y = x[kp], y[kp]
            wgt = 1 / np.sqrt(max(len(x), 1))
            R = np.zeros((len(x), nv))
            r = np.zeros(len(x))
            for k, v, sgn in ((i, x, 1), (j, y, -1)):
                if k == 0:
                    r -= sgn * v
                else:
                    R[:, 2 * (k - 1)] = sgn * v
                    R[:, 2 * (k - 1) + 1] = sgn
            rows.append(R * wgt)
            rhs.append(r * wgt)
        # weak priors (a = 1, b = 0) keep isolated panels and flat overlaps well posed
        P = np.zeros((nv, nv))
        pr = np.zeros(nv)
        for k in range(1, n):
            P[2 * (k - 1), 2 * (k - 1)] = (1e-3 if scale else 1e3) * levels[k]
            pr[2 * (k - 1)] = (1e-3 if scale else 1e3) * levels[k]
            P[2 * (k - 1) + 1, 2 * (k - 1) + 1] = 1e-6
        A = np.vstack(rows + [P])
        sol = np.linalg.lstsq(A, np.concatenate(rhs + [pr]), rcond=None)[0]
        a = np.concatenate([[1.0], sol[0::2]])
        b = np.concatenate([[0.0], sol[1::2]])
        for q, ((i, j, x, y), kp) in enumerate(zip(pairs, keep)):
            res = (a[i] * x + b[i]) - (a[j] * y + b[j])
            s = 1.4826 * np.median(np.abs(res[kp] - np.median(res[kp]))) + 1e-12
            keep[q] = np.abs(res - np.median(res[kp])) < 3 * s
    return a, b


def blend(panels, transforms, canvas_shape, match_background=True, scale=True, feather=None, fill=0.0,
          return_info=False):
    """Warp each panel to the canvas (2x3 panel -> canvas, e.g. from plan), match backgrounds in the overlaps
    (a_k I_k + b_k per panel and channel, global least squares, panel 0 fixed) and feather blend with
    distance-to-edge weights (capped at `feather` px when given). Returns float32 (H, W, C)."""
    H, W = canvas_shape
    warped = []
    for img, M in zip(panels, transforms):
        img = np.asarray(img, np.float32)
        img = img[..., None] if img.ndim == 2 else img
        c = register._affine(_corners(img.shape), np.asarray(M, np.float64))
        x0, y0 = max(0, int(np.floor(c[:, 0].min()))), max(0, int(np.floor(c[:, 1].min())))
        x1, y1 = min(W, int(np.ceil(c[:, 0].max())) + 1), min(H, int(np.ceil(c[:, 1].max())) + 1)
        Ms = np.asarray(M, np.float64).copy()
        Ms[:, 2] -= (x0, y0)
        out = register.warp(img, Ms, (y1 - y0, x1 - x0))
        warped.append((y0, x0, out))
    n, C = len(warped), warped[0][2].shape[2]
    a = np.ones((n, C))
    b = np.zeros((n, C))
    if match_background and n > 1:
        for ch in range(C):
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    (yi, xi, A), (yj, xj, B) = warped[i], warped[j]
                    ty0, tx0 = max(yi, yj), max(xi, xj)
                    ty1, tx1 = min(yi + A.shape[0], yj + B.shape[0]), min(xi + A.shape[1], xj + B.shape[1])
                    if ty1 - ty0 < 16 or tx1 - tx0 < 16:
                        continue
                    s = _blocks(A[ty0 - yi:ty1 - yi, tx0 - xi:tx1 - xi, ch], B[ty0 - yj:ty1 - yj, tx0 - xj:tx1 - xj, ch])
                    if s is not None:
                        pairs.append((i, j, s[0], s[1]))
            if pairs:
                levels = [float(np.nanmedian(w[2][::8, ::8, ch])) for w in warped]
                a[:, ch], b[:, ch] = _solve(pairs, n, scale, levels)
    num = np.zeros((H, W, C), np.float32)
    den = np.zeros((H, W), np.float32)
    for k, (y0, x0, img) in enumerate(warped):
        valid = np.isfinite(img).all(-1)
        d = cv2.distanceTransform(np.pad(valid, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
        if feather:
            d = np.minimum(d, feather)
        d[~valid] = 0
        corr = np.where(valid[..., None], img * a[k].astype(np.float32) + b[k].astype(np.float32), 0)
        h, w = d.shape
        num[y0:y0 + h, x0:x0 + w] += corr * d[..., None]
        den[y0:y0 + h, x0:x0 + w] += d
    out = np.where(den[..., None] > 0, num / np.maximum(den, 1e-12)[..., None], fill).astype(np.float32)
    if return_info:
        return out, {"scale": a.tolist(), "offset": b.tolist(), "coverage": den > 0}
    return out
