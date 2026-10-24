"""Comet stacking.

Two integrations are made from the same calibrated frames:
- star-aligned (the normal stack): stars sharp, comet smeared along its path;
- comet-aligned: star transforms plus a translation that keeps the nucleus fixed; with sigma rejection
  the moving stars are mostly rejected, the comet is sharp.
`combine_star_comet` merges them: the smeared comet is removed from the star stack and the sharp comet
(from the comet stack, residual star trails removed) is added back.

Transforms are 2x3 affine matrices mapping frame pixel coords -> reference pixel coords, as returned by
register.match and used by cv2.warpAffine(img, M, ...) without WARP_INVERSE_MAP. Positions are (x, y)
in reference coordinates (i.e. measured on registered frames).
"""
from datetime import datetime

import numpy as np
from scipy import ndimage


def _to_seconds(times):
    out = []
    for t in times:
        if isinstance(t, (int, float, np.integer, np.floating)):
            out.append(float(t))
            continue
        if isinstance(t, str):
            t = datetime.fromisoformat(t.strip().replace("Z", "+00:00"))
        if isinstance(t, np.datetime64):
            out.append(float(t.astype("datetime64[ms]").astype(np.int64)) / 1000.0)
        else:
            out.append(t.timestamp())
    return np.asarray(out, np.float64)


def _lum(img):
    a = np.asarray(img, np.float32)
    return np.nanmean(a, -1) if a.ndim == 3 else a


def _inv(M):
    return np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]


def _apply(M, xy):
    return M[:, :2] @ np.asarray(xy, np.float64) + M[:, 2]


def _subpix(v, i, j):
    """Sub-pixel peak around integer max (i, j) with a log-parabola (exact for Gaussians)."""
    def off(a, b, c):
        if min(a, b, c) > 0:
            a, b, c = np.log(a), np.log(b), np.log(c)
        den = a - 2 * b + c
        return 0.0 if den >= 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    h, w = v.shape
    if not (0 < i < h - 1 and 0 < j < w - 1):
        return float(j), float(i)
    dy = off(v[i - 1, j], v[i, j], v[i + 1, j])
    dx = off(v[i, j - 1], v[i, j], v[i, j + 1])
    return j + dx, i + dy


def nucleus(img, xy, radius=12, star_size=5, smooth=1.5):
    """Refined (x, y) of the brightest diffuse blob within `radius` px of xy (None if not found)."""
    lum = _lum(img)
    H, W = lum.shape
    x, y = xy
    pad = int(radius + 3 * star_size + 4 * smooth + 2)
    x0, x1 = max(0, int(x) - pad), min(W, int(x) + pad + 1)
    y0, y1 = max(0, int(y) - pad), min(H, int(y) + pad + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = lum[y0:y1, x0:x1].astype(np.float32)
    fin = np.isfinite(crop)
    if fin.mean() < 0.5:
        return None
    crop = np.where(fin, crop, np.nanmedian(crop))
    # point sources (stars, hot pixels) vanish under a median larger than them; the coma survives
    starless = ndimage.median_filter(crop, size=star_size, mode="nearest")
    sm = ndimage.gaussian_filter(starless, smooth, mode="nearest")
    sm = sm - float(np.percentile(sm, 20))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    if not inside.any():
        return None
    v = np.where(inside, sm, -np.inf)
    i, j = np.unravel_index(int(np.argmax(v)), v.shape)
    if sm[i, j] <= 0:
        return None
    px, py = _subpix(sm, i, j)
    return px + x0, py + y0


def _predict(n, first_xy, last_xy, t):
    first = np.asarray(first_xy, np.float64)
    last = np.asarray(last_xy, np.float64)
    span = t[-1] - t[0]
    u = (t - t[0]) / span if span != 0 else np.linspace(0, 1, n)
    return first[None] + u[:, None] * (last - first)[None]


def comet_positions(frames_lum, transforms, first_xy, last_xy=None, times=None, radius=12,
                    registered=True, fit_motion=True, max_dev=1.5, star_size=5):
    """Nucleus position (N, 2) float64, (x, y) in reference coordinates, one per frame.

    frames_lum: sequence of N images (H, W) or (H, W, C) (luminance is taken), registered on the stars
                (registered=True) or raw frames (registered=False: `transforms` maps them to the reference).
    transforms: N 2x3 matrices (list) or None when registered=True.
    first_xy / last_xy: user-given nucleus position (reference coords) in the first / last frame.
                Without last_xy the comet is tracked frame by frame from first_xy.
    times: N times (seconds, datetime, numpy datetime64 or ISO DATE-OBS strings); None = equally spaced.
    fit_motion: fit uniform motion x(t), y(t) through the measurements and replace those farther than
                max_dev px (or not found) with the fit.
    """
    n = len(frames_lum)
    t = _to_seconds(times) if times is not None else np.arange(n, dtype=np.float64)
    t = t - t[0]   # absolute epochs (~1e9 s) make the motion fits ill-conditioned
    if transforms is not None:
        transforms = [np.asarray(M, np.float64) for M in transforms]

    def measure(i, pred):
        img = frames_lum[i]
        if registered or transforms is None:
            return nucleus(img, pred, radius, star_size)
        M = transforms[i]
        p = nucleus(img, _apply(_inv(M), pred), radius, star_size)
        return None if p is None else tuple(_apply(M, p))

    meas = np.full((n, 2), np.nan)
    if last_xy is not None:
        pred = _predict(n, first_xy, last_xy, t)
        for i in range(n):
            p = measure(i, pred[i])
            if p is not None:
                meas[i] = p
    else:
        pred = np.zeros((n, 2))
        cur = np.asarray(first_xy, np.float64)
        vel = np.zeros(2)
        for i in range(n):
            if i > 0:
                cur = cur + vel * (t[i] - t[i - 1])
            pred[i] = cur
            p = measure(i, cur)
            if p is not None:
                meas[i] = p
                good = np.flatnonzero(np.isfinite(meas[:i + 1, 0]))
                if len(good) >= 2 and t[good[-1]] != t[good[0]]:
                    A = np.stack([np.ones(len(good)), t[good]], 1)
                    vel = np.linalg.lstsq(A, meas[good], rcond=None)[0][1]
                cur = meas[i]
    if not fit_motion:
        return np.where(np.isfinite(meas), meas, pred)
    good = np.isfinite(meas[:, 0])
    if good.sum() < 2:
        return pred
    A = np.stack([np.ones(n), t - t[0]], 1)
    use = good.copy()
    for _ in range(5):
        sol = np.linalg.lstsq(A[use], meas[use], rcond=None)[0]
        fit = A @ sol
        dev = np.hypot(*(meas - fit).T)
        new = good & (dev <= max(max_dev, 3 * float(np.median(dev[use])) * 1.4826))
        if new.sum() < 2 or (new == use).all():
            break
        use = new
    return np.where(use[:, None], meas, fit)


def comet_transforms(star_transforms, positions, ref_index):
    """2x3 matrices (frame -> reference) that keep the comet nucleus on positions[ref_index].

    star_transforms and positions are indexed alike: lists, or dicts keyed by frame index.
    positions are in reference coordinates (as from comet_positions on registered frames).
    """
    keys = list(star_transforms.keys()) if isinstance(star_transforms, dict) else range(len(star_transforms))
    ref = np.asarray(positions[ref_index], np.float64)
    out = {}
    for k in keys:
        M = np.array(star_transforms[k], np.float64, copy=True)
        M[:, 2] += ref - np.asarray(positions[k], np.float64)
        out[k] = M
    return out if isinstance(star_transforms, dict) else [out[k] for k in keys]


def _disk_mask(shape, centers, radius, feather):
    """Soft mask (H, W): 1 within `radius` of the polyline through centers, smooth edge of `feather` px."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    c = np.atleast_2d(np.asarray(centers, np.float64))
    d = np.hypot(xx - c[0, 0], yy - c[0, 1])
    for a, b in zip(c[:-1], c[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 == 0:
            continue
        u = np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / L2, 0, 1)
        d = np.minimum(d, np.hypot(xx - a[0] - u * ab[0], yy - a[1] - u * ab[1]))
    e = np.clip((radius + feather - d) / max(feather, 1e-6), 0, 1)
    return (e * e * (3 - 2 * e)).astype(np.float32)


def _ring_level(img, m):
    ring = (m == 0)
    return float(np.nanmedian(img[ring])) if ring.any() else float(np.nanmedian(img))


def combine_star_comet(star_stack_img, comet_stack_img, comet_mask_radius, comet_xy=None, trail=None,
                       star_size=7, feather=None, star_kappa=4.0):
    """Final image with sharp stars (star stack) and a sharp comet (comet stack).

    star_stack_img, comet_stack_img: (H, W) or (H, W, C), same reference grid.
    comet_mask_radius: radius (px) of the region holding the comet (coma + inner tail).
    comet_xy: nucleus in the comet stack (default: brightest diffuse blob of the comet stack).
    trail: nucleus positions (N, 2) during the session: the smeared comet in the star stack lies along
           this path (default: a disk at comet_xy).
    """
    S = np.asarray(star_stack_img, np.float32)
    Cm = np.asarray(comet_stack_img, np.float32)
    if S.shape != Cm.shape:
        raise ValueError("star and comet stacks differ in shape")
    two_d = S.ndim == 2
    S3, C3 = (S[..., None], Cm[..., None]) if two_d else (S, Cm)
    H, W, nc = S3.shape
    R = float(comet_mask_radius)
    feather = 0.3 * R if feather is None else float(feather)
    if comet_xy is None:
        lum = _lum(Cm)
        sm = ndimage.gaussian_filter(ndimage.median_filter(np.nan_to_num(lum), size=star_size), 2)
        i, j = np.unravel_index(int(np.argmax(sm)), sm.shape)
        comet_xy = (float(j), float(i))
    path = np.asarray(trail, np.float64) if trail is not None else np.asarray([comet_xy], np.float64)
    pad = int(R + feather + star_size + 2)
    bx0 = max(0, int(min(path[:, 0].min(), comet_xy[0])) - pad)
    bx1 = min(W, int(max(path[:, 0].max(), comet_xy[0])) + pad + 1)
    by0 = max(0, int(min(path[:, 1].min(), comet_xy[1])) - pad)
    by1 = min(H, int(max(path[:, 1].max(), comet_xy[1])) + pad + 1)
    off = np.array([bx0, by0], np.float64)
    shape = (by1 - by0, bx1 - bx0)
    mt = _disk_mask(shape, path - off, R, feather)                 # smeared comet in the star stack
    mc = _disk_mask(shape, np.asarray([comet_xy]) - off, R, feather)  # sharp comet in the comet stack
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    core = np.hypot(xx - (comet_xy[0] - bx0), yy - (comet_xy[1] - by0)) <= max(3.0, 0.15 * R)

    out = S3.copy()
    for c in range(nc):
        s = np.nan_to_num(S3[by0:by1, bx0:bx1, c])
        k = np.nan_to_num(C3[by0:by1, bx0:bx1, c])
        # star stack: remove the smooth (comet) component inside the trail mask, keep the stars
        s_smooth = ndimage.median_filter(s, size=star_size, mode="nearest")
        s_clean = s - mt * (s_smooth - _ring_level(s_smooth, mt))
        # comet stack: residual star trails replaced by the local median (not in the nucleus core)
        k_smooth = ndimage.median_filter(k, size=star_size, mode="nearest")
        hp = k - k_smooth
        z = hp[::2, ::2]
        nz = float(np.median(np.abs(z - np.median(z)))) * 1.4826 + 1e-12
        stars = ndimage.binary_dilation(hp > star_kappa * nz, iterations=1) & ~core
        k_clean = np.where(stars, k_smooth, k)
        comet = mc * (k_clean - _ring_level(k_smooth, mc))
        out[by0:by1, bx0:bx1, c] = s_clean + comet
    return out[..., 0] if two_d else out
