"""Local normalization (like PixInsight LocalNormalization).

A registered frame is matched to a reference with smooth, spatially varying scale and offset maps:

    normalized = A(x, y) * frame + B(x, y)

so that gradients that change from frame to frame (Moon, light pollution, high clouds, sky glow
moving with the altitude) are removed before pixel rejection and integration.

The model is a coarse grid (one node per `tile` pixels, typically 128 px): a few KB per frame,
saved to .npz with `save`/`load`. `apply` evaluates the maps for a band of rows only (the Loader of
the integration reads rows y0:y1), with separable cubic interpolation of the grid.

Method:
- background: per tile, star-rejected (high-pass mask, dilated) and sigma-clipped median of frame
  and reference; tiles with too few valid pixels (NaN borders of registered frames) or whose offset
  disagrees with their neighbours (satellite trails, a galaxy core only in one frame...) are rejected,
  filled from the neighbours and the grids smoothed;
- scale: ratio of the background-subtracted signal (stars, nebula) of reference and frame over the
  pixels where the reference has signal (ratio of sums: insensitive to seeing/FWHM changes), per tile
  when there is enough signal (`local_scale`), otherwise global;
- offset: B = bg_ref - A * bg_frame.
"""
import warnings

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------- helpers

def _as3(a):
    a = np.asarray(a, np.float32)
    return a[..., None] if a.ndim == 2 else a


def _nan_fill(img):
    med = float(np.nanmedian(img[::4, ::4])) if np.isfinite(img[::4, ::4]).any() else 0.0
    return np.where(np.isfinite(img), img, med).astype(np.float32)


def _star_mask(img, kappa=3.0, grow=2):
    """True on stars/compact sources (high-pass above kappa*noise), dilated by `grow` pixels."""
    f = _nan_fill(img)
    hp = f - cv2.GaussianBlur(f, (0, 0), 4)
    s = hp[::2, ::2]
    noise = float(np.median(np.abs(s - np.median(s)))) * 1.4826 + 1e-12
    m = (hp > kappa * noise).astype(np.uint8)
    if grow > 0:
        m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return m.astype(bool)


def _tiles(img, t, step):
    """(H, W) -> (gh, gw, n) samples per tile (NaN padded, subsampled by `step`)."""
    H, W = img.shape
    gh, gw = -(-H // t), -(-W // t)
    pad = np.full((gh * t, gw * t), np.nan, np.float32)
    pad[:H, :W] = img
    sub = pad.reshape(gh, t, gw, t)[:, ::step, :, ::step]
    return sub.transpose(0, 2, 1, 3).reshape(gh, gw, -1)


def _clipped_stats(v, iters=4, lo=4.0, hi=2.5):
    """Per-tile sigma-clipped median, MAD sigma and valid fraction of v (gh, gw, n)."""
    n0 = v.shape[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = v.copy()
        for _ in range(iters):
            med = np.nanmedian(v, -1, keepdims=True)
            mad = np.nanmedian(np.abs(v - med), -1, keepdims=True) * 1.4826
            v = np.where((v > med + hi * mad) | (v < med - lo * mad), np.nan, v)
        med = np.nanmedian(v, -1)
        mad = np.nanmedian(np.abs(v - med[..., None]), -1) * 1.4826
    frac = np.isfinite(v).sum(-1) / n0
    return med, mad, frac


def _fill_grid(g):
    """Fill NaN nodes from their neighbours (iterative normalized convolution)."""
    g = g.astype(np.float64).copy()
    if not np.isfinite(g).any():
        return np.zeros_like(g)
    k = np.array([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]])
    for _ in range(max(g.shape) * 2):
        bad = ~np.isfinite(g)
        if not bad.any():
            break
        v = np.where(bad, 0, g)
        w = (~bad).astype(np.float64)
        num = ndimage.convolve(v, k, mode="nearest")
        den = ndimage.convolve(w, k, mode="nearest")
        upd = bad & (den > 0)
        g[upd] = num[upd] / den[upd]
    return g


def _smooth_grid(g, sigma):
    if sigma <= 0:
        return g
    return ndimage.gaussian_filter(g, sigma, mode="nearest")


def _keys(x):
    x = np.abs(x)
    return np.where(x <= 1, 1.5 * x ** 3 - 2.5 * x ** 2 + 1,
                    np.where(x < 2, -0.5 * x ** 3 + 2.5 * x ** 2 - 4 * x + 2, 0.0))


def _interp_matrix(pix, n, t):
    """(len(pix), n) cubic-convolution weights from grid nodes (centres of tiles of size t) to pixels."""
    u = (np.asarray(pix, np.float64) + 0.5) / t - 0.5
    u = np.clip(u, 0, n - 1)
    i0 = np.floor(u).astype(int)
    f = u - i0
    Wm = np.zeros((len(u), n), np.float64)
    rows = np.arange(len(u))
    for k in (-1, 0, 1, 2):
        np.add.at(Wm, (rows, np.clip(i0 + k, 0, n - 1)), _keys(f - k))
    return Wm


def _eval(grid, t, y0, y1, x0, x1, step=1):
    """Evaluate a (gh, gw, C) grid on pixel rows y0:y1, cols x0:x1 (every `step`)."""
    gh, gw, C = grid.shape
    Wy = _interp_matrix(np.arange(y0, y1, step), gh, t)
    Wx = _interp_matrix(np.arange(x0, x1, step), gw, t)
    out = np.empty((Wy.shape[0], Wx.shape[0], C), np.float32)
    for c in range(C):
        out[..., c] = Wy @ grid[..., c] @ Wx.T
    return out


def _background_grids(img3, t, mask=None):
    """Clipped tile medians (gh, gw, C) and valid fractions."""
    step = max(1, t // 64)
    meds, fracs, sig = [], [], []
    for c in range(img3.shape[-1]):
        ch = img3[..., c]
        if mask is not None:
            ch = np.where(mask, np.nan, ch)
        med, mad, frac = _clipped_stats(_tiles(ch, t, step))
        meds.append(med)
        fracs.append(frac)
        sig.append(mad)
    return np.stack(meds, -1), np.stack(fracs, -1), np.stack(sig, -1)


# ---------------------------------------------------------------- public API

def fit(frame, reference, scale_px=512, reject_stars=True, local_scale=True, min_valid=0.3,
        outlier_sigma=3.0, smooth=1.0):
    """Model so that apply(frame) matches `reference` locally.

    frame, reference: registered images (H, W) or (H, W, C) float32, NaN = no data.
    scale_px: size of the structures that are normalized (larger = only very smooth gradients).
    Returns a dict {"a": (gh, gw, C), "b": (gh, gw, C), "tile": int, "shape": (H, W), "scale": float}.
    """
    f3, r3 = _as3(frame), _as3(reference)
    if f3.shape != r3.shape:
        raise ValueError(f"frame {f3.shape} and reference {r3.shape} differ")
    H, W, C = f3.shape
    t = int(max(16, scale_px // 4))
    mask = None
    if reject_stars:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mask = _star_mask(np.nanmean(r3, -1)) | _star_mask(np.nanmean(f3, -1))
    bf, ff, _ = _background_grids(f3, t, mask)
    br, fr, sr = _background_grids(r3, t, mask)

    # outlier tiles: offset inconsistent with the neighbourhood, or too little valid sky
    off = br - bf
    bad = (ff < min_valid) | (fr < min_valid) | ~np.isfinite(off)
    for c in range(C):
        o = np.where(bad[..., c], np.nan, off[..., c])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loc = ndimage.generic_filter(o, np.nanmedian, size=3, mode="nearest") if np.isfinite(o).any() else o
            dev = np.abs(o - loc)
            s = np.nanmedian(dev) * 1.4826 if np.isfinite(dev).any() else 0.0
            noise = np.nanmedian(sr[..., c]) if np.isfinite(sr[..., c]).any() else 0.0
        thr = outlier_sigma * max(s, 0.5 * noise, 1e-7)
        bad[..., c] |= dev > thr
    bf_s = np.empty_like(bf, dtype=np.float64)
    br_s = np.empty_like(br, dtype=np.float64)
    for c in range(C):
        bf_s[..., c] = _smooth_grid(_fill_grid(np.where(bad[..., c], np.nan, bf[..., c])), smooth)
        br_s[..., c] = _smooth_grid(_fill_grid(np.where(bad[..., c], np.nan, br[..., c])), smooth)

    # scale from the background-subtracted signal (on a 2x decimated grid)
    s2 = 2 if min(H, W) >= 256 else 1
    gh, gw = bf.shape[:2]
    bgf = _eval(bf_s, t, 0, H, 0, W, s2)
    bgr = _eval(br_s, t, 0, H, 0, W, s2)
    fd, rd = f3[::s2, ::s2] - bgf, r3[::s2, ::s2] - bgr
    yy, xx = np.mgrid[0:fd.shape[0], 0:fd.shape[1]]
    tid = (yy * s2 // t) * gw + (xx * s2 // t)
    A = np.ones((gh, gw, C), np.float64)
    a_glob = np.ones(C)
    for c in range(C):
        r_ = np.nan_to_num(rd[..., c])
        f_ = np.nan_to_num(fd[..., c])
        sm = cv2.GaussianBlur(r_.astype(np.float32), (0, 0), 1.0)
        z = sm[::2, ::2]
        nz = float(np.median(np.abs(z - np.median(z)))) * 1.4826 + 1e-12
        hi = max(float(np.nanmax(r3[..., c])), float(np.nanmax(f3[..., c])))
        ok = np.isfinite(rd[..., c]) & np.isfinite(fd[..., c])
        ok &= (r3[::s2, ::s2, c] < 0.95 * hi) & (f3[::s2, ::s2, c] < 0.95 * hi)
        sig = ok & (sm > 5 * nz)
        if sig.sum() > 50 and f_[sig].sum() > 0:
            a_glob[c] = float(r_[sig].sum() / f_[sig].sum())
        a_glob[c] = float(np.clip(a_glob[c], 0.05, 20))
        A[..., c] = a_glob[c]
        if local_scale and sig.sum() > 50:
            cnt = np.bincount(tid[sig], minlength=gh * gw).reshape(gh, gw)
            sr_ = np.bincount(tid[sig], r_[sig], minlength=gh * gw).reshape(gh, gw)
            sf_ = np.bincount(tid[sig], f_[sig], minlength=gh * gw).reshape(gh, gw)
            with np.errstate(divide="ignore", invalid="ignore"):
                loc = np.where((cnt >= 30) & (sf_ > 0), sr_ / sf_, np.nan)
            loc = np.where((loc > 0.5 * a_glob[c]) & (loc < 2 * a_glob[c]), loc, np.nan)
            if np.isfinite(loc).sum() >= 4:
                # shrink towards the global value where the local estimate rests on few pixels
                w = np.where(np.isfinite(loc), cnt / (cnt + 200.0), 0)
                loc = np.where(np.isfinite(loc), w * loc + (1 - w) * a_glob[c], a_glob[c])
                A[..., c] = _smooth_grid(loc, max(smooth, 1.0))
    B = br_s - A * bf_s
    return {"a": A.astype(np.float32), "b": B.astype(np.float32), "tile": t, "shape": (H, W),
            "scale": float(np.mean(a_glob))}


def maps(model, y0, y1, x0=0, x1=None):
    """Scale and offset maps (y1-y0, x1-x0, C) for a band of rows."""
    x1 = model["shape"][1] if x1 is None else x1
    return _eval(model["a"], model["tile"], y0, y1, x0, x1), _eval(model["b"], model["tile"], y0, y1, x0, x1)


def apply(frame_rows, model, y0, y1):
    """Normalize rows y0:y1 of a frame: frame_rows (y1-y0, W) or (y1-y0, W, C). Same shape out."""
    fr = np.asarray(frame_rows, np.float32)
    a, b = maps(model, y0, y1, 0, fr.shape[1])
    if fr.ndim == 2:
        return fr * a[..., 0] + b[..., 0]
    return fr * a + b


def save(model, path):
    np.savez_compressed(path, a=model["a"], b=model["b"], tile=model["tile"],
                        shape=np.asarray(model["shape"]), scale=model.get("scale", 1.0))


def load(path):
    z = np.load(path)
    return {"a": z["a"], "b": z["b"], "tile": int(z["tile"]), "shape": tuple(int(v) for v in z["shape"]),
            "scale": float(z["scale"])}


def frame_stats(img, scale_px=512):
    """Gradient (spread of the smooth background, in noise units), noise and SNR of a frame.

    The result is what `select_reference` expects (one dict per frame).
    """
    im3 = _as3(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lum = np.nanmean(im3, -1)
    t = int(max(16, scale_px // 4))
    bg, frac, sig = _background_grids(lum[..., None], t, _star_mask(lum))
    bg = np.where(frac >= 0.3, bg, np.nan)[..., 0]
    f = _nan_fill(lum)
    hp = (f - cv2.GaussianBlur(f, (0, 0), 1.0))[::2, ::2]
    noise = float(np.median(np.abs(hp - np.median(hp)))) * 1.4826 * 1.6 + 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grad = float(np.nanpercentile(bg, 95) - np.nanpercentile(bg, 5)) if np.isfinite(bg).any() else 0.0
        level = float(np.nanmedian(bg)) if np.isfinite(bg).any() else 0.0
        sig_ = float(np.nanpercentile(lum[::2, ::2], 99.5)) - level
    return {"gradient": grad / noise, "noise": noise, "snr": max(sig_, 0.0) / noise, "background": level}


def reference_scores(frames_stats):
    """Score of every frame as a local normalization reference (higher = better, see select_reference)."""
    g = np.array([float(s.get("gradient", 0.0)) for s in frames_stats])
    q = np.array([float(s.get("snr", 1.0)) for s in frames_stats])
    gn = g / max(float(np.median(g)), 1e-12)
    qn = q / max(float(np.median(q)), 1e-12)
    return (qn / (0.5 + gn)).tolist()


def select_reference(frames_stats):
    """Index of the best reference: low gradient and high SNR.

    frames_stats: list of dicts with "gradient" (lower is better) and optionally "snr" (higher is better),
    e.g. from `frame_stats`. Values are compared relative to the median of the set.
    """
    if not frames_stats:
        raise ValueError("no frames")
    g = np.array([float(s.get("gradient", 0.0)) for s in frames_stats])
    q = np.array([float(s.get("snr", 1.0)) for s in frames_stats])
    gn = g / max(float(np.median(g)), 1e-12)
    qn = q / max(float(np.median(q)), 1e-12)
    score = qn / (0.5 + gn)
    return int(np.argmax(score))
