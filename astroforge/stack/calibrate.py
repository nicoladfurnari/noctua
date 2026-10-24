"""Raw frame loading, master frames, calibration, cosmetic correction and debayer."""
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

from . import combine, xisf

# ---------------------------------------------------------------- loading

def load_raw(path):
    """Raw CFA/mono frame as float32 (H, W) in [0, 1], top-down row order (BAYERPAT applies as is)."""
    path = Path(path)
    if path.suffix.lower() == ".xisf":
        a, _ = xisf.read(path)
        return a[..., 0], {}
    with fits.open(path, memmap=False) as hdul:
        h = hdul[0].header
        d = hdul[0].data
    if np.issubdtype(d.dtype, np.integer):
        d = d.astype(np.float32) / 65535.0
    else:
        d = d.astype(np.float32)
        if d.max() > 2:            # float data in ADU
            d /= 65535.0
    if d.ndim == 3:
        d = d[0]
    # capture programs (APT, NINA, SharpCap) write the first row at the top, as PixInsight reads it
    if str(h.get("ROWORDER", "TOP-DOWN")).upper() == "BOTTOM-UP":
        d = d[::-1]
    return np.ascontiguousarray(d), h


_HSHIFT = {"GRBG": "RGGB", "RGGB": "GRBG", "GBRG": "BGGR", "BGGR": "GBRG"}
_VSHIFT = {"RGGB": "GBRG", "GBRG": "RGGB", "GRBG": "BGGR", "BGGR": "GRBG"}


def vertical_shift(pattern):
    """The pattern seen one row down (what a BAYERPAT written for the bottom-up stored order becomes once the
    rows are flipped): its green diagonal is the same as the horizontal shift's, only R and B differ."""
    return _VSHIFT.get(pattern, pattern)


def verify_pattern(pattern, cfa):
    """Check the header Bayer pattern against the data: the two green sites must match.

    Some camera/software combinations write a pattern shifted by one column (e.g. QHY168C with APT
    says GRBG for RGGB data). Returns (pattern, corrected?).
    """
    h, w = cfa.shape
    y0, x0 = h // 4 // 2 * 2, w // 4 // 2 * 2       # even: an odd start (e.g. width 300 or 3004) swaps the sites
    c = cfa[y0:3 * h // 4, x0:3 * w // 4]
    m = [[float(np.median(c[y::2, x::2][::4, ::4])) for x in (0, 1)] for y in (0, 1)]
    diag_a = abs(m[0][0] - m[1][1]) / max(m[0][0] + m[1][1], 1e-9)   # greens at (0,0),(1,1)
    diag_b = abs(m[0][1] - m[1][0]) / max(m[0][1] + m[1][0], 1e-9)   # greens at (0,1),(1,0)
    header_greens_a = pattern[0] == "G"
    data_greens_a = diag_a < diag_b
    if header_greens_a == data_greens_a:
        return pattern, False
    return _HSHIFT[pattern], True


def load_master(path):
    if str(path).lower().endswith(".xisf"):
        a, _ = xisf.read(path)
        return a[..., 0] if a.shape[-1] == 1 else a
    return load_raw(path)[0]


# ---------------------------------------------------------------- masters

def _row_reader(path):
    """(read(y0, y1, sy=1, sx=1, x0=0) -> float32 rows [y0:y1:sy, x0::sx] in [0, 1], top-down; height; handle to
    close) of a raw frame. FITS are read from the memory map band by band (BZERO/BSCALE applied here, bottom-up
    files flipped by index), so nothing of the frame stays in RAM; an XISF is kept whole."""
    p = Path(path)
    if p.suffix.lower() == ".xisf":
        a = np.ascontiguousarray(xisf.read(p)[0][..., 0], np.float32)
        return (lambda y0, y1, sy=1, sx=1, x0=0: a[y0:y1:sy, x0::sx]), a.shape[0], None
    hd = fits.open(p, memmap=True, do_not_scale_image_data=True)
    h, d = hd[0].header, hd[0].data
    if d.ndim == 3:
        d = d[0]
    H = d.shape[0]
    flip = str(h.get("ROWORDER", "TOP-DOWN")).upper() == "BOTTOM-UP"
    if np.issubdtype(d.dtype, np.integer):
        k, b = float(h.get("BSCALE", 1.0)) / 65535.0, float(h.get("BZERO", 0.0)) / 65535.0
    else:
        k, b = (1.0 / 65535.0 if float(np.nanmax(d[::16, ::16])) > 2 else 1.0), 0.0   # float data in ADU

    def read(y0, y1, sy=1, sx=1, x0=0):
        blk = d[H - y1:H - y0][::-1] if flip else d[y0:y1]
        return np.asarray(blk[::sy, x0::sx], np.float32) * k + b
    return read, H, hd


class _RowLoader:
    """Rows of the calibration frames for combine_chunked. Each frame is opened once (at the warm-up, in one
    thread) and then read by bands from the memory map: 60 flats of 16 Mpx kept whole as float32 were 4 GB, the
    bands are planned on the available RAM. Flats: bias / dark subtracted per band and normalised by the
    frame's own median (or one per Bayer site), measured once on a subsample of the whole frame."""

    def __init__(self, height, kind, bias=None, dark=None, mode="global"):
        self.height, self.kind, self.mode = height, kind, mode
        self.sub = dark if dark is not None else bias          # what a flat gets subtracted
        self.frames = {}

    def _open(self, item):
        read, h, handle = _row_reader(item)
        if h != self.height:
            raise ValueError(f"{Path(item).name}: height {h} differs from the first frame ({self.height})")
        norm = None
        if self.kind == "flat":
            if self.mode == "cfa":            # the median of each 2x2 site, as normalize_flat measures it
                norm = [[0.0, 0.0], [0.0, 0.0]]
                for dy in range(2):
                    for dx in range(2):
                        a = read(dy, h, 4, 4, dx)
                        if self.sub is not None:
                            a = a - self.sub[dy::4, dx::4]
                        norm[dy][dx] = max(float(np.median(a)), 1e-6)
            else:
                a = read(0, h, 3, 3)
                if self.sub is not None:
                    a = a - self.sub[::3, ::3]
                norm = max(float(np.median(a)), 1e-6)
        self.frames[item] = (read, norm, handle)
        return self.frames[item]

    def __call__(self, item, y0, y1):
        read, norm, _ = self.frames.get(item) or self._open(item)
        a = read(y0, y1)
        if self.kind == "flat":
            if self.sub is not None:
                a = a - self.sub[y0:y1]
            if self.mode == "cfa":
                for dy in range(2):
                    for dx in range(2):
                        a[(dy - y0) % 2::2, dx::2] /= norm[dy][dx]
            else:
                a = a / norm
        return a

    def close(self):
        for _, _, handle in self.frames.values():
            if handle is not None:
                handle.close()
        self.frames.clear()


def normalize_flat(flat, mode="global"):
    """Normalize a master flat to unit median.

    mode "global": one median for the whole frame (default, as before). On a colour (OSC) sensor the
    flat panel / sky tint then ends up in the calibrated lights: each Bayer site is divided by its own
    colour level, so the colour balance changes.
    mode "cfa": each of the four 2x2 Bayer sites is divided by its own median, so the flat corrects
    only vignetting and dust and leaves the colour balance of the lights untouched. Use it for OSC.
    """
    flat = np.asarray(flat, np.float32)
    if mode == "cfa" and flat.ndim == 2:
        out = flat.copy()
        for dy in range(2):
            for dx in range(2):
                site = out[dy::2, dx::2]
                site /= max(float(np.median(site[::2, ::2])), 1e-6)
        return out
    if mode not in ("global", "cfa"):
        raise ValueError(f"unknown flat normalisation {mode!r}")
    # odd stride: a stride of 4 would sample a single Bayer site (e.g. only red) on a CFA flat
    return flat / max(float(np.median(flat[::3, ::3])), 1e-6)


def build_master(paths, kind, bias=None, dark=None, method="auto", log=print, cfa_normalize=False, low=4.0, high=3.0,
                 **combine_kw):
    """Integrate calibration frames into a master. Flats are bias/dark-subtracted and normalized.

    cfa_normalize (flats of OSC cameras): normalize each Bayer site by its own median instead of one
    global median (see normalize_flat), so the flat does not change the colour balance of the lights.
    """
    mode = "cfa" if cfa_normalize else "global"
    log(f"master {kind}: {len(paths)} frames, rejection {method}" + (", CFA normalisation" if cfa_normalize and kind == "flat" else ""))
    _, h, handle = _row_reader(paths[0])
    if handle is not None:
        handle.close()
    loader = _RowLoader(h, kind, bias, dark, mode)
    try:
        master, _, _ = combine.combine_chunked(list(paths), loader, method=method, rows=512, low=low, high=high,
                                               **combine_kw)
    finally:
        loader.close()
    if kind == "flat":
        master = normalize_flat(master, mode)
    return master


def save_master(master, path):
    xisf.write(master[..., None] if master.ndim == 2 else master, path)


# ---------------------------------------------------------------- calibration

def calibrate(light, bias=None, dark=None, flat=None, dark_scale=1.0):
    out = light.copy()
    if dark is not None:
        if dark_scale != 1.0 and bias is not None:
            out -= bias + (dark - bias) * dark_scale
        else:
            out -= dark
    elif bias is not None:
        out -= bias
    if flat is not None:
        out /= np.maximum(flat, 0.05)
    return out


def _masked_median(a, reject, axis):
    """Median along ``axis`` ignoring rejected samples (vectorised: sort with NaN last). NaN if none left."""
    x = np.where(reject, np.nan, a)
    x = np.sort(x, axis=axis)
    n = (~reject).sum(axis=axis)
    lo = np.maximum((n - 1) // 2, 0)
    hi = np.maximum(n // 2, 0)
    x = np.moveaxis(x, axis, -1)
    v = 0.5 * (np.take_along_axis(x, lo[..., None], -1)[..., 0] + np.take_along_axis(x, hi[..., None], -1)[..., 0])
    return np.where(n > 0, v, np.nan)


def _pad_linear(a, k, axis):
    """Pad ``a`` by ``k`` along ``axis`` extrapolating a straight line fitted to the k border samples."""
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    m = max(2, min(n, max(6, k)))
    t = np.arange(m, dtype=np.float64)
    tc = t - t.mean()

    def side(block, steps):        # block[0] is the border sample, steps > 0 go outward
        mean = block.mean(axis=0)
        slope = np.tensordot(tc, block - mean, axes=(0, 0)) / (tc ** 2).sum()
        at = mean + slope * (-t.mean())                       # fitted value at the border sample
        return at[None] - slope[None] * steps[:, None] if a.ndim == 2 else at - slope * steps

    steps = np.arange(1, k + 1, dtype=np.float64)
    top = side(a[:m].astype(np.float64), steps)[::-1]
    bottom = side(a[::-1][:m].astype(np.float64), steps)
    out = np.concatenate([top, a, bottom]).astype(np.float32)
    return np.moveaxis(out, 0, axis)


def _smooth_background(p, f=8, sigma=1.0, reject=None):
    """Large-scale background of a plane: f x f bin medians (rejected pixels left out) -> median 5 ->
    Gaussian, upsampled.

    The binned image is padded by linear extrapolation (a line fitted to the border bins), so linear
    gradients stay exact up to the borders (a reflection would bend them there)."""
    h, w = p.shape
    hb, wb = max(1, h // f), max(1, w // f)
    blocks = p[:hb * f, :wb * f].reshape(hb, f, wb, f).transpose(0, 2, 1, 3).reshape(hb, wb, f * f)
    if reject is not None:
        rb = reject[:hb * f, :wb * f].reshape(hb, f, wb, f).transpose(0, 2, 1, 3).reshape(hb, wb, f * f)
        small = _masked_median(blocks, rb, 2)
        fallback = np.median(blocks, axis=2)
        small = np.where(np.isfinite(small), small, fallback).astype(np.float32)
    else:
        small = np.median(blocks, axis=2).astype(np.float32)
    k = int(3 * sigma) + 3
    pad = _pad_linear(_pad_linear(small, k, 0), k, 1)
    pad = cv2.medianBlur(np.ascontiguousarray(pad, np.float32), 5)
    pad = cv2.GaussianBlur(pad, (0, 0), sigma)
    pad = pad[k - 1:k + hb + 1, k - 1:k + wb + 1]        # one padded bin each side for the upsampling
    big = cv2.resize(pad, ((wb + 2) * f, (hb + 2) * f), interpolation=cv2.INTER_LINEAR)
    big = big[f:, f:]
    if big.shape[0] < h or big.shape[1] < w:
        big = np.pad(big, ((0, max(0, h - big.shape[0])), (0, max(0, w - big.shape[1]))), mode="edge")
    return big[:h, :w]


def _band_plane(p, axis, amount, protect_sigma, scale):
    """Remove per-row (axis 0) / per-column (axis 1) offsets from one colour plane in place."""
    h, w = p.shape
    if min(h, w) < 32:
        return p
    across = 1 - axis
    n_min = 0.1 * p.shape[across]
    # noise from the differences along the bands: insensitive to the banding itself
    dd = p[:, 1:] - p[:, :-1] if axis == 0 else p[1:] - p[:-1]
    dd = dd[::2, ::2]
    noise = float(np.median(np.abs(dd - np.median(dd)))) * 1.4826 / np.sqrt(2) + 1e-12
    # offsets are measured against a smooth 2-D background (gradients, nebulae, vignetting residuals
    # are kept). It is smooth on ~scale rows, so only banding slower than that is left in place.
    f = int(np.clip(round(scale / 6), 2, 8))               # bin size: ~scale/6 px
    d = p - _smooth_background(p, f, scale / f)
    plain = np.median(d, axis=across)
    line = plain.copy()                                    # first guess (the median ignores stars)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for it in range(3):                                # protect stars/nebulae: clip vs. the current offsets
            # last pass: background again, without stars/nebulae and without the banding found so far
            # (more passes drift: the background bias near nebulae would be re-added each time)
            if it == 2:
                d = p - _smooth_background(p - (line[:, None] if axis == 0 else line[None, :]), f,
                                           scale / f, reject)
                plain = np.median(d, axis=across)
            dev = d - (line[:, None] if axis == 0 else line[None, :])
            reject = np.abs(dev) > protect_sigma * noise
            reject = cv2.dilate(reject.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            # faint extended structure (nebula wings, the background model's ring around them) is below
            # the per-pixel threshold but not below the threshold of a 9x9 mean (noise / 9)
            clean = np.where(reject, 0, dev).astype(np.float32)
            frac = cv2.blur((~reject).astype(np.float32), (9, 9))
            mean9 = cv2.blur(clean, (9, 9)) / np.maximum(frac, 1e-3)
            reject |= (np.abs(mean9) > protect_sigma * noise / (9 * np.sqrt(np.maximum(frac, 1e-3)))) & (frac > 0.3)
            n_ok = (~reject).sum(axis=across)
            line = _masked_median(d, reject, across)
            few = (n_ok < n_min) | ~np.isfinite(line)        # row mostly covered: plain median of the residual
            line[few] = plain[few]
    line -= np.median(line)
    # offsets within the noise of a line median are not banding: shrink them (soft threshold), so a
    # clean frame is left (almost) untouched
    sig = 1.25 * noise / np.sqrt(np.maximum(n_ok, 1))
    line = np.sign(line) * np.maximum(np.abs(line) - sig, 0)
    off = (amount * line).astype(np.float32)
    p -= off[:, None] if axis == 0 else off[None, :]
    return p


def reduce_banding(cal, pattern=None, axis="rows", amount=1.0, protect_sigma=3.0, scale=48):
    """Remove horizontal (and/or vertical) banding: per-row / per-column offsets, as PixInsight
    CanonBandingReduction or Siril "banding reduction".

    cal: calibrated frame, CFA/mono (H, W) or debayered (H, W, C). pattern: Bayer pattern of a CFA
    frame (or True): the four Bayer planes are corrected separately (their offsets differ).
    axis: "rows", "cols" or "both". amount: 0..1 fraction of the estimated offsets removed.
    protect_sigma: pixels deviating more than this many sigma from the background + offsets (stars,
    nebulae, found against a binned-median background; faint extended structure via a 9x9 mean) are
    excluded from the row statistics.
    scale: sensor pixels of the background smoothing. Offsets are measured against that background, so
    gradients and structures larger than ``scale`` are kept, and banding slower than ``scale`` (which
    cannot be told apart from a gradient) stays too. Larger frames allow a larger scale.
    On a clean frame the estimated offsets are at the noise of a row median (~1.25 sigma / sqrt(width)):
    the frame is unchanged within noise. Cost ~0.3 s per megapixel and axis. Returns a new float32 array.

    Use it on calibrated frames, before debayering (pattern = the Bayer pattern) or on mono frames.
    """
    out = np.array(cal, dtype=np.float32, copy=True)
    axes = {"rows": (0,), "cols": (1,), "columns": (1,), "both": (0, 1)}[axis]
    if out.ndim == 3:
        planes = [out[..., c] for c in range(out.shape[2])]
    elif pattern:
        planes = [out[dy::2, dx::2] for dy in (0, 1) for dx in (0, 1)]
    else:
        planes = [out]
    plane_scale = scale / 2.0 if (out.ndim == 2 and pattern) else float(scale)   # scale is in sensor pixels
    for view in planes:
        p = np.ascontiguousarray(view)
        for ax in axes:
            _band_plane(p, ax, amount, protect_sigma, plane_scale)
        view[...] = p
    return out


def active_area(flat=None, light=None, level=0.2):
    """Bounding box (y0, y1, x0, x1) of the illuminated sensor area, excluding overscan/black columns.

    Origin is kept even so the Bayer pattern does not shift.
    """
    ref = flat if flat is not None else light
    if ref is None:
        return None
    if flat is None:   # on a light, overscan sits near the bias level: compare to the image median
        level = 0.5 * float(np.median(ref[::4, ::4]))
    cols = np.median(ref[::4, :], axis=0) > level
    rows = np.median(ref[:, ::4], axis=1) > level
    if not cols.any() or not rows.any():
        return None
    x0, x1 = int(np.argmax(cols)), len(cols) - int(np.argmax(cols[::-1]))
    y0, y1 = int(np.argmax(rows)), len(rows) - int(np.argmax(rows[::-1]))
    x0 += x0 % 2
    y0 += y0 % 2
    return y0, y1, x0, x1


def defect_map(dark=None, flat=None, hot_sigma=10.0, hot_min=0.003, cold_frac=0.5, columns_sigma=0.0):
    """Hot pixels from the (bias-subtracted) master dark, dead/cold pixels from the master flat.

    Warm pixels are already removed by dark subtraction: only pixels above both hot_sigma noise
    and hot_min (~200 ADU) are treated as defects. columns_sigma > 0 also marks whole columns whose dark level
    stands out from the neighbouring columns by that many sigmas (bad / hot columns of CMOS sensors).
    """
    bad = None
    if dark is not None:
        # compare with the local same-color median: amp glow and gradients are smooth, hot pixels are not
        bad = np.zeros(dark.shape, bool)
        for dy in range(2):
            for dx in range(2):
                plane = np.ascontiguousarray(dark[dy::2, dx::2], dtype=np.float32)
                excess = plane - cv2.medianBlur(plane, 5)
                e = excess[::2, ::2]
                mad = float(np.median(np.abs(e - np.median(e)))) * 1.4826
                # high-gain sensors have many warm pixels that dark subtraction already handles:
                # only the most extreme 0.1% are treated as defects
                cap = float(np.percentile(e, 99.9))
                bad[dy::2, dx::2] = excess > max(hot_sigma * mad, hot_min, cap)
    if flat is not None:
        # dead/cold = much darker than the same-color neighbours (vignetted corners are not defects)
        cold = np.zeros(flat.shape, bool)
        for dy in range(2):
            for dx in range(2):
                plane = np.ascontiguousarray(flat[dy::2, dx::2], dtype=np.float32)
                local = cv2.medianBlur(plane, 5)
                cold[dy::2, dx::2] = plane < cold_frac * local
        bad = cold if bad is None else bad | cold
    if dark is not None and columns_sigma > 0:
        col = np.median(dark[::4], axis=0)
        ref = cv2.medianBlur(np.ascontiguousarray(col[None, :], dtype=np.float32), 5)[0]
        dev = col - ref
        mad = float(np.median(np.abs(dev - np.median(dev)))) * 1.4826 + 1e-9
        badcols = np.abs(dev) > columns_sigma * mad
        if badcols.any():
            bad = (np.zeros(dark.shape, bool) if bad is None else bad)
            bad[:, badcols] = True
    return bad


def fix_defects(cfa, bad, cfa_period=2):
    """Replace bad pixels with the median of same-color neighbours."""
    if bad is None or not bad.any():
        return cfa
    out = cfa.copy()
    p = cfa_period
    for dy in range(p):
        for dx in range(p):
            plane = np.ascontiguousarray(cfa[dy::p, dx::p])
            med = cv2.medianBlur(plane, 3) if plane.dtype == np.float32 else plane
            b = bad[dy::p, dx::p]
            sub = out[dy::p, dx::p]
            sub[b] = med[b]
    return out


# ---------------------------------------------------------------- debayer

_CV_CODES = {}


def _cv_code(pattern):
    """OpenCV names Bayer codes differently from the FITS convention: find the right one by test."""
    if pattern in _CV_CODES:
        return _CV_CODES[pattern]
    rng = np.random.default_rng(0)
    rgb = np.zeros((8, 8, 3), np.uint16)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 1000, 20000, 50000
    rgb += rng.integers(0, 10, rgb.shape, dtype=np.uint16)
    idx = {"R": 0, "G": 1, "B": 2}
    mosaic = np.zeros((8, 8), np.uint16)
    for i, ch in enumerate(pattern):
        y, x = divmod(i, 2)
        mosaic[y::2, x::2] = rgb[y::2, x::2, idx[ch]]
    best = None
    for name in ("BayerRG2RGB", "BayerGR2RGB", "BayerBG2RGB", "BayerGB2RGB"):
        code = getattr(cv2, "COLOR_" + name)
        out = cv2.cvtColor(mosaic, code)[2:6, 2:6].astype(np.float64)
        err = np.abs(out - rgb[2:6, 2:6]).mean()
        if best is None or err < best[0]:
            best = (err, code)
    _CV_CODES[pattern] = best[1]
    return best[1]


def _rcd_band(c, cmap):
    """Ratio Corrected Demosaicing (L. Sanz Rodríguez, RCD v2.3 as in RawTherapee/Siril) on a padded band.
    c: CFA > 0, cmap: colour index per pixel (0 R, 1 G, 2 B). Returns R, G, B planes of the same size."""
    def sh(a, dy, dx):   # sh(a, dy, dx)[y, x] = a[y + dy, x + dx]; wrap-around only touches the margin
        return np.roll(a, (-dy, -dx), axis=(0, 1))

    eps, epssq = 1e-5, 1e-10
    R, G, B = cmap == 0, cmap == 1, cmap == 2
    # step 1: vertical / horizontal discrimination from high-pass filters of colour differences
    bv = ((sh(c, -3, 0) - sh(c, -1, 0) - sh(c, 1, 0) + sh(c, 3, 0)) - 3 * (sh(c, -2, 0) + sh(c, 2, 0)) + 6 * c) ** 2
    bh = ((sh(c, 0, -3) - sh(c, 0, -1) - sh(c, 0, 1) + sh(c, 0, 3)) - 3 * (sh(c, 0, -2) + sh(c, 0, 2)) + 6 * c) ** 2
    vs = np.maximum(epssq, sh(bv, -1, 0) + bv + sh(bv, 1, 0))
    hs = np.maximum(epssq, sh(bh, 0, -1) + bh + sh(bh, 0, 1))
    vh = vs / (vs + hs)
    del bv, bh, vs, hs
    # step 2: low-pass filter, step 3: green at red/blue sites from ratio-corrected cardinal estimates
    lpf = (c + 0.5 * (sh(c, -1, 0) + sh(c, 1, 0) + sh(c, 0, -1) + sh(c, 0, 1))
           + 0.25 * (sh(c, -1, -1) + sh(c, -1, 1) + sh(c, 1, -1) + sh(c, 1, 1)))
    vhn = 0.25 * (sh(vh, -1, -1) + sh(vh, -1, 1) + sh(vh, 1, -1) + sh(vh, 1, 1))
    vhd = np.where(np.abs(0.5 - vh) < np.abs(0.5 - vhn), vhn, vh)
    del vh, vhn
    a = np.abs
    ng = eps + a(sh(c, -1, 0) - sh(c, 1, 0)) + a(c - sh(c, -2, 0)) + a(sh(c, -1, 0) - sh(c, -3, 0)) + a(sh(c, -2, 0) - sh(c, -4, 0))
    sg = eps + a(sh(c, -1, 0) - sh(c, 1, 0)) + a(c - sh(c, 2, 0)) + a(sh(c, 1, 0) - sh(c, 3, 0)) + a(sh(c, 2, 0) - sh(c, 4, 0))
    wg = eps + a(sh(c, 0, -1) - sh(c, 0, 1)) + a(c - sh(c, 0, -2)) + a(sh(c, 0, -1) - sh(c, 0, -3)) + a(sh(c, 0, -2) - sh(c, 0, -4))
    eg = eps + a(sh(c, 0, -1) - sh(c, 0, 1)) + a(c - sh(c, 0, 2)) + a(sh(c, 0, 1) - sh(c, 0, 3)) + a(sh(c, 0, 2) - sh(c, 0, 4))
    n_est = sh(c, -1, 0) * 2 * lpf / (eps + lpf + sh(lpf, -2, 0))
    s_est = sh(c, 1, 0) * 2 * lpf / (eps + lpf + sh(lpf, 2, 0))
    w_est = sh(c, 0, -1) * 2 * lpf / (eps + lpf + sh(lpf, 0, -2))
    e_est = sh(c, 0, 1) * 2 * lpf / (eps + lpf + sh(lpf, 0, 2))
    v_est = (sg * n_est + ng * s_est) / (ng + sg)
    h_est = (eg * w_est + wg * e_est) / (eg + wg)
    g = np.where(G, c, np.clip(vhd * h_est + (1 - vhd) * v_est, 0, 1))
    del lpf, ng, sg, wg, eg, n_est, s_est, w_est, e_est, v_est, h_est
    # step 4.1-4.2: red at blue sites and blue at red sites along the better diagonal
    bp = ((sh(c, -3, -3) - sh(c, -1, -1) - sh(c, 1, 1) + sh(c, 3, 3)) - 3 * (sh(c, -2, -2) + sh(c, 2, 2)) + 6 * c) ** 2
    bq = ((sh(c, -3, 3) - sh(c, -1, 1) - sh(c, 1, -1) + sh(c, 3, -3)) - 3 * (sh(c, -2, 2) + sh(c, 2, -2)) + 6 * c) ** 2
    ps = np.maximum(epssq, sh(bp, -1, -1) + bp + sh(bp, 1, 1))
    qs = np.maximum(epssq, sh(bq, -1, 1) + bq + sh(bq, 1, -1))
    pq = ps / (ps + qs)
    pqn = 0.25 * (sh(pq, -1, -1) + sh(pq, -1, 1) + sh(pq, 1, -1) + sh(pq, 1, 1))
    pqd = np.where(np.abs(0.5 - pq) < np.abs(0.5 - pqn), pqn, pq)
    del bp, bq, ps, qs, pq, pqn
    out = []
    for own, other in ((R, B), (B, R)):
        x = np.where(own, c, 0).astype(np.float32)
        nwg = eps + a(sh(x, -1, -1) - sh(x, 1, 1)) + a(sh(x, -1, -1) - sh(x, -3, -3)) + a(g - sh(g, -2, -2))
        neg = eps + a(sh(x, -1, 1) - sh(x, 1, -1)) + a(sh(x, -1, 1) - sh(x, -3, 3)) + a(g - sh(g, -2, 2))
        swg = eps + a(sh(x, 1, -1) - sh(x, -1, 1)) + a(sh(x, 1, -1) - sh(x, 3, -3)) + a(g - sh(g, 2, -2))
        seg = eps + a(sh(x, 1, 1) - sh(x, -1, -1)) + a(sh(x, 1, 1) - sh(x, 3, 3)) + a(g - sh(g, 2, 2))
        d = x - g    # colour differences
        p_est = (nwg * sh(d, 1, 1) + seg * sh(d, -1, -1)) / (nwg + seg)
        q_est = (neg * sh(d, 1, -1) + swg * sh(d, -1, 1)) / (neg + swg)
        x = np.where(other, np.clip(g + (1 - pqd) * p_est + pqd * q_est, 0, 1), x)
        out.append(x)
    # step 4.3: red and blue at green sites (all non-green sites are known now)
    for k, x in enumerate(out):
        ng = eps + a(g - sh(g, -2, 0)) + a(sh(x, -1, 0) - sh(x, 1, 0)) + a(sh(x, -1, 0) - sh(x, -3, 0))
        sg = eps + a(g - sh(g, 2, 0)) + a(sh(x, 1, 0) - sh(x, -1, 0)) + a(sh(x, 1, 0) - sh(x, 3, 0))
        wg = eps + a(g - sh(g, 0, -2)) + a(sh(x, 0, -1) - sh(x, 0, 1)) + a(sh(x, 0, -1) - sh(x, 0, -3))
        eg = eps + a(g - sh(g, 0, 2)) + a(sh(x, 0, 1) - sh(x, 0, -1)) + a(sh(x, 0, 1) - sh(x, 0, 3))
        d = x - g
        v = (ng * sh(d, 1, 0) + sg * sh(d, -1, 0)) / (ng + sg)
        h = (eg * sh(d, 0, -1) + wg * sh(d, 0, 1)) / (eg + wg)
        out[k] = np.where(G, np.clip(g + (1 - vhd) * v + vhd * h, 0, 1), x)
    return out[0], g, out[1]


def rcd(cfa, pattern, band=512):
    """RCD demosaicing of a CFA in [0, 1] -> (H, W, 3). Works by row bands to bound memory."""
    idx = {"R": 0, "G": 1, "B": 2}
    tile = np.array([[idx[pattern[0]], idx[pattern[1]]], [idx[pattern[2]], idx[pattern[3]]]])
    m = 8                                   # even margin: keeps the Bayer phase
    p = np.pad(cfa.astype(np.float32), m, mode="reflect")
    cmap = np.tile(tile, (p.shape[0] // 2 + 1, p.shape[1] // 2 + 1))[:p.shape[0], :p.shape[1]]
    h, w = cfa.shape
    out = np.empty((h, w, 3), np.float32)

    def work(y0):                           # band starts are even, so the phase is the same in every band
        y1 = min(h, y0 + band)
        r, g, b = _rcd_band(np.maximum(p[y0:y1 + 2 * m], 1e-6), cmap[y0:y1 + 2 * m])
        out[y0:y1] = np.stack([r, g, b], -1)[m:-m, m:-m]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, min(8, (os.cpu_count() or 2) // 2))) as ex:   # numpy ufuncs release the GIL
        list(ex.map(work, range(0, h, band)))
    return out


def debayer(cfa, pattern, method="bilinear"):
    """CFA float32 (H, W) -> RGB float32 (H, W, 3). method: bilinear, ea (edge aware) or rcd."""
    pattern = pattern.upper()
    if pattern not in ("RGGB", "BGGR", "GRBG", "GBRG"):
        raise ValueError(f"unsupported Bayer pattern {pattern}")
    # OpenCV debayers integers only. Calibrated sky background sits near 0 with negative noise:
    # clipping it at 0 would raise the background (worse with short subs or dual-band filters),
    # so shift by a pedestal first and remove it after
    pedestal = min(max(0.0, -float(cfa.min())), 0.1)
    if method == "rcd":
        return rcd(np.clip(cfa + pedestal, 0, 1), pattern) - pedestal
    u16 = (np.clip(cfa + pedestal, 0, 1) * 65535 + 0.5).astype(np.uint16)
    code = _cv_code(pattern)
    if method == "ea":
        code = code + (cv2.COLOR_BayerBG2RGB_EA - cv2.COLOR_BayerBG2RGB)
    rgb = cv2.cvtColor(u16, code)
    return rgb.astype(np.float32) / 65535.0 - pedestal


def looks_like_cfa(raw, ratio=1.12):
    """True when the four 2x2 sites have clearly different levels, as on a colour (Bayer) sensor."""
    h, w = raw.shape[0] // 2 * 2, raw.shape[1] // 2 * 2
    meds = [float(np.median(raw[y:h:2, x:w:2][::4, ::4])) for y in (0, 1) for x in (0, 1)]
    lo = min(meds)
    return lo > 0 and max(meds) / lo > ratio


def superpixel_lum(cfa):
    """Fast 2x2 binned luminance of a CFA frame (for star detection)."""
    h, w = cfa.shape[0] // 2 * 2, cfa.shape[1] // 2 * 2
    c = cfa[:h, :w]
    return (c[0::2, 0::2] + c[0::2, 1::2] + c[1::2, 0::2] + c[1::2, 1::2]) / 4
