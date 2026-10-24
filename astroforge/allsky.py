"""All-sky reference for gradient correction (idea of PixInsight MultiscaleGradientCorrection / MARS).

The large-scale background of a survey (DSS2, Pan-STARRS DR1) taken from the CDS hips2fits service tells what
the sky really looks like at scales of several arcminutes. The image's own large-scale structure minus the
(scaled) reference structure is the gradient: extended faint nebulosity present in the survey is kept.

hips2fits is queried with the image WCS reduced by a factor k (1/8..1/16, at most ~1000 px), so the reference
lands on a coarse version of the image pixel grid. Downloads are cached in ~/.astroforge/allsky. urllib only.
"""
import hashlib
import io as _io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import net

import numpy as np
from astropy.io import fits
from scipy import ndimage

HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
CACHE = Path.home() / ".astroforge" / "allsky"
MAX_PX = 1000

# survey -> HiPS id per R, G, B ("mean" = average of the other two). DSS2 red/blue and PS1 g/r are FITS
# (16-bit / float); CDS/P/DSS2/color is an 8-bit stretched colour composite (planes R, G, B, alpha).
SURVEYS = {
    "dss2_redblue": {"R": "CDS/P/DSS2/red", "G": "mean", "B": "CDS/P/DSS2/blue"},
    "dss2_red": {"R": "CDS/P/DSS2/red", "G": "CDS/P/DSS2/red", "B": "CDS/P/DSS2/red"},
    "dss2_color": {"RGB": "CDS/P/DSS2/color"},
    "panstarrs": {"R": "CDS/P/PanSTARRS/DR1/r", "G": "mean", "B": "CDS/P/PanSTARRS/DR1/g"},
}
SURVEY_CHOICES = ["auto"] + list(SURVEYS)
PS1_MIN_DEC = -29.5   # Pan-STARRS 3pi covers dec > -30


def reduction_factor(shape, max_px=MAX_PX):
    """k such that the reference is 1/8..1/16 of the image (more for huge images), at most max_px."""
    L = max(shape[:2])
    k = max(8, math.ceil(L / max_px))
    return max(1, min(k, L // 32)) if L >= 32 else 1


def reduced_wcs(wcs, shape, k):
    """hips2fits WCS dict (FITS keywords) of the image grid reduced by k, plus the reduced (h, w).

    Pixel x of the image maps to (x + 0.5) / kx - 0.5 of the reduced grid; SIP is dropped (sub-pixel at /k).
    """
    H, W = shape[:2]
    w2, h2 = max(2, int(round(W / k))), max(2, int(round(H / k)))
    kx, ky = W / w2, H / h2
    c = wcs.celestial
    cd = np.array(c.pixel_scale_matrix, float) * np.array([kx, ky])[None, :]
    crpix = (np.array(c.wcs.crpix, float) - 0.5) / np.array([kx, ky]) + 0.5
    ctype = [str(t).replace("-SIP", "") for t in c.wcs.ctype]
    d = {"NAXIS1": w2, "NAXIS2": h2, "CTYPE1": ctype[0], "CTYPE2": ctype[1],
         "CRVAL1": round(float(c.wcs.crval[0]), 8), "CRVAL2": round(float(c.wcs.crval[1]), 8),
         "CRPIX1": round(float(crpix[0]), 5), "CRPIX2": round(float(crpix[1]), 5),
         "CD1_1": float(f"{cd[0, 0]:.10e}"), "CD1_2": float(f"{cd[0, 1]:.10e}"),
         "CD2_1": float(f"{cd[1, 0]:.10e}"), "CD2_2": float(f"{cd[1, 1]:.10e}")}
    return d, (h2, w2)


def hips_url(hips, wcs_dict):
    q = {"hips": hips, "wcs": json.dumps(wcs_dict, separators=(",", ":")), "format": "fits"}
    return HIPS2FITS + "?" + urllib.parse.urlencode(q)


def _download(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "AstroForge"})
    with net.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_hips(hips, wcs_dict, timeout=120, cache_dir=None, log=print):
    """One hips2fits cutout on the grid of `wcs_dict`: float32 (h, w) or (h, w, 3) for colour HiPS, None on failure."""
    cache_dir = CACHE if cache_dir is None else Path(cache_dir)
    key = hashlib.md5((hips + json.dumps(wcs_dict, sort_keys=True)).encode()).hexdigest()[:20]
    path = cache_dir / f"{hips.replace('/', '_')}_{key}.npy"
    if path.exists():
        try:
            return np.load(path)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
    url = hips_url(hips, wcs_dict)
    try:
        raw = _download(url, timeout)
        with fits.open(_io.BytesIO(raw)) as hd:
            data = np.asarray(hd[0].data, np.float32)
    except (urllib.error.URLError, OSError, ValueError, TypeError, TimeoutError) as e:
        log(f"allsky: download of {hips} failed ({type(e).__name__}: {e})")
        return None
    if data.ndim == 3:
        data = np.moveaxis(data[:3], 0, -1)
    shape = (wcs_dict["NAXIS2"], wcs_dict["NAXIS1"])
    if data.shape[:2] != shape:
        log(f"allsky: {hips} answered {data.shape}, expected {shape}")
        return None
    data = np.ascontiguousarray(data)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data)
    except OSError as e:
        log(f"allsky: cache not written ({e})")
    log(f"allsky: {hips} {shape[1]}x{shape[0]} downloaded ({len(raw) / 1024:.0f} KB)")
    return data


def pick_survey(survey, wcs, shape):
    if survey in SURVEYS:
        return survey
    if survey != "auto":
        raise KeyError(f"unknown survey '{survey}' (choices: {', '.join(SURVEY_CHOICES)})")
    # DSS2 plates keep the extended nebulosity better than the sky-subtracted PS1 stacks; whole sky
    return "dss2_redblue"


def _fill(a):
    a = np.asarray(a, np.float32).copy()
    bad = ~np.isfinite(a)
    if bad.any():
        a[bad] = np.nanmedian(a) if (~bad).any() else 0.0
    return a


def starless(a, size=3):
    """Remove stars at the coarse resolution: grey opening + median, per channel. (h, w[, C]) float32."""
    a = np.asarray(a, np.float32)
    chans = a[..., None] if a.ndim == 2 else a
    out = np.empty_like(chans)
    for c in range(chans.shape[-1]):
        x = _fill(chans[..., c])
        x = ndimage.grey_opening(x, size=(size, size), mode="reflect")
        out[..., c] = ndimage.median_filter(x, size=size + 2, mode="reflect")
    return out if a.ndim == 3 else out[..., 0]


def reference_background(wcs, shape, survey="auto", scale_px=None, k=None, timeout=120, cache_dir=None,
                         log=print):
    """Star-free survey reference of the field on the image grid reduced by k: (H/k, W/k, 3) float32, or None.

    `scale_px` (full-resolution px): when given the reference is also smoothed (sigma = scale/16) so only
    structures of a few coarse pixels and more survive. Values are in survey units (scaled later by the fit).
    """
    survey = pick_survey(survey, wcs, shape)
    k = k or reduction_factor(shape)
    wd, (h2, w2) = reduced_wcs(wcs, shape, k)
    spec = SURVEYS[survey]
    if "RGB" in spec:
        cube = fetch_hips(spec["RGB"], wd, timeout, cache_dir, log)
        if cube is None:
            return None
        if cube.ndim == 2:
            cube = np.repeat(cube[..., None], 3, -1)
        ref = (np.clip(cube[..., :3], 0, 255) / 255.0) ** 2.2   # undo the display stretch (approximately)
    else:
        planes, got = {}, {}
        for ch in "RGB":
            hid = spec[ch]
            if hid == "mean":
                continue
            if hid not in got:
                got[hid] = fetch_hips(hid, wd, timeout, cache_dir, log)
                if got[hid] is None:
                    return None
            planes[ch] = got[hid] if got[hid].ndim == 2 else got[hid].mean(-1)
        for ch in "RGB":
            if spec[ch] == "mean":
                # bands can come in different units (PS1 g vs r): average after a robust normalisation
                others = []
                for o in "RGB":
                    if o != ch and o in planes:
                        p = _fill(planes[o])
                        lo, hi = np.percentile(p, [5, 99])
                        others.append((p - lo) / max(hi - lo, 1e-12))
                planes[ch] = np.mean(others, axis=0)
        ref = np.stack([planes[c] for c in "RGB"], -1)
    ref = starless(ref.astype(np.float32))
    if scale_px:
        s = max(0.0, scale_px / k / 16)
        if s >= 0.5:
            ref = ndimage.gaussian_filter(ref, (s, s, 0), mode="reflect")
    log(f"allsky: reference {survey} {w2}x{h2} (1/{k} of the image)")
    return ref.astype(np.float32)


def _design(x, y, degree):
    return np.stack([(x ** i) * (y ** j) for i in range(degree + 1) for j in range(degree + 1 - i)], -1)


def _robust_poly(d, degree=2, iters=5):
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = _design(xx.ravel() / max(w - 1, 1) * 2 - 1, yy.ravel() / max(h - 1, 1) * 2 - 1, degree)
    v = d.ravel().astype(np.float64)
    keep = np.isfinite(v)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        res = v - A @ coef
        s = 1.4826 * np.median(np.abs(res[keep])) + 1e-12
        keep = np.abs(res) < 3 * s
    return (A @ coef).reshape(h, w)


def _huber_slope(x, y, iters=15):
    """Robust slope through the origin of y = a*x (Huber IRLS). Returns (a, correlation)."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    a = float((x * y).sum() / max((x * x).sum(), 1e-30))
    for _ in range(iters):
        r = y - a * x
        s = 1.4826 * np.median(np.abs(r)) + 1e-30
        wgt = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-30))
        a_new = float((wgt * x * y).sum() / max((wgt * x * x).sum(), 1e-30))
        if abs(a_new - a) <= 1e-6 * max(abs(a), 1e-12):
            a = a_new
            break
        a = a_new
    c = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 0.0
    return a, float(c)


def _bands(a, sigma):
    """Band-pass layers G(s) - G(2s) for s = sigma/8, /4, /2, border-trimmed and median-centred, flattened."""
    out = []
    for f in (8, 4, 2):
        s = max(0.7, sigma / f)
        d = ndimage.gaussian_filter(a, s, mode="reflect") - ndimage.gaussian_filter(a, 2 * s, mode="reflect")
        m = int(math.ceil(4 * s))
        if d.shape[0] > 2 * m + 4 and d.shape[1] > 2 * m + 4:
            d = d[m:-m, m:-m]
        d = d.ravel()
        out.append(d - np.median(d))
    return np.concatenate(out)


def default_scale_px(shape):
    """Like MGC's 1024 px gradient scale, but never more than half the long side of the image."""
    return float(min(1024, max(shape[:2]) / 2))


def gradient_model(img, reference, scale_px=None, scale_arcmin=None, wcs=None, protect=0.0, log=print):
    """Full-resolution background model (sky + gradient) of `img` given the coarse `reference`.

    Per channel, at the coarse grid (both star-free): the reference is scaled by `a`, fitted robustly on the
    band-pass layers below the gradient scale (where gradients have no power and the nebulae do); the
    difference D = image - a*reference is the gradient plus the sky; its large-scale part (robust quadratic +
    Gaussian of the residual, sigma = scale/4) is the model. `protect` (0..1) blends, where the reference shows
    nebulosity, towards a quadratic fitted only on reference-dark sky (safer where the survey saturates).
    Returns (model (H, W, C) float32, info dict).
    """
    import cv2
    H, W, nc = img.shape
    h2, w2 = reference.shape[:2]
    k = W / w2
    if scale_px is None and scale_arcmin and wcs is not None:
        from .astrometry import pixel_scale_arcsec
        scale_px = scale_arcmin * 60.0 / pixel_scale_arcsec(wcs)
    scale_px = float(scale_px or default_scale_px(img.shape))
    sigma = max(1.0, scale_px / k / 4)
    small = cv2.resize(np.ascontiguousarray(img, np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
    small = small.reshape(h2, w2, nc)
    small = starless(small)
    ref = np.asarray(reference, np.float32)
    if nc == 1:
        ref = ref.mean(-1, keepdims=True)
    elif ref.shape[-1] != nc:
        ref = np.repeat(ref.mean(-1, keepdims=True), nc, -1)
    model_small = np.empty((h2, w2, nc), np.float32)
    info = {"scale_px": scale_px, "sigma_coarse": sigma, "k": k, "a": [], "corr": []}
    for c in range(nc):
        I, R = small[..., c].astype(np.float64), _fill(ref[..., c]).astype(np.float64)
        a, corr = _huber_slope(_bands(R, sigma), _bands(I, sigma))
        if not np.isfinite(a) or a < 0 or corr < 0.2:
            log(f"allsky: channel {c}: reference does not match the image (corr {corr:.2f}); "
                "model from the image only")
            a = 0.0
        R0 = float(np.percentile(R, 5))
        neb = a * (R - R0)
        D = I - neb
        trend = _robust_poly(D, 2)
        m = trend + ndimage.gaussian_filter(D - trend, sigma, mode="reflect")
        if protect > 0 and a > 0:
            thr = max(float(np.percentile(neb, 90)), 1e-12)
            wgt = np.clip(neb / thr, 0, 1)
            dark = wgt < 0.1
            if dark.mean() > 0.1:
                m = (1 - protect * wgt) * m + protect * wgt * _poly_on(D, dark)
        model_small[..., c] = m
        info["a"].append(a)
        info["corr"].append(corr)
    model = cv2.resize(model_small, (W, H), interpolation=cv2.INTER_CUBIC).reshape(H, W, nc)
    return model.astype(np.float32), info


def _poly_on(d, mask, degree=2):
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = _design(xx.ravel() / max(w - 1, 1) * 2 - 1, yy.ravel() / max(h - 1, 1) * 2 - 1, degree)
    keep = mask.ravel()
    coef, *_ = np.linalg.lstsq(A[keep], d.ravel()[keep], rcond=None)
    return (A @ coef).reshape(h, w)


def apply(img, model, correction="subtraction"):
    """Remove the model keeping the median sky level."""
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        ref = float(np.median(model[::8, ::8, c]))
        if str(correction).startswith("div"):
            out[..., c] = img[..., c] / np.maximum(model[..., c], 1e-6) * ref
        else:
            out[..., c] = img[..., c] - model[..., c] + ref
    np.clip(out, 0, 1, out=out)
    return out
