"""Several sources of one target: sessions shot with different cameras (merged into one integration) or the
hybrid mode (a colour image plus narrowband channels, e.g. RGB + Ha, combined afterwards with a slider).

A source is {"name", "role", "lights": [...], "calibration": [...], "master_dirs": [...], "filters": bool,
"file": path}: with "file" an already integrated image is used as it is, otherwise the source is stacked with its
own calibration frames (each camera needs its own bias/dark/flat). Every result is registered on the first
source by star matching (triangles: different pixel scales and rotations are fine) and cropped to the area all
sources cover.

- mode "multi": sources with the same channels (e.g. two colour cameras, or two mono Ha sets); each one is
  brought to the reference levels with a robust linear fit and the images are averaged with noise weights
  (1 / sigma^2 of the background, like ImageIntegration's noise evaluation), only where they overlap.
- mode "hybrid": the first source is the colour image (role "rgb"), the others narrowband (role "ha", "oiii",
  "sii"): they are saved registered on it, and the step color.nb_rgb (NBRGBCombination) puts them into the
  colour image in post-production, where its sliders set how much of each one shows.
"""
import re
from pathlib import Path

import numpy as np

from .. import io
from . import analyze, register, xisf

NB_ROLES = ("ha", "oiii", "sii")


def _slug(name, i):
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "")).strip("_")
    return s[:24] or f"src{i}"


def load(path):
    return load_header(path)[0]


def load_header(path):
    a, h = io.load(path)
    return (a if a.ndim == 3 else a[..., None]), h


def lum(img):
    return img[..., 0] if img.shape[-1] == 1 else img.mean(-1)


def stars(img, n=400, highpass=False):
    """Star positions. highpass: on the image minus its blurred copy, for stretched (finished) images where the
    nebula is as bright as the stars and the brightest "sources" are knots of nebulosity."""
    L = np.ascontiguousarray(np.nan_to_num(lum(img)), np.float32)
    if highpass:
        import cv2
        L = np.clip(L - cv2.GaussianBlur(L, (0, 0), 8), 0, None)
    st = analyze.detect(L, n, saturation=2.0)
    return np.stack([st["x"], st["y"]], 1)


def wcs_match(img, ref, img_header=None, ref_header=None, tools=None, log=print, name="", cache=None):
    """Affine img -> ref from plate solving both (ASTAP): for cameras with different fields, where the brightest
    stars of the larger field fall outside the smaller one and the triangles cannot match. The image without
    RA/DEC hints is solved near the reference's centre (same target). Refined on the stars it then lands on.
    Returns (M, n, rms) or None."""
    from .. import astrometry

    def solve(im, hdr, **kw):
        # finished files often lack FOCALLEN / XPIXSZ: without the field size ASTAP fails, so try a range
        _, _, fov = astrometry.header_hints(hdr, im.shape[0])
        for fv in ([fov] if fov else []) + [0, 1.0, 2.0, 3.0, 4.5]:
            got = astrometry.solve(lum(im), tools, hdr, fov_deg=fv, log=log, retry_auto_fov=False, **kw)
            if got is not None:
                return got
        return None
    cache = {} if cache is None else cache
    if "ref" not in cache:
        cache["ref"] = solve(ref, ref_header)
    wr = cache["ref"]
    if wr is None:
        log(f"{name}: plate solve of the first source failed")
        return None
    ra, dec = astrometry.field_center(wr, ref.shape[:2])
    hra, _, _ = astrometry.header_hints(img_header, img.shape[0])
    wi = solve(img, img_header, ra_hours=None if hra is not None else ra / 15.0,
               dec_deg=None if hra is not None else dec, radius_deg=15 if hra is None else 30)
    if wi is None:
        log(f"{name}: plate solve failed")
        return None
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.linspace(0, w - 1, 12), np.linspace(0, h - 1, 12))
    src = np.stack([gx.ravel(), gy.ravel()], 1)
    sky = wi.all_pix2world(src, 0)
    dst = wr.all_world2pix(sky, 0)
    A = np.hstack([src, np.ones((len(src), 1))])
    M = np.linalg.lstsq(A, dst, rcond=None)[0].T
    log(f"{name}: placed by plate solving ({astrometry.pixel_scale_arcsec(wi):.2f}\"/px on "
        f"{astrometry.pixel_scale_arcsec(wr):.2f}\"/px)")
    return refine(M, img, ref)


def refine(M, img, ref, tols=(6.0, 2.5)):
    """Affine img -> ref refined on many stars (high-pass) near where M puts them: a match made on few control
    points (or on plate solutions) gets the whole overlap. Returns (M, n, rms)."""
    from scipy.spatial import cKDTree
    xy, rxy = stars(img, 1500, True), stars(ref, 3000, True)
    tree = cKDTree(rxy)
    for tol in tols:
        d, j = tree.query(xy @ M[:, :2].T + M[:, 2])
        good = d < tol
        if good.sum() < 10:
            break
        M = np.linalg.lstsq(np.hstack([xy[good], np.ones((good.sum(), 1))]), rxy[j[good]], rcond=None)[0].T
    d, _ = tree.query(xy @ M[:, :2].T + M[:, 2])
    inl = d < 2.0
    return M, int(inl.sum()), float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("nan")


def register_on(img, ref_xy, shape, log=print, name="", ref=None, solve=None):
    """img warped onto the reference frame (shape = reference (H, W)); pixels it does not cover are NaN.
    ref: the reference image, for the retries on finished (stretched) images when the first match fails.
    solve: callable () -> (M, n, rms) | None, the last resort (plate solving both images)."""
    # (stars of the reference, how many, high-pass, triangle control points): the last one is for cameras with
    # different fields, where the brightest stars of the larger field fall outside the smaller one (M8 drizzle
    # 2x QHY168C + M20 Ha QHY163M: only 300 control points find it, ~20 s; 500 take 7 min)
    tries = [(None, 400, False, 80), (ref, 800, True, 80), (ref, 300, True, 300)] if ref is not None         else [(None, 400, False, 80)]
    err = None
    for r, n, hp, cp in tries:
        rxy = ref_xy if r is None else stars(r, n, hp)
        xy = stars(img, n, hp)
        if len(xy) < 8 or len(rxy) < 8:
            err = ValueError(f"{name}: too few stars to register ({len(xy)} / {len(rxy)})")
            continue
        try:
            M, n, rms = register.match(xy, rxy, max_points=cp)
            if r is not None:
                M, n, rms = refine(M, img, ref)
            break
        except Exception as e:                 # astroalign: no matching triangles
            err = e
            log(f"{name}: star match failed ({str(e)[:80]}), retrying on the stars only (high-pass, more stars)")
    else:
        got = solve() if solve is not None else None
        if got is None:
            raise ValueError(f"{name}: could not register on the first source: {err}")
        M, n, rms = got
    a, b = M[0, :2], M[1, :2]
    scale = float(np.sqrt(abs(a[0] * b[1] - a[1] * b[0])))
    log(f"{name}: registered on {n} stars, rms {rms:.2f} px, scale x{scale:.3f}, "
        f"rotation {np.degrees(np.arctan2(b[0], a[0])):+.1f} deg")
    out = register.warp(img, M, shape)
    return out if out.ndim == 3 else out[..., None]


def common_area(images):
    """Largest axis-aligned rectangle (greedy) where every image has data: finite (the registration leaves NaN
    outside) and, on linear data, > 0 (black borders of the integrations; a finished image can have a sky at 0)."""
    from ..pipeline import looks_stretched
    cov = None
    for im in images:
        c = np.all(np.isfinite(im), -1)
        if not looks_stretched(np.nan_to_num(im, nan=0.0)):
            c &= np.all(np.nan_to_num(im, nan=0.0) > 0, -1)
        cov = c if cov is None else cov & c
    ys, xs = np.nonzero(cov)
    if len(ys) < 0.05 * cov.size:
        raise ValueError("the sources do not overlap")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    while y1 - y0 > 8 and x1 - x0 > 8:
        sub = cov[y0:y1, x0:x1]
        if sub.mean() > 0.998:
            break
        edges = {"top": sub[0].mean(), "bottom": sub[-1].mean(), "left": sub[:, 0].mean(), "right": sub[:, -1].mean()}
        worst = min(edges, key=edges.get)
        step = max(1, int((1 - edges[worst]) * 4))          # faster where the edge is mostly empty
        if worst == "top":
            y0 += step
        elif worst == "bottom":
            y1 -= step
        elif worst == "left":
            x0 += step
        else:
            x1 -= step
    return int(y0), int(y1), int(x0), int(x1)


def background_noise(ch):
    """sigma of the background (high-pass MAD on the darker half), for the noise weights."""
    import cv2
    x = np.nan_to_num(ch).astype(np.float32)
    hp = x - cv2.GaussianBlur(x, (0, 0), 1.5)
    dark = x < np.nanpercentile(x, 50)
    return float(1.4826 * np.median(np.abs(hp[dark] - np.median(hp[dark])))) or 1e-9


def fit_levels(ch, ref):
    """a + b*ch matched to ref on the pixels both have (robust, iterated)."""
    m = np.isfinite(ch) & np.isfinite(ref)
    x, y = ch[m][::7].astype(np.float64), ref[m][::7].astype(np.float64)
    keep = np.ones(len(x), bool)
    b, a = 1.0, 0.0
    for _ in range(5):
        b, a = np.polyfit(x[keep], y[keep], 1)
        r = y - (a + b * x)
        keep = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r[keep])) + 1e-12
    return float(a), float(b)


def merge(images, names, log=print):
    """Noise-weighted average of registered images with the same channels (multi-camera)."""
    ref = images[0]
    out_num = np.zeros(ref.shape, np.float64)
    out_den = np.zeros(ref.shape, np.float64)
    report = []
    for im, name in zip(images, names):
        if im.shape[-1] != ref.shape[-1]:
            raise ValueError(f"{name}: {im.shape[-1]} channels, the first source has {ref.shape[-1]} "
                             "(a colour image and a mono one go together with the hybrid mode)")
        ws = []
        for c in range(im.shape[-1]):
            a, b = fit_levels(im[..., c], ref[..., c]) if im is not ref else (0.0, 1.0)
            ch = a + b * im[..., c]
            w = 1.0 / background_noise(ch) ** 2
            ok = np.isfinite(ch)
            out_num[..., c][ok] += w * ch[ok]
            out_den[..., c][ok] += w
            ws.append(w)
        report.append({"name": name, "weight": float(np.mean(ws))})
    tot = sum(r["weight"] for r in report)
    for r in report:
        r["share"] = round(r["weight"] / tot, 3)
        log(f"merge: {r['name']} weight {100 * r['share']:.0f}%")
    with np.errstate(invalid="ignore"):
        out = (out_num / out_den).astype(np.float32)
    return out, report


def result_file(rep, drizzle):
    """Output file of a Runner.run / run_filters report."""
    if rep.get("combined"):
        return rep["combined"]
    out = rep["output"]
    return f"{out}_drizzle{drizzle}x.fits" if drizzle and drizzle > 1 else f"{out}_integration.fits"


def source_name(src, i):
    """Name of a source for logs and outputs: the one given, else role + file or folder name."""
    if src.get("name"):
        return src["name"]
    role = {"rgb": "RGB", "ha": "Ha", "oiii": "OIII", "sii": "SII"}.get(src.get("role") or "", "")
    where = src.get("file") or (src.get("lights") or [""])[0]
    stem = Path(where).stem if where else f"sorgente {i + 1}"
    return f"{role} {stem}".strip()


def run_sources(runner, sources, mode="multi"):
    """Stack (or load) every source, register them on the first one, then merge (multi) or prepare the hybrid
    combination. Returns (image, report); the report has "sources" and, for hybrid, "nb" {role: file}."""
    from .project import Runner
    s = runner.s
    base_output = s["output"]
    srcs = [x for x in sources if x.get("file") or x.get("lights")]
    if len(srcs) < 2:
        raise ValueError("this mode needs at least 2 sources (each with light folders or an integrated file)")
    if mode == "hybrid":
        # the colour source can be anywhere in the list (deleted and re-added cards): it goes first
        colour = [x for x in srcs if (x.get("role") or "rgb") not in NB_ROLES]
        if len(colour) != 1:
            raise ValueError("modalità ibrida: serve una sola sorgente a colori (RGB), le altre Ha / OIII / SII "
                             f"(ora a colori: {len(colour)})")
        srcs = colour + [x for x in srcs if x is not colour[0]]
    files, runs = [], []
    for i, src in enumerate(srcs):
        name = source_name(src, i)
        runner._step("sources", i / (len(srcs) + 1), f"===== {name} =====")
        if src.get("file"):
            runner.log(f"{name}: integrated file {src['file']}")
            files.append(src["file"])
            runs.append({"name": name, "file": src["file"]})
            continue
        sub = dict(s)
        sub.update(lights=list(src["lights"]), calibration=list(src.get("calibration") or s.get("calibration") or []),
                   master_dirs=list(src.get("master_dirs") or []) + list(s.get("master_dirs") or []),
                   output=f"{base_output}_{_slug(name, i + 1)}", light_group=None, masters={},
                   exclude=s.get("exclude") or [])
        sub.pop("sources", None)
        r = Runner(sub, progress=runner.progress, log=runner.log, cancel=runner.cancel)
        _, rep = r.run_filters(sub.get("palette", "auto")) if src.get("filters") else r.run()
        f = result_file(rep, sub.get("drizzle", 1))
        files.append(f)
        runs.append({"name": name, "file": f, "frames": rep.get("lights_total"), "setup": rep.get("light_setup")})
        if getattr(r, "header", None) is not None and not hasattr(runner, "header"):
            runner.header = r.header

    runner._step("sources", len(srcs) / (len(srcs) + 1), "registering the sources on the first one")
    ref, ref_header = load_header(files[0])
    if mode == "multi":
        for f, run in zip(files[1:], runs[1:]):
            ch = load(f).shape[-1]
            if ch != ref.shape[-1]:
                raise ValueError(f"{run['name']}: {ch} channels, the first source has {ref.shape[-1]} "
                                 "(a colour image and a mono one go together with the hybrid mode)")
    ref_xy = stars(ref)
    imgs = [ref]
    from ..pipeline import load_config
    tools, solved = load_config().get("tools", {}), {}
    for src, f, run in zip(srcs[1:], files[1:], runs[1:]):
        im, ih = load_header(f)
        solve = (lambda im=im, ih=ih, nm=run["name"]: wcs_match(im, ref, ih, ref_header, tools, runner.log, nm, solved))             if tools.get("astap") else None
        imgs.append(register_on(im, ref_xy, ref.shape[:2], runner.log, run["name"], ref=ref, solve=solve))
    y0, y1, x0, x1 = common_area(imgs)
    imgs = [im[y0:y1, x0:x1] for im in imgs]
    runner.log(f"area covered by every source: {x1 - x0}x{y1 - y0} (x {x0}-{x1}, y {y0}-{y1})")
    work = Path(s["work_dir"])
    hdr = getattr(runner, "header", None)
    from ..pipeline import looks_stretched
    report = {"mode": mode, "sources": runs, "crop": [int(y0), int(y1), int(x0), int(x1)],
              "stretched": looks_stretched(np.nan_to_num(imgs[0], nan=0.0))}   # finished files: no linear profile
    if mode == "multi":
        out, w = merge(imgs, [r["name"] for r in runs], runner.log)
        out = np.nan_to_num(out, nan=0.0)
        base = work / f"{base_output}_multi"
        report["weights"] = w
    else:
        out = np.nan_to_num(imgs[0], nan=0.0)
        base = work / f"{base_output}_hybrid_rgb"
        nb = {}
        for src, im in zip(srcs[1:], imgs[1:]):
            role = src["role"]
            path = work / f"{base_output}_hybrid_{role}.fits"
            io.save_fits(np.nan_to_num(lum(im), nan=0.0)[..., None].astype(np.float32), path, hdr)
            nb[role] = str(path)
            runner.log(f"{role.upper()} registered on the colour image: {path.name}")
        report["nb"] = nb
    xisf.write(out, f"{base}.xisf")
    io.save_fits(out, f"{base}.fits", hdr)
    runner._preview(out, f"{base}_preview.jpg")
    report.update(output=str(base), combined=f"{base}.fits")
    runner.report.update(report)
    runner._step("done", 1.0, f"{len(srcs)} sources " + ("merged" if mode == "multi" else "ready for the hybrid combination"))
    return out, runner.report
