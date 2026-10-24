"""Plate solving (ASTAP) and SpectroPhotometric Color Calibration against Gaia DR3."""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .. import astrometry, spectra
from ..color import background_levels
from ..stats import luminance
from .base import Param, Plugin, register

CACHE = Path.home() / ".astroforge" / "gaia"
WL = spectra.WL  # nm
SENSORS = spectra.SENSORS            # name -> R/G/B response curves (see astroforge/spectra.py)
WHITE_REFERENCES = spectra.WHITE_REFERENCES
bp_rp_to_teff = spectra.bp_rp_to_teff
planck_photons = spectra.planck_photons


def rgb_response(teff, sensor="generic_cmos", filters=None, qe="sensor"):
    """Integrated photon flux in R, G, B of model stars of temperature teff: shape (N, 3)."""
    return spectra.fluxes(spectra.star_spectrum(teff), spectra.response(sensor, filters, qe))


def expected_ratios(bp_rp, sensor="generic_cmos", white_reference="G2V", filters=None, qe="sensor"):
    """Expected R/G and B/G (columns 0 and 2, column 1 = 1) relative to the white reference."""
    return spectra.expected_ratios(bp_rp, sensor, white_reference, filters, qe)


def _fwhm(lum, xs, ys, half=6):
    vals = []
    h, w = lum.shape
    for x, y in zip(xs[:200], ys[:200]):
        xi, yi = int(round(x)), int(round(y))
        if xi < half or yi < half or xi >= w - half or yi >= h - half:
            continue
        cut = lum[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(np.float64)
        cut = cut - np.median(np.concatenate([cut[0], cut[-1], cut[:, 0], cut[:, -1]]))
        cut[cut < 0] = 0
        s = cut.sum()
        if s <= 0:
            continue
        g = np.arange(-half, half + 1)
        mx, my = (cut.sum(0) * g).sum() / s, (cut.sum(1) * g).sum() / s
        var = ((cut.sum(0) * (g - mx) ** 2).sum() + (cut.sum(1) * (g - my) ** 2).sum()) / (2 * s)
        if var > 0:
            vals.append(2.3548 * np.sqrt(var))
    return float(np.median(vals)) if vals else 3.0


def photometry(img, xs, ys, radius, saturation=0.95):
    """Aperture photometry with a local annulus. Returns (flux (N, C), ok mask, refined x, y)."""
    h, w, nc = img.shape
    r_in, r_out = radius + 2.0, radius + 6.0
    half = int(np.ceil(r_out)) + 1
    g = np.arange(-half, half + 1)
    lum = luminance(img)
    flux = np.zeros((len(xs), nc))
    ok = np.zeros(len(xs), bool)
    rx, ry = np.array(xs, float), np.array(ys, float)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xi, yi = int(round(x)), int(round(y))
        if xi < half + 2 or yi < half + 2 or xi >= w - half - 2 or yi >= h - half - 2:
            continue
        # recenter on the light (WCS + catalogue are good to ~1 px)
        win = lum[yi - 2:yi + 3, xi - 2:xi + 3].astype(np.float64)
        win = win - win.min()
        if win.sum() > 0:
            x = xi + (win.sum(0) * np.arange(-2, 3)).sum() / win.sum()
            y = yi + (win.sum(1) * np.arange(-2, 3)).sum() / win.sum()
            xi, yi = int(round(x)), int(round(y))
        cut = img[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(np.float64)
        if cut.shape[:2] != (2 * half + 1, 2 * half + 1):
            continue
        d = np.hypot(g[None, :] - (x - xi), g[:, None] - (y - yi))
        ap, ann = d <= radius, (d >= r_in) & (d <= r_out)
        if cut[ap].max() >= saturation:
            continue
        sky = np.median(cut[ann], axis=0)
        f = (cut[ap] - sky).sum(axis=0)
        noise = np.std(cut[ann], axis=0) * np.sqrt(ap.sum()) + 1e-12
        if (f > 5 * noise).all():
            flux[i], ok[i], rx[i], ry[i] = f, True, x, y
    return flux, ok, rx, ry


def fit_scales(measured, expected, iters=10, clip=3.0):
    """Robust per-channel factors k (k_G = 1) so that k*measured/G matches expected/G (in log space)."""
    m = measured / measured[:, 1:2]
    lr = np.log(expected) - np.log(m)          # log of per-star correction
    keep = np.isfinite(lr).all(axis=1)
    k = np.zeros(3)
    for _ in range(iters):
        k = np.median(lr[keep], axis=0)
        res = lr - k
        mad = 1.4826 * np.median(np.abs(res[keep]), axis=0) + 1e-6
        new = np.isfinite(lr).all(axis=1) & (np.abs(res) <= clip * mad).all(axis=1)
        if new.sum() < 5 or (new == keep).all():
            keep = new if new.sum() >= 5 else keep
            break
        keep = new
    k = np.median(lr[keep], axis=0)
    scatter = 1.4826 * np.median(np.abs(lr[keep] - k), axis=0)
    return np.exp(k), keep, scatter


def _tools(ctx):
    if ctx.tools:
        return ctx.tools
    from ..pipeline import load_config
    return load_config().get("tools", {})


def _get_wcs(img, ctx, fov_deg, ra_hours, dec_deg, radius_deg, timeout, force=False):
    w = ctx.layers.get("wcs")
    if w is not None and not force:
        return w
    w = astrometry.solve(img, _tools(ctx), header=ctx.header, fov_deg=fov_deg,
                         ra_hours=None if ra_hours < 0 else ra_hours, dec_deg=None if dec_deg < -90 else dec_deg,
                         radius_deg=radius_deg, timeout=timeout, log=ctx.log)
    if w is not None:
        ctx.layers["wcs"] = w
    return w


SOLVE_PARAMS = {
    "fov_deg": Param(0.0, 0.0, 30.0, "field height in degrees, 0 = from FOCALLEN/XPIXSZ or automatic"),
    "ra_hours": Param(-1.0, -1.0, 24.0, "approximate RA in hours, -1 = from header or blind"),
    "dec_deg": Param(-99.0, -99.0, 90.0, "approximate Dec in degrees, -99 = from header or blind"),
    "radius_deg": Param(30.0, 1.0, 180.0, "search radius around the hint"),
    "timeout": Param(300, 30, 1800, "seconds before ASTAP is stopped"),
}


@register
class PlateSolve(Plugin):
    id = "analysis.plate_solve"
    name = "Plate solve (ASTAP)"
    category = "analysis"
    description = "Finds the sky coordinates of the image with ASTAP; the WCS is kept for later steps (SPCC)."
    params = dict(SOLVE_PARAMS)

    def run(self, img, ctx, fov_deg, ra_hours, dec_deg, radius_deg, timeout):
        w = _get_wcs(img, ctx, fov_deg, ra_hours, dec_deg, radius_deg, timeout, force=True)
        if w is None:
            ctx.log("plate solve failed: image unchanged, no WCS")
            return img
        ra, dec = astrometry.field_center(w, img.shape)
        rot, flip = astrometry.rotation_deg(w)
        ctx.log(f"solved: center RA {ra / 15:.4f}h ({ra:.4f}) Dec {dec:+.4f}, "
                f"{astrometry.pixel_scale_arcsec(w):.3f}\"/px, rotation {rot:.1f} deg{' (mirrored)' if flip else ''}")
        return img


@register
class SPCC(Plugin):
    id = "color.spcc"
    name = "Spectrophotometric color calibration"
    category = "color"
    stage = "linear"
    description = ("Plate solves, measures Gaia DR3 stars and scales R/G/B so star colours match their "
                   "BP-RP (model spectrum through sensor x filter x QE curves of astroforge/spectra.py); "
                   "also neutralises the background. Curves are approximations; own CSV curves go in "
                   "~/.astroforge/spectra.")
    params = {**SOLVE_PARAMS,
              "sensor": Param("generic_cmos", choices=spectra.sensor_names(),
                              help="camera R/G/B response (sony_color = IMX071/IMX294 e.g. QHY168C, "
                                   "panasonic_color = MN34230 QHY163C, mono_* = mono + RGB filters)"),
              "filter": Param("none", choices=spectra.filter_names(),
                              help="filter in front of the sensor (several: 'uv_ir_cut+l_pro')"),
              "qe": Param("sensor", choices=spectra.qe_names(),
                          help="quantum efficiency: 'sensor' = native (colour curves already include it), "
                               "'ideal' = 1, or a QE curve"),
              "white_reference": Param("average_spiral", choices=list(WHITE_REFERENCES),
                                       help="what is rendered as white"),
              "mag_limit": Param(14.0, 9.0, 18.0, "faintest Gaia G magnitude used"),
              "aperture": Param(0.0, 0.0, 30.0, "aperture radius in px, 0 = 1.5 x FWHM"),
              "saturation": Param(0.95, 0.5, 1.0, "stars with a pixel above this are rejected"),
              "neutralize_background": Param(True, help="bring the sky to the same level in every channel"),
              "graph": Param(True, help="save spcc_graph.png (measured vs expected colours) in the output folder")}

    def run(self, img, ctx, fov_deg, ra_hours, dec_deg, radius_deg, timeout, sensor, white_reference,
            mag_limit, aperture, saturation, neutralize_background, filter="none", qe="sensor", graph=True):
        if img.shape[-1] != 3:
            ctx.log("SPCC needs a colour image: skipped")
            return img
        filters = None if str(filter).strip().lower() in ("", "none") else filter
        try:
            spectra.response(sensor, filters, qe)
            spectra.white_reference(white_reference)
        except KeyError as e:
            ctx.log(f"SPCC: {e.args[0]}; image unchanged")
            return img
        narrow = spectra.is_narrowband(filters)
        if narrow:
            ctx.log(f"SPCC warning: '{filter}' is a narrowband filter "
                    f"(~{spectra.equivalent_width(filters):.0f} nm): stars get balanced, but the colours of "
                    "nebular emission lines are not calibrated by star colours")
        w = _get_wcs(img, ctx, fov_deg, ra_hours, dec_deg, radius_deg, timeout)
        if w is None:
            ctx.log("SPCC: no plate solution, image unchanged")
            return img
        cat = astrometry.gaia_stars(w, img.shape, mag_limit=mag_limit, cache_dir=CACHE, log=ctx.log)
        if cat is None or len(cat["x"]) < 10:
            ctx.log("SPCC: Gaia catalogue unavailable or too few stars, image unchanged")
            return img
        xs, ys = cat["x"], cat["y"]
        # blended: another catalogue star less than 2.5 mag fainter within the outer annulus
        lum = luminance(img)
        fw = _fwhm(lum, xs[np.argsort(cat["g"])], ys[np.argsort(cat["g"])])
        radius = aperture or max(2.0, 1.5 * fw)
        tree = cKDTree(np.c_[xs, ys])
        blended = np.zeros(len(xs), bool)
        for i, nb in enumerate(tree.query_ball_point(np.c_[xs, ys], radius + 6)):
            blended[i] = any(j != i and cat["g"][j] < cat["g"][i] + 2.5 for j in nb)
        flux, ok, _, _ = photometry(img, xs, ys, radius, saturation)
        ok &= ~blended & (cat["bp_rp"] > -0.4) & (cat["bp_rp"] < 3.5)
        if ok.sum() < 10:
            ctx.log(f"SPCC: only {ok.sum()} usable stars, image unchanged")
            return img
        exp = expected_ratios(cat["bp_rp"][ok], sensor, white_reference, filters, qe)
        k, keep, scatter = fit_scales(flux[ok], exp)
        slope = fit_slopes(flux[ok][keep], exp[keep])
        bg = background_levels(img)
        kn = k / k.max()
        if neutralize_background:
            out = (img - bg) * kn.astype(np.float32) + float((bg * kn).mean())
        else:
            out = img * kn.astype(np.float32)
        np.clip(out, 0, 1, out=out)
        info = {"factors": k.tolist(), "applied": kn.tolist(), "stars": int(keep.sum()),
                "measured": int(ok.sum()), "catalog": int(len(xs)), "scatter": scatter.tolist(),
                "aperture": float(radius), "background": bg.tolist(), "sensor": sensor,
                "white_reference": white_reference, "filter": filter, "qe": qe, "narrowband": bool(narrow),
                "slope": slope.tolist()}
        pct = (np.exp(scatter) - 1) * 100
        ctx.log(f"SPCC: factors R/G/B {np.round(k, 4)} from {keep.sum()}/{ok.sum()} stars "
                f"(catalogue {len(xs)}, aperture {radius:.1f}px), scatter R {scatter[0]:.3f} B {scatter[2]:.3f} (ln) "
                f"= {pct[0]:.1f}% / {pct[2]:.1f}%, slope R/G {slope[0]:.2f} B/G {slope[2]:.2f} (1 = curves fit)")
        if graph and ctx.out_dir:
            path = Path(ctx.out_dir) / "spcc_graph.png"
            try:
                save_graph(path, flux[ok], exp, k, keep, scatter,
                           title=f"SPCC  {sensor}  filter {filter}  white {white_reference}")
                info["graph"] = str(path)
                ctx.log(f"SPCC graph: {path}")
            except Exception as e:  # plotting must never break the calibration
                ctx.log(f"SPCC graph not saved ({type(e).__name__}: {e})")
        ctx.layers["spcc"] = info
        return out.astype(np.float32, copy=False)


def fit_slopes(measured, expected):
    """Slope of ln(measured ratio) vs ln(expected ratio) per channel (column 1 = 1). ~1 means the
    sensor/filter curves describe the camera well; < 1 means the camera sees smaller colour differences."""
    m = np.log(measured / measured[:, 1:2])
    e = np.log(expected)
    out = np.ones(3)
    for c in (0, 2):
        x, y = e[:, c], m[:, c]
        if len(x) >= 5 and np.ptp(x) > 1e-6:
            out[c] = np.polyfit(x, y, 1)[0]
    return out


def save_graph(path, measured, expected, k, keep, scatter, title="SPCC"):
    """PNG like the PixInsight SPCC graph: measured vs expected ln(R/G) and ln(B/G) per star with the
    fitted line y = x - ln k (matplotlib if installed, otherwise drawn with OpenCV)."""
    m = np.log(measured / measured[:, 1:2])
    e = np.log(expected)
    lk = np.log(k)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [(0, "R/G", (200, 40, 40)), (2, "B/G", (40, 70, 200))]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None:
        fig, axs = plt.subplots(1, 2, figsize=(11, 5), dpi=100)
        for ax, (c, name, col) in zip(axs, panels):
            colr = tuple(v / 255 for v in col)
            ax.scatter(e[~keep, c], m[~keep, c], s=8, c="0.7", label="rejected")
            ax.scatter(e[keep, c], m[keep, c], s=10, color=colr, label="used")
            xs = np.linspace(e[:, c].min(), e[:, c].max(), 2)
            ax.plot(xs, xs - lk[c], "k-", lw=1.2, label=f"fit  k={k[c]:.4f}")
            ax.set_xlabel(f"expected ln({name})  (catalogue)")
            ax.set_ylabel(f"measured ln({name})")
            ax.set_title(f"{name}: scatter {scatter[c]:.3f} ln ({(np.exp(scatter[c]) - 1) * 100:.1f}%)")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
    import cv2
    pw, ph, pad = 520, 460, 60
    canvas = np.full((ph + 40, 2 * pw, 3), 255, np.uint8)
    cv2.putText(canvas, title, (pad, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    for p, (c, name, col) in enumerate(panels):
        x0, y0 = p * pw + pad, 40
        w, h = pw - pad - 20, ph - pad - 30
        xv, yv = e[:, c], m[:, c]
        lo = min(xv.min(), (yv[keep] + lk[c]).min() if keep.any() else xv.min())
        hi = max(xv.max(), (yv[keep] + lk[c]).max() if keep.any() else xv.max())
        span = max(hi - lo, 1e-3)
        lo, hi = lo - 0.08 * span, hi + 0.08 * span
        ylo, yhi = lo - lk[c], hi - lk[c]

        def px(x, y, _x0=x0, _y0=y0, _w=w, _h=h, _lo=lo, _hi=hi, _ylo=ylo, _yhi=yhi):
            return (int(_x0 + (x - _lo) / (_hi - _lo) * _w), int(_y0 + _h - (y - _ylo) / (_yhi - _ylo) * _h))

        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), 1)
        for i in range(5):
            tx = lo + (hi - lo) * i / 4
            ty = ylo + (yhi - ylo) * i / 4
            gx, _ = px(tx, ylo)
            _, gy = px(lo, ty)
            cv2.line(canvas, (gx, y0), (gx, y0 + h), (225, 225, 225), 1)
            cv2.line(canvas, (x0, gy), (x0 + w, gy), (225, 225, 225), 1)
            cv2.putText(canvas, f"{tx:.2f}", (gx - 16, y0 + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)
            cv2.putText(canvas, f"{ty:.2f}", (x0 - 46, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)
        bgr = col[::-1]
        for i in np.flatnonzero(~keep):
            cv2.circle(canvas, px(xv[i], yv[i]), 2, (180, 180, 180), -1, cv2.LINE_AA)
        for i in np.flatnonzero(keep):
            cv2.circle(canvas, px(xv[i], yv[i]), 3, bgr, -1, cv2.LINE_AA)
        cv2.line(canvas, px(lo, lo - lk[c]), px(hi, hi - lk[c]), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{name}: k={k[c]:.4f}  scatter {scatter[c]:.3f} ln ({(np.exp(scatter[c]) - 1) * 100:.1f}%)",
                    (x0, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"x: expected ln({name})   y: measured ln({name})", (x0, y0 + h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise OSError("PNG encoding failed")
    path.write_bytes(buf.tobytes())
    return path


# ================================================================== PCC (photometric, APASS)
#
# PixInsight PhotometricColorCalibration does not need filter curves: it measures R/G and B/G of the
# catalogue stars and relates them to the stars' catalogue colour indices; the white point is where that
# relation crosses the catalogue colour of the white reference. Equivalent implemented here:
#
#   ln(R/G)_measured = aR + bR * (g'-r')      robust line (Theil-Sen start, iterative 3-sigma MAD clip)
#   ln(B/G)_measured = aB + bB * (B-V)
#
# evaluated at the white reference indices (B-V)_w, (g'-r')_w gives the colour a white-reference star
# would have in the image; the factors k_R = exp(-(aR + bR (g'-r')_w)), k_B = exp(-(aB + bB (B-V)_w)),
# k_G = 1 make it neutral. Stars without g'/r' use g'-r' = 1.05 (B-V) - 0.23 (Jester et al. 2005, stars).

APASS_CACHE = Path.home() / ".astroforge" / "apass"
# B-V of the white references: Sun 0.653 (Ramirez et al. 2012); average spiral 0.69 (Sb/Sc, the same
# population mix as spectra.white_reference); elliptical 0.96 (Fukugita et al. 1995, E template).
PCC_WHITE_BV = {"average_spiral": 0.69, "g2v": 0.653, "elliptical": 0.96}
PCC_R_INDICES = ["g-r", "B-V"]


def bv_to_gr(bv):
    """SDSS g'-r' from Johnson B-V for stars (Jester et al. 2005, Rc-Ic < 1.15)."""
    return 1.05 * np.asarray(bv, float) - 0.23


def pcc_white(white_reference):
    """(B-V, g'-r') of a white reference: a name (average_spiral, G2V, elliptical) or a B-V number."""
    key = str(white_reference).strip().lower()
    key = {"sun": "g2v", "solar": "g2v", "spiral": "average_spiral", "galaxy": "average_spiral",
           "average_elliptical": "elliptical"}.get(key, key)
    if key in PCC_WHITE_BV:
        bv = PCC_WHITE_BV[key]
    else:
        try:
            bv = float(key)
        except ValueError:
            raise KeyError(f"unknown white reference '{white_reference}' "
                           "(average_spiral, G2V, elliptical or a B-V value such as 0.65)") from None
        if not -0.5 <= bv <= 2.5:
            raise KeyError(f"white reference B-V {bv} outside -0.5..2.5")
    return float(bv), float(bv_to_gr(bv))


def robust_line(x, y, clip=3.0, iters=15, min_keep=5, floor=0.01):
    """y = a + b x robust to outliers: Theil-Sen start, then least squares on the stars within
    clip x max(MAD, floor) of the line, iterated (floor = 1% photometric error, so a very tight
    sequence does not reject its own natural curvature). Returns a, b, keep mask, scatter (1.4826 MAD of the residuals)."""
    from scipy.stats import theilslopes
    x, y = np.asarray(x, float), np.asarray(y, float)
    fin = np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(fin)
    if len(idx) > 1500:
        idx = np.random.default_rng(0).choice(idx, 1500, replace=False)
    if np.ptp(x[idx]) > 1e-6:
        b, a = theilslopes(y[idx], x[idx])[:2]
    else:
        b, a = 0.0, float(np.median(y[idx]))
    keep = fin.copy()
    for _ in range(iters):
        res = y - (a + b * x)
        mad = max(1.4826 * np.median(np.abs(res[fin & keep])), floor)
        new = fin & (np.abs(res) <= clip * mad)
        if new.sum() < min_keep:
            break
        same = (new == keep).all()
        keep = new
        if np.ptp(x[keep]) > 1e-6:
            b, a = np.polyfit(x[keep], y[keep], 1)
        else:
            b, a = 0.0, float(np.median(y[keep]))
        if same:
            break
    keep &= fin
    scatter = 1.4826 * np.median(np.abs(y[keep] - (a + b * x[keep])))
    return float(a), float(b), keep, float(scatter)


@register
class PCC(Plugin):
    id = "color.pcc"
    name = "Photometric color calibration (APASS)"
    category = "color"
    stage = "linear"
    description = ("PixInsight-style PCC: plate solves, measures APASS DR9 stars (VizieR II/336/apass9) and fits "
                   "measured ln(R/G) against g'-r' and ln(B/G) against B-V; the factors make a star with the "
                   "colour indices of the white reference neutral. No sensor or filter curves needed. "
                   "Also neutralises the background.")
    params = {**SOLVE_PARAMS,
              "white_reference": Param("average_spiral",
                                       help="what is rendered as white: average_spiral, G2V, elliptical, "
                                            "or a B-V value (e.g. 0.65)"),
              "r_index": Param("g-r", choices=PCC_R_INDICES,
                               help="catalogue index fitted against R/G: APASS g'-r' (from B-V when missing) "
                                    "or B-V"),
              "mag_limit": Param(15.0, 9.0, 17.0, "faintest APASS V magnitude used"),
              "aperture": Param(0.0, 0.0, 30.0, "aperture radius in px, 0 = 1.5 x FWHM"),
              "saturation": Param(0.95, 0.5, 1.0, "stars with a pixel above this are rejected"),
              "neutralize_background": Param(True, help="bring the sky to the same level in every channel"),
              "graph": Param(True, help="save pcc_graph.png (measured colour vs catalogue index) in the output folder")}

    def run(self, img, ctx, fov_deg, ra_hours, dec_deg, radius_deg, timeout, white_reference="average_spiral",
            r_index="g-r", mag_limit=15.0, aperture=0.0, saturation=0.95, neutralize_background=True, graph=True):
        if img.ndim != 3 or img.shape[-1] != 3:
            ctx.log("PCC needs a colour image: skipped")
            return img
        try:
            bv_w, gr_w = pcc_white(white_reference)
        except KeyError as e:
            ctx.log(f"PCC: {e.args[0]}; image unchanged")
            return img
        if r_index not in PCC_R_INDICES:
            ctx.log(f"PCC: unknown r_index '{r_index}' ({', '.join(PCC_R_INDICES)}); image unchanged")
            return img
        w = _get_wcs(img, ctx, fov_deg, ra_hours, dec_deg, radius_deg, timeout)
        if w is None:
            ctx.log("PCC: no plate solution, image unchanged")
            return img
        cat = astrometry.apass_stars(w, img.shape, mag_limit=mag_limit, cache_dir=APASS_CACHE, log=ctx.log)
        if cat is None or len(cat["x"]) < 10:
            ctx.log("PCC: APASS catalogue unavailable or too few stars, image unchanged")
            return img
        n = len(cat["x"])
        nan = np.full(n, np.nan)
        xs, ys = np.asarray(cat["x"], float), np.asarray(cat["y"], float)
        V, B = np.asarray(cat.get("V", nan), float), np.asarray(cat.get("B", nan), float)
        g, r = np.asarray(cat.get("g", nan), float), np.asarray(cat.get("r", nan), float)
        mag = np.where(np.isfinite(V), V, r)
        mag = np.where(np.isfinite(mag), mag, 99.0)
        bv, gr = B - V, g - r
        n_from_bv = int((~np.isfinite(gr) & np.isfinite(bv)).sum()) if r_index == "g-r" else 0
        if r_index == "g-r":
            xr, xr_w = np.where(np.isfinite(gr), gr, bv_to_gr(bv)), gr_w
        else:
            xr, xr_w = bv, bv_w
        lum = luminance(img)
        order = np.argsort(mag)
        fw = _fwhm(lum, xs[order], ys[order])
        radius = aperture or max(2.0, 1.5 * fw)
        tree = cKDTree(np.c_[xs, ys])
        blended = np.zeros(n, bool)
        for i, nb in enumerate(tree.query_ball_point(np.c_[xs, ys], radius + 6)):
            blended[i] = any(j != i and mag[j] < mag[i] + 2.5 for j in nb)
        flux, ok, _, _ = photometry(img, xs, ys, radius, saturation)
        ok &= ~blended & np.isfinite(bv) & np.isfinite(xr) & (bv > -0.4) & (bv < 2.2)
        if ok.sum() < 10:
            ctx.log(f"PCC: only {ok.sum()} usable stars, image unchanged")
            return img
        f = flux[ok]
        with np.errstate(divide="ignore", invalid="ignore"):
            lrg, lbg = np.log(f[:, 0] / f[:, 1]), np.log(f[:, 2] / f[:, 1])
        aR, bR, keepR, sR = robust_line(xr[ok], lrg)
        aB, bB, keepB, sB = robust_line(bv[ok], lbg)
        k = np.array([np.exp(-(aR + bR * xr_w)), 1.0, np.exp(-(aB + bB * bv_w))])
        if not np.isfinite(k).all() or (k <= 0).any():
            ctx.log("PCC: fit failed, image unchanged")
            return img
        bg = background_levels(img)
        kn = k / k.max()
        if neutralize_background:
            out = (img - bg) * kn.astype(np.float32) + float((bg * kn).mean())
        else:
            out = img * kn.astype(np.float32)
        np.clip(out, 0, 1, out=out)
        keep = keepR & keepB
        info = {"catalog_name": astrometry.APASS_NAME, "factors": k.tolist(), "applied": kn.tolist(),
                "white_reference": str(white_reference), "white_bv": bv_w, "white_gr": gr_w, "r_index": r_index,
                "fit_r": {"intercept": aR, "slope": bR, "scatter": sR, "stars": int(keepR.sum()), "index": r_index},
                "fit_b": {"intercept": aB, "slope": bB, "scatter": sB, "stars": int(keepB.sum()), "index": "B-V"},
                "stars": int(keep.sum()), "measured": int(ok.sum()), "catalog": int(n),
                "gr_from_bv": n_from_bv, "aperture": float(radius), "background": bg.tolist()}
        ctx.log(f"PCC ({astrometry.APASS_NAME}): factors R/G/B {np.round(k, 4)}, white {white_reference} "
                f"(B-V {bv_w:.3f}, g'-r' {gr_w:.3f}); R/G fit on {keepR.sum()}/{ok.sum()} stars slope {bR:.3f} "
                f"scatter {sR:.3f} ln, B/G fit on {keepB.sum()}/{ok.sum()} slope {bB:.3f} scatter {sB:.3f} ln "
                f"(catalogue {n}, aperture {radius:.1f}px"
                + (f", g'-r' from B-V for {n_from_bv}" if n_from_bv else "") + ")")
        if graph and ctx.out_dir:
            path = Path(ctx.out_dir) / "pcc_graph.png"
            try:
                save_pcc_graph(path, [(xr[ok], lrg, keepR, aR, bR, xr_w, sR, f"R/G vs {r_index}"),
                                      (bv[ok], lbg, keepB, aB, bB, bv_w, sB, "B/G vs B-V")],
                               title=f"PCC  {astrometry.APASS_NAME}  white {white_reference}")
                info["graph"] = str(path)
                ctx.log(f"PCC graph: {path}")
            except Exception as e:  # plotting must never break the calibration
                ctx.log(f"PCC graph not saved ({type(e).__name__}: {e})")
        ctx.layers["pcc"] = info
        return out.astype(np.float32, copy=False)


def save_pcc_graph(path, panels, title="PCC"):
    """PNG like the PixInsight PCC graph: per star measured ln(ratio) vs catalogue colour index, the robust
    line and the white reference index (vertical green line). panels: (x, y, keep, a, b, x_white, scatter, name)."""
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pw, ph, pad = 560, 460, 60
    canvas = np.full((ph + 40, pw * len(panels), 3), 255, np.uint8)
    cv2.putText(canvas, title, (pad, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    colours = [(40, 40, 200), (200, 70, 40)]   # BGR: red panel, blue panel
    for p, (xv, yv, keep, a, b, xw, sc, name) in enumerate(panels):
        x0, y0 = p * pw + pad, 40
        w, h = pw - pad - 20, ph - pad - 30
        fin = np.isfinite(xv) & np.isfinite(yv)
        xlo, xhi = float(np.min(np.r_[xv[fin], xw])), float(np.max(np.r_[xv[fin], xw]))
        sel = keep & fin if (keep & fin).any() else fin
        yw = a + b * xw
        ylo, yhi = min(float(yv[sel].min()), yw), max(float(yv[sel].max()), yw)
        xs_, ys_ = max(xhi - xlo, 1e-3), max(yhi - ylo, 1e-3)
        lim = (xlo - 0.08 * xs_, xhi + 0.08 * xs_, ylo - 0.15 * ys_, yhi + 0.15 * ys_)

        def px(x, y, _x0=x0, _y0=y0, _w=w, _h=h, _a=lim):
            return (int(_x0 + (x - _a[0]) / (_a[1] - _a[0]) * _w),
                    int(np.clip(_y0 + _h - (y - _a[2]) / (_a[3] - _a[2]) * _h, _y0, _y0 + _h)))

        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), 1)
        for i in range(5):
            tx, ty = lim[0] + (lim[1] - lim[0]) * i / 4, lim[2] + (lim[3] - lim[2]) * i / 4
            gx, _ = px(tx, lim[2])
            _, gy = px(lim[0], ty)
            cv2.line(canvas, (gx, y0), (gx, y0 + h), (225, 225, 225), 1)
            cv2.line(canvas, (x0, gy), (x0 + w, gy), (225, 225, 225), 1)
            cv2.putText(canvas, f"{tx:.2f}", (gx - 16, y0 + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)
            cv2.putText(canvas, f"{ty:.2f}", (x0 - 46, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)
        for i in np.flatnonzero(fin & ~keep):
            cv2.circle(canvas, px(xv[i], yv[i]), 2, (180, 180, 180), -1, cv2.LINE_AA)
        for i in np.flatnonzero(fin & keep):
            cv2.circle(canvas, px(xv[i], yv[i]), 3, colours[p % 2], -1, cv2.LINE_AA)
        lx = np.r_[xv[sel], xw]       # line over the used stars (+ white index): stays inside the y range
        cv2.line(canvas, px(lx.min(), a + b * lx.min()), px(lx.max(), a + b * lx.max()), (0, 0, 0), 2, cv2.LINE_AA)
        wx, _ = px(xw, lim[2])
        cv2.line(canvas, (wx, y0), (wx, y0 + h), (0, 160, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{name}: slope {b:.3f} scatter {sc:.3f} ln, white {xw:.2f}: k={np.exp(-yw):.4f}",
                    (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        ratio, _, index = name.partition(" vs ")
        cv2.putText(canvas, f"x: catalogue {index}   y: measured ln({ratio})",
                    (x0, y0 + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise OSError("PNG encoding failed")
    path.write_bytes(buf.tobytes())
    return path
