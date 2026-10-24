"""Offline tests for plate-solve helpers and SPCC (catalogue monkeypatched)."""
import numpy as np
from astropy.wcs import WCS

from astroforge import astrometry, plugins
from astroforge.plugins import spcc

WCS_TEXT = """SIMPLE  =                    T / file does conform to FITS standard
BITPIX  =                   16 / number of bits per data pixel
NAXIS   =                    0 / number of data axes
CTYPE1  = 'RA---TAN-SIP'       / TAN (gnomic) projection + SIP distortions
CTYPE2  = 'DEC--TAN-SIP'       / TAN (gnomic) projection + SIP distortions
CUNIT1  = 'deg     '           / Unit of coordinates
CUNIT2  = 'deg     '           / Unit of coordinates
EQUINOX =               2000.0 / Equinox of coordinates
CRPIX1  =  2.001000000000E+002 / X of reference pixel
CRPIX2  =  1.505000000000E+002 / Y of reference pixel
CRVAL1  =  2.709000000000E+002 / RA of reference pixel (deg)
CRVAL2  = -2.438000000000E+001 / DEC of reference pixel (deg)
CDELT1  = -5.555555555556E-004 / X pixel size (deg)
CDELT2  =  5.555555555556E-004 / Y pixel size (deg)
CROTA1  =  1.000000000000E+001 / Image twist X axis (deg)
CROTA2  =  1.000000000000E+001 / Image twist Y axis (deg)
CD1_1   = -5.471156315000E-004 / CD matrix to convert (x,y) to (Ra, Dec)
CD1_2   =  9.647180000000E-005 / CD matrix to convert (x,y) to (Ra, Dec)
CD2_1   =  9.647180000000E-005 / CD matrix to convert (x,y) to (Ra, Dec)
CD2_2   =  5.471156315000E-004 / CD matrix to convert (x,y) to (Ra, Dec)
A_ORDER =                    2 / Polynomial order, axis 1
A_0_2   =  1.000000000000E-007 / SIP coefficient
A_1_1   =  0.000000000000E+000 / SIP coefficient
A_2_0   =  0.000000000000E+000 / SIP coefficient
B_ORDER =                    2 / Polynomial order, axis 2
B_0_2   =  0.000000000000E+000 / SIP coefficient
B_1_1   =  0.000000000000E+000 / SIP coefficient
B_2_0   =  1.000000000000E-007 / SIP coefficient
PLTSOLVD=                    T / Astrometric solved by ASTAP v2026.09.19.
END
"""


def synth_wcs(ra=270.9, dec=-24.38, scale=2.0, w=400, h=300):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [w / 2 + 0.5, h / 2 + 0.5]
    wcs.wcs.cd = np.array([[-scale, 0], [0, scale]]) / 3600
    return wcs


def test_parse_wcs_text():
    w = astrometry.parse_wcs_text(WCS_TEXT)
    assert w.sip is not None
    assert abs(astrometry.pixel_scale_arcsec(w) - 2.0) < 0.01
    ra, dec = w.all_pix2world([[199.1, 149.5]], 0)[0]
    assert abs(ra - 270.9) < 1e-6 and abs(dec + 24.38) < 1e-6
    rot, flip = astrometry.rotation_deg(w)
    assert abs(rot - 10.0) < 0.1 and not flip
    # same header as raw 80-char cards (the .wcs written with -wcs has no newlines)
    raw = "".join(ln.ljust(80) for ln in WCS_TEXT.splitlines())
    assert abs(astrometry.parse_wcs_text(raw.encode()).wcs.crval[0] - 270.9) < 1e-9
    assert "CRVAL1" in astrometry.wcs_to_header(w)


def test_header_hints():
    hdr = {"OBJCTRA": "18 03 36", "OBJCTDEC": "-24 22 48", "FOCALLEN": 400, "XPIXSZ": 3.76}
    ra, dec, fov = astrometry.header_hints(hdr, 3000)
    assert abs(ra - 18.06) < 1e-6 and abs(dec + 24.38) < 1e-6
    assert abs(fov - 3000 * 3.76 / 400 * 206.265 / 3600) < 1e-9


def test_colour_model_monotonic():
    bp_rp = np.linspace(-0.2, 3.0, 30)
    teff = spcc.bp_rp_to_teff(bp_rp)
    assert np.all(np.diff(teff) < 0)
    assert abs(spcc.bp_rp_to_teff(0.82) - 5778) < 150      # the Sun
    for sensor in spcc.SENSORS:
        e = spcc.expected_ratios(bp_rp, sensor)
        rb = e[:, 0] / e[:, 2]
        # colour CFAs leak NIR without an IR cut: R/B flattens for the coolest M dwarfs (BP-RP > 2.5)
        assert np.all(np.diff(rb[bp_rp <= 2.5]) > 0), sensor
    sun = spcc.expected_ratios([0.82], "generic_cmos", "G2V")[0]
    assert np.allclose(sun, 1, atol=0.03)


def _star_field(cast, n=250, w=400, h=300, seed=1):
    rng = np.random.default_rng(seed)
    wcs = synth_wcs(w=w, h=h)
    x, y = rng.uniform(15, w - 15, n), rng.uniform(15, h - 15, n)
    g = rng.uniform(8, 13, n)
    bp_rp = rng.uniform(0.2, 2.2, n)
    ra, dec = wcs.all_pix2world(x, y, 0)
    rgb = spcc.expected_ratios(bp_rp, "generic_cmos", "G2V") * np.asarray(cast)
    amp = 0.4 * 10 ** (-0.4 * (g - 8))
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.array([0.02, 0.015, 0.03]) * np.asarray(cast)
    yy, xx = np.mgrid[0:h, 0:w]
    sig = 1.3
    for i in range(n):
        x0, y0 = int(x[i]), int(y[i])
        sl = (slice(max(y0 - 8, 0), y0 + 9), slice(max(x0 - 8, 0), x0 + 9))
        prof = np.exp(-((xx[sl] - x[i]) ** 2 + (yy[sl] - y[i]) ** 2) / (2 * sig ** 2))
        img[sl] += (amp[i] * prof)[..., None] * rgb[i]
    img += rng.normal(0, 3e-4, img.shape).astype(np.float32)
    cat = {"ra": ra, "dec": dec, "x": x, "y": y, "g": g, "bp_rp": bp_rp}
    return img, wcs, cat


def test_spcc_removes_cast(monkeypatch):
    cast = (1.3, 1.0, 0.7)
    img, wcs, cat = _star_field(cast)
    monkeypatch.setattr(astrometry, "gaia_stars", lambda *a, **k: cat)
    ctx = plugins.Context(log=lambda m: None)
    ctx.layers["wcs"] = wcs
    pl = plugins.get("color.spcc")
    out = pl.run(img, ctx, **pl.resolve({"white_reference": "G2V"}))
    st = ctx.layers["spcc"]
    k = np.array(st["factors"])
    assert np.allclose(k * cast, 1.0, rtol=0.03), k
    assert st["stars"] > 50
    # background neutral and star colours back to the catalogue ones
    bg = out[5:20, 5:20].reshape(-1, 3).mean(0)
    assert np.ptp(bg) < 2e-3
    flux, ok, _, _ = spcc.photometry(out, cat["x"], cat["y"], 3.0)
    meas = flux[ok] / flux[ok][:, 1:2]
    exp = spcc.expected_ratios(cat["bp_rp"][ok])
    assert np.allclose(np.median(meas / exp, axis=0), 1, atol=0.03)


def test_spcc_without_solution_returns_input(monkeypatch):
    img = np.full((50, 60, 3), 0.1, np.float32)
    ctx = plugins.Context(log=lambda m: None, tools={"astap": "C:/nonexistent/astap_cli.exe"})
    pl = plugins.get("color.spcc")
    assert pl.run(img, ctx, **pl.resolve({})) is img
    pl = plugins.get("analysis.plate_solve")
    assert pl.run(img, ctx, **pl.resolve({})) is img and "wcs" not in ctx.layers
