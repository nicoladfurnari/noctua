"""PhotometricColorCalibration (color.pcc) against APASS: synthetic fields + one live VizieR query."""
import numpy as np
import pytest
from astropy.wcs import WCS

from astroforge import astrometry, plugins, spectra
from astroforge.plugins import spcc


def synth_wcs(ra=270.9, dec=-24.38, scale=2.0, w=400, h=300):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(w + 1) / 2, (h + 1) / 2]
    wcs.wcs.cdelt = [-scale / 3600, scale / 3600]
    return wcs


def _band(c, s):
    return np.exp(-0.5 * ((spectra.WL - c) / s) ** 2)


# approximate Johnson B, V and SDSS g', r' passbands (Gaussians) to give the model stars catalogue indices
BANDS = {"B": _band(440, 40), "V": _band(550, 37), "g": _band(475, 55), "r": _band(620, 55)}
SUN_BV = spcc.PCC_WHITE_BV["g2v"]


def _mags(sp):
    """Magnitudes in B, V, g, r with zero points set so the G2V model has B-V 0.653 and g'-r' from Jester."""
    sun = spectra.star_spectrum(spectra.SUN_TEFF)
    out = {}
    for k, band in BANDS.items():
        out[k] = -2.5 * np.log10(sp @ band) + 2.5 * np.log10(sun @ band)[0]
    out["B"] += SUN_BV
    out["g"] += float(spcc.bv_to_gr(SUN_BV))
    return out


def _field(cast, n=260, w=400, h=300, seed=3, outliers=0.0):
    """Black-body-like stars through the generic sensor, colour cast applied; APASS-like catalogue."""
    rng = np.random.default_rng(seed)
    wcs = synth_wcs(w=w, h=h)
    x, y = rng.uniform(15, w - 15, n), rng.uniform(15, h - 15, n)
    vmag = rng.uniform(9, 13.5, n)
    teff = spectra.bp_rp_to_teff(rng.uniform(0.3, 1.8, n))
    sp = spectra.star_spectrum(teff)
    resp = spectra.response("generic_cmos")
    sun = spectra.fluxes(spectra.star_spectrum(spectra.SUN_TEFF), resp)[0]
    rgb = spectra.fluxes(sp, resp) / sun
    rgb = rgb / rgb[:, 1:2] * np.asarray(cast)
    m = _mags(sp)
    bv, gr = m["B"] - m["V"], m["g"] - m["r"]
    bad = rng.random(n) < outliers
    bv[bad] += rng.choice([-1, 1], bad.sum()) * rng.uniform(0.4, 1.0, bad.sum())
    gr[bad] += rng.choice([-1, 1], bad.sum()) * rng.uniform(0.4, 1.0, bad.sum())
    amp = 0.4 * 10 ** (-0.4 * (vmag - 9))
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.array([0.02, 0.015, 0.03]) * np.asarray(cast)
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        x0, y0 = int(x[i]), int(y[i])
        sl = (slice(max(y0 - 8, 0), y0 + 9), slice(max(x0 - 8, 0), x0 + 9))
        prof = np.exp(-((xx[sl] - x[i]) ** 2 + (yy[sl] - y[i]) ** 2) / (2 * 1.3 ** 2))
        img[sl] += (amp[i] * prof)[..., None] * rgb[i]
    img += rng.normal(0, 3e-4, img.shape).astype(np.float32)
    ra, dec = wcs.all_pix2world(x, y, 0)
    r = vmag - 0.3
    nan = np.full(n, np.nan)
    cat = {"ra": ra, "dec": dec, "x": x, "y": y, "V": vmag, "B": vmag + bv, "r": r, "g": r + gr, "i": nan,
           "e_V": nan, "e_B": nan, "e_g": nan, "e_r": nan, "e_i": nan}
    # a quarter of the stars without g'/r' (as in APASS): the plugin derives g'-r' from B-V
    cat["g"][::4] = np.nan
    return img, wcs, cat


def _run(img, wcs, cat, monkeypatch, out_dir=None, **params):
    monkeypatch.setattr(astrometry, "apass_stars", lambda *a, **k: cat)
    ctx = plugins.Context(log=lambda m: None, out_dir=out_dir)
    ctx.layers["wcs"] = wcs
    pl = plugins.get("color.pcc")
    return pl.run(img, ctx, **pl.resolve(params)), ctx


def test_pcc_white_references():
    assert spcc.pcc_white("G2V") == pytest.approx((0.653, 1.05 * 0.653 - 0.23))
    assert spcc.pcc_white("average_spiral")[0] == pytest.approx(0.69)
    assert spcc.pcc_white("elliptical")[0] > spcc.pcc_white("average_spiral")[0]
    assert spcc.pcc_white("0.5") == pytest.approx((0.5, 1.05 * 0.5 - 0.23))
    assert spcc.pcc_white(0.8)[0] == 0.8
    with pytest.raises(KeyError):
        spcc.pcc_white("no_such_reference")


def test_robust_line_outliers():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1.5, 300)
    y = 0.2 - 0.9 * x + rng.normal(0, 0.01, 300)
    y[:60] += rng.uniform(0.3, 1.0, 60)
    a, b, keep, sc = spcc.robust_line(x, y)
    assert a == pytest.approx(0.2, abs=0.01) and b == pytest.approx(-0.9, abs=0.02)
    assert not keep[:60].any() and keep[60:].mean() > 0.95 and sc < 0.02


@pytest.mark.parametrize("outliers", [0.0, 0.2])
def test_pcc_removes_cast(monkeypatch, tmp_path, outliers):
    cast = (1.3, 1.0, 0.7)
    img, wcs, cat = _field(cast, outliers=outliers)
    out, ctx = _run(img, wcs, cat, monkeypatch, out_dir=tmp_path, white_reference="G2V")
    st = ctx.layers["pcc"]
    k = np.array(st["factors"])
    assert np.allclose(k * cast, 1.0, rtol=0.02), k * np.asarray(cast)
    assert st["stars"] > 40 and st["gr_from_bv"] > 20
    assert st["stars"] >= (0.55 if outliers else 0.8) * st["measured"]
    assert st["fit_b"]["slope"] < 0 < st["fit_r"]["slope"]      # redder stars: lower B/G, higher R/G
    assert "APASS" in st["catalog_name"]
    assert (tmp_path / "pcc_graph.png").stat().st_size > 1000
    # background neutral
    bg = out[5:20, 5:20].reshape(-1, 3).mean(0)
    assert np.ptp(bg) < 2e-3


def test_pcc_white_reference_shifts_balance(monkeypatch):
    cast = (1.0, 1.0, 1.0)
    img, wcs, cat = _field(cast)
    _, c1 = _run(img, wcs, cat, monkeypatch, white_reference="G2V", r_index="B-V")
    _, c2 = _run(img, wcs, cat, monkeypatch, white_reference="0.653", r_index="B-V")
    _, c3 = _run(img, wcs, cat, monkeypatch, white_reference="elliptical")
    k1, k2, k3 = (np.array(c.layers["pcc"]["factors"]) for c in (c1, c2, c3))
    assert np.allclose(k1, 1, rtol=0.02) and np.allclose(k1, k2)
    # a redder white: red is turned down and blue up
    assert k3[0] < k1[0] and k3[2] > k1[2]


def test_pcc_fallbacks(monkeypatch):
    img = np.full((50, 60, 3), 0.1, np.float32)
    pl = plugins.get("color.pcc")
    ctx = plugins.Context(log=lambda m: None, tools={"astap": "C:/nonexistent/astap_cli.exe"})
    assert pl.run(img, ctx, **pl.resolve({})) is img and "pcc" not in ctx.layers
    monkeypatch.setattr(astrometry, "apass_stars", lambda *a, **k: None)     # offline
    ctx = plugins.Context(log=lambda m: None)
    ctx.layers["wcs"] = synth_wcs(w=60, h=50)
    assert pl.run(img, ctx, **pl.resolve({})) is img and "pcc" not in ctx.layers
    assert pl.run(img[..., :1], ctx, **pl.resolve({})).shape[-1] == 1
    assert pl.run(img, ctx, **pl.resolve({"white_reference": "bogus"})) is img


def test_apass_live_m8(tmp_path):
    """Live VizieR APASS DR9 cone search on a small M8 field (skipped when offline)."""
    wcs = synth_wcs()
    logs = []
    cat = astrometry.apass_stars(wcs, (300, 400), mag_limit=15, cache_dir=tmp_path, timeout=30, log=logs.append)
    if cat is None:
        pytest.skip(f"VizieR not reachable: {logs}")
    assert len(cat["x"]) >= 20
    for k in ("x", "y", "V", "B", "g", "r", "i"):
        assert len(cat[k]) == len(cat["x"])
    assert ((cat["x"] >= 0) & (cat["x"] <= 399) & (cat["y"] >= 0) & (cat["y"] <= 299)).all()
    v = cat["V"][np.isfinite(cat["V"])]
    assert len(v) > 0.8 * len(cat["x"]) and v.max() < 15
    bv = (cat["B"] - cat["V"])[np.isfinite(cat["B"] - cat["V"])]
    assert -0.5 < np.median(bv) < 2.0
    assert list(tmp_path.glob("apass_*.npz"))      # cached
    again = astrometry.apass_stars(wcs, (300, 400), mag_limit=15, cache_dir=tmp_path, log=logs.append)
    assert np.array_equal(again["x"], cat["x"])
