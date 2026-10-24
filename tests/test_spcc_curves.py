"""Spectral library (astroforge/spectra.py) and SPCC with sensor / filter / QE curves."""
import numpy as np
import pytest
from astropy.wcs import WCS

from astroforge import astrometry, plugins, spectra
from astroforge.plugins import spcc


def _at(curve, nm):
    return float(curve[int(round(nm - spectra.WL[0]))])


def test_curves_normalised_and_in_range():
    assert spectra.WL[0] == 330 and spectra.WL[-1] == 1100
    for name, s in spectra.SENSORS.items():
        for c in "RGB":
            a = s[c]
            assert a.shape == spectra.WL.shape and np.isfinite(a).all(), (name, c)
            assert a.min() >= 0 and 0.5 <= a.max() <= 1.0 + 1e-9, (name, c)
    for name, a in {**spectra.FILTERS, **spectra.QE}.items():
        assert a.min() >= 0 and 0.5 <= a.max() <= 1.0 + 1e-9, name
    # colour channels peak in the right order
    for name in ("sony_color", "panasonic_color", "generic_cmos"):
        r = spectra.response(name, "uv_ir_cut")
        peaks = spectra.WL[np.argmax(r, axis=1)]
        assert peaks[2] < peaks[1] < peaks[0], (name, peaks)
    # narrowband filters centred on their lines
    for name, nm in (("ha_7nm", 656.3), ("oiii_7nm", 500.7), ("sii_7nm", 672.0)):
        a = spectra.FILTERS[name]
        assert _at(a, nm) > 0.8 and _at(a, nm - 8) < 0.05 and _at(a, nm + 8) < 0.05
        assert 5 < spectra.equivalent_width(name) < 9


def test_combine_is_product():
    a = spectra.combine("uv_ir_cut", "l_pro")
    assert np.allclose(a, spectra.FILTERS["uv_ir_cut"] * spectra.FILTERS["l_pro"])
    assert np.allclose(spectra.combine("uv_ir_cut+l_pro"), a)
    assert np.allclose(spectra.combine(), 1) and np.allclose(spectra.combine("none"), 1)
    assert np.allclose(spectra.combine(np.full_like(spectra.WL, 0.5), "ideal"), 0.5)
    with pytest.raises(KeyError):
        spectra.combine("no_such_filter")


def test_response_l_pro_blocks_sodium():
    plain = spectra.response("sony_color")
    lp = spectra.response("sony_color", "l_pro")
    for c in (0, 1):   # R and G see 589 nm
        assert _at(plain[c], 589) > 0.2
        assert _at(lp[c], 589) < 0.02 * _at(plain[c], 589)
    # passes OIII / Hb / Ha, cuts the NIR leak of the colour sensor
    assert _at(lp[2], 500.7) > 0.3 and _at(lp[0], 656.3) > 0.5
    assert _at(plain[0], 850) > 0.2 and _at(lp[0], 850) < 1e-3
    # L-eNhance / L-eXtreme are narrowband, L-Pro and UV/IR cut are not
    assert not spectra.is_narrowband("l_pro") and not spectra.is_narrowband("uv_ir_cut")
    assert spectra.is_narrowband("l_enhance") and spectra.is_narrowband("l_extreme")
    assert not spectra.is_narrowband(None)


def test_mono_and_qe():
    ideal = spectra.response("mono_baader_rgb", qe="ideal")
    native = spectra.response("mono_baader_rgb")      # MN34230 mono QE
    assert np.allclose(native, ideal * spectra.QE["mn34230_mono"])
    assert np.allclose(spectra.response("sony_color", qe="sensor"), spectra.response("sony_color", qe="ideal"))


def test_g2v_white_and_colour_order():
    sun = spectra.fluxes(spectra.star_spectrum(spectra.SUN_TEFF), spectra.response("generic_cmos"))[0]
    ref = spectra.fluxes(spectra.white_reference("G2V"), spectra.response("generic_cmos"))[0]
    assert np.allclose(sun / ref / (sun[1] / ref[1]), 1, atol=1e-9)
    assert np.allclose(spcc.expected_ratios([0.82], "generic_cmos", "G2V")[0], 1, atol=0.03)
    bp_rp = np.linspace(-0.2, 3.2, 40)
    teff = spectra.bp_rp_to_teff(bp_rp)
    assert np.all(np.diff(teff) < 0) and 9000 < spectra.bp_rp_to_teff(0.0) < 10500
    for sensor in spectra.SENSORS:
        for filt in (None, "uv_ir_cut", "l_pro"):
            e = spectra.expected_ratios(bp_rp, sensor, "average_spiral", filt)
            rb = e[:, 0] / e[:, 2]
            # without an IR cut the NIR leak of colour CFAs (B and G open again above ~800 nm) flattens
            # R/B for the coolest M dwarfs: require monotonic up to BP-RP 2.5 there
            n = len(rb) if filt else int(np.searchsorted(bp_rp, 2.5))
            assert np.all(np.diff(rb[:n]) > 0), (sensor, filt)
    # the spiral reference is redder than the Sun: a G2V star looks bluish against it
    g = spectra.expected_ratios([0.82], "sony_color", "average_spiral")[0]
    assert g[0] < 1 and g[0] / g[2] < 0.99


def test_user_curves_override(tmp_path):
    wl = np.arange(300, 1101, 5)
    t = np.where((wl > 450) & (wl < 600), 90.0, 0.0)            # percent
    (tmp_path / "MyFilter.csv").write_text("# test\nwavelength,T\n" + "\n".join(f"{a},{b}" for a, b in zip(wl, t)))
    rgb = np.c_[wl, np.exp(-0.5 * ((wl - 610) / 30) ** 2), np.exp(-0.5 * ((wl - 530) / 30) ** 2),
                np.exp(-0.5 * ((wl - 460) / 30) ** 2)]
    np.savetxt(tmp_path / "mycam.csv", rgb, delimiter=",")
    try:
        spectra.load(user_dir=tmp_path)
        assert "myfilter" in spectra.FILTERS and "mycam" in spectra.SENSORS
        assert abs(_at(spectra.FILTERS["myfilter"], 520) - 0.9) < 1e-6
        e = spectra.expected_ratios([0.3, 1.5], "mycam", "G2V", "myfilter")
        assert e[1, 0] / e[1, 2] > e[0, 0] / e[0, 2]
    finally:
        spectra.load()
    assert "myfilter" not in spectra.FILTERS


def _wcs(w, h, scale=2.0):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [83.8, -5.4]
    wcs.wcs.crpix = [w / 2 + 0.5, h / 2 + 0.5]
    wcs.wcs.cd = np.array([[-scale, 0], [0, scale]]) / 3600
    return wcs


def _field(cast, sensor, white, filt, n=260, w=420, h=320, seed=3):
    rng = np.random.default_rng(seed)
    wcs = _wcs(w, h)
    x, y = rng.uniform(15, w - 15, n), rng.uniform(15, h - 15, n)
    g = rng.uniform(8, 13, n)
    bp_rp = rng.uniform(0.0, 2.6, n)
    ra, dec = wcs.all_pix2world(x, y, 0)
    rgb = spectra.expected_ratios(bp_rp, sensor, white, filt) * np.asarray(cast)
    amp = 0.4 * 10 ** (-0.4 * (g - 8))
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.array([0.03, 0.02, 0.025]) * np.asarray(cast)
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        x0, y0 = int(x[i]), int(y[i])
        sl = (slice(max(y0 - 8, 0), y0 + 9), slice(max(x0 - 8, 0), x0 + 9))
        prof = np.exp(-((xx[sl] - x[i]) ** 2 + (yy[sl] - y[i]) ** 2) / (2 * 1.4 ** 2))
        img[sl] += (amp[i] * prof)[..., None] * rgb[i]
    img += rng.normal(0, 3e-4, img.shape).astype(np.float32)
    return img, wcs, {"ra": ra, "dec": dec, "x": x, "y": y, "g": g, "bp_rp": bp_rp}


def test_spcc_l_pro_recovers_cast(monkeypatch, tmp_path):
    cast = (1.25, 1.0, 0.75)
    img, wcs, cat = _field(cast, "sony_color", "average_spiral", "l_pro")
    monkeypatch.setattr(astrometry, "gaia_stars", lambda *a, **k: cat)
    logs = []
    ctx = plugins.Context(log=logs.append, out_dir=tmp_path)
    ctx.layers["wcs"] = wcs
    pl = plugins.get("color.spcc")
    assert pl.params["white_reference"].default == "average_spiral"
    out = pl.run(img, ctx, **pl.resolve({"sensor": "sony_color", "filter": "l_pro"}))
    st = ctx.layers["spcc"]
    k = np.array(st["factors"])
    assert np.allclose(k * cast, 1.0, rtol=0.02), k
    assert st["stars"] > 40 and st["filter"] == "l_pro" and not st["narrowband"]
    assert np.allclose(np.array(st["slope"])[[0, 2]], 1, atol=0.1)
    assert out.shape == img.shape and out.dtype == np.float32
    g = tmp_path / "spcc_graph.png"
    assert g.exists() and g.stat().st_size > 1000 and st["graph"] == str(g)
    assert any("scatter" in m and "%" in m for m in logs)
    # a wrong filter assumption gives a visibly worse calibration than the right one
    ctx2 = plugins.Context(log=lambda m: None)
    ctx2.layers["wcs"] = wcs
    pl.run(img, ctx2, **pl.resolve({"sensor": "sony_color", "filter": "none", "graph": False}))
    k2 = np.array(ctx2.layers["spcc"]["factors"])
    assert np.abs(np.log(k2 * cast)).max() > np.abs(np.log(k * cast)).max()


def test_spcc_narrowband_warns_and_bad_names(monkeypatch):
    cast = (1.1, 1.0, 0.9)
    img, wcs, cat = _field(cast, "sony_color", "G2V", "l_extreme", n=200)
    monkeypatch.setattr(astrometry, "gaia_stars", lambda *a, **k: cat)
    logs = []
    ctx = plugins.Context(log=logs.append)
    ctx.layers["wcs"] = wcs
    pl = plugins.get("color.spcc")
    pl.run(img, ctx, **pl.resolve({"sensor": "sony_color", "filter": "l_extreme", "white_reference": "G2V"}))
    assert any("narrowband" in m for m in logs)
    assert ctx.layers["spcc"]["narrowband"]
    ctx = plugins.Context(log=lambda m: None)
    ctx.layers["wcs"] = wcs
    assert pl.run(img, ctx, **pl.resolve({"filter": "no_such_filter"})) is img
    assert "spcc" not in ctx.layers


def test_save_graph_fallback(tmp_path):
    rng = np.random.default_rng(0)
    exp = np.c_[np.exp(rng.normal(0, .2, 50)), np.ones(50), np.exp(rng.normal(0, .2, 50))]
    meas = exp / np.array([1.2, 1, 0.8]) * np.exp(rng.normal(0, .02, (50, 3)))
    keep = np.ones(50, bool)
    keep[:5] = False
    p = spcc.save_graph(tmp_path / "g.png", meas, exp, np.array([1.2, 1, .8]), keep, np.array([.02, 0, .02]))
    import cv2
    im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert im is not None and im.shape[0] > 300 and im.shape[1] > 600
