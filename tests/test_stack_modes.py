"""End-to-end stacking modes on synthetic FITS: local normalization, dual-band OSC, comet, dark optimization,
mosaic. Small frames so the whole file runs in well under a minute on a CPU."""
import numpy as np
import pytest
from astropy.io import fits

from astroforge import io
from astroforge.stack.project import Runner

H, W = 320, 400
RNG = np.random.default_rng(11)
SKY_W, SKY_H = 700, 420
STARS = [(RNG.uniform(10, SKY_W - 10), RNG.uniform(10, SKY_H - 10), RNG.uniform(0.05, 0.6)) for _ in range(260)]


def _write(path, data, **hdr):
    h = fits.Header()
    h.update({"INSTRUME": "TESTCAM", "GAIN": 100, "OFFSET": 30, "XBINNING": 1, "SET-TEMP": -10.0, **hdr})
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU((np.clip(data, 0, 1) * 65535).astype(np.uint16), h).writeto(path, overwrite=True)


def _sky(x0, y0, dx=0.0, dy=0.0, h=H, w=W, fwhm=2.4):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w), np.float32)
    s2 = 2 * (fwhm / 2.355) ** 2
    for x, y, a in STARS:
        x, y = x - x0 + dx, y - y0 + dy
        if -10 < x < w + 10 and -10 < y < h + 10:
            img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / s2)
    return img


def _cal(root):
    for k in range(4):
        _write(root / "bias" / f"B_{k}.fits", 0.01 + RNG.normal(0, 0.001, (H, W)), IMAGETYP="Bias Frame", EXPTIME=0.0)


def _runner(root, tmp_path, **kw):
    return Runner({"lights": [str(root / "lights")], "calibration": [str(root / "bias")],
                   "work_dir": str(tmp_path / "work"), "auto_reject": False, "cosmetic": False, **kw},
                  log=lambda m: None)


def test_local_normalization_removes_changing_gradients(tmp_path):
    root = tmp_path / "ln"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for k in range(8):
        # every frame but one has a strong gradient in a different direction (Moon moving, light pollution)
        ang = k * np.pi / 16          # the Moon moves slowly: similar directions, they do not cancel out
        amp = 0.0 if k == 3 else 0.04
        grad = amp * ((xx - W / 2) * np.cos(ang) + (yy - H / 2) * np.sin(ang)) / W
        img = 0.05 + grad + _sky(100, 50, k * 1.3, -k * 0.9) + RNG.normal(0, 0.002, (H, W))
        _write(root / "lights" / f"L_{k}.fits", img + 0.01, IMAGETYP="Light Frame", EXPTIME=60.0)
    _cal(root)
    res = {}
    for ln in (False, True):
        img, rep = _runner(root, tmp_path / f"w{ln}", local_norm=ln, local_norm_scale=128, coverage_crop=0,
                           rejection="winsorized").run()
        bg = img[..., 0][16:-16, 16:-16]
        tiles = bg.reshape(bg.shape[0] // 16, 16, bg.shape[1] // 16, 16).min(axis=(1, 3))   # star-free minima
        res[ln] = (float(np.percentile(tiles, 95) - np.percentile(tiles, 5)), rep["rejected_pct"])
    # frames brought onto the gradient-free one: flat background and far fewer wrongly rejected pixels
    assert res[True][0] < 0.5 * 0.04 * 0.5, res
    assert res[True][1] < res[False][1], res


def test_dualband_osc_gives_ha_and_oiii(tmp_path):
    root = tmp_path / "db"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ha_neb = 0.08 * np.exp(-(((xx - 120) / 40) ** 2 + ((yy - 160) / 40) ** 2))
    o_neb = 0.08 * np.exp(-(((xx - 290) / 40) ** 2 + ((yy - 160) / 40) ** 2))
    for k in range(5):
        st = _sky(100, 50, k * 1.4, k * 0.8)
        r = 0.03 + ha_neb + st
        g = 0.03 + o_neb + st
        b = 0.03 + o_neb + st
        cfa = np.empty((H, W), np.float32)
        cfa[0::2, 0::2], cfa[0::2, 1::2], cfa[1::2, 0::2], cfa[1::2, 1::2] = r[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], b[1::2, 1::2]
        _write(root / "lights" / f"L_{k}.fits", cfa + RNG.normal(0, 0.002, cfa.shape) + 0.01,
               IMAGETYP="Light Frame", EXPTIME=120.0, BAYERPAT="RGGB")
    _cal(root)
    _, rep = _runner(root, tmp_path, dualband=True, coverage_crop=0).run()
    ha, _ = io.load(rep["dualband"]["Ha"])
    o, _ = io.load(rep["dualband"]["OIII"])
    a, b = (slice(150, 170), slice(110, 130)), (slice(150, 170), slice(280, 300))
    assert np.median(ha[a]) - np.median(ha[b]) > 0.04
    assert np.median(o[b]) - np.median(o[a]) > 0.04


def test_comet_stack_keeps_comet_sharp(tmp_path):
    root = tmp_path / "comet"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    n = 8
    start, step = np.array([150.0, 140.0]), np.array([4.0, 2.0])     # comet moves 4,2 px per frame on the sky
    for k in range(n):
        dx, dy = k * 1.1, -k * 0.7                                    # telescope dither
        cx, cy = start + k * step + [dx, dy]
        comet = 0.5 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))
        img = 0.04 + _sky(100, 50, dx, dy) + comet + RNG.normal(0, 0.002, (H, W))
        _write(root / "lights" / f"L_{k}.fits", img + 0.01, IMAGETYP="Light Frame", EXPTIME=60.0,
               **{"DATE-OBS": f"2026-10-01T22:{k:02d}:00"})
    _cal(root)
    # the reference frame is chosen by quality: give the positions in its coordinates via the first frame
    r = _runner(root, tmp_path, comet=True, comet_first_xy=list(start), comet_last_xy=list(start + (n - 1) * step),
                coverage_crop=0, rejection="sigma")
    img, rep = r.run()
    pos = np.asarray(rep["comet"]["positions"])
    assert len(pos) == n
    cstack, _ = io.load(rep["comet"]["comet_stack"])
    peak = float(cstack[..., 0].max())
    assert peak > 0.3      # 8 frames smeared along a 28 px path would peak near 0.5/4


def test_dark_optimization_reports_scale(tmp_path):
    root = tmp_path / "dark"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hot = (RNG.random((H, W)) < 0.002) * 0.05
    glow = 0.02 * np.exp(-((xx - W) ** 2 + (yy) ** 2) / (2 * 60 ** 2))
    dark_current = 0.004 + hot + glow
    for k in range(5):
        _write(root / "lights" / f"L_{k}.fits", 0.01 + 0.03 + dark_current + _sky(100, 50, k, k) + RNG.normal(0, 0.002, (H, W)),
               IMAGETYP="Light Frame", EXPTIME=300.0)
        _write(root / "darks" / f"D_{k}.fits", 0.01 + dark_current + RNG.normal(0, 0.001, (H, W)),
               IMAGETYP="Dark Frame", EXPTIME=300.0)
    _cal(root)
    r = Runner({"lights": [str(root / "lights")], "calibration": [str(root / "bias"), str(root / "darks")],
                "work_dir": str(tmp_path / "work"), "auto_reject": False, "cosmetic": False, "dark_optimize": True},
               log=lambda m: None)
    _, rep = r.run()
    assert abs(rep["dark_optimize"]["k"] - 1.0) < 0.1


def test_exposure_tolerance_uses_each_exposures_dark(tmp_path):
    root = tmp_path / "exp"
    hot = (np.random.default_rng(3).random((H, W)) < 0.003) * 0.08      # fixed pattern: grows with exposure
    for exp in (120.0, 300.0):
        for k in range(3):
            light = 0.01 + 0.0001 * exp + hot * exp / 300 + _sky(100, 50, k, -k) * exp / 300 + RNG.normal(0, 0.002, (H, W))
            _write(root / "lights" / f"L_{exp:g}_{k}.fits", light, IMAGETYP="Light Frame", EXPTIME=exp)
            _write(root / "darks" / f"D_{exp:g}_{k}.fits", 0.01 + hot * exp / 300 + RNG.normal(0, 0.001, (H, W)),
                   IMAGETYP="Dark Frame", EXPTIME=exp)
    _cal(root)
    r = Runner({"lights": [str(root / "lights")], "calibration": [str(root / "bias"), str(root / "darks")],
                "work_dir": str(tmp_path / "work"), "auto_reject": False, "cosmetic": False, "coverage_crop": 0,
                "exposure_tolerance": 200}, log=lambda m: None)
    img, rep = r.run()
    assert rep["lights_total"] == 6 and set(rep["darks"]) == {"120", "300"}
    assert all(abs(d["k"] - 1) < 0.01 and not d["optimized"] for d in rep["darks"].values())
    assert {e: d["dark_exptime"] for e, d in rep["darks"].items()} == {"120": 120.0, "300": 300.0}
    sky = img[..., 0]
    dev = np.abs(sky[(hot > 0) & (sky > 0)] - np.median(sky))     # covered hot pixels (0 = outside coverage)
    assert float(np.median(dev)) < 0.002          # without the right dark per exposure they stay at ~0.03-0.08


def test_subframe_weighting_formula_and_approval(tmp_path):
    from astroforge.stack import analyze, subframes
    qs = [analyze.Quality(stars=100, fwhm=f, eccentricity=e, background=0.05, noise=0.002, snr=s, psf_signal=p)
          for f, e, s, p in ((3.8, 0.55, 1.54, 50), (5.0, 0.7, 1.45, 40), (7.0, 0.89, 1.37, 20))]
    vals = subframes.apply(qs, "nebula", approval=70)
    assert vals[0] == pytest.approx(100) and vals[2] == pytest.approx(50)
    assert qs[0].accepted and not qs[2].accepted and "weight" in qs[2].reason
    assert qs[0].weight == pytest.approx(1.0)
    with pytest.raises(ValueError):
        subframes.evaluate("__import__('os')", {})


def test_mosaic_two_panels(tmp_path):
    root = tmp_path / "mosaic"
    panels = []
    for p, x0 in enumerate((0, 280)):           # 120 px overlap
        for k in range(4):
            img = 0.04 + 0.01 * p + _sky(x0, 50, k * 1.2, -k * 0.6) + RNG.normal(0, 0.002, (H, W))
            _write(root / f"panel{p}" / f"L_{k}.fits", img + 0.01, IMAGETYP="Light Frame", EXPTIME=60.0)
        panels.append([str(root / f"panel{p}")])
    _cal(root)
    r = Runner({"lights": [], "calibration": [str(root / "bias")], "work_dir": str(tmp_path / "work"),
                "auto_reject": False, "cosmetic": False}, log=lambda m: None)
    img, rep = r.run_mosaic(panels)
    assert img.shape[1] > W + 200 and abs(img.shape[0] - H) < 40
    assert len(rep["panels"]) == 2
    assert [r["name"] for r in rep["runs"]] == ["Pannello 1", "Pannello 2"] and all(r["frames"] for r in rep["runs"])


def test_satellite_trail_removed_in_1x(tmp_path):
    root = tmp_path / "sat"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    line = np.abs((yy - 40) - 0.55 * xx) < 1.5          # a straight trail across the frame
    rng = np.random.default_rng(7)
    for k in range(6):
        # no dither: the trail mask below is in frame coordinates = reference coordinates
        img = 0.05 + _sky(100, 50) + rng.normal(0, 0.003, (H, W))
        if k == 2:
            img = img + 0.006 * line                     # faint: 2 sigma per pixel
        _write(root / "lights" / f"L_{k}.fits", img + 0.01, IMAGETYP="Light Frame", EXPTIME=60.0)
    _cal(root)
    res = {}
    for sat in (False, True):
        # plain mean, no rejection: only the trail detection can remove the trail
        img, rep = _runner(root, tmp_path / f"w{sat}", satellites=sat, coverage_crop=0, rejection="none",
                           reject_grow=0).run()
        sky = img[..., 0]
        on = np.median(sky[line & (xx > 40) & (xx < W - 40)])
        off = np.median(sky[~line])
        res[sat] = float(on - off)
    assert res[True] < 0.35 * res[False], res
    assert rep["trails"]


def test_drizzle_with_local_norm_and_kernel(tmp_path):
    root = tmp_path / "dz"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for k in range(6):
        grad = (0.0 if k == 0 else 0.03) * xx / W
        img = 0.05 + grad + _sky(100, 50, k * 0.6, -k * 0.4) + RNG.normal(0, 0.002, (H, W))
        _write(root / "lights" / f"L_{k}.fits", img + 0.01, IMAGETYP="Light Frame", EXPTIME=60.0)
    _cal(root)
    img, rep = _runner(root, tmp_path, drizzle=2, local_norm=True, local_norm_scale=128, drizzle_kernel="gaussian",
                       coverage_crop=0).run()
    assert img.shape[0] == 2 * H and np.isfinite(img).all()


def test_master_library_reuse_and_defective_flat(tmp_path):
    lib = tmp_path / "lib"
    root = tmp_path / "s1"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = 1 - 0.3 * ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W / 2) ** 2
    for k in range(3):
        _write(root / "lights" / f"L_{k}.fits", (0.05 + _sky(100, 50, k, -k)) * vign + 0.01 + RNG.normal(0, 0.002, (H, W)),
               IMAGETYP="Light Frame", EXPTIME=60.0, FILTER="L")
        _write(root / "flats" / f"F_{k}.fits", 0.5 * vign + 0.01 + RNG.normal(0, 0.001, (H, W)),
               IMAGETYP="Flat Field", EXPTIME=1.0, FILTER="L")
    _cal(root)
    base = {"work_dir": str(tmp_path / "w1"), "auto_reject": False, "cosmetic": False, "master_library": str(lib)}
    Runner({**base, "lights": [str(root / "lights")], "calibration": [str(root / "bias"), str(root / "flats")]},
           log=lambda m: None).run()
    # second session, same camera and filter, no calibration frames at all
    root2 = tmp_path / "s2"
    for k in range(3):
        _write(root2 / "lights" / f"L_{k}.fits", (0.05 + _sky(100, 50, k, k)) * vign + 0.01 + RNG.normal(0, 0.002, (H, W)),
               IMAGETYP="Light Frame", EXPTIME=60.0, FILTER="L")
    r = Runner({**base, "work_dir": str(tmp_path / "w2"), "lights": [str(root2 / "lights")]}, log=lambda m: None)
    img, rep = r.run()
    assert set(rep["library_masters"]) == {"bias", "flat"}
    sky = img[..., 0][20:-20, 20:-20]
    corner, centre = float(np.median(sky[:30, :30])), float(np.median(sky[H // 2 - 30:H // 2, W // 2 - 30:W // 2]))
    assert abs(corner - centre) < 0.1 * centre          # vignetting corrected by the library flat
    # a defective master (like the QHY168C flat at ~37 ADU) in an external #MASTERS folder is not used
    ext = tmp_path / "MASTERS"
    ext.mkdir()
    h = fits.Header(); h.update(IMAGETYP="Master Flat", EXPTIME=1.0, FILTER="R", INSTRUME="TESTCAM", NCOMBINE=20)
    fits.PrimaryHDU(np.full((H, W), 37, np.uint16), h).writeto(ext / "MasterFlat_R.fits")
    from astroforge.stack import masterlib
    L = masterlib.MasterLibrary(tmp_path / "lib2", [ext], log=lambda m: None)
    assert L.find("flat", {"camera": "TESTCAM", "width": W, "height": H, "filter": "R"}) is None


def test_osc_shifted_bayer_header_with_cfa_normalised_flats(tmp_path):
    # QHY168C + APT: data are RGGB, the header says GRBG. With CFA-normalised master flats the pattern must be
    # checked on a raw frame (on the flat every site is ~1 and the wrong pattern was kept: green nebula).
    # Also covers the frame measures: stars fitted on the superpixel image with superpixel coordinates.
    root = tmp_path / "osc"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    neb = 0.06 * np.exp(-(((xx - 200) / 50) ** 2 + ((yy - 160) / 50) ** 2))
    for k in range(6):
        st = _sky(100, 50, k * 1.3, -k * 0.9, fwhm=4.0)
        r, g, b = 0.03 + neb + st, 0.03 + 0.15 * neb + st, 0.03 + 0.1 * neb + st
        cfa = np.empty((H, W), np.float32)
        cfa[0::2, 0::2], cfa[0::2, 1::2], cfa[1::2, 0::2], cfa[1::2, 1::2] = r[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], b[1::2, 1::2]
        _write(root / "lights" / f"L_{k}.fits", cfa + RNG.normal(0, 0.001, cfa.shape) + 0.01,
               IMAGETYP="Light Frame", EXPTIME=120.0, BAYERPAT="GRBG")
    panel = np.empty((H, W), np.float32)                     # green-heavy flat panel, as in the real flats
    panel[0::2, 0::2], panel[0::2, 1::2], panel[1::2, 0::2], panel[1::2, 1::2] = 0.30, 0.47, 0.47, 0.36
    for k in range(4):
        _write(root / "flats" / f"F_{k}.fits", 0.01 + panel + RNG.normal(0, 0.002, (H, W)),
               IMAGETYP="Flat Frame", EXPTIME=2.0)
    _cal(root)
    r = Runner({"lights": [str(root / "lights")], "calibration": [str(root / "bias"), str(root / "flats")],
                "work_dir": str(tmp_path / "work"), "auto_reject": True, "cosmetic": False, "coverage_crop": 0},
               log=lambda m: None)
    img, rep = r.run()
    assert rep["bayer"] == "RGGB"
    assert all(f["accepted"] for f in rep["frames"]), [(f["fwhm"], f["eccentricity"]) for f in rep["frames"]]
    assert max(f["eccentricity"] for f in rep["frames"]) < 0.45
    c = img[150:170, 190:210]
    assert float(np.median(c[..., 0] - c[..., 1])) > 0.02      # the nebula is red, not green


@pytest.mark.parametrize("raw,canon", [("ha", "Ha"), ("Ha 7nm", "Ha"), ("H-Alpha_7nm", "Ha"), ("Halpha 6.5 nm", "Ha"),
                                       ("Ha 3,5nm", "Ha"), ("OIII 6.5nm", "OIII"), ("O3", "OIII"),
                                       ("SII-3nm (Antlia)", "SII"), ("S2 7nm", "SII"), ("Lum", "L"),
                                       ("L-eNhance", "L-eNhance")])
def test_filter_names_with_bandwidth(raw, canon):
    # capture programs write the bandwidth in FILTER: 7 nm narrowband sets must still group as Ha/OIII/SII
    # (a dual-band filter such as L-eNhance must not become luminance)
    from astroforge.stack.channels import canonical
    assert canonical(raw) == canon
