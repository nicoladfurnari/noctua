"""Stacking workflow on small synthetic FITS sets (mono + filters, calibration selection)."""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from astroforge.stack import calibrate, frames
from astroforge.stack.project import Runner

H, W = 320, 400
RNG = np.random.default_rng(3)
STARS = [(RNG.uniform(20, W - 20), RNG.uniform(20, H - 20), RNG.uniform(0.05, 0.5)) for _ in range(80)]


def _write(path, data, **hdr):
    h = fits.Header()
    h.update({"INSTRUME": "QHY163M", "GAIN": 100, "OFFSET": 30, "XBINNING": 1, "SET-TEMP": -15.0, **hdr})
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU((np.clip(data, 0, 1) * 65535).astype(np.uint16), h).writeto(path, overwrite=True)


def _light(dx, dy, exp, level):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.full((H, W), 0.02 + 0.00005 * exp, np.float32)
    for x, y, a in STARS:
        img += a * level * np.exp(-((xx - x - dx) ** 2 + (yy - y - dy) ** 2) / (2 * 1.5 ** 2))
    return img + RNG.normal(0, 0.003, img.shape)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("sky")
    vignette = 1 - 0.2 * ((np.mgrid[0:H, 0:W][1] - W / 2) / W) ** 2
    n = 0
    for filt, exp in (("Ha", 300.0), ("Ha", 180.0), ("OIII", 300.0)):
        for k in range(3):
            n += 1
            # measured temperature wobbles around the -15 setpoint: must stay one group
            _write(root / "lights" / f"L_{filt}_{exp:g}_{k}.fits",
                   _light(k * 1.7, k * -1.1, exp, exp / 300) * vignette + 0.01,
                   IMAGETYP="Light Frame", EXPTIME=exp, FILTER=filt, **{"CCD-TEMP": -15.0 + (0.1 if k % 2 else -0.1)})
    for exp in (300.0, 180.0):
        for k in range(3):
            _write(root / "darks" / f"D_{exp:g}_{k}.fits", 0.01 + RNG.normal(0, 0.002, (H, W)),
                   IMAGETYP="Dark Frame", EXPTIME=exp)
    for k in range(3):
        _write(root / "bias" / f"B_{k}.fits", 0.01 + RNG.normal(0, 0.002, (H, W)), IMAGETYP="Bias Frame", EXPTIME=0.0)
    for filt, exp in (("Ha", 1.0), ("OIII", 3.0)):
        for k in range(3):
            _write(root / "flats" / f"F_{filt}_{k}.fits", 0.5 * vignette + 0.01 + RNG.normal(0, 0.002, (H, W)),
                   IMAGETYP="Flat Field", EXPTIME=exp, FILTER=filt)
            _write(root / "darkflats" / f"DF_{exp:g}_{k}.fits", 0.01 + 0.001 * exp + RNG.normal(0, 0.002, (H, W)),
                   IMAGETYP="Dark Flat", EXPTIME=exp)
    return root


def _runner(root, tmp_path, **kw):
    return Runner({"lights": [str(root / "lights")],
                   "calibration": [str(root / d) for d in ("darks", "bias", "flats", "darkflats")],
                   "work_dir": str(tmp_path / "work"), "auto_reject": False, "cosmetic": False, **kw},
                  log=lambda m: None)


def test_setpoint_keeps_one_group(dataset, tmp_path):
    sc = _runner(dataset, tmp_path).scan()
    assert sorted(len(fs) for _, fs in sc["light_groups"]) == [3, 3, 3]


def test_darkflat_matches_flat_exposure(dataset, tmp_path):
    r = _runner(dataset, tmp_path)
    sc = r.scan()
    for key, _ in sc["light_groups"]:
        picked = r._pick(sc["groups"], "darkflat", key, 3.0 if key[8] == "OIII" else 1.0)
        assert {f.exptime for f in picked} == {3.0 if key[8] == "OIII" else 1.0}


def test_flat_cache_depends_on_bias_and_darkflat(tmp_path):
    r = Runner({"work_dir": str(tmp_path)}, log=lambda m: None)
    paths = ["a.fits", "b.fits"]
    assert r._master_path("flat", paths, ("bias1", None)) != r._master_path("flat", paths, ("bias1", "df1"))
    assert r._master_path("bias", paths) == r._master_path("bias", list(reversed(paths)))


def test_pick_ties_with_none_keys(tmp_path):
    r = Runner({"work_dir": str(tmp_path)}, log=lambda m: None)
    f = frames.Frame(path="x", type="bias")
    key = ("cam", 10, 10, 1, None, None, None, 1.0, "")
    groups = {"bias": {("cam", 10, 10, 1, None, None, None): [f], ("cam", 10, 10, 1, None, None, -10): [f]}}
    assert r._pick(groups, "bias", key) == [f]


def test_all_excluded_is_a_clear_error(dataset, tmp_path):
    r = _runner(dataset, tmp_path)
    lights = [str(p) for p in (dataset / "lights").glob("*.fits")]
    r.s["exclude"] = lights
    with pytest.raises(ValueError, match="exclusions"):
        r.run()


def test_debayer_keeps_negative_background():
    cfa = np.random.default_rng(0).normal(0, 0.01, (64, 64)).astype(np.float32)
    assert abs(float(calibrate.debayer(cfa, "RGGB").mean())) < 0.002


def test_looks_like_cfa():
    flat = np.full((64, 64), 0.5, np.float32)
    assert not calibrate.looks_like_cfa(flat)
    flat[0::2, 1::2] = flat[1::2, 0::2] = 0.8          # greens brighter, as on a colour flat
    assert calibrate.looks_like_cfa(flat)


def test_run_filters_merges_exposures(dataset, tmp_path):
    r = _runner(dataset, tmp_path, drizzle=1)
    rgb, rep = r.run_filters("HOO")
    work = tmp_path / "work"
    assert (work / "stack_Ha_300s_integration.fits").exists()
    assert (work / "stack_Ha_180s_integration.fits").exists()
    assert (work / "stack_OIII_integration.fits").exists()
    assert rep["filters"]["Ha"].endswith("stack_Ha_merged.fits") and Path(rep["filters"]["Ha"]).exists()
    assert rgb.shape[-1] == 3 and np.isfinite(rgb).all()
    assert len(list((work / "masters").glob("master_darkflat_*.xisf"))) == 2


def test_flat_per_session(tmp_path):
    """Two nights, dust in different places: each night must be divided by its own flat."""
    root = tmp_path / "sky2"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dust = {"2026-09-01": (100, 120), "2026-09-05": (300, 220)}
    for night, (dx, dy) in dust.items():
        spot = 1 - 0.3 * np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * 12 ** 2))
        for k in range(3):
            date = f"{night}T23:{10 + k:02d}:00"
            _write(root / "lights" / f"L_{night}_{k}.fits", _light(k * 1.3, k * 0.7, 300.0, 1.0) * spot + 0.01,
                   IMAGETYP="Light Frame", EXPTIME=300.0, FILTER="L", **{"DATE-OBS": date})
            _write(root / "flats" / f"F_{night}_{k}.fits", 0.5 * spot + 0.01 + RNG.normal(0, 0.001, (H, W)),
                   IMAGETYP="Flat Field", EXPTIME=1.0, FILTER="L", **{"DATE-OBS": f"{night}T19:0{k}:00"})
    for k in range(3):
        _write(root / "bias" / f"B_{k}.fits", 0.01 + RNG.normal(0, 0.001, (H, W)), IMAGETYP="Bias Frame", EXPTIME=0.0)
    r = Runner({"lights": [str(root / "lights")], "calibration": [str(root / "flats"), str(root / "bias")],
                "work_dir": str(tmp_path / "work"), "auto_reject": False, "cosmetic": False,
                "coverage_crop": 0}, log=lambda m: None)
    r.run()
    # check every calibrated frame (in the integration the errors of the two nights would cancel out)
    for cal in sorted((tmp_path / "work" / "calibrated" / "stack").glob("*.npy")):
        a = np.load(cal)
        sky = float(np.median(a))
        for dx, dy in dust.values():
            patch = float(np.median(a[dy - 4:dy + 4, dx - 4:dx + 4]))
            assert abs(patch - sky) < 0.05 * sky, (cal.name, dx, dy, patch, sky)   # 16% without per-session flats
    assert len(list((tmp_path / "work" / "masters").glob("master_flat_*.xisf"))) == 2


def test_run_filters_reports_every_run(dataset, tmp_path):
    rgb, rep = _runner(dataset, tmp_path, drizzle=1).run_filters("HOO")
    names = [r["name"] for r in rep["runs"]]
    assert sorted(names) == ["Ha_180s", "Ha_300s", "OIII"]
    assert all(len(r["frames"]) == 3 and r["frames"][0]["calibrated"].endswith(".npy") for r in rep["runs"])
    # each run keeps its own calibrated frames
    assert len({str(__import__("pathlib").Path(r["frames"][0]["calibrated"]).parent) for r in rep["runs"]}) == 3


def test_bias_groups_by_exposure_and_uses_the_shortest(tmp_path):
    """One calibration folder with 'Bias 0.001s' and 'Bias 0.3s' (like #MASTERS): never mixed, shortest used."""
    root = tmp_path / "MASTERS"
    for k in range(3):
        _write(root / "Bias 0.001s" / f"B_short_{k}.fits", 0.010 + RNG.normal(0, 0.001, (H, W)),
               IMAGETYP="Bias Frame", EXPTIME=0.001)
        _write(root / "Bias 0.3s" / f"B_long_{k}.fits", 0.030 + RNG.normal(0, 0.001, (H, W)),
               IMAGETYP="Bias Frame", EXPTIME=0.3)
        _write(tmp_path / "lights" / f"L_{k}.fits", _light(k, -k, 300.0, 1.0), IMAGETYP="Light Frame", EXPTIME=300.0,
               FILTER="L")
    logs = []
    r = Runner({"lights": [str(tmp_path / "lights")], "calibration": [str(root)], "work_dir": str(tmp_path / "w")},
               log=logs.append)
    sc = r.scan()
    assert len(sc["groups"]["bias"]) == 2                       # two groups, not one mixed set
    picked = r._pick(sc["groups"], "bias", sc["light_groups"][0][0])
    assert {f.exptime for f in picked} == {0.001} and len(picked) == 3
    assert any("using the shortest (0.001 s)" in m for m in logs)


def test_one_light_night_uses_the_flats_of_the_closest_night(tmp_path):
    """Lights of one night, flats of two nights with different exposures (whole #MASTERS): the same night's set."""
    for k in range(3):
        _write(tmp_path / "lights" / f"L_{k}.fits", _light(k, -k, 300.0, 1.0), IMAGETYP="Light Frame", EXPTIME=300.0,
               FILTER="Ha", **{"DATE-OBS": f"2020-01-05T21:0{k}:00"})
        _write(tmp_path / "M" / "Flat 20k" / f"F_old_{k}.fits", 0.6 + RNG.normal(0, 0.002, (H, W)), IMAGETYP="Flat Field",
               EXPTIME=0.008, FILTER="Ha", **{"DATE-OBS": f"2020-01-04T22:4{k}:00"})
        _write(tmp_path / "M" / "Flat 10k" / f"F_new_{k}.fits", 0.3 + RNG.normal(0, 0.002, (H, W)), IMAGETYP="Flat Field",
               EXPTIME=0.003, FILTER="Ha", **{"DATE-OBS": f"2020-01-06T00:1{k}:00"})   # after midnight: night of the 5th
    logs = []
    r = Runner({"lights": [str(tmp_path / "lights")], "calibration": [str(tmp_path / "M")], "work_dir": str(tmp_path / "w"),
                "master_reuse": False}, log=logs.append)
    sc = r.scan()
    key, lights = sc["light_groups"][0]
    seen = []
    orig = r._build_or_load
    r._build_or_load = lambda kind, fs, *a, **kw: (seen.append((kind, fs)), orig(kind, fs, *a, **kw))[1]
    r.masters(sc["groups"], key, 300.0, lights)
    flat_sets = [fs for kind, fs in seen if kind == "flat"]
    assert len(flat_sets) == 1 and {f.exptime for f in flat_sets[0]} == {0.003}
    assert any("using 2020-01-05" in m for m in logs)


def test_calibration_folder_with_raw_frames_and_ready_masters(tmp_path):
    """Raw darks and a ready master bias (PixInsight style) in one folder named MASTERS: the master bias is used as
    it is, the raw darks are integrated (and are not taken for masters because of the folder name)."""
    cal = tmp_path / "MASTERS"
    for k in range(3):
        _write(cal / f"D_{k}.fits", 0.012 + RNG.normal(0, 0.002, (H, W)), IMAGETYP="Dark Frame", EXPTIME=300.0)
        _write(tmp_path / "lights" / f"L_{k}.fits", _light(k, -k, 300.0, 1.0), IMAGETYP="Light Frame", EXPTIME=300.0,
               FILTER="L")
    _write(cal / "bias-BINNING_1.fits", np.full((H, W), 0.0101, np.float32), IMAGETYP="Master Bias", EXPTIME=0.0,
           NCOMBINE=40)
    logs = []
    r = Runner({"lights": [str(tmp_path / "lights")], "calibration": [str(cal)], "work_dir": str(tmp_path / "w"),
                "master_library": str(tmp_path / "lib")}, log=logs.append)
    sc = r.scan()
    assert not sc["groups"]["bias"] and len(next(iter(sc["groups"]["dark"].values()))) == 3
    key, lights = sc["light_groups"][0]
    m = r.masters(sc["groups"], key, 300.0, lights)
    assert abs(float(np.median(m["bias"])) - 0.0101) < 2e-4             # the ready master, not rebuilt
    assert any("master bias from the library: bias-BINNING_1.fits" in x for x in logs)
    assert not any("indexed dark" in x for x in logs)                   # raw darks are not masters


def test_integrated_stack_among_lights_is_not_restacked(tmp_path):
    for k in range(3):
        _write(tmp_path / "lights" / f"L_{k}.fits", _light(k, -k, 300.0, 1.0), IMAGETYP="Light Frame", EXPTIME=300.0)
    _write(tmp_path / "lights" / "integration.fits", _light(0, 0, 300.0, 1.0), IMAGETYP="Light Frame", EXPTIME=300.0,
           NCOMBINE=3)
    found = frames.scan(tmp_path / "lights")
    assert sorted(f.type for f in found) == ["light", "light", "light", "master"]


def test_wbpp_options_end_to_end(dataset, tmp_path):
    logs = []
    r = Runner({"lights": [str(dataset / "lights")],
                "calibration": [str(dataset / d) for d in ("darks", "bias", "flats", "darkflats")],
                "work_dir": str(tmp_path / "work"), "auto_reject": False, "cosmetic": False,
                "exposure_tolerance": 200, "local_norm": True, "local_norm_ref_frames": 4, "pedestal": "auto",
                "large_scale_rejection": {"high": 2, "growth": 1}, "plate_solve": False}, log=logs.append)
    img, rep = r.run()
    assert rep["lights_total"] == 6 and "pedestal_adu" in rep
    assert any("integration of" in m for m in logs)                     # LN reference made of the best frames
    assert any("large-scale pixel rejection" in m for m in logs)
    assert {"calibrate", "register", "integrate"} <= set(rep["timings"])
    assert np.isfinite(img).all()


def test_run_groups_integrates_every_group_with_its_settings(dataset, tmp_path):
    r = _runner(dataset, tmp_path, group_settings={"2": {"rejection": "percentile"}}, plate_solve=False)
    logs = []
    r.log = logs.append
    _, rep = r.run_groups()
    assert len(rep["groups"]) == 3 and all(Path(f).exists() for f in rep["groups"])
    assert any("group settings" in m and "percentile" in m for m in logs)


def test_verify_pattern_with_odd_quarter_offsets():
    # the central crop of the check started at w // 4: odd for width 300 / 3004, which swapped the sites
    from astroforge.stack import calibrate
    for h, w in ((200, 300), (2004, 3004), (3346, 5040)):
        cfa = np.full((h, w), 0.02, np.float32)
        cfa[0::2, 0::2] *= 1.6
        cfa[1::2, 1::2] *= 0.7                              # RGGB data
        assert calibrate.verify_pattern("RGGB", cfa) == ("RGGB", False), (h, w)
        assert calibrate.verify_pattern("GRBG", cfa) == ("RGGB", True), (h, w)
