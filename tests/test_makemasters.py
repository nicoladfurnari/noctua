"""Building masters on their own (astroforge.stack.makemasters) from a synthetic #MASTERS folder."""
import numpy as np
import pytest
from astropy.io import fits

from astroforge.stack import frames, makemasters

H, W = 120, 160
RNG = np.random.default_rng(7)
HOT = [(10, 20), (50, 70), (90, 140)]
BADCOL = 33


def _write(path, data, **hdr):
    h = fits.Header({"INSTRUME": "QHY163M", "GAIN": 121, "OFFSET": 30, "XBINNING": 1, "SET-TEMP": -30.0, **hdr})
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU((np.clip(data, 0, 1) * 65535).astype(np.uint16), h).writeto(path, overwrite=True)


@pytest.fixture(scope="module")
def masters_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("MASTERS")
    cols = 0.001 * np.sin(np.arange(W) / 3.0)[None, :]                  # fixed column pattern of the bias
    for k in range(6):
        _write(root / "Bias 0.001s" / f"B_{k}.fit", 0.02 + cols + RNG.normal(0, 0.002, (H, W)),
               IMAGETYP="Bias Frame", EXPTIME=0.001)
        _write(root / "Bias 0.3s" / f"B03_{k}.fit", 0.025 + cols + RNG.normal(0, 0.002, (H, W)),
               IMAGETYP="Bias Frame", EXPTIME=0.3)
        d = 0.02 + cols + 0.002 + RNG.normal(0, 0.002, (H, W))
        for y, x in HOT:
            d[y, x] = 0.6
        d[:, BADCOL] += 0.03
        _write(root / "Dark 300s" / f"D_{k}.fit", d, IMAGETYP="Dark Frame", EXPTIME=300.0)
    yy, xx = np.mgrid[:H, :W]
    vign = 1 - 0.3 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    for night, date in (("n1", "2020-01-04T22:4"), ("n2", "2020-01-14T23:1")):
        for k in range(4):
            _write(root / f"Flat {night}" / f"F_ha_{night}_{k}.fit", 0.02 + 0.5 * vign + RNG.normal(0, 0.003, (H, W)),
                   IMAGETYP="Flat Field", EXPTIME=0.003, FILTER="Ha", **{"DATE-OBS": f"{date}{k}:00"})
    _write(root / "Flat n1" / "F_lonely.fit", 0.5 * vign, IMAGETYP="Flat Field", EXPTIME=0.003, FILTER="OIII",
           **{"DATE-OBS": "2020-01-04T22:50:00"})
    return root


@pytest.fixture(scope="module")
def built(masters_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    logs = []
    rep = makemasters.make_masters([masters_dir], out, {"library": False, "format": "fits"}, log=logs.append)
    return rep, out, logs


def test_one_master_per_group_named_and_recognised(built):
    rep, out, _ = built
    kinds = sorted(e["kind"] for e in rep["masters"])
    assert kinds == ["bias", "bias", "dark", "flat", "flat"]             # two bias exposures, flats per night
    names = [e["file"] for e in rep["masters"]]
    assert any("master_flat_Ha_0.003s" in n and "2020-01-04" in n for n in names)
    assert any("master_dark_300s_-30C_g121" in n for n in names)
    assert len(set(names)) == len(names)                                 # 0.001 s and 0.3 s bias: two files
    assert any("master_bias_0.3s" in n for n in names) and any("master_bias_0.001s" in n for n in names)
    for n in names:                                                      # a later scan sees them as ready masters
        assert frames.read_frame(n).type == "master"
    assert rep["skipped"] and rep["skipped"][0]["kind"] == "flat"        # the single OIII flat
    assert (out / "masters_report.txt").exists() and (out / "masters_report.json").exists()


def test_flats_are_normalised_and_bias_removed(built):
    rep, _, _ = built
    flat = [e for e in rep["masters"] if e["kind"] == "flat"][0]
    a = fits.getdata(flat["file"]).astype(np.float32)
    assert abs(float(np.median(a)) - 1.0) < 0.02
    assert 10 < flat["stats"]["vignetting_pct"] < 25          # synthetic corners at 0.85 of the centre


def test_defect_map_has_hot_pixels_and_the_bad_column(built):
    rep, _, _ = built
    assert len(rep["defect_maps"]) == 1
    bad = fits.getdata(rep["defect_maps"][0]["file"]).astype(bool)
    assert all(bad[y, x] for y, x in HOT)
    assert bad[:, BADCOL].all() and rep["defect_maps"][0]["columns"] >= 1
    assert bad.mean() < 0.05


def test_superbias_keeps_pattern_and_drops_noise(masters_dir, tmp_path):
    rep = makemasters.make_masters([masters_dir / "Bias 0.001s"], tmp_path, {"library": False, "superbias": True,
                                                                              "defect_map": False}, log=lambda m: None)
    from astroforge.stack import calibrate
    sb = calibrate.load_master(rep["masters"][0]["file"])
    plain = makemasters.make_masters([masters_dir / "Bias 0.001s"], tmp_path / "p", {"library": False, "defect_map": False},
                                     log=lambda m: None)
    pb = calibrate.load_master(plain["masters"][0]["file"])
    truth = 0.02 + 0.001 * np.sin(np.arange(W) / 3.0)[None, :] * np.ones((H, 1))
    err_sb = float(np.std(sb - truth * 65535 / 65535))
    err_pb = float(np.std(pb - truth))
    assert err_sb < 0.4 * err_pb                                          # much closer to the real bias


def test_min_frames_and_rejection_options(masters_dir, tmp_path):
    rep = makemasters.make_masters([masters_dir / "Flat n1"], tmp_path, {"library": False, "min_frames": 5,
                                                                         "rejection": "sigma", "low": 3, "high": 3},
                                   log=lambda m: None)
    assert not rep["masters"] and len(rep["skipped"]) == 2


def test_gui_job(masters_dir, tmp_path):
    import time
    from fastapi.testclient import TestClient
    from astroforge.gui.server import create_app
    c = TestClient(create_app())
    r = c.post("/api/masters", json={"folders": [str(masters_dir / "Bias 0.001s")], "out_dir": str(tmp_path / "m"),
                                     "options": {"library": False, "defect_map": False, "nonsense": 1}}).json()
    for _ in range(100):
        j = c.get(f"/api/jobs/{r['job']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j["log"][-3:]
    assert len(j["result"]["masters"]) == 1 and (tmp_path / "m" / "masters_report.txt").exists()
    assert c.post("/api/masters", json={"folders": []}).status_code == 400


def test_per_type_options_and_legacy_keys():
    t = makemasters.type_options({"rejection": "sigma", "low": 2.5, "flat": {"combination": "median"}}, "flat")
    assert t["rejection"] == "sigma" and t["low"] == 2.5 and t["combination"] == "median"
    assert makemasters.combine_kw(t)["method"] == "median"
    b = makemasters.type_options({"bias": {"rejection": "esd", "esd_outliers": 0.2}}, "bias")
    assert makemasters.combine_kw(b)["method"] == "esd" and b["esd_outliers"] == 0.2
    assert makemasters.type_options({"flat_cfa": False}, "flat")["normalize"] == "global"
    assert makemasters.type_options({}, "dark") == makemasters.TYPE_DEFAULTS["dark"]


def test_flat_options_calibration_and_min_frames(masters_dir, tmp_path):
    rep = makemasters.make_masters([masters_dir], tmp_path, {"library": False, "defect_map": False, "format": "fits",
                                   "flat": {"calibrate_with": "none", "per_night": False, "min_frames": 5},
                                   "bias": {"combination": "median"}, "dark": {"rejection": "sigma", "low": 3}},
                                   log=lambda m: None)
    flats = [e for e in rep["masters"] if e["kind"] == "flat"]
    assert len(flats) == 1 and flats[0]["frames"] == 8 and not flats[0]["night"]     # both nights together
    assert rep["types"]["bias"]["combination"] == "median" and rep["types"]["dark"]["low"] == 3
    assert rep["skipped"] and rep["skipped"][0]["frames"] == 1                       # lonely OIII < 5


def test_gui_defaults_and_per_type_options(masters_dir, tmp_path):
    import time
    from fastapi.testclient import TestClient
    from astroforge.gui.server import create_app
    c = TestClient(create_app())
    d = c.get("/api/masters/defaults").json()
    assert set(d["types"]) == {"bias", "dark", "darkflat", "flat"} and d["types"]["flat"]["calibrate_with"] == "auto"
    r = c.post("/api/masters", json={"folders": [str(masters_dir / "Bias 0.001s")], "out_dir": str(tmp_path / "m"),
                                     "options": {"library": False, "defect_map": False,
                                                 "bias": {"combination": "median", "junk": 1}}}).json()
    for _ in range(100):
        j = c.get(f"/api/jobs/{r['job']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j["log"][-3:]
    assert j["result"]["types"]["bias"]["combination"] == "median" and "junk" not in j["result"]["types"]["bias"]
