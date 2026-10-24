"""WBPP features: grouping keywords, calibration keywords, large-scale rejection, pedestal, plan, plate solve skip."""
import numpy as np
from astropy.io import fits

from astroforge.stack import combine, frames
from astroforge.stack.project import Runner

RNG = np.random.default_rng(4)


def write(path, data, **hdr):
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU((np.clip(data, 0, 1) * 65535).astype(np.uint16),
                    fits.Header({"INSTRUME": "CAM", "GAIN": 100, "XBINNING": 1, **hdr})).writeto(path, overwrite=True)


def session(root):
    for k in range(6):
        tel = "A" if k < 3 else "B"
        write(root / "lights" / f"L_{k}.fits", 0.05 + RNG.normal(0, 0.002, (48, 64)), IMAGETYP="Light Frame",
              EXPTIME=60.0, TELESCOP=tel, FILTER="L")
    for k in range(3):
        write(root / "bias" / f"B_{k}.fits", 0.02 + RNG.normal(0, 0.001, (48, 64)), IMAGETYP="Bias Frame", EXPTIME=0.0)
        for rot in ("0", "90"):
            write(root / "flats" / f"F_{rot}_{k}.fits", 0.5 + RNG.normal(0, 0.003, (48, 64)), IMAGETYP="Flat Field",
                  EXPTIME=1.0, FILTER="L", ROTATOR=rot)


def test_grouping_keywords_split_lights(tmp_path):
    session(tmp_path)
    fs = frames.scan(tmp_path, keywords=["TELESCOP", "NIGHT"])
    g = frames.group(fs, ["TELESCOP"])
    keys = list(g["light"])
    assert len(keys) == 2 and {dict(k[-1])["TELESCOP"] for k in keys} == {"A", "B"}
    assert "TELESCOP=A" in frames.describe_key(keys[0]) or "TELESCOP=B" in frames.describe_key(keys[0])
    assert len(frames.group(fs)["light"]) == 1                     # no keywords: one group as before
    assert keys[0][8] == "L"                                       # filter still at its position


def test_calibration_keywords_pick_matching_flats(tmp_path):
    session(tmp_path)
    r = Runner({"lights": [str(tmp_path / "lights")], "calibration": [str(tmp_path / "bias"), str(tmp_path / "flats")],
                "work_dir": str(tmp_path / "w"), "calibration_keywords": ["ROTATOR"]}, log=lambda m: None)
    sc = r.scan()
    key = sc["light_groups"][0][0]
    r._light_kw = {"ROTATOR": "90"}
    flats = r._pick(sc["groups"], "flat", key)
    assert len(flats) == 3 and all(f.keywords["ROTATOR"] == "90" for f in flats)
    r._light_kw = {}
    assert len(r._pick(sc["groups"], "flat", key)) == 6


def test_large_scale_rejection_takes_the_whole_structure():
    n, h, w = 8, 60, 80
    stack = RNG.normal(0.1, 0.002, (n, h, w)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    blob = np.exp(-((xx - 40) ** 2 + (yy - 30) ** 2) / 120.0)       # a wide glint in one frame
    stack[3] += 0.03 * blob
    plain, _, rh0 = combine.combine(stack, "sigma", 3, 3)
    ls, _, rh1 = combine.combine(stack, "sigma", 3, 3, large_scale={"high": 2, "growth": 2})
    edge = (blob > 0.05) & (blob < 0.3)                              # faint wings the per-pixel test keeps
    assert rh1[edge].mean() > rh0[edge].mean() + 0.3
    assert abs(float(ls[edge].mean()) - 0.1) < abs(float(plain[edge].mean()) - 0.1)
    assert rh1[:10, :10].sum() == rh0[:10, :10].sum()               # nothing more rejected far from it


def test_pedestal_auto_and_fixed(tmp_path):
    r = Runner({"work_dir": str(tmp_path)}, log=lambda m: None)
    r._lights_for_pedestal = [0, 1, 2]
    neg = lambda i: np.full((40, 40), 0.001, np.float32) - np.eye(40, dtype=np.float32) * 0.0015   # noqa: E731
    p = r._pedestal(neg, "auto")
    assert 0.0005 / 1 < p < 0.002 and r.report["pedestal_adu"] >= 33
    assert r._pedestal(neg, 100) == 100 / 65535 and r._pedestal(neg, 0) == 0.0


def test_plan_lists_groups_masters_and_time(tmp_path):
    session(tmp_path)
    r = Runner({"lights": [str(tmp_path / "lights")], "calibration": [str(tmp_path / "bias"), str(tmp_path / "flats")],
                "work_dir": str(tmp_path / "w"), "grouping_keywords": ["TELESCOP"], "all_groups": True,
                "group_settings": {"1": {"drizzle": 2}}, "master_reuse": False}, log=lambda m: None)
    p = r.plan()
    assert len(p["groups"]) == 2 and p["total_s"] >= 0 and p["groups"][0]["stages"][0]["stage"] == "masters"
    assert not any(st["stage"] == "solve" for st in p["groups"][0]["stages"])       # no RA/DEC: no plate solve
    kinds = {m["kind"]: m["how"] for m in p["groups"][0]["masters"]}
    assert kinds["bias"] == "crea" and kinds["flat"] == "crea" and kinds["dark"] == "manca"
    assert any(st["stage"] == "drizzle" for st in p["groups"][1]["stages"])        # the per-group override
    assert not any(st["stage"] == "drizzle" for st in p["groups"][0]["stages"])


def test_plate_solve_skipped_without_hints(tmp_path):
    r = Runner({"work_dir": str(tmp_path), "plate_solve": "auto"}, log=lambda m: None)
    assert r._solve_output(np.zeros((20, 20, 1), np.float32), str(tmp_path / "x.fits"), fits.Header()) is None
    r.s["plate_solve"] = False
    assert r._solve_output(np.zeros((20, 20, 1), np.float32), str(tmp_path / "x.fits"), fits.Header()) is None


def test_plate_solve_writes_wcs(tmp_path, monkeypatch):
    from astropy.wcs import WCS
    from astroforge import astrometry, pipeline
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [270.9, -24.38]
    w.wcs.crpix = [10.5, 10.5]
    w.wcs.cd = np.array([[-2.0, 0], [0, 2.0]]) / 3600
    monkeypatch.setattr(pipeline, "load_config", lambda: {"tools": {"astap": "astap.exe"}})
    monkeypatch.setattr(astrometry, "solve", lambda *a, **k: w)
    r = Runner({"work_dir": str(tmp_path), "plate_solve": True}, log=lambda m: None)
    path = tmp_path / "x.fits"
    assert r._solve_output(np.zeros((20, 20, 1), np.float32), str(path), fits.Header()) is w
    sol = r.report["wcs"]["x.fits"]
    assert abs(sol["scale"] - 2.0) < 0.01 and isinstance(sol["rotation"], float) and sol["flipped"] is False
    assert abs(fits.getheader(path)["CRVAL1"] - 270.9) < 1e-9
