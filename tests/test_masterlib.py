"""Master library (reuse, ranking, sanity flags, external masters), CFA flat normalisation, banding."""
from datetime import datetime, timedelta

import numpy as np
import pytest
from astropy.io import fits

from astroforge.stack import calibrate, frames
from astroforge.stack.masterlib import MasterLibrary, quality

H, W = 120, 160
RNG = np.random.default_rng(7)


def _frame(path, kind, **kw):
    base = dict(path=str(path), type=kind, width=W, height=H, camera="QHY168C", gain=10.0, offset=40.0,
                temp=-10.0, binning=1, date="2026-10-20T22:00:00")
    base.update(kw)
    return frames.Frame(**base)


def _dark(level=0.01):
    return (level + RNG.normal(0, 0.001, (H, W))).astype(np.float32)


def _flat(tint=(1.0, 1.0, 1.0, 1.0), level=1.0):
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.3 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    f = np.empty((H, W), np.float32)
    for i, t in enumerate(tint):
        dy, dx = divmod(i, 2)
        f[dy::2, dx::2] = vig[dy::2, dx::2] * t
    return (f * level + RNG.normal(0, 0.002 * level, f.shape)).astype(np.float32)


@pytest.fixture
def lib(tmp_path):
    return MasterLibrary(tmp_path / "lib", log=lambda *a: None)


LIGHT = {"camera": "QHY168C", "width": W, "height": H, "binning": 1, "gain": 10.0, "offset": 40.0,
         "temp": -10.0, "filter": "", "night": "2026-10-20"}


# ---------------------------------------------------------------- add / find

def test_exact_match_beats_near_match(lib, tmp_path):
    exact = lib.add("dark", _dark(), [_frame(tmp_path / f"d300_{k}.fit", "dark", exptime=300.0) for k in range(5)])
    lib.add("dark", _dark(), [_frame(tmp_path / f"dw_{k}.fit", "dark", exptime=300.0, temp=-8.0) for k in range(20)])
    lib.add("dark", _dark(), [_frame(tmp_path / f"d120_{k}.fit", "dark", exptime=120.0) for k in range(20)])
    lib.add("dark", _dark(), [_frame(tmp_path / f"dg_{k}.fit", "dark", exptime=300.0, gain=20.0) for k in range(20)])
    m = lib.find("dark", LIGHT, exptime=300.0)
    assert m.entry["id"] == exact["id"]
    assert any("matches" in r for r in m.reasons)
    ids = [c.entry["id"] for c in lib.candidates("dark", LIGHT, exptime=300.0)]
    assert len(ids) == 2                                          # other gain, and 120 s without bias: excluded
    # with a bias the 120 s dark becomes usable, scaled
    c = lib.candidates("dark", LIGHT, exptime=300.0, scalable=True)
    assert len(c) == 3 and c[0].entry["id"] == exact["id"]
    scaled = [x for x in c if x.entry["exptime"] == 120.0][0]
    assert scaled.dark_scale == pytest.approx(2.5) and any("scaled" in r for r in scaled.reasons)
    # persisted and reloaded
    again = MasterLibrary(lib.root, log=lambda *a: None)
    assert again.find("dark", LIGHT, exptime=300.0).entry["id"] == exact["id"]
    assert again.load(exact["id"]).shape == (H, W)
    assert again.lookup("dark", [f"{tmp_path / f'd300_{k}.fit'}" for k in range(5)]) is not None


def test_light_as_frame_or_key(lib, tmp_path):
    e = lib.add("bias", _dark(0.005), [_frame(tmp_path / f"b{k}.fit", "bias", exptime=0.0) for k in range(9)])
    light = _frame(tmp_path / "l.fit", "light", exptime=300.0)
    assert lib.find("bias", light).entry["id"] == e["id"]
    assert lib.find("bias", frames.setup_key(light)).entry["id"] == e["id"]
    assert lib.find("bias", {**LIGHT, "width": W + 2}) is None


def test_flats_filter_and_night(lib, tmp_path):
    same = lib.add("flat", _flat(), [_frame(tmp_path / f"fa{k}.fit", "flat", exptime=1.0, filter="Ha")
                                     for k in range(5)], {"flat_level": 25000})
    other_night = lib.add("flat", _flat(), [_frame(tmp_path / f"fb{k}.fit", "flat", exptime=1.0, filter="Ha",
                                                   date="2026-09-01T21:00:00") for k in range(30)])
    lib.add("flat", _flat(), [_frame(tmp_path / f"fc{k}.fit", "flat", exptime=1.0, filter="OIII") for k in range(30)])
    m = lib.find("flat", {**LIGHT, "filter": "Ha"})
    assert m.entry["id"] == same["id"] and "same night" in m.reasons
    c = lib.candidates("flat", {**LIGHT, "filter": "Ha"})
    assert [x.entry["id"] for x in c] == [same["id"], other_night["id"]]       # OIII never offered
    assert any("another night" in r for r in c[1].reasons)
    assert lib.find("flat", {**LIGHT, "filter": "SII"}) is None


def test_expiry_warning(lib, tmp_path):
    e = lib.add("dark", _dark(), [_frame(tmp_path / f"d{k}.fit", "dark", exptime=60.0) for k in range(5)])
    fresh = lib.find("dark", LIGHT, exptime=60.0)
    assert not any("days old" in r for r in fresh.reasons)
    later = datetime.now() + timedelta(days=400)
    old = lib.find("dark", LIGHT, exptime=60.0, now=later)
    assert old.entry["id"] == e["id"] and any("days old" in r for r in old.reasons)
    assert old.score < fresh.score


def test_flagged_masters_are_skipped(lib, tmp_path):
    bad = lib.add("flat", _flat(), [_frame(tmp_path / f"f{k}.fit", "flat", filter="L") for k in range(5)],
                  {"flat_level": 37})
    assert bad["flags"] and "implausibly low" in bad["flags"][0]
    assert lib.find("flat", {**LIGHT, "filter": "L"}) is None
    m = lib.find("flat", {**LIGHT, "filter": "L"}, allow_flagged=True)
    assert m and any("FLAGGED" in r for r in m.reasons)


def test_quality_flags_defective_and_cfa_imbalance():
    # the user's QHY168C master flat: median ~37 ADU of 65535, not normalized
    broken = _flat(level=37 / 65535)
    st, flags = quality("flat", broken)
    assert st["level_adu"] == pytest.approx(37, rel=0.1)
    assert any("implausibly low" in f for f in flags)
    good = _flat(tint=(0.5, 1.0, 1.0, 0.4), level=0.4)
    assert quality("flat", good, bayer="RGGB")[1] == []
    imbalanced = _flat(tint=(0.05, 1.0, 1.0, 0.8), level=0.4)
    assert any("CFA imbalance" in f for f in quality("flat", imbalanced, bayer="RGGB")[1])


# ---------------------------------------------------------------- external masters

def _write_master(path, data, **hdr):
    h = fits.Header()
    h.update({"INSTRUME": "QHY168C", "XBINNING": 1, **hdr})
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data.astype(np.float32), h).writeto(path, overwrite=True)


def test_scan_external_pixinsight_headers(lib, tmp_path):
    ext = tmp_path / "#MASTERS"
    _write_master(ext / "masterDark_BIN-1_EXPOSURE-300.00s.xisf.fits", _dark() * 65535,
                  IMAGETYP="Master Dark", EXPTIME=300.0, GAIN=10, OFFSET=40, **{"SET-TEMP": -10.0, "NCOMBINE": 30})
    _write_master(ext / "bias.fit", _dark(0.005), IMAGETYP="MasterBias", GAIN=10, OFFSET=40,
                  **{"CCD-TEMP": -10.3, "NCOMBINE": 50})
    _write_master(ext / "masterFlat_FILTER-L.fit", _flat(level=37 / 65535), IMAGETYP="Master Flat", FILTER="L",
                  BAYERPAT="RGGB", **{"DATE-OBS": "2026-10-20T20:00:00", "NCOMBINE": 25})
    _write_master(ext / "masterFlat_FILTER-Ha.fit", _flat(level=0.4) * 65535, IMAGETYP="'Master Flat'",
                  FILTER="Ha", **{"DATE-OBS": "2026-10-20T20:00:00", "NCOMBINE": 25})
    _write_master(ext / "sub" / "light_001.fit", _dark(), IMAGETYP="Light Frame")   # not a master
    got = lib.scan_external(ext)
    kinds = sorted(e["kind"] for e in got)
    assert kinds == ["bias", "dark", "flat", "flat"]
    dark = lib.find("dark", LIGHT, exptime=300.0)
    assert dark.entry["external"] and dark.entry["nframes"] == 30
    assert dark.entry["stats"]["median"] == pytest.approx(0.01 * 65535, rel=0.05)
    assert lib.find("bias", LIGHT).entry["temp"] == pytest.approx(-10.3)
    assert lib.find("flat", {**LIGHT, "filter": "L"}) is None                  # the 37 ADU flat is flagged
    ha = lib.find("flat", {**LIGHT, "filter": "Ha"})
    assert ha and "same night" in ha.reasons
    assert np.median(lib.load(ha.entry)) == pytest.approx(1.0, abs=0.01)       # normalized on load
    # rescan: unchanged files are not re-read, external files survive remove()
    assert len(lib.scan_external(ext)) == 4
    lib.remove(dark.entry["id"])
    assert (ext / "masterDark_BIN-1_EXPOSURE-300.00s.xisf.fits").exists()


# ---------------------------------------------------------------- CFA flat normalisation

def test_cfa_normalisation_keeps_colour_ratios():
    pattern = "RGGB"
    sky = {"R": 0.10, "G": 0.20, "B": 0.05}          # true colour of the target
    tint = (0.5, 1.0, 1.0, 0.35)                     # flat panel: reddish-blue deficit, RGGB order
    flat_raw = _flat(tint=tint, level=0.45)
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.3 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    light = np.empty((H, W), np.float32)
    for i, ch in enumerate(pattern):
        dy, dx = divmod(i, 2)
        light[dy::2, dx::2] = sky[ch] * vig[dy::2, dx::2] * tint[i]   # same optics and sensor response as the flat
    light += RNG.normal(0, 0.0005, light.shape).astype(np.float32)

    def ratios(cal):
        m = {ch: float(np.median(cal[i // 2::2, i % 2::2])) for i, ch in enumerate(pattern)}
        return m["R"] / m["G"], m["B"] / m["G"]

    cfa = calibrate.calibrate(light, flat=calibrate.normalize_flat(flat_raw, "cfa"))
    glob = calibrate.calibrate(light, flat=calibrate.normalize_flat(flat_raw, "global"))
    sensor = (sky["R"] * tint[0] / (sky["G"] * tint[1]), sky["B"] * tint[3] / (sky["G"] * tint[1]))
    assert ratios(cfa) == pytest.approx(sensor, rel=0.02)            # colour balance of the light untouched
    assert ratios(glob)[0] == pytest.approx(sky["R"] / sky["G"], rel=0.02)   # global: flat tint removed
    # vignetting flattened either way
    for cal in (cfa, glob):
        g = cal[0::2, 1::2]
        assert abs(np.median(g[:8, :8]) / np.median(g[25:35, 35:45]) - 1) < 0.03


def test_build_master_cfa_normalize(tmp_path):
    paths = []
    for k in range(3):
        p = tmp_path / f"F_{k}.fits"
        fits.PrimaryHDU((np.clip(_flat(tint=(0.5, 1, 1, 0.35), level=0.5), 0, 1) * 65535).astype(np.uint16)
                        ).writeto(p)
        paths.append(str(p))
    m = calibrate.build_master(paths, "flat", method="median", log=lambda *a: None, cfa_normalize=True)
    sites = [np.median(m[dy::2, dx::2]) for dy in (0, 1) for dx in (0, 1)]
    assert np.allclose(sites, 1.0, atol=0.01)
    g = calibrate.build_master(paths, "flat", method="median", log=lambda *a: None)
    assert np.median(g) == pytest.approx(1.0, abs=0.01) and np.median(g[0::2, 0::2]) < 0.8


# ---------------------------------------------------------------- banding

def _sky(h=800, w=1200, noise=0.002, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 0.05 + 0.02 * yy / h + 0.01 * xx / w                                 # gradients: must be kept
    img += 0.03 * np.exp(-(((xx - 700) / 160) ** 2 + ((yy - 300) / 120) ** 2))  # nebula
    for _ in range(500):
        x, y, a = rng.uniform(5, w - 5), rng.uniform(5, h - 5), rng.uniform(0.02, 0.6)
        img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.3 ** 2))
    return (img + rng.normal(0, noise, img.shape)).astype(np.float32)


def _banding(n, seed, std=0.004):
    """Random per-line offsets (2x the noise). Banding slower than ~16 lines is indistinguishable from a
    gradient and is kept on purpose, so the synthetic banding has none."""
    from scipy.ndimage import gaussian_filter1d
    w = np.random.default_rng(seed).normal(0, std, n)
    return (w - gaussian_filter1d(w, 16, mode="reflect")).astype(np.float32)


def _offsets(x, ref, axis=1):
    d = x - ref
    r = np.median(d, axis=axis)
    return r - np.median(r)


@pytest.fixture(scope="module")
def sky():
    return _sky()


@pytest.mark.parametrize("pattern", [None, "RGGB"])
def test_banding_removed(sky, pattern):
    band = _banding(sky.shape[0], 5)
    if pattern:
        # each Bayer plane sees every other row: a sensor-row banding subsampled that way aliases into slow
        # (gradient-like) per-plane offsets, which are kept by design. Inject banding fast on each plane.
        band[0::2], band[1::2] = _banding(sky.shape[0] // 2, 6), _banding(sky.shape[0] // 2, 7)
    banded = sky + band[:, None]
    out = calibrate.reduce_banding(banded, pattern=pattern)
    assert _offsets(banded, sky).std() == pytest.approx(band.std(), rel=0.01)
    assert _offsets(out, sky).std() < 0.1 * band.std()
    # gradients, stars and nebula kept: the image changed only by the banding
    assert abs(float(np.median(out - sky))) < 0.0005
    assert out[300, 700] == pytest.approx(banded[300, 700] - band[300], abs=0.002)   # nebula core


def test_banding_clean_frame_unchanged(sky):
    noise = 0.002
    out = calibrate.reduce_banding(sky, pattern="RGGB", axis="both")
    diff = out - sky
    assert np.abs(diff).max() < 0.75 * noise          # within noise
    assert diff.std() < 0.15 * noise


def test_banding_columns_and_rgb(sky):
    col = _banding(sky.shape[1], 9)
    out = calibrate.reduce_banding(sky + col[None, :], axis="cols")
    assert _offsets(out, sky, axis=0).std() < 0.1 * col.std()
    stripes = 0.004 * np.sign(np.sin(np.arange(sky.shape[0]) * 1.3)).astype(np.float32)
    rgb = np.stack([sky + stripes[:, None]] * 3, -1)
    out = calibrate.reduce_banding(rgb, amount=1.0)
    assert out.shape == rgb.shape
    assert _offsets(out[..., 1], sky).std() < 0.1 * stripes.std()
    half = calibrate.reduce_banding(rgb[..., :1], amount=0.5)
    assert _offsets(half[..., 0], sky).std() == pytest.approx(0.5 * stripes.std(), rel=0.15)
