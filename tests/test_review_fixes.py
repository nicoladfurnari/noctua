"""Defects found by the code review of 2026-10-24 (pipeline masks and star target, star mask on a star layer,
px scaling of flat points and octave lists, recombine reduce while linear)."""
import numpy as np

from astroforge import masks, pipeline, plugins, stars

RNG = np.random.default_rng(7)
H, W = 96, 128


def ctx(**kw):
    return plugins.Context(log=lambda m: None, **kw)


def linear_field():
    img = 0.01 + RNG.normal(0, 0.0005, (H, W, 3))
    yy, xx = np.mgrid[:H, :W]
    img += 0.05 * np.exp(-((xx - 40) ** 2 + (yy - 50) ** 2) / 600)[..., None]
    for x, y in ((90, 20), (20, 80), (100, 70)):
        img += 0.9 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 3)[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def test_masked_stretch_uses_the_mask_of_the_input():
    img = linear_field()
    plain = pipeline.apply_step(img, {"plugin": "stretch.auto"}, ctx(linear=True))
    masked = pipeline.apply_step(img, {"plugin": "stretch.auto", "mask": {"type": "lum"}}, ctx(linear=True))
    # the luminance mask is high on the nebula: there the masked result follows the stretch (before the fix the
    # mask was measured as if the input were already stretched: all zeros, the stretch cancelled)
    y, x = 50, 40
    assert masked[y, x].mean() > 0.5 * plain[y, x].mean()
    assert masked[y, x].mean() > 5 * img[y, x].mean()


def test_stretch_of_the_star_layer_keeps_the_image_linear():
    img = linear_field()
    c = ctx(linear=True)
    c.layers["stars"] = img.copy()
    out = pipeline.apply_step(img, {"plugin": "stretch.auto", "target": "stars"}, c)
    assert c.linear is True
    assert np.array_equal(out, img)
    assert c.layers["stars"].mean() > img.mean()


def test_validate_ignores_disabled_and_star_target_steps():
    preset = {"steps": [{"plugin": "stretch.auto", "enabled": False},
                        {"plugin": "stretch.auto", "target": "stars"},
                        {"plugin": "background.poly"}]}
    assert pipeline.validate(preset) == []
    preset["steps"][0]["enabled"] = True
    assert len(pipeline.validate(preset)) == 1


def test_star_mask_on_a_layer_with_a_zero_sky_is_finite():
    layer = np.zeros((256, 256, 3), np.float32)
    yy, xx = np.mgrid[:256, :256]
    for x, y in RNG.integers(10, 246, (30, 2)):
        layer += 0.8 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2)[..., None]
    m = stars.star_mask(np.clip(layer, 0, 1))
    assert np.isfinite(m).all()
    assert m.max() > 0.9 and m.mean() < 0.2
    m2 = masks.star_mask(layer, False)
    assert np.isfinite(m2).all()


def test_px_scaling_of_flat_points_and_octave_lists():
    ls = plugins.get("sharpen.larson_sekanina")
    assert ls.scaled(ls.resolve({"center": [100, 200]}), 0.5)["center"] == [50, 100]
    assert ls.scaled(ls.resolve({"center": [[100, 200]]}), 0.5)["center"] == [[50, 100]]
    ws = plugins.get("sharpen.wavelet")
    assert ws.scaled(ws.resolve({"layers": [2, 3]}), 0.25)["layers"] == [1]
    assert ws.scaled(ws.resolve({"layers": [2, 3]}), 1.0)["layers"] == [2, 3]


def test_recombine_reduce_works_while_linear():
    img = np.full((H, W, 3), 0.01, np.float32)
    layer = np.zeros((H, W, 3), np.float32)
    layer[40:44, 60:64] = 0.5
    rec = plugins.get("stars.recombine")
    outs = []
    for reduce in (0.0, 1.0):
        c = ctx(linear=True)
        c.layers["stars"], c.layers["stars_mode"] = layer, "add"
        outs.append(rec.run(img, c, **rec.resolve({"reduce": reduce})))
    assert outs[1].sum() < outs[0].sum()


# ---------------------------------------------------------------- stacker
def test_light_groups_merge_within_two_degrees():
    from astroforge.stack import frames as fr
    fs = [fr.Frame(path=f"a{i}.fits", type="light", width=10, height=10, exptime=60, temp=t, camera="cam")
          for i, t in enumerate((-9.4, -8.9, -9.1, -8.8, -10.0))]
    fs.append(fr.Frame(path="warm.fits", type="light", width=10, height=10, exptime=60, temp=5.0, camera="cam"))
    g = fr.group(fs)["light"]
    assert sorted(len(v) for v in g.values()) == [1, 5]


def test_night_of_a_date_only_frame():
    from astroforge.stack import frames as fr
    assert fr.night(fr.Frame(path="x", type="light", date="2024-10-12")) == "2024-10-12"
    assert fr.night(fr.Frame(path="x", type="light", date="2024-10-13T02:30:00")) == "2024-10-12"


def test_xisf_keywords_are_xml_escaped(tmp_path):
    from astroforge.stack import xisf
    p = tmp_path / "m.xisf"
    xisf.write(np.zeros((4, 4, 1), np.float32), p, {"INSTRUME": 'ZWO & co "x" <y>', "FILTER": "Ha<OIII"})
    _, kw = xisf.read(p)
    assert kw["INSTRUME"] == 'ZWO & co "x" <y>' and kw["FILTER"] == "Ha<OIII"


def test_master_exptime_from_fits(tmp_path):
    from astropy.io import fits
    from astroforge.stack import frames as fr
    p = tmp_path / "dark.fits"
    fits.PrimaryHDU(np.zeros((4, 4), np.float32), fits.Header({"EXPTIME": 120.0})).writeto(p)
    assert fr.master_exptime(p) == 120.0
    assert fr.master_exptime(tmp_path / "missing.fits") is None


def test_gui_full_shape_reads_the_fits_header_only(tmp_path):
    from astropy.io import fits
    from astroforge.gui import server
    p = tmp_path / "big.fits"
    fits.PrimaryHDU(np.zeros((30, 50), np.float32)).writeto(p)
    server._FULL_SHAPE.clear()
    assert server._full_shape(p) == (50, 30)
    assert server._full_shape(str(p)) == (50, 30)


def test_build_master_by_rows_matches_whole_frames(tmp_path):
    """Masters read band by band from the memory map (int16 + BZERO, one file bottom-up, a dark to subtract,
    CFA normalisation) give what the whole-frame path gave."""
    from astropy.io import fits
    from astroforge.stack import calibrate, combine
    rng = np.random.default_rng(5)
    H, W = 64, 80
    yy, xx = np.mgrid[:H, :W]
    base = 0.6 - 0.2 * ((xx - 40) ** 2 + (yy - 32) ** 2) / 2500
    tint = np.ones((H, W), np.float32)
    tint[0::2, 0::2], tint[1::2, 1::2] = 0.5, 0.35
    dark = np.full((H, W), 0.02, np.float32)
    paths, frames = [], []
    for k in range(4):
        f = np.clip(base * tint + dark + rng.normal(0, 0.003, (H, W)), 0, 1).astype(np.float32)
        frames.append(f)
        u16 = (f * 65535 + 0.5).astype(np.uint16)
        hdu = fits.PrimaryHDU((u16[::-1] if k == 1 else u16))      # astropy stores uint16 as int16 + BZERO 32768
        if k == 1:
            hdu.header["ROWORDER"] = "BOTTOM-UP"
        p = tmp_path / f"flat{k}.fits"
        hdu.writeto(p)
        paths.append(str(p))
    for cfa in (False, True):
        got = calibrate.build_master(paths, "flat", dark=dark, method="median", log=lambda *a: None, cfa_normalize=cfa)
        mode = "cfa" if cfa else "global"
        ref = combine.combine(np.stack([calibrate.normalize_flat(f - dark, mode) for f in frames]), "median")[0]
        ref = calibrate.normalize_flat(ref, mode)
        assert got.shape == (H, W) and np.allclose(got, ref, atol=2e-3), np.abs(got - ref).max()
    raw1 = calibrate.load_raw(paths[1])[0]
    assert np.allclose(raw1, frames[1], atol=1e-4)      # the bottom-up file is read top-down by both paths


def test_fix_saturated_leaves_unsaturated_cores_alone():
    yy, xx = np.mgrid[:128, :128]
    star = np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / 200)
    img = np.stack([0.01 + 0.3 * star, 0.01 + 0.2 * star, 0.01 + 0.1 * star], -1).astype(np.float32)
    pl = plugins.get("color.fix_saturated")
    out = pl.run(img, ctx(), **pl.resolve({}))
    assert np.array_equal(out, img)
    sat = np.clip(img * 4, 0, 1).astype(np.float32)           # a clipped core, magenta: red and blue saturate first
    out = pl.run(sat, ctx(), **pl.resolve({}))
    assert out[64, 64].std() < sat[64, 64].std()


def test_vertical_shift_of_the_bayer_pattern():
    from astroforge.stack import calibrate
    assert calibrate.vertical_shift("RGGB") == "GBRG" and calibrate.vertical_shift("GRBG") == "BGGR"


# ---------------------------------------------------------------- fourth review pass
def test_palette_names_in_any_case():
    from astroforge.stack import channels
    ha = np.full((8, 8), 0.2, np.float32)
    chans = {"Ha": ha, "OIII": ha * 0.5, "SII": ha * 0.3}
    out, pal = channels.combine(chans, "sho")
    assert pal == "SHO" and out.shape == (8, 8, 3)


def test_frame_type_from_the_path():
    from pathlib import PureWindowsPath as P
    from astroforge.stack import frames as fr
    assert fr._classify(P(r"D:\Astro\Dark Doodad\frame_001.fit"), {}) == "light"
    assert fr._classify(P(r"D:\Astro\Dark Horse\light_001.fit"), {}) == "light"
    assert fr._classify(P(r"D:\Astro\#Dark\frame_001.fit"), {}) == "dark"
    assert fr._classify(P(r"D:\Astro\QHY163M\Dark 300s gain 121 -30C\frame_001.fit"), {}) == "dark"
    assert fr._classify(P(r"D:\Astro\QHY163M\Flat 10k ADU gain 121 -30C\frame_001.fit"), {}) == "flat"
    assert fr._classify(P(r"D:\Astro\M8\flat_001.fit"), {}) == "flat"
    assert fr._classify(P(r"D:\Astro\M8\D_0001.fit"), {}) == "dark"
    assert fr._classify(P(r"D:\Astro\M8\x.fit"), {"IMAGETYP": "Bias Frame"}) == "bias"


def test_make_masters_cancel_is_a_cancelled_job(tmp_path):
    from astropy.io import fits
    from astroforge.stack import makemasters, project
    src = tmp_path / "src"
    src.mkdir()
    for k in range(3):
        h = fits.Header({"IMAGETYP": "Bias Frame", "EXPTIME": 0.001})
        fits.PrimaryHDU(np.zeros((16, 16), np.uint16) + 100, h).writeto(src / f"B_{k}.fits")
    import pytest
    with pytest.raises(project.Cancelled):
        makemasters.make_masters([str(src)], str(tmp_path / "out"), {}, log=lambda m: None, cancel=lambda: True)


def test_start_gui_runs_updater_and_gui_on_one_line():
    import pytest
    from pathlib import Path
    cmd = Path(__file__).resolve().parents[1] / "start-gui.cmd"
    if not cmd.exists():
        pytest.skip("start-gui.cmd not in this checkout (dev copy on Hawk)")
    lines = [l for l in cmd.read_text(encoding="utf-8").splitlines() if "astroforge.updater" in l]
    assert len(lines) == 1 and "astroforge gui" in lines[0]
