"""Tests for the PixInsight-workflow processes (masks, TGV, MMT, HT, linked MTF, colour, HDR, heal, geometry)."""
import numpy as np
import pytest

from astroforge import enhance, io, masks, plugins, tgv
from astroforge.plugins import workflow  # noqa: F401  (registers the plugins)
from astroforge.stats import luminance  # noqa: F401


def ctx(**kw):
    return plugins.Context(log=lambda m: None, **kw)


def run(pid, img, c=None, **params):
    pl = plugins.get(pid)
    return pl.run(img, c or ctx(linear=False), **pl.resolve(params))


def field(h=200, w=240, stars=((60, 50, 0.6), (150, 120, 0.5), (200, 160, 0.4)), fwhm=3.0, bg=0.1):
    yy, xx = np.mgrid[:h, :w]
    img = np.full((h, w), bg, np.float32)
    for x, y, a in stars:
        img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (fwhm / 2.355) ** 2))
    return img


def patches():
    """Red, green, blue, yellow and grey patches on a dark background (nonlinear)."""
    img = np.full((60, 300, 3), 0.05, np.float32)
    for i, col in enumerate([(0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8), (0.8, 0.8, 0.1), (0.5, 0.5, 0.5)]):
        img[10:50, 10 + 60 * i:50 + 60 * i] = col
    return img


SL = [np.s_[10:50, 10 + 60 * i:50 + 60 * i] for i in range(5)]   # red, green, blue, yellow, grey


# ------------------------------------------------------------------ masks
def test_range_mask_selects_bright():
    img = np.repeat(np.linspace(0, 1, 101, dtype=np.float32)[None, :, None], 4, 0).repeat(3, -1)
    m = masks.range_mask(img, 0.5, 1.0)
    L = masks.cie_lightness(img)
    assert np.all(m[L >= 0.5] == 1) and np.all(m[L < 0.5] == 0)
    soft = masks.range_mask(img, 0.4, 0.8, fuzziness=0.5)
    assert soft.max() == 1 and 0 < soft[0, 45] < 1 and soft[0, 95] == 0


def test_color_mask_selects_hue():
    img = patches()
    m = masks.color_mask(img, "red")
    assert m[SL[0]].mean() > 0.9
    assert m[SL[1]].max() < 0.01 and m[SL[2]].max() < 0.01 and m[SL[4]].max() < 0.01
    assert masks.color_mask(img, "blue")[SL[2]].mean() > 0.9


def test_luminance_mask_and_make_mask():
    img = patches()
    m = masks.luminance_mask(img, 0.0, 0.5, 1.0)
    assert np.allclose(m, masks.cie_lightness(img), atol=1e-5)
    spec = {"type": "color", "color": "green", "invert": True, "blur": 2}
    mm = masks.make_mask(img, spec, ctx(linear=False))
    assert mm.shape == img.shape[:2] and mm[25, 90] < 0.05 and mm[25, 30] > 0.95
    with pytest.raises(ValueError):
        masks.make_mask(img, {"type": "range", "nope": 1})
    c = ctx(linear=False)
    c.layers["hdr_mask_1"] = np.ones((30, 150), np.float32)
    assert masks.make_mask(img, {"type": "layer", "name": "hdr_mask_1"}, c).shape == img.shape[:2]


# ------------------------------------------------------------------ noise
def test_tgv_lowers_noise_keeps_stars():
    rng = np.random.default_rng(1)
    clean = np.repeat(field()[..., None], 3, -1)
    noisy = np.clip(clean + rng.normal(0, 0.01, clean.shape), 0, 1).astype(np.float32)
    out = run("noise.tgv", noisy, ctx(linear=True))
    bg = np.s_[100:190, 10:100]
    assert np.std((out - clean)[bg]) < 0.6 * np.std((noisy - clean)[bg])
    for x, y, a in ((60, 50, 0.6), (150, 120, 0.5)):
        assert out[y, x, 1] - 0.1 > 0.85 * a


def test_tgv_speed_tile_stitch():
    rng = np.random.default_rng(2)
    f = (0.2 + rng.normal(0, 0.02, (700, 900))).astype(np.float32)
    u, _ = tgv.tgv_channel(f, 2.0, iterations=60)
    assert u.shape == f.shape and np.std(u) < 0.3 * np.std(f)
    # no seams at tile borders
    assert abs(float(np.mean(np.abs(np.diff(u, axis=1)[:, 510:514]))) - float(np.mean(np.abs(np.diff(u, axis=1))))) < 0.01


def test_mmt_and_first_layer():
    rng = np.random.default_rng(3)
    clean = np.repeat(field()[..., None], 3, -1)
    noisy = (clean + rng.normal(0, 0.01, clean.shape)).astype(np.float32)
    bg = np.s_[100:190, 10:100]
    out = run("noise.mmt", noisy, protect=0.0)
    assert np.std((out - clean)[bg]) < 0.6 * np.std((noisy - clean)[bg])
    fl = run("noise.first_layer_subtract", noisy, amount=1.0)
    assert np.std((fl - clean)[bg]) < 0.7 * np.std((noisy - clean)[bg])
    assert np.array_equal(run("noise.first_layer_subtract", noisy, amount=0.0), np.clip(noisy, 0, 1))


# ------------------------------------------------------------------ stretch
def test_histogram_transform():
    img = np.random.default_rng(4).random((20, 20, 3)).astype(np.float32)
    assert np.allclose(run("stretch.histogram", img), img, atol=1e-6)
    out = run("stretch.histogram", img, shadows=[0.1, 0.2, 0.3], midtones=0.25)
    assert out[..., 0].min() == 0 and np.all(out[..., 2][img[..., 2] <= 0.3] == 0)
    assert out.mean() > img.mean()


def test_linked_mtf_target_and_linked():
    rng = np.random.default_rng(5)
    grey = np.repeat((0.02 + 0.002 * rng.standard_normal((150, 150)))[..., None], 3, -1).astype(np.float32)
    grey[70:80, 70:80] = 0.5
    c = ctx(linear=True)
    out = run("stretch.linked_mtf", grey, c)
    assert abs(float(np.median(out)) - 0.2) < 0.03
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    assert not c.linear


def test_arcsinh_options():
    img = np.repeat(field()[..., None], 3, -1) * np.array([1.0, 0.7, 0.4], np.float32)
    img[48:53, 58:63] = [1.0, 0.95, 0.9]
    base = run("stretch.arcsinh", img, ctx(linear=True))
    same = run("stretch.arcsinh", img, ctx(linear=True), protect_highlights=False, use_rgb_working_space=True,
               estimate_black_point=False)
    assert np.array_equal(base, same)
    prot = run("stretch.arcsinh", img, ctx(linear=True), beta=50.0, protect_highlights=True)
    assert prot.max() <= 1.0
    eq = run("stretch.arcsinh", img, ctx(linear=True), use_rgb_working_space=False, estimate_black_point=True)
    assert eq.max() <= 1.0 and not np.allclose(eq, base)


# ------------------------------------------------------------------ curves / colour
def test_curves_default_and_identity():
    img = patches() + np.random.default_rng(6).random((60, 300, 3)).astype(np.float32) * 0.1
    pts = [[0, 0], [0.15, 0.14], [0.5, 0.52], [1, 1]]
    assert np.array_equal(run("contrast.curves", img), enhance.apply_curve(img, [tuple(p) for p in pts]))
    for ch in ["rgb", "r", "g", "b", "l", "a", "b_star", "c", "s", "h"]:
        out = run("contrast.curves", img, points=[[0, 0], [1, 1]], channel=ch)
        assert np.abs(out - img).max() < 2e-3, ch


def test_curves_channels_act():
    img = patches()
    r = run("contrast.curves", img, points=[[0, 0], [0.5, 0.7], [1, 1]], channel="r")
    assert np.array_equal(r[..., 1:], img[..., 1:]) and r[..., 0].mean() > img[..., 0].mean()
    s = run("contrast.curves", img, points=[[0, 0], [0.5, 0.25], [1, 0.5]], channel="s")
    sat = lambda a: (a.max(-1) - a.min(-1)).mean()  # noqa: E731
    assert sat(s[SL[0]]) < 0.7 * sat(img[SL[0]])


def test_selective_color_moves_only_chosen_hue():
    img = patches()
    out = run("color.selective", img, color="red", saturation=-0.6, hue_shift=20)
    assert np.abs(out[SL[0]] - img[SL[0]]).max() > 0.1
    for k in (1, 2, 4):
        assert np.abs(out[SL[k]] - img[SL[k]]).max() < 1e-6
    assert np.abs(out[0:5] - img[0:5]).max() < 1e-6


def test_linear_fit_matches_channels():
    rng = np.random.default_rng(7)
    g = (0.05 + 0.3 * rng.random((100, 100))).astype(np.float32)
    img = np.stack([0.6 * g + 0.02, g, 1.3 * g - 0.01], -1).astype(np.float32)
    img[::17, ::13, 0] = 0.9     # outliers (stars of another colour)
    out = run("color.linear_fit", img, ctx(linear=True))
    good = img[..., 0] < 0.85
    assert np.abs(out[..., 0] - g)[good].max() < 1e-3 and np.abs(out[..., 2] - g).max() < 1e-3


# ------------------------------------------------------------------ HDR, heal, geometry
def test_hdr_composition_restores_core(tmp_path):
    rng = np.random.default_rng(8)
    yy, xx = np.mgrid[:160, :200]
    scene = field(160, 200, stars=[(30 + 37 * i % 150, 20 + 53 * i % 130, 0.4) for i in range(12)], bg=0.05)
    scene = scene + 3.0 * np.exp(-((xx - 100) ** 2 + (yy - 80) ** 2) / (2 * 12 ** 2))   # bright core, peak ~3
    long_ = np.clip(scene, 0, 1)[..., None].astype(np.float32)
    short = np.clip(scene / 8 + rng.normal(0, 0.0005, scene.shape), 0, 1)[..., None].astype(np.float32)
    p = tmp_path / "short.fits"
    io.save_fits(short, p)
    c = ctx(linear=True)
    out = run("tools.hdr_composition", long_, c, images=[str(p)])
    ref = scene / scene.max()
    core = np.s_[70:90, 90:110]
    assert np.abs(out[..., 0][core] - ref[core]).max() < 0.03 * ref[core].max()
    assert "hdr_mask_1" in c.layers and c.layers["hdr_mask_1"][80, 100] > 0.99


def test_hdr_registers_shifted_image(tmp_path):
    from astroforge.plugins.workflow import align_to
    rng = np.random.default_rng(11)
    stars = list(zip(rng.uniform(10, 230, 40), rng.uniform(10, 170, 40), rng.uniform(0.2, 0.6, 40)))
    ref = field(180, 240, stars=stars)[..., None]
    moved = field(180, 240, stars=[(x + 3.4, y - 2.2, a) for x, y, a in stars])[..., None]
    back = align_to(moved, ref, log=lambda m: None)
    inner = np.s_[10:-10, 10:-10]
    assert np.abs(back - ref)[inner].max() < 0.08 < 0.3 < np.abs(moved - ref)[inner].max()


def test_heal_removes_dust_donut():
    rng = np.random.default_rng(9)
    yy, xx = np.mgrid[:120, :160]
    clean = (0.2 + 0.0005 * xx + rng.normal(0, 0.005, xx.shape)).astype(np.float32)
    d = np.hypot(xx - 80, yy - 60)
    donut = clean * (1 - 0.15 * ((d > 5) & (d < 11)))
    img = np.repeat(donut[..., None], 3, -1).astype(np.float32)
    out = run("cosmetic.heal", img, spots=[[80, 60, 12]])
    region = d < 11
    err0 = np.abs(img[..., 0][region].mean() - clean[region].mean())
    err1 = np.abs(out[..., 0][region].mean() - clean[region].mean())
    assert err1 < 0.15 * err0
    assert np.std(out[..., 0][region]) > 0.5 * np.std(clean[region])     # noise matched, not a flat blob
    assert np.array_equal(out[:, :40], img[:, :40])
    pl = plugins.get("cosmetic.heal")
    assert pl.scaled({"spots": [[80, 60, 12]]}, 0.5)["spots"] == [[40, 30, 6]]


def test_rotate_and_crop():
    img = np.random.default_rng(10).random((40, 60, 3)).astype(np.float32)
    assert np.array_equal(run("geometry.rotate", img, angle=90), np.rot90(img))
    assert np.array_equal(run("geometry.rotate", img, angle=180, flip="horizontal"), np.rot90(img, 2)[:, ::-1])
    assert np.array_equal(run("geometry.rotate", img, flip="vertical"), img[::-1])
    r = run("geometry.rotate", np.full((100, 150, 3), 0.5, np.float32), angle=17)
    assert r.min() > 0.45 and r.shape[0] < 100
    assert run("geometry.rotate", img, angle=17, crop=False).shape[0] > 40
    c = run("crop.manual", img, x0=5, y0=10, x1=25, y1=30)
    assert np.array_equal(c, img[10:30, 5:25])
    pl = plugins.get("crop.manual")
    assert pl.scaled({"x0": 10, "y0": 4, "x1": 30, "y1": None}, 0.5) == {"x0": 5, "y0": 2, "x1": 15, "y1": None}


def test_pipeline_step_mask_limits_the_process():
    from astroforge import pipeline
    from astroforge.plugins.base import Context
    img = np.full((40, 60, 3), 0.1, np.float32)
    img[:, 30:] = 0.8                                     # bright half = "nebula"
    step = {"plugin": "tools.pixelmath", "params": {"expression": "T*0.5"},
            "mask": {"type": "range", "lower": 0.5, "upper": 1.0, "lightness": False}}
    out = pipeline.run_steps(img, [step], Context(log=lambda m: None, linear=False))
    assert np.allclose(out[:, :20], 0.1) and np.allclose(out[:, 40:], 0.4, atol=0.02)
    step["mask"]["invert"] = True
    out = pipeline.run_steps(img, [step], Context(log=lambda m: None, linear=False))
    assert np.allclose(out[:, :20], 0.05, atol=0.01) and np.allclose(out[:, 40:], 0.8, atol=0.02)


def test_step_on_star_layer_and_metadata(tmp_path):
    from astropy.io import fits
    from astroforge import io, pipeline
    from astroforge.plugins.base import Context
    img = np.full((30, 30, 3), 0.2, np.float32)
    ctx = Context(log=lambda m: None, linear=False)
    ctx.layers.update(stars=np.full((30, 30, 3), 0.1, np.float32), stars_mode="screen")
    step = {"plugin": "tools.pixelmath", "params": {"expression": "T*2"}, "target": "stars"}
    out = pipeline.run_steps(img, [step], ctx)
    assert np.allclose(out, img) and np.allclose(ctx.layers["stars"], 0.2)
    msgs = []
    out = pipeline.run_steps(img, [step], Context(log=msgs.append, linear=False))   # no layer: skipped
    assert np.allclose(out, img) and "no star layer" in msgs[0]
    io.save_fits(img, tmp_path / "m.fits", None, {"observer": "Nicola", "COPYRIGHT": "All rights reserved",
                                                  "comment": "Askar 180\n56 x 120 s"})
    h = fits.getheader(tmp_path / "m.fits")
    assert h["OBSERVER"] == "Nicola" and h["COPYRIGHT"] == "All rights reserved" and "Askar 180" in str(h["COMMENT"])
