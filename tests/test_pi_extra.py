"""More PixInsight processes (astroforge.plugins.pi_extra) on synthetic images."""
import numpy as np
import pytest
from astropy.io import fits

from astroforge import plugins
from astroforge.stats import luminance

RNG = np.random.default_rng(3)
H, W = 96, 128


def ctx(**kw):
    return plugins.Context(log=lambda m: None, **kw)


def run(pid, img, c=None, **params):
    pl = plugins.get(pid)
    return pl.run(img, c or ctx(), **pl.resolve(params))


def linear_field():
    img = 0.01 + RNG.normal(0, 0.0005, (H, W, 3))
    yy, xx = np.mgrid[:H, :W]
    img += 0.02 * np.exp(-((xx - 40) ** 2 + (yy - 50) ** 2) / 600)[..., None] * np.array([1, 0.6, 0.4])
    for x, y in ((90, 20), (20, 80), (100, 70)):
        img += 0.9 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 3)[..., None] * np.array([0.8, 0.9, 1.0])
    return np.clip(img, 0, 1).astype(np.float32)


def test_masked_stretch_hits_target_and_protects_stars():
    img = linear_field()
    c = ctx()
    out = run("stretch.masked", img, c, target=0.15, iterations=40)
    assert abs(float(np.median(luminance(out))) - 0.15) < 0.03
    assert not c.linear
    plain = run("stretch.statistical", img, target=0.15)          # one global stretch for comparison
    blue = lambda o: float(o[20, 90, 2] - o[20, 90, 0])         # noqa: E731
    assert blue(out) > 0.1 and blue(out) > 3 * blue(plain)       # the star core keeps its color, not white


def test_adaptive_stretch_brightens_signal_not_noise():
    img = linear_field()
    out = run("stretch.adaptive", img)
    neb, sky = luminance(out)[50, 40], float(np.median(luminance(out)[:, 110:]))
    assert neb > sky + 0.05 and sky < 0.5


@pytest.mark.parametrize("mode", ["average", "median", "min", "max"])
def test_integer_resample_down_up(mode):
    img = RNG.random((H, W, 3)).astype(np.float32)
    c = ctx(layers={"stars": img.copy()})
    down = run("geometry.integer_resample", img, c, factor=4, mode=mode)
    assert down.shape == (H // 4, W // 4, 3) and c.layers["stars"].shape == down.shape
    if mode == "average":
        assert np.isclose(down[0, 0, 0], img[:4, :4, 0].mean(), atol=1e-6)
    up = run("geometry.integer_resample", down, factor=2, direction="up")
    assert up.shape == (H // 2, W // 2, 3) and np.all(up[0, 0] == up[1, 1])


def test_larson_sekanina_shows_a_jet_and_finds_the_nucleus():
    yy, xx = np.mgrid[:H, :W]
    r, phi = np.hypot(xx - 64, yy - 48), np.arctan2(yy - 48, xx - 64)
    coma = 0.6 * np.exp(-r / 12) * (1 + 0.15 * np.exp(-(phi - 0.5) ** 2 / 0.01))      # faint jet at 0.5 rad
    img = np.repeat(coma[..., None], 3, -1).astype(np.float32)
    f = run("sharpen.larson_sekanina", img, angle=10, output="filter", amount=2)
    on = f[int(48 + 15 * np.sin(0.5)), int(64 + 15 * np.cos(0.5)), 0]
    off = f[int(48 + 15 * np.sin(2.5)), int(64 + 15 * np.cos(2.5)), 0]
    assert on > off + 0.05


def test_convolution_kinds_keep_flux():
    img = np.zeros((H, W, 1), np.float32)
    img[48, 64] = 1
    for kind in ("gaussian", "box", "moffat"):
        out = run("tools.convolution", img, kind=kind, sigma=2)
        assert out.shape == img.shape and abs(float(out.sum()) - 1) < 0.02 and out.max() < 0.2


def test_invert_binarize_grayscale_extract():
    img = RNG.random((H, W, 3)).astype(np.float32)
    assert np.allclose(run("tools.invert", run("tools.invert", img)), img, atol=1e-6)
    b = run("tools.binarize", img, threshold=0.5)
    assert set(np.unique(b)) <= {0.0, 1.0} and np.all(b[..., 0] == b[..., 1])
    g = run("tools.grayscale", img)
    assert g.shape == (H, W, 1) and np.allclose(g[..., 0], luminance(img), atol=1e-6)
    assert run("tools.grayscale", img, keep_rgb=True).shape == (H, W, 3)
    c = ctx()
    r = run("tools.channel_extract", img, c, channel="R", save_layers=True)
    assert np.allclose(r[..., 0], img[..., 0]) and "channel_b*" in c.layers
    for ch in ("L*", "a*", "b*", "H", "S", "V"):
        v = run("tools.channel_extract", img, channel=ch)
        assert v.shape == (H, W, 1) and 0 <= v.min() and v.max() <= 1


def _save(path, a):
    fits.PrimaryHDU(np.moveaxis(a, -1, 0) if a.shape[-1] == 3 else a[..., 0]).writeto(path, overwrite=True)
    return str(path)


def test_lrgb_takes_detail_from_l_and_color_from_rgb(tmp_path):
    yy, xx = np.mgrid[:H, :W]
    detail = (0.3 + 0.3 * (np.sin(xx / 2) > 0)).astype(np.float32)        # sharp stripes only in L
    rgb = np.stack([np.full((H, W), 0.5), np.full((H, W), 0.3), np.full((H, W), 0.2)], -1).astype(np.float32)
    path = _save(tmp_path / "L.fits", detail[..., None])
    out = run("color.lrgb", rgb, luminance=path, match=False)
    assert np.std(luminance(out)) > 0.05                                   # stripes arrived
    assert np.mean(out[..., 0]) > np.mean(out[..., 2])                     # still orange
    with pytest.raises(ValueError):
        run("color.lrgb", rgb)


def test_channel_combination(tmp_path):
    base = RNG.random((H, W, 3)).astype(np.float32)
    ha = RNG.random((H, W, 1)).astype(np.float32)
    out = run("color.channel_combine", base, R=_save(tmp_path / "ha.fits", ha))
    assert np.allclose(out[..., 0], ha[..., 0], atol=1e-6) and np.allclose(out[..., 1:], base[..., 1:])


def test_all_new_plugins_listed():
    ids = {"stretch.masked", "stretch.adaptive", "geometry.integer_resample", "sharpen.larson_sekanina",
           "tools.convolution", "tools.invert", "tools.binarize", "tools.grayscale", "tools.channel_extract",
           "color.lrgb", "color.channel_combine"}
    assert ids <= set(plugins.REGISTRY)


def test_stf_linked_keeps_tint_separate_neutralizes():
    img = linear_field() * np.array([1.0, 0.8, 0.5], np.float32)          # warm background cast
    lk = run("stretch.stf", img, channels="linked")
    sp = run("stretch.stf", img, channels="separate")
    sky = (slice(0, 30), slice(100, 128))
    med = lambda o: np.median(o[sky].reshape(-1, 3), 0)                     # noqa: E731
    assert med(lk)[0] > med(lk)[2] + 0.05                                   # cast kept
    assert np.ptp(med(sp)) < 0.02 and abs(med(sp).mean() - 0.25) < 0.04    # neutral, at the target
    boosted = run("stretch.stf", img, channels="separate", boost=True)
    assert med(boosted).mean() > med(sp).mean() + 0.05
    mono = run("stretch.stf", img[..., :1], channels="separate")
    assert mono.shape == (H, W, 1)


def _scene(seed=5):
    """RGB field + Ha of the same field from another camera: 0.8x scale, rotated 7 deg, emission only in Ha."""
    import cv2
    rng = np.random.default_rng(seed)
    h, w = 240, 320
    yy, xx = np.mgrid[:h, :w]
    neb = np.exp(-((xx - 160) ** 2 + (yy - 120) ** 2) / 2500.0)
    stars = np.zeros((h, w), np.float32)
    for x, y, f in zip(rng.uniform(10, w - 10, 120), rng.uniform(10, h - 10, 120), rng.uniform(0.05, 0.6, 120)):
        stars += f * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 3.0)
    sky = 0.01
    rgb = np.stack([sky + stars + 0.003 * neb, sky + stars * 0.95, sky + stars * 0.9], -1)
    rgb = (rgb + rng.normal(0, 0.0004, rgb.shape)).astype(np.float32)
    ha_full = sky * 0.2 + stars * 0.07 + 0.05 * neb              # narrow filter: 7% of the star light, strong line
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 7, 0.8)
    ha = cv2.warpAffine(ha_full.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.002)
    ha = (ha + rng.normal(0, 0.0004, ha.shape)).astype(np.float32)
    return rgb, ha[..., None], neb, stars


def test_nbrgb_adds_emission_not_stars_and_registers_another_camera(tmp_path):
    rgb, ha, neb, stars = _scene()
    path = _save(tmp_path / "ha.fits", ha)
    c = ctx(linear=True)
    out = run("color.nb_rgb", rgb, c, ha=path, ha_amount=1.0)
    gain_neb = out[120, 160, 0] - rgb[120, 160, 0]
    sy, sx = np.unravel_index(np.argmax(stars * (neb < 0.05)), stars.shape)    # a bright star off the nebula
    gain_star = out[sy, sx, 0] - rgb[sy, sx, 0]
    assert gain_neb > 0.003 and gain_star < 0.2 * gain_neb                     # emission yes, stars no
    assert np.allclose(out[..., 1], np.clip(rgb[..., 1], 0, 1), atol=1e-6)     # Ha only in red (ha_blue 0)
    more = run("color.nb_rgb", rgb, ctx(), ha=path, ha_amount=2.0)
    assert more[120, 160, 0] - rgb[120, 160, 0] > 1.8 * gain_neb               # the slider scales it
    mx = run("color.nb_rgb", rgb, ctx(), ha=path, mode="max")
    assert mx[120, 160, 0] >= rgb[120, 160, 0]
    with pytest.raises(ValueError):
        run("color.nb_rgb", rgb)


def test_nbrgb_on_finished_stretched_files_responds_to_the_slider(tmp_path):
    # finished files: the starless narrowband, levels matched to the red channel, is blended in: the slider
    # visibly moves the nebula, the sky is not lifted
    rgb, ha, neb, stars = _scene()
    st = lambda a: np.clip(np.clip(a, 0, 1) ** 0.3, 0, 1).astype(np.float32)          # noqa: E731
    rgb_f, path = st(rgb), _save(tmp_path / "ha_f.fits", st(ha))
    sy, sx = np.unravel_index(np.argmax(stars * (neb < 0.05)), stars.shape)
    gains = []
    for mode in ("screen", "nbrgb", "max"):
        for amt in (0.5, 1.0, 2.0):
            out = run("color.nb_rgb", rgb_f, ctx(linear=False), ha=path, ha_amount=amt, mode=mode)
            assert np.all(out >= rgb_f - 1e-6)                                         # never darker
            gains.append(out[120, 160, 0] - rgb_f[120, 160, 0])
            assert abs(out[5, 5, 0] - rgb_f[5, 5, 0]) < 0.02                           # sky corner unchanged
        if mode == "max":         # lighten: only where the narrowband gets brighter than the channel
            assert gains[-3] <= gains[-2] <= gains[-1] and gains[-1] > 0.01, (mode, gains[-3:])
        else:
            assert gains[-3] < gains[-2] < gains[-1] and gains[-2] > 0.03, (mode, gains[-3:])


def test_nbrgb_dual_band_colour_image_still_responds_to_the_slider(tmp_path):
    """Reported by Nicola (Rosetta, dual-band colour + Ha): the red channel already holds the Ha line, so the stars
    and the nebula have the same Ha/R ratio, the continuum subtraction left ~0 and Ha 5 changed nothing."""
    rgb, ha, neb, stars = _scene()
    rgb[..., 0] += 0.02 * neb                                   # the line is in the colour image too
    import cv2
    M = cv2.getRotationMatrix2D((160, 120), 7, 0.8)
    ha = cv2.warpAffine(0.3 * (rgb[..., 0] - 0.01) + 0.002, M, (320, 240), borderValue=0.002)[..., None]
    path = _save(tmp_path / "ha.fits", ha)
    logs = []
    gains = []
    for amt in (1.0, 5.0):
        out = run("color.nb_rgb", rgb, plugins.Context(linear=True, log=logs.append), ha=path, ha_amount=amt)
        gains.append(out[120, 160, 0] - rgb[120, 160, 0])
    assert any("already contains" in m for m in logs)
    assert gains[0] > 0.005 and gains[1] > 3 * gains[0], gains
    sy, sx = np.unravel_index(np.argmax(stars * (neb < 0.05)), stars.shape)    # stars not bloated
    assert out[sy, sx, 0] - rgb[sy, sx, 0] < 0.5 * gains[1]


def test_nbrgb_flatten_removes_the_narrowband_gradient(tmp_path):
    # an Ha file with a bright corner (vignetting / light pollution) must not tint the sky red
    rgb, ha, neb, stars = _scene()
    h, w = ha.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    grad = (0.03 * (xx / w) * (yy / h)).astype(np.float32)[..., None]
    path = _save(tmp_path / "ha_grad.fits", ha + grad)
    flat = run("color.nb_rgb", rgb, ctx(linear=True), ha=path, ha_amount=2.0)
    raw = run("color.nb_rgb", rgb, ctx(linear=True), ha=path, ha_amount=2.0, flatten=False)
    corner = (slice(h - 25, h - 5), slice(w - 25, w - 5))
    assert np.median(raw[corner][..., 0] - rgb[corner][..., 0]) > 3 * np.median(flat[corner][..., 0] - rgb[corner][..., 0]) + 1e-4
    assert flat[120, 160, 0] - rgb[120, 160, 0] > 0.003                       # the nebula is still added


def test_curves_one_curve_per_channel():
    # one Curves step holds a curve per channel (the editor's tabs), applied RGB first then R, G, B...;
    # older profiles (points on one channel) give the same result as before
    rng = np.random.default_rng(2)
    img = rng.uniform(0.1, 0.9, (40, 50, 3)).astype(np.float32)
    up = [[0, 0], [0.5, 0.7], [1, 1]]
    old = run("contrast.curves", img, points=up, channel="r")
    new = run("contrast.curves", img, curves={"r": up})
    assert np.allclose(old, new) and np.allclose(new[..., 1:], img[..., 1:]) and new[..., 0].mean() > img[..., 0].mean() + 0.05
    both = run("contrast.curves", img, curves={"r": up, "b": [[0, 0], [0.5, 0.3], [1, 1]], "g": [[0, 0], [1, 1]]})
    assert both[..., 0].mean() > img[..., 0].mean() + 0.05 and both[..., 2].mean() < img[..., 2].mean() - 0.05
    assert np.allclose(both[..., 1], img[..., 1])                                  # identity curve: untouched
    k_then_r = run("contrast.curves", img, curves={"rgb": up, "r": up})
    r_only = run("contrast.curves", run("contrast.curves", img, curves={"rgb": up}), curves={"r": up})
    assert np.allclose(k_then_r, r_only, atol=1e-6)                               # RGB first, then R
