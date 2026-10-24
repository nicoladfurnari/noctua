"""Tests for the processes of the Trifid video workflow (star stretch/size, magenta, USM, NBN, gradient fit,
GHS parameters, text box)."""
import numpy as np

from astroforge import plugins, stretch
from astroforge.plugins import stretch_extra, video_workflow  # noqa: F401  (registers the plugins)
from astroforge.stats import luminance


def ctx(**kw):
    return plugins.Context(log=lambda m: None, **kw)


def run(pid, img, c=None, **params):
    pl = plugins.get(pid)
    return pl.run(img, c or ctx(linear=False), **pl.resolve(params))


def star_layer(h=64, w=64, fwhm=4.0, amp=0.3, rgb=(1.0, 0.7, 0.4)):
    yy, xx = np.mgrid[:h, :w]
    g = amp * np.exp(-((xx - w // 2) ** 2 + (yy - h // 2) ** 2) / (2 * (fwhm / 2.355) ** 2))
    return (g[..., None] * np.array(rgb)).astype(np.float32)


def fwhm_of(layer):
    row = layer.max(-1)[layer.shape[0] // 2]
    return int((row >= row.max() / 2).sum())


# ------------------------------------------------------------------ stars
def test_star_stretch_on_layer_brightens_and_saturates():
    st = star_layer()
    img = np.full((64, 64, 3), 0.2, np.float32)
    c = ctx(linear=False, layers={"stars": st, "stars_mode": "add"})
    out = run("stars.stretch", img, c, reduce=0.0)
    assert out is img
    new = c.layers["stars"]
    assert c.layers["stars_mode"] == "screen"
    assert new.dtype == np.float32 and new.min() >= 0 and new.max() <= 1
    assert new.max() > 2 * st.max()
    wing = st.max(-1) > 0.01
    chroma = lambda a: (a.max(-1) - a.min(-1)) / np.maximum(a.max(-1), 1e-6)  # noqa: E731
    assert chroma(new)[wing].mean() > chroma(st)[wing].mean() * 1.05
    red = run("stars.stretch", img, ctx(layers={"stars": st}), reduce=0.0, color_boost=0.0)
    assert np.isclose(red.mean(), img.mean())
    # without a layer: the image itself is stretched
    c2 = ctx(linear=True)
    assert run("stars.stretch", st, c2).max() > st.max() and not c2.linear


def test_star_size_changes_fwhm_direction():
    st = star_layer(fwhm=10.0, amp=0.8)
    w0 = fwhm_of(st)
    c = ctx(layers={"stars": st.copy()})
    run("stars.size", np.zeros_like(st), c, amount=0.8)
    big = c.layers["stars"]
    c = ctx(layers={"stars": st.copy()})
    run("stars.size", np.zeros_like(st), c, amount=-0.8)
    small = c.layers["stars"]
    assert fwhm_of(big) > w0 > fwhm_of(small)
    assert big.max() <= 1 and small.min() >= 0
    ratio = big[32, 30] / big[32, 30, 0]
    assert np.allclose(ratio, [1.0, 0.7, 0.4], atol=1e-3)     # colours kept


# ------------------------------------------------------------------ colour
def test_fix_magenta_neutralises_magenta_only():
    img = np.full((40, 80, 3), 0.1, np.float32)
    img[10:30, 5:25] = (0.9, 0.4, 0.85)       # magenta star
    img[10:30, 45:65] = (0.3, 0.7, 0.5)       # green/teal nebula
    out = run("color.fix_magenta", img)
    mag, grn = out[20, 15], out[20, 55]
    assert mag[1] > 0.8 and (mag.max() - mag.min()) < (0.9 - 0.4) / 3
    assert np.allclose(grn, img[20, 55], atol=1e-6)
    assert np.allclose(out[0, 0], img[0, 0], atol=1e-6)
    plain = run("color.fix_magenta", img, invert=False)
    assert plain[20, 55, 1] < img[20, 55, 1]                  # plain SCNR green instead


def test_nbn_normalises_oiii_and_keeps_range():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:120, :120]
    neb = np.exp(-((xx - 60) ** 2 + (yy - 60) ** 2) / (2 * 25 ** 2)).astype(np.float32)
    H = 0.1 + 0.7 * neb
    O = 0.05 + 0.12 * neb
    img = np.clip(np.stack([H, O, O], -1) + rng.normal(0, 0.003, (120, 120, 3)), 0, 1).astype(np.float32)
    out = run("color.nb_normalization", img, mode="HOO", lightness="off")
    assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1
    sig = lambda a, c: a[60, 60, c] - np.median(a[:10, :10, c])  # noqa: E731
    before, after = sig(img, 1) / sig(img, 0), sig(out, 1) / sig(out, 0)
    assert before < 0.2 and 0.8 < after < 1.25
    lit = run("color.nb_normalization", img, mode="HOO", highlight_reduction=1.0, boost=3.0)
    assert lit.min() >= 0 and lit.max() <= 1
    assert abs(luminance(lit).mean() - luminance(img).mean()) < 0.05
    same = run("color.nb_normalization", img, blend=0.0)
    assert np.allclose(same, img, atol=1e-6)


def test_gradient_fit_removes_blue_gradient_keeps_grey():
    rng = np.random.default_rng(1)
    h, w = 200, 300
    xx = np.mgrid[:h, :w][1] / w
    grey = np.repeat((0.08 + 0.04 * xx)[..., None], 3, -1).astype(np.float32)   # brightness gradient only
    for y, x in rng.integers(10, 190, (40, 2)):                                  # a few stars
        grey[y - 1:y + 2, x - 1:x + 2] += 0.5
    grey = np.clip(grey + rng.normal(0, 0.002, grey.shape), 0, 1).astype(np.float32)
    out = run("color.gradient_fit", grey)
    assert np.abs(out - grey).max() < 2e-3
    blue = grey.copy()
    blue[..., 2] += 0.06 * xx
    blue[..., 0] += 0.01 * xx
    out = run("color.gradient_fit", blue)
    slope = lambda a, f: np.polyfit(np.arange(w), np.median(f(a[..., 2], a[..., 1]), 0), 1)[0]  # noqa: E731
    sub, div = np.subtract, np.divide
    assert abs(slope(out, sub)) < 0.2 * abs(slope(blue, sub))
    assert abs(slope(out, div)) < 0.5 * abs(slope(blue, div))
    unclipped = blue.max(-1) < 0.95
    assert np.abs(luminance(out) - luminance(blue))[unclipped].max() < 1e-4      # luminance kept
    for m in ("ratio",):
        r = run("color.gradient_fit", blue, method=m, strength="high")
        assert abs(slope(r, div)) < 0.2 * abs(slope(blue, div)) and r.max() <= 1


# ------------------------------------------------------------------ sharpen
def test_unsharp_increases_local_contrast():
    rng = np.random.default_rng(2)
    import cv2
    base = cv2.GaussianBlur(rng.random((100, 100)).astype(np.float32), (0, 0), 2)
    img = np.repeat((0.2 + 0.7 * (base - base.min()) / np.ptp(base))[..., None], 3, -1).astype(np.float32)
    hp = lambda a: (a - cv2.GaussianBlur(a, (0, 0), 3)).std()  # noqa: E731
    for lum in (True, False):
        out = run("sharpen.unsharp", img, sigma=2.0, amount=0.6, luminance=lum)
        assert out.max() <= 1 and out.min() >= 0 and out.dtype == np.float32
        assert hp(out[..., 1]) > 1.5 * hp(img[..., 1])
    # dark deringing: a bright spot on a flat field makes no dark ring
    spot = np.full((41, 41, 1), 0.3, np.float32)
    spot[18:23, 18:23] = 0.9
    ring = run("sharpen.unsharp", spot, sigma=3.0, amount=0.7, dering_dark=0.0).min()
    clean = run("sharpen.unsharp", spot, sigma=3.0, amount=0.7, dering_dark=1.0).min()
    assert ring < 0.25 and clean >= 0.3 - 1e-6


# ------------------------------------------------------------------ GHS
def test_ghs_identity_and_backward_compatible():
    x = np.linspace(0, 1, 2001, dtype=np.float32)
    assert np.allclose(stretch.ghs(x, 0.0, 3.0, 0.2), x)
    for b in (-3.0, -1.0, -0.5, 0.0, 1.0, 4.0):
        D = np.expm1(6.0)
        assert np.allclose(stretch.ghs(x, D, b, 0.1), stretch_extra._ghs(x, D, b, 0.1), atol=2e-5)
    img = (np.random.default_rng(3).random((20, 20, 3)) * 0.3).astype(np.float32)
    old = run("stretch.ghs", img, D=5.0, b=2.0, SP=0.05)
    lum = luminance(img)
    ref = stretch_extra._preserve_color(img, lum, stretch_extra._ghs(np.clip(lum, 0, 1), np.expm1(5.0), 2.0, 0.05))
    assert np.allclose(old, ref, atol=1e-5)
    assert np.allclose(run("stretch.ghs", img, D=0.0, color="rgb"), img, atol=1e-6)


def test_ghs_monotonic_continuous_and_inverse():
    x = np.linspace(0, 1, 200001)
    LP, SP, HP = 0.05, 0.2, 0.7
    for b in (-5.0, -1.0, -0.3, 0.0, 2.0, 12.0):
        y = stretch.ghs(x, np.expm1(5.0), b, SP, LP, HP).astype(np.float64)
        assert y[0] == 0 and abs(y[-1] - 1) < 1e-6
        assert np.all(np.diff(y) >= -1e-7)
        assert np.abs(np.diff(y)).max() < 0.01             # no jumps at LP / SP / HP
        for p in (LP, SP, HP):                               # continuous slope at the junctions
            i = int(round(p * (len(x) - 1)))
            dl, dr = (y[i] - y[i - 5]) / 5, (y[i + 5] - y[i]) / 5
            assert abs(dl - dr) <= 0.05 * max(dl, dr) + 1e-7
        lin_lo = y[: int(LP * (len(x) - 1))]
        assert np.allclose(np.diff(lin_lo, 2), 0, atol=1e-6)  # linear shadows
        inv = stretch.ghs(y.astype(np.float32), np.expm1(5.0), b, SP, LP, HP, inverse=True)
        ok = y < 1 - 1e-4          # where the curve is not flat to float precision (b = 0 saturates)
        assert np.abs(inv - x)[ok].max() < 2e-3
    img = np.random.default_rng(4).random((16, 16, 3)).astype(np.float32)
    c = ctx(linear=True)
    lin = run("stretch.ghs", img, c, mode="lin", BP=0.2, color="rgb")
    assert np.allclose(lin, np.clip((img - 0.2) / 0.8, 0, 1), atol=1e-6) and c.linear
    back = run("stretch.ghs", run("stretch.ghs", img, D=4.0, b=1.0, SP=0.3, color="rgb"), mode="inverse",
               D=4.0, b=1.0, SP=0.3, color="rgb")
    assert np.abs(back - img).max() < 3e-3
    auto = run("stretch.ghs", img, D=3.0, SP="auto", LP=0.1, HP=0.9)
    assert auto.min() >= 0 and auto.max() <= 1


# ------------------------------------------------------------------ text
def test_annotate_text_draws_in_the_corner_only():
    img = np.full((200, 300, 3), 0.25, np.float32)
    out = run("annotate.text", img, text="M20 Trifid\\nAskar 180", position="bottom-right")
    diff = np.abs(out - img).max(-1) > 1e-6
    ys, xs = np.nonzero(diff)
    assert diff.sum() > 200
    assert ys.min() > 100 and xs.min() > 150                 # bottom-right quadrant only
    assert out.max() <= 1 and out.dtype == np.float32
    tl = np.abs(run("annotate.text", img, text="x", position="top-left", box=False) - img).max(-1) > 1e-6
    ys, xs = np.nonzero(tl)
    assert ys.max() < 50 and xs.max() < 60
    assert run("annotate.text", img, text="") is img
    c = ctx()
    assert run("annotate.text", img, c, text="a", apply=False) is img and "text_image" in c.layers
    mono = run("annotate.text", img[..., :1], text="abc", position=[0.4, 0.4])
    assert mono.shape == (200, 300, 1) and (np.abs(mono - img[..., :1]) > 1e-6).any()
