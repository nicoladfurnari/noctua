"""Quality tests for DBE-like background, Moffat PSF + RL-TV deconvolution and colour-preserving GHS."""
import cv2
import numpy as np

from astroforge import background, deconv, plugins
from astroforge.plugins import background_dbe  # noqa: F401  (registers background.dbe)
from astroforge.stats import luminance

CTX = plugins.Context(log=lambda m: None)


def run(pid, img, ctx=CTX, **params):
    pl = plugins.get(pid)
    return pl.run(img, ctx, **pl.resolve(params))


def moffat(shape, cx, cy, fwhm, beta, amp, ell=0.0, angle=0.0):
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float64)
    a = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    dx, dy = xx - cx, yy - cy
    c, s = np.cos(angle), np.sin(angle)
    u, v = dx * c + dy * s, -dx * s + dy * c
    q = (u / a) ** 2 + (v / (a * (1 - ell))) ** 2
    return amp * (1 + q) ** -beta


def star_field(h, w, n, fwhm, beta, rng, amp=(0.05, 0.5), ell=0.0, angle=0.0):
    img = np.zeros((h, w))
    xs, ys = rng.uniform(15, w - 15, n), rng.uniform(15, h - 15, n)
    for x, y, a in zip(xs, ys, rng.uniform(*amp, n)):
        x0, y0 = int(x) - 15, int(y) - 15
        img[y0:y0 + 31, x0:x0 + 31] += moffat((31, 31), x - x0, y - y0, fwhm, beta, a, ell, angle)
    return img, xs, ys


def sky_with_gradient(rng, h=400, w=600):
    yy, xx = np.mgrid[:h, :w] / max(h, w)
    grad = 0.02 + 0.03 * xx + 0.015 * yy ** 2 + 0.01 * np.sin(2.5 * xx + 1)
    stars, *_ = star_field(h, w, 150, 3.0, 3.0, rng)
    blob = 0.04 * np.exp(-(((xx - 0.5) * 600 / 70) ** 2 + ((yy - 0.33) * 600 / 50) ** 2) / 2)
    tint = np.array([1.0, 0.9, 1.1])
    img = (0.05 + grad + stars + blob)[..., None] * tint + rng.normal(0, 0.002, (h, w, 3))
    return img.astype(np.float32), grad, blob


def test_dbe_removes_gradient():
    rng = np.random.default_rng(1)
    img, grad, blob = sky_with_gradient(rng)
    out = run("background.dbe", img, ctx := plugins.Context(log=lambda m: None))
    assert ctx.layers["background_model"].shape == img.shape
    def blocks(a, b=20):
        h, w = a.shape[0] // b * b, a.shape[1] // b * b
        return np.median(luminance(a)[:h, :w].reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3)
                         .reshape(h // b, w // b, -1), axis=-1)
    flat = blocks(blob[..., None].repeat(3, -1)) < 3e-4
    before, after = blocks(img)[flat], blocks(out)[flat]
    assert np.subtract(*np.percentile(after, [95, 5])) < 0.1 * np.ptp(before)
    err = luminance(ctx.layers["background_model"]) - (0.05 + grad) * luminance(np.float32([[[1.0, 0.9, 1.1]]]))
    err = np.abs(err - np.median(err))
    assert err[blob < 3e-4].max() < 0.05 * np.ptp(grad) and err.max() < 0.006
    # the nebula blob is kept (not absorbed by the model)
    peak = luminance(out)[int(0.33 * 600) - 3:int(0.33 * 600) + 4, 297:304].mean() - np.median(after)
    assert peak > 0.6 * 0.04


def test_dbe_division_and_poly_unchanged():
    rng = np.random.default_rng(2)
    img, _, _ = sky_with_gradient(rng, 200, 300)
    out = run("background.dbe", img, correction="division")
    assert out.shape == img.shape and np.isfinite(out).all()
    ref, _, _ = background.extract(img, degree=2, tiles=40, reject_sigma=1.5)
    assert np.array_equal(run("background.poly", img), ref)


def test_dbe_user_samples_honoured():
    rng = np.random.default_rng(3)
    img, _, _ = sky_with_gradient(rng, 200, 300)
    samples = [[20, 20], [280, 20], [20, 180], [280, 180], [150, 100]]
    _, model, pts = background.extract_rbf(img, samples=samples, smoothing=0.0)
    assert np.allclose(pts, samples)
    for x, y in samples:
        box = img[y - 12:y + 13, x - 12:x + 13]
        assert np.allclose(model[y, x], np.median(box.reshape(-1, 3), axis=0), atol=0.004)
    out = run("background.dbe", img, samples=samples)
    assert out.shape == img.shape


def test_estimate_psf_moffat():
    rng = np.random.default_rng(4)
    for fwhm, beta in ((3.0, 2.5), (5.0, 4.0)):
        field, *_ = star_field(500, 500, 60, fwhm, beta, rng, amp=(0.1, 0.7), ell=0.15, angle=0.6)
        img = 0.05 + field + rng.normal(0, 0.001, field.shape)
        psf = deconv.estimate_psf(img.astype(np.float32))
        assert abs(psf["fwhm_major"] - fwhm) / fwhm < 0.1, psf
        assert abs(psf["fwhm"] - fwhm * np.sqrt(0.85)) / fwhm < 0.1, psf
        assert abs((psf["angle"] - 0.6 + np.pi / 2) % np.pi - np.pi / 2) < 0.2, psf
        assert abs(psf["ell"] - 0.15) < 0.07, psf
        assert psf["n"] >= 10


def test_rl_tv_sharpens_without_ringing():
    rng = np.random.default_rng(5)
    sharp, xs, ys = star_field(300, 300, 40, 1.5, 4.0, rng, amp=(0.2, 0.6))
    bg = 0.05
    kernel = deconv.moffat_kernel(3.5, 3.0)
    blurred = cv2.filter2D(bg + sharp, -1, kernel, borderType=cv2.BORDER_REFLECT)
    obs = (blurred + rng.normal(0, 0.0005, blurred.shape)).astype(np.float32)
    psf = deconv.estimate_psf(obs)
    assert abs(psf["fwhm"] - 3.9) < 0.6, psf
    out = deconv.richardson_lucy(obs, deconv.moffat_kernel(psf["fwhm"], psf["beta"]), 30, tv_weight=0.002,
                                 dering=1.0)
    pk = [(out[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].max() - bg) /
          (obs[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].max() - bg) for x, y in zip(xs, ys)]
    assert np.median(pk) > 1.5
    noise = 0.0005
    assert out.min() > bg - 6 * noise
    # plugin, colour image, luminance mode and per-channel mode
    col = np.repeat(obs[..., None], 3, -1) * np.array([1.0, 0.8, 0.6], np.float32)
    for ch in ("luminance", "rgb"):
        res = run("sharpen.deconvolution", col, channels=ch, iterations=20, amount=1.0)
        assert res.shape == col.shape and res.dtype == np.float32
        assert res.max() > col.max() * 1.2
        assert res.min() >= 0.6 * bg - 6 * noise
    res = run("sharpen.deconvolution", obs[..., None], tiles=2, iterations=10, psf="gaussian")
    assert res.shape == (300, 300, 1)


def test_ghs_luminance_preserves_star_colour():
    h = w = 64
    star = moffat((h, w), 32, 32, 6.0, 2.5, 0.6)
    rgb = np.array([1.0, 0.55, 0.3])
    img = (0.01 + star[..., None] * rgb).astype(np.float32)
    out = run("stretch.ghs", img, D=6.0, b=2.0, SP=0.01)
    assert out.max() <= 1.0 and out.min() >= 0
    wing = (star > 0.002) & (out.max(-1) < 0.98)
    assert wing.sum() > 50
    ratio = (out[wing] / out[wing][:, :1])
    ref = (img[wing] / img[wing][:, :1])
    assert np.abs(ratio - ref).max() < 0.02
    core = out[32, 32]
    assert core[0] >= core[1] >= core[2]                 # hue order kept even where desaturated
    assert np.isclose(luminance(core[None, None])[0, 0], luminance(out[31:34, 31:34]).max(), atol=0.05)
    per = run("stretch.ghs", img, D=6.0, b=2.0, SP=0.01, color="rgb")
    assert per.shape == img.shape and per.max() <= 1.0
