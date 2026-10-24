"""PSF fit, PSF signal weights, clamped Lanczos, distortion correction and mosaics on synthetic data."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from astroforge.stack import analyze, mosaic, register


def _moffat_field(fwhm, rng, beta=3.0, n=60, H=600, W=800, noise=0.002, amp=(0.05, 0.6), xy=None):
    a = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    img = np.full((H, W), 0.05, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    R = int(5 * fwhm) + 3
    xy = xy if xy is not None else np.column_stack([rng.uniform(40, W - 40, n), rng.uniform(40, H - 40, n),
                                                    rng.uniform(*amp, n)])
    for x, y, A in xy:
        sl = np.s_[max(0, int(y) - R):int(y) + R + 1, max(0, int(x) - R):int(x) + R + 1]
        img[sl] += A * (1 + ((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / a ** 2) ** -beta
    return img + rng.normal(0, noise, img.shape).astype(np.float32), xy


def _gauss(xy, shape, sigma=1.5, amp=None, bg=0.0):
    H, W = shape
    img = np.full(shape, bg, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for k, (x, y) in enumerate(xy):
        x0, y0 = int(round(x)), int(round(y))
        sl = np.s_[max(0, y0 - 8):max(0, y0 + 9), max(0, x0 - 8):max(0, x0 + 9)]
        img[sl] += (1.0 if amp is None else amp[k]) * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * sigma ** 2))
    return img


@pytest.mark.parametrize("fwhm", [3.0, 12.0])
def test_moffat_fwhm(fwhm):
    img, _ = _moffat_field(fwhm, np.random.default_rng(1))
    st = analyze.detect(img)
    q = analyze.quality(img, st)
    assert len(st["psf"]["fwhm"]) >= 20
    assert abs(q.fwhm - fwhm) / fwhm < 0.05
    assert abs(np.median(st["psf"]["beta"]) - 3.0) < 0.5
    assert q.eccentricity < 0.3


def test_psf_weight_orders_by_snr():
    rng = np.random.default_rng(2)
    _, xy = _moffat_field(4.0, rng, n=80)
    qs = []
    for noise, transp in ((0.004, 1.0), (0.002, 1.0), (0.004, 0.6), (0.008, 1.0)):
        img, _ = _moffat_field(4.0, rng, noise=noise, xy=xy * [1, 1, transp])
        qs.append(analyze.quality(img, analyze.detect(img)))
    analyze.score_frames(qs)
    assert all(q.psf_signal > 0 for q in qs)
    snr = [transp / noise for noise, transp in ((0.004, 1.0), (0.002, 1.0), (0.004, 0.6), (0.008, 1.0))]
    assert np.argsort([q.weight for q in qs]).tolist() == np.argsort(snr).tolist()
    assert max(q.weight for q in qs) == pytest.approx(1.0)
    assert isinstance(qs[0].to_dict()["psf_signal"], float)


def test_lanczos_clamping():
    img = np.full((80, 80), 0.1, np.float32)
    yy, xx = np.mgrid[0:80, 0:80]
    img[np.hypot(xx - 40, yy - 40) < 4] = 1.0          # saturated, flat topped star
    M = np.array([[1, 0, 0.47], [0, 1, -0.31]])
    raw = register.warp(img, M, img.shape, clamp=False)
    cl = register.warp(img, M, img.shape)
    assert np.nanmin(raw) < 0.1 - 0.02               # Lanczos undershoot (dark ring)
    assert np.nanmin(cl) >= 0.1 - 1e-6 and np.nanmax(cl) <= 1.0 + 1e-6
    assert np.array_equal(np.isnan(raw), np.isnan(cl))
    assert np.isnan(cl[:, 0]).all() or np.isnan(cl[0]).all()     # NaN outside coverage, as before
    rgb = register.warp(np.dstack([img] * 3), M, img.shape)
    assert rgb.shape == (80, 80, 3) and np.nanmin(rgb) >= 0.1 - 1e-6


def _barrel(p, c, s, k):
    d = p - c
    return c + d * (1 + k * (d ** 2).sum(1, keepdims=True) / s ** 2)


@pytest.mark.parametrize("order", [3, "tps"])
def test_distortion_star_lists(order):
    rng = np.random.default_rng(4)
    W, H = 3000, 2000
    ref = np.column_stack([rng.uniform(0, W, 500), rng.uniform(0, H, 500)])
    th = np.radians(0.7)
    A = np.array([[np.cos(th), -np.sin(th), 12.3], [np.sin(th), np.cos(th), -7.8]])
    c = np.array([W / 2, H / 2])
    src = register._affine(_barrel(ref, c, 1800, 0.0017), register._inv(A)) + rng.normal(0, 0.01, (500, 2))
    keep = rng.random(500) > 0.1                        # some stars missing, some spurious
    src_l = np.vstack([src[keep], np.column_stack([rng.uniform(0, W, 30), rng.uniform(0, H, 30)])])
    M, n, rms = register.match(src_l, ref)
    model = register.fit_distortion(src_l, ref, M, order=order)
    pred = model.source(ref[keep], M)
    err = np.sqrt(np.mean(np.sum((pred - src[keep]) ** 2, 1)))
    assert err < 0.1
    assert model.info["distortion_rms_improvement"] > 0.1
    assert model.info["rms"] < 0.1


def test_distortion_warp_image():
    rng = np.random.default_rng(5)
    H, W = 700, 900
    ref = np.column_stack([rng.uniform(15, W - 15, 160), rng.uniform(15, H - 15, 160)])
    amp = rng.uniform(0.2, 0.8, 160)
    c = np.array([W / 2, H / 2])
    A = np.array([[1, 0, 3.4], [0, 1, -2.2]])
    src_true = register._affine(_barrel(ref, c, 560, 0.005), register._inv(A))
    img = _gauss(src_true, (H, W), amp=amp, bg=0.05) + rng.normal(0, 0.0005, (H, W)).astype(np.float32)
    st = analyze.detect(img)
    src_xy = np.stack([st["x"], st["y"]], 1)
    M, n, rms = register.match(src_xy, ref)
    model = register.fit_distortion(src_xy, ref, M, order=3)
    assert model.info["distortion_rms_improvement"] > 0.1

    def residual(out):
        s2 = analyze.detect(np.nan_to_num(out[..., 0], nan=0.05))
        d, j = cKDTree(np.stack([s2["x"], s2["y"]], 1)).query(ref)
        inner = (ref[:, 0] > 20) & (ref[:, 0] < W - 20) & (ref[:, 1] > 20) & (ref[:, 1] < H - 20) & (d < 3)
        return np.sqrt(np.mean(d[inner] ** 2))

    plain = residual(register.warp(img[..., None], M, (H, W)))
    corr = residual(register.warp(img[..., None], M, (H, W), distortion=model))
    assert corr < 0.1 < plain


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_mosaic_2x2_seamless():
    rng = np.random.default_rng(6)
    SH, SW = 600, 800
    stars = np.column_stack([rng.uniform(0, SW, 500), rng.uniform(0, SH, 500)])
    amp = rng.uniform(0.02, 0.5, 500)

    def scene(ox, oy, shape):
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        neb = 0.08 + 0.04 * np.exp(-(((xx + ox) - 420) ** 2 + ((yy + oy) - 280) ** 2) / (2 * 150 ** 2))
        return neb + _gauss(stars - [ox, oy], shape, amp=amp)

    sigma = 0.002
    origins = [(0.0, 0.0), (319.6, 0.4), (0.3, 239.7), (320.2, 240.5)]
    offsets = [0.0, 0.03, -0.02, 0.05]
    shape = (360, 480)
    panels = [scene(ox, oy, shape) + off + rng.normal(0, sigma, shape).astype(np.float32)
              for (ox, oy), off in zip(origins, offsets)]
    T = mosaic.register_panels(panels)
    for (ox, oy), M in zip(origins, T):
        assert np.allclose(M[:, 2], [ox, oy], atol=0.1) and np.allclose(M[:, :2], np.eye(2), atol=1e-3)
    canvas, CT = mosaic.plan(T, [p.shape for p in panels])
    assert abs(canvas[0] - 601) <= 1 and abs(canvas[1] - 801) <= 1
    out, info = mosaic.blend(panels, CT, canvas, return_info=True)
    assert out.shape == canvas + (1,) and out.dtype == np.float32
    truth = scene(-CT[0][0, 2], -CT[0][1, 2], canvas)
    starry = _gauss(stars + CT[0][:, 2], canvas, amp=amp) > 3 * sigma

    def steps(img):
        r = np.where(starry, np.nan, img[..., 0] - truth)[8:-8, 8:-8]
        worst = 0.0
        for x in (319, 320, 480, 481):     # vertical seams: panel edges inside the canvas
            worst = max(worst, np.nanmax(np.abs(np.nanmean(r[:, x - 8 - 3:x - 8], 1) - np.nanmean(r[:, x - 8:x - 8 + 3], 1))
                                            [: (r.shape[0] // 16) * 16].reshape(-1, 16).mean(1)))
        for y in (239, 240, 360, 361):
            worst = max(worst, np.nanmax(np.abs(np.nanmean(r[y - 8 - 3:y - 8], 0) - np.nanmean(r[y - 8:y - 8 + 3], 0))
                                            [: (r.shape[1] // 16) * 16].reshape(-1, 16).mean(1)))
        blocks = r[: (r.shape[0] // 16) * 16, : (r.shape[1] // 16) * 16]
        bm = np.nanmean(blocks.reshape(blocks.shape[0] // 16, 16, -1, 16), (1, 3))
        return worst, np.nanmax(np.abs(bm))

    seam, level = steps(out)
    assert seam < 3 * sigma and level < 3 * sigma
    assert np.allclose(np.array(info["offset"])[:, 0], -np.array(offsets), atol=0.005)
    seam_raw, level_raw = steps(mosaic.blend(panels, CT, canvas, match_background=False))
    assert level_raw > 3 * sigma


def test_eccentricity_rejection_is_relative_to_the_session():
    def q(ecc):
        return analyze.Quality(stars=1000, fwhm=4.0, eccentricity=ecc, background=0.01, noise=0.001, snr=3000)
    # whole session slightly elongated: nothing rejected; an outlier still is
    qs = [q(e) for e in (0.74, 0.75, 0.76, 0.77, 0.75, 0.76)] + [q(0.9)]
    analyze.score_frames(qs)
    assert [x.accepted for x in qs] == [True] * 6 + [False]
    # round stars: the fixed 0.75 limit still applies
    qs = [q(e) for e in (0.40, 0.42, 0.45, 0.80)]
    analyze.score_frames(qs)
    assert [x.accepted for x in qs] == [True, True, True, False]
