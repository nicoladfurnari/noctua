"""Drizzle extras: distortion models, local normalization, gaussian / lanczos3 kernels."""
import math
import time

import cv2
import numpy as np
import pytest
from scipy.special import erf

from astroforge.stack import localnorm, register
from astroforge.stack.drizzle import Drizzle, kernel_lut, outlier_mask

S = 240                       # square frames
C0 = (S - 1) / 2
KB = 10 / 170 ** 3            # barrel: 10 px at r = 170 px
RNG = np.random.default_rng(11)
SKY = np.stack([RNG.uniform(30, S - 30, 220), RNG.uniform(30, S - 30, 220)], 1)
AMP = RNG.uniform(0.2, 1.0, len(SKY))


def _affine(dx, dy, rot_deg):
    t = math.radians(rot_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, dx], [s, c, dy]])


def _undist(xy):
    """Sensor pixel -> ideal (undistorted) sensor coordinates."""
    d = xy - C0
    return xy + KB * (d ** 2).sum(-1, keepdims=True) * d


def _dist(xy):
    p = xy.copy()
    for _ in range(30):
        p = p - (_undist(p) - xy)
    return p


def _frame_params(n=8, shift=25, rot=1.5):
    r = np.random.default_rng(5)
    return [_affine(r.uniform(-shift, shift), r.uniform(-shift, shift), r.uniform(-rot, rot)) for _ in range(n)]


def _stars_in(M):
    """Sky -> sensor positions of a frame whose ideal sensor -> sky transform is M."""
    return _dist((SKY - M[:, 2]) @ np.linalg.inv(M[:, :2]).T)


def _render(xy, sigma, n=S, step=1.0, bg=0.05):
    """Pixel-integrated gaussian stars on cells of size `step` (centres at (i + 0.5) * step - 0.5), per px^2."""
    e = (np.arange(int(n / step) + 1)) * step - 0.5
    k = 1 / (sigma * math.sqrt(2))
    img = np.full((len(e) - 1, len(e) - 1), bg, np.float64)
    for (sx, sy), a in zip(xy, AMP):
        gx = 0.5 * np.diff(erf((e - sx) * k))
        gy = 0.5 * np.diff(erf((e - sy) * k))
        img += a * np.outer(gy, gx) / step ** 2
    return img.astype(np.float32)


def _bordered(f, m=10):
    f = f.copy()
    f[:m] = f[-m:] = np.nan
    f[:, :m] = f[:, -m:] = np.nan
    return f


def _fwhm(img, scale):
    vals = []
    for sx, sy in SKY:
        cx, cy = sx * scale + (scale - 1) / 2, sy * scale + (scale - 1) / 2
        r = int(3 * scale)
        x0, y0 = int(round(cx)) - r, int(round(cy)) - r
        box = img[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1].astype(np.float64)
        box = np.clip(box - 0.05, 0, None)
        yy, xx = np.mgrid[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
        m = box.sum()
        mx, my = (box * xx).sum() / m, (box * yy).sum() / m
        var = (box * ((xx - mx) ** 2 + (yy - my) ** 2)).sum() / m / 2
        vals.append(2.3548 * math.sqrt(var) / scale)
    return float(np.median(vals))


def _distorted_set(sigma=0.6, **kw):
    frames, models, affs = [], [], []
    r = np.random.default_rng(2)
    for M in _frame_params(**kw):
        src = _stars_in(M)
        frames.append(_render(src, sigma))
        meas = src + r.normal(0, 0.02, src.shape)
        A = np.hstack([meas, np.ones((len(meas), 1))])
        Ma = np.linalg.lstsq(A, SKY, rcond=None)[0].T
        affs.append(Ma)
        models.append(register.fit_distortion(meas, SKY, Ma))
    return frames, affs, models


@pytest.mark.parametrize("kernel", ["gaussian", "lanczos3"])
def test_kernel_lut_rows(kernel):
    for A, pf in ((np.eye(2) * 2, 0.9), (_affine(0, 0, 25)[:, :2] * 3, 0.6), (np.diag([1.0, -1.0]), 1.0)):
        mn, (ny, nx), lut = kernel_lut(A, pf, kernel)
        assert np.allclose(lut.sum(1), 1)
        # centroid of the kernel = pixel centre (bounding-box corner at f -> centre at f - mn)
        j, i = np.mgrid[0:ny, 0:nx]
        cx = (lut * (i.ravel() + 0.5)).sum(1)
        assert abs(cx[0] - (-mn[0])) < 0.02 and abs(cx[16] - (1 - mn[0])) < 0.02


@pytest.mark.parametrize("kernel", ["square", "gaussian", "lanczos3"])
@pytest.mark.parametrize("dist", [False, True])
def test_flux_conservation(kernel, dist):
    frames, affs, models = _distorted_set(shift=3, rot=0.3)
    frames = [_bordered(f, 16) for f in frames[:4]]
    dz = Drizzle((S, S), 2, 1.0 if kernel == "lanczos3" else 0.8, 1, kernel=kernel)
    tot = 0.0
    for f, M, d in zip(frames, affs, models):
        dz.add(f[..., None], M, 0.7, distortion=d if dist else None)
        tot += float(np.nansum(f)) * 0.7
    assert abs(dz.flux.sum() / tot - 1) < 1e-6
    n = sum(np.isfinite(f).sum() for f in frames) * 0.7
    assert abs(dz.weight.sum() / n - 1) < 1e-6
    out, _ = dz.result(smooth=0)
    truth = _render(SKY, 0.6)
    m = 50
    got = out[2 * m:2 * (S - m), 2 * m:2 * (S - m), 0].sum() / 4
    assert abs(got / truth[m:S - m, m:S - m].sum() - 1) < 0.01


def test_distortion_sharper_and_closer_to_truth():
    frames, affs, models = _distorted_set()
    assert all(d.info["distortion_rms_improvement"] > 0.1 for d in models)
    truth = _render(SKY, 0.6, 2 * S, 0.5)[:2 * S, :2 * S]
    res = {}
    for name, use in (("affine", False), ("distortion", True)):
        dz = Drizzle((S, S), 2, 0.7, 1)
        t = time.perf_counter()
        dz.add_many({"img": f[..., None], "M": M, "distortion": d if use else None}
                    for f, M, d in zip(frames, affs, models))
        dt = time.perf_counter() - t
        out = dz.result(smooth=0)[0][..., 0]
        c = slice(70, 2 * S - 70)
        res[name] = (_fwhm(out, 2), float(np.sqrt(np.mean((out[c, c] - truth[c, c]) ** 2))), dt)
    print("drizzle 2x barrel field: " + ", ".join(f"{k} FWHM {v[0]:.3f}px rms {v[1]:.4f} ({v[2]:.2f}s)"
                                                   for k, v in res.items()))
    assert res["distortion"][0] < 0.95 * res["affine"][0]
    assert res["distortion"][1] < 0.8 * res["affine"][1]


def test_distortion_speed_report():
    r = np.random.default_rng(0)
    img = r.random((2000, 3000), np.float32)[..., None]
    M = _affine(0.3, -0.2, 0.05)
    xy = np.stack([r.uniform(0, 3000, 600), r.uniform(0, 2000, 600)], 1)
    c = np.array([1500, 1000])
    d = xy - c
    und = xy + 3e-10 * (d ** 2).sum(1, keepdims=True) * d
    ref = und @ M[:, :2].T + M[:, 2]
    dm = register.fit_distortion(xy, ref, M)
    for kernel in ("square", "gaussian"):
        dz = Drizzle((2000, 3000), 2, 0.9, 1, kernel=kernel)
        dz.add(img, M)
        t = time.perf_counter()
        dz.add(img, M)
        a = time.perf_counter() - t
        t = time.perf_counter()
        dz.add(img, M, distortion=dm)
        b = time.perf_counter() - t
        print(f"3000x2000 2x {kernel}: affine {1 / a:.2f} fr/s, distortion {1 / b:.2f} fr/s ({len(dz._luts)} LUTs)")


def _gradient_frames():
    Ms = [_affine(*v) for v in ((0, 0, 0), (3.2, -2.1, 0.3), (-2.4, 1.7, -0.2), (1.1, 2.9, 0.1))]
    yy, xx = np.mgrid[0:S, 0:S] / S
    grads = [0 * xx, 0.08 * xx, 0.06 * yy + 0.02, 0.05 * (1 - xx) * yy]
    r = np.random.default_rng(4)
    frames = []
    for M, g in zip(Ms, grads):
        src = (SKY - M[:, 2]) @ np.linalg.inv(M[:, :2]).T
        frames.append((_render(src, 1.2) * 0.9 + g + r.normal(0, 0.003, (S, S))).astype(np.float32))
    return Ms, frames


def _plane_spread(img, truth, m):
    """Spread (max - min) over the field of the best plane through img - truth (background trend)."""
    d = (img - truth)[m:-m, m:-m]
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(d.size)], 1)
    c = np.linalg.lstsq(A, d.ravel(), rcond=None)[0]
    return float(np.ptp(A @ c))


def test_local_norm_in_drizzle_matches_1x():
    Ms, frames = _gradient_frames()
    truth = _render(SKY, 1.2)
    reg = [register.warp(f[..., None], M, (S, S), cv2.INTER_LINEAR, False) for f, M in zip(frames, Ms)]
    models = [localnorm.fit(r, truth[..., None], 128) for r in reg]
    one = np.nanmean([localnorm.apply(r, m, 0, S) for r, m in zip(reg, models)], 0)[..., 0]
    out = {}
    for name, ln in (("none", False), ("local_norm", True)):
        dz = Drizzle((S, S), 1, 1.0, 1)
        for f, M, m in zip(frames, Ms, models):
            dz.add(f[..., None], M, local_norm=m if ln else None)
        out[name] = dz.result(smooth=0)[0][..., 0]
    g = {k: _plane_spread(v, truth, 12) for k, v in out.items()}
    g["1x"] = _plane_spread(one, truth, 12)
    diff = float(np.abs(out["local_norm"] - one)[12:-12, 12:-12].mean())
    print(f"background trend left: none {g['none']:.4f}, drizzle+LN {g['local_norm']:.4f}, 1x+LN {g['1x']:.4f}; "
          f"mean |drizzle1x - 1x| {diff:.5f}")
    assert g["local_norm"] < 0.25 * g["none"]
    assert g["local_norm"] < g["1x"] + 0.004
    # 2x as well
    dz = Drizzle((S, S), 2, 0.8, 1)
    for f, M, m in zip(frames, Ms, models):
        dz.add(f[..., None], M, local_norm=m)
    o2 = cv2.resize(dz.result()[0][..., 0], (S, S), interpolation=cv2.INTER_AREA)
    assert _plane_spread(o2, truth, 12) < 0.25 * g["none"]


@pytest.mark.parametrize("cfa", ["RGGB", None])
def test_local_norm_constant_equals_scale_offset(cfa):
    r = np.random.default_rng(9)
    so = [(1.3, 0.02), (0.8, -0.01), (1.1, 0.03)]
    model = {"a": np.tile(np.array([1.3, 0.8, 1.1], np.float32), (2, 2, 1)),
             "b": np.tile(np.array([0.02, -0.01, 0.03], np.float32), (2, 2, 1)),
             "tile": 128, "shape": (160, 200)}
    img = r.random((160, 200) if cfa else (160, 200, 3)).astype(np.float32)
    M = _affine(1.3, -0.7, 4.0)
    a, b = Drizzle((160, 200), 2, 0.8, 3), Drizzle((160, 200), 2, 0.8, 3)
    a.add(img, M, 1.0, cfa, so)
    b.add(img, M, 1.0, cfa, local_norm=model)
    assert np.allclose(a.flux, b.flux, rtol=1e-5, atol=1e-6) and np.allclose(a.weight, b.weight)
    c = Drizzle((160, 200), 2, 0.8, 3)
    c._add_points(img, M, 1.0, cfa, local_norm=model)
    d = Drizzle((160, 200), 2, 0.8, 3)
    d._add_points(img, M, 1.0, cfa, so)
    assert np.allclose(c.flux, d.flux, rtol=1e-5, atol=1e-6)
    ref = r.random((160, 200, 3)).astype(np.float32)
    m1 = outlier_mask(img, M, ref, [0.05] * 3, [0.1] * 3, so, cfa)
    m2 = outlier_mask(img, M, ref, [0.05] * 3, [0.1] * 3, so, cfa, local_norm=model)
    assert (m1 != m2).mean() < 0.01


def test_points_path_with_distortion():
    frames, affs, models = _distorted_set()
    a, b = Drizzle((S, S), 2, 0.8, 1), Drizzle((S, S), 2, 0.8, 1)
    for f, M, d in zip(frames[:3], affs, models):
        a.add(f[..., None], M, distortion=d)
        b._add_points(f[..., None], M, distortion=d)
    ia, ib = a.result(smooth=1.0)[0], b.result(smooth=1.0)[0]
    assert np.abs(ia - ib)[60:-60, 60:-60].mean() < 0.01 * np.abs(ib).mean()


def test_outlier_mask_distortion():
    frames, affs, models = _distorted_set()
    truth = _render(SKY, 0.6)[..., None]
    f = frames[3].copy()
    f[100, 60:180] += 0.5          # satellite
    so = [(1.0, 0.0)]
    with_d = outlier_mask(f[..., None], affs[3], truth, [0.003], [0.05], so, distortion=models[3])
    without = outlier_mask(f[..., None], affs[3], truth, [0.003], [0.05], so)
    trail = np.zeros_like(with_d)
    trail[100, 60:180] = True
    assert with_d[trail].mean() > 0.95
    assert with_d[~trail].mean() < without[~trail].mean()
