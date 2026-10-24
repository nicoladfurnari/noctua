"""Drizzle: exact footprint, flux conservation, resolution gain, threading, speed."""
import math
import time

import cv2
import numpy as np
import pytest
from scipy.special import erf

from astroforge.stack.drizzle import Drizzle, footprint_lut

RNG = np.random.default_rng(7)
H, W = 160, 200
STARS = [(RNG.uniform(15, W - 15), RNG.uniform(15, H - 15), RNG.uniform(0.2, 1.0)) for _ in range(40)]


def _affine(dx, dy, rot_deg):
    t = math.radians(rot_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, dx], [s, c, dy]])


def _render(M, sigma, bg=0.05):
    """Pixel-integrated gaussian stars as seen by a frame whose frame->reference transform is M."""
    Minv = cv2.invertAffineTransform(M)
    img = np.full((H, W), bg, np.float64)
    x = np.arange(W)
    y = np.arange(H)
    k = 1 / (sigma * math.sqrt(2))
    for sx, sy, a in STARS:
        fx, fy = Minv @ [sx, sy, 1]
        gx = 0.5 * (erf((x + 0.5 - fx) * k) - erf((x - 0.5 - fx) * k))
        gy = 0.5 * (erf((y + 0.5 - fy) * k) - erf((y - 0.5 - fy) * k))
        img += a * np.outer(gy, gx)
    return img.astype(np.float32)


def _bordered(f, m=6):
    f = f.copy()
    f[:m] = f[-m:] = np.nan
    f[:, :m] = f[:, -m:] = np.nan
    return f


def _dithers(n=8):
    r = np.random.default_rng(1)
    return [_affine(r.uniform(-3, 3), r.uniform(-3, 3), r.uniform(-0.4, 0.4)) for _ in range(n)]


def _fwhm(img, scale):
    vals = []
    for sx, sy, _ in STARS:
        cx, cy = sx * scale + (scale - 1) / 2, sy * scale + (scale - 1) / 2
        r = int(4 * scale)
        x0, y0 = int(round(cx)) - r, int(round(cy)) - r
        box = img[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1].astype(np.float64)
        box = np.clip(box - np.median(img), 0, None)
        yy, xx = np.mgrid[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
        m = box.sum()
        mx, my = (box * xx).sum() / m, (box * yy).sum() / m
        var = (box * ((xx - mx) ** 2 + (yy - my) ** 2)).sum() / m / 2
        vals.append(2.3548 * math.sqrt(var) / scale)
    return float(np.median(vals))


def test_footprint_lut_exact():
    for A, pf in ((np.eye(2) * 2, 0.9), (_affine(0, 0, 25)[:, :2] * 3, 0.6), (np.diag([2.0, -2.0]), 1.0)):
        mn, (ny, nx), lut = footprint_lut(A, pf, 16)
        assert np.allclose(lut.sum(1), 1) and lut.min() > -1e-12
        k = 1200
        o = ((np.arange(k) + 0.5) / k - 0.5) * pf
        oy, ox = np.meshgrid(o, o, indexing="ij")
        for gx, gy in ((0, 0), (5, 11), (16, 16)):
            p = np.stack([ox.ravel(), oy.ravel()], 1) @ A.T - mn + [gx / 16, gy / 16]
            h, _, _ = np.histogram2d(p[:, 1], p[:, 0], bins=[ny, nx], range=[[0, ny], [0, nx]])
            assert np.abs(h.ravel() / len(p) - lut[gy * 17 + gx]).max() < 2e-3


@pytest.mark.parametrize("scale", [2, 3])
@pytest.mark.parametrize("pixfrac", [0.6, 0.8, 1.0])
def test_flux_conservation(scale, pixfrac):
    Ms = _dithers()
    frames = [_bordered(_render(M, 1.2)) for M in Ms]
    dz = Drizzle((H, W), scale, pixfrac, 1)
    for f, M in zip(frames, Ms):
        dz.add(f[..., None], M)
    # accumulator: every valid input pixel deposits exactly its value (NaN border keeps all inside)
    assert abs(dz.flux.sum() / sum(float(np.nansum(f)) for f in frames) - 1) < 1e-6
    # image: surface brightness integrated over the interior matches the 1x truth
    out, _ = dz.result(smooth=0)
    truth = _render(_affine(0, 0, 0), 1.2)
    m = 10
    got = out[m * scale:(H - m) * scale, m * scale:(W - m) * scale].sum() / scale ** 2
    assert abs(got / truth[m:H - m, m:W - m].sum() - 1) < 0.005


def test_bayer_flux_per_channel():
    Ms = _dithers(4)
    dz = Drizzle((H, W), 2, 0.8, 3)
    tot = np.zeros(3)
    ch = np.zeros((H, W), int)
    ch[0::2, 1::2] = ch[1::2, 0::2] = 1
    ch[1::2, 1::2] = 2
    for M in Ms:
        f = _bordered(_render(M, 1.2))
        dz.add(f, M, cfa_pattern="RGGB")
        ok = np.isfinite(f)
        tot += np.bincount(ch[ok], f[ok].astype(np.float64), 3)
    assert np.allclose(dz.flux.reshape(-1, 3).sum(0), tot, rtol=1e-6)


def test_drizzle_sharper_than_integration():
    Ms = _dithers()
    frames = [_render(M, 0.6) for M in Ms]
    integ = np.mean([cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR) for f, M in zip(frames, Ms)], 0)
    dz = Drizzle((H, W), 2, 0.6, 1)
    n = dz.add_many((f[..., None], M) for f, M in zip(frames, Ms))
    assert n == len(frames)
    out, _ = dz.result(smooth=0)
    f1, f2 = _fwhm(integ, 1), _fwhm(out[..., 0], 2)
    print(f"FWHM 1x integration {f1:.2f}px, drizzle 2x {f2:.2f}px (1x units)")
    assert f2 < 0.95 * f1


def test_threads_match_single():
    r = np.random.default_rng(3)
    M = _affine(2.3, -1.7, 12.0)
    rej = r.random((H, W)) < 0.01
    so = [(1.1, 0.01), (0.9, -0.02), (1.0, 0.0)]
    for img, cfa in ((r.random((H, W)).astype(np.float32), "GRBG"), (r.random((H, W, 3)).astype(np.float32), None)):
        img[5, 7] = np.nan
        a, b = Drizzle((H, W), 3, 0.7, 3), Drizzle((H, W), 3, 0.7, 3)
        a.add(img, M, 0.8, cfa, so, rej, workers=1)
        b.add(img, M, 0.8, cfa, so, rej, workers=8)
        assert np.allclose(a.flux, b.flux, rtol=1e-12, atol=1e-12)
        assert np.allclose(a.weight, b.weight, rtol=1e-12, atol=1e-12)
        # same result as the old sub-point sampling, within its approximation
        c = Drizzle((H, W), 3, 0.7, 3)
        c._add_points(img, M, 0.8, cfa, so, rej)
        ia, ic = a.result(smooth=1.0)[0], c.result(smooth=1.0)[0]
        assert np.abs(ia - ic)[20:-20, 20:-20].mean() < 0.01 * np.abs(ic).mean()


def test_speed_report():
    img = np.random.default_rng(0).random((2000, 3000), np.float32)[..., None]
    M = _affine(0.3, -0.2, 0.05)
    dz = Drizzle((2000, 3000), 2, 0.9, 1)
    dz.add(img, M)                      # first touch of the accumulator pages
    t = time.perf_counter()
    dz.add(img, M)
    new = 1 / (time.perf_counter() - t)
    t = time.perf_counter()
    dz.add(img, M, workers=1)
    one = 1 / (time.perf_counter() - t)
    t = time.perf_counter()
    dz._add_points(img, M)
    old = 1 / (time.perf_counter() - t)
    print(f"3000x2000 mono 2x: sub-points {old:.3f} fr/s, exact 1 thread {one:.2f} fr/s, "
          f"exact {dz.workers_for()} threads {new:.2f} fr/s")
    assert new > old
