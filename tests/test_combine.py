"""Pixel rejection (combine.py) and satellite trail detection (satellites.py) on synthetic stacks."""
import cv2
import numpy as np
import pytest

from astroforge.stack import combine as C
from astroforge.stack.satellites import trail_mask

SD = 0.01
H, W = 60, 90
REJECTORS = ("percentile", "sigma", "winsorized", "linear_fit", "esd")


def _line_dist(shape, p0, p1):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    u = (np.float32(p1[0] - p0[0]), np.float32(p1[1] - p0[1]))
    n = np.hypot(*u)
    return np.abs((xx - p0[0]) * u[1] / n - (yy - p0[1]) * u[0] / n)


def _stack(n, seed=0, wings=0.0):
    rng = np.random.default_rng(seed)
    truth = (1.0 + 0.05 * np.sin(np.mgrid[0:H, 0:W][1] / 9.0)).astype(np.float32)
    clean = truth + rng.normal(0, SD, (n, H, W)).astype(np.float32)
    st = clean.copy()
    dist = _line_dist((H, W), (0, 10), (W, 50))
    trail = dist < 1.0
    st[n // 2][trail] += 0.3                                    # satellite: 30 sigma
    if wings:
        st[n // 2][(dist >= 1.0) & (dist < 4.0)] += wings          # faint wings
    # hot pixels: one per pixel at most, never on the trail
    cand = np.flatnonzero(~(dist < 5).ravel())
    pix = rng.choice(cand, int(0.01 * H * W), replace=False)
    st[rng.integers(0, n, pix.size), pix // W, pix % W] += rng.uniform(0.15, 1.0, pix.size)
    return st, clean, truth, trail, dist, pix


@pytest.mark.parametrize("n", [8, 20])
@pytest.mark.parametrize("method", REJECTORS + ("auto",))
def test_rejection_removes_outliers_unbiased(method, n):
    st, clean, truth, trail, _, pix = _stack(n, seed=n)
    out, rl, rh = C.combine(st, method)
    ref = clean.mean(0)
    res = (out - ref) / (SD / np.sqrt(n))         # in sigma of the integration
    assert np.abs(res[trail]).max() < 4, method
    assert np.abs(res.ravel()[pix]).max() < 4, method
    assert abs(float((out - truth).mean())) < 0.1 * SD / np.sqrt(n)      # unbiased within noise
    assert float((out - truth).std()) < 1.2 * float((ref - truth).std())  # little SNR lost
    assert rh[trail].min() >= 1 and rh.ravel()[pix].min() >= 1
    assert (rl.sum() + rh.sum()) / st.size < 0.03


def test_no_rejection_and_median():
    st, clean, truth, trail, _, _ = _stack(8)
    out, rl, rh = C.combine(st, "none")
    assert np.allclose(out, st.mean(0), atol=1e-6) and rl.sum() == rh.sum() == 0
    med, _, _ = C.combine(st, "median")
    assert np.allclose(med, np.median(st, 0), atol=1e-6)


@pytest.mark.parametrize("method", REJECTORS + ("none", "median"))
def test_nan_coverage(method):
    st, clean, truth, trail, _, _ = _stack(10, seed=3)
    st[0, :, :20] = np.nan
    st[:6, :5] = np.nan
    st[:, 50:53, 60:64] = np.nan                  # nobody covers this
    out, rl, rh = C.combine(st, method)
    hole = np.zeros((H, W), bool)
    hole[50:53, 60:64] = True
    assert np.isnan(out[hole]).all() and np.isfinite(out[~hole]).all()
    cnt = (~np.isnan(st)).sum(0)
    assert ((rl.astype(int) + rh) <= cnt).all()
    if method != "none":
        assert np.abs(out[~hole] - truth[~hole]).max() < 6 * SD / np.sqrt(4)


def test_weights_and_rgb_shape():
    st, *_ = _stack(12, seed=4)
    rgb = np.stack([st, st * 0.9, st * 1.1], -1)
    w = np.linspace(0.5, 1.5, 12)
    out, rl, rh = C.combine(rgb, "winsorized", weights=w)
    assert out.shape == rgb.shape[1:] and rl.shape == rgb.shape[1:] and rl.dtype == np.uint16
    clean = C.combine(rgb[1:], "none", weights=w[1:])[0]
    assert np.isfinite(out).all() and np.abs(out - clean).mean() < 0.01


def test_auto_choice():
    assert [C.auto_method(n) for n in (2, 4, 8, 15, 19, 20, 60)] == \
        ["none", "percentile", "winsorized", "winsorized", "winsorized", "linear_fit", "linear_fit"]
    with pytest.raises(ValueError):
        C.combine(np.zeros((3, 4, 4), np.float32), "bogus")


def test_grow_rejects_trail_wings():
    n = 8
    st, clean, truth, trail, dist, _ = _stack(n, seed=5, wings=0.6 * SD)
    wing = (dist >= 1.0) & (dist < 4.0)
    base = C.combine(st, "winsorized")[0]
    grown, _, rh = C.combine(st, "winsorized", grow_px=4)
    ref = clean.mean(0)
    bias0 = float((base - ref)[wing].mean())
    bias1 = float((grown - ref)[wing].mean())
    assert bias0 > 0.5 * 0.6 * SD / n                # wings are not rejected per pixel...
    assert abs(bias1) < 0.25 * abs(bias0)            # ...but grown from the trail core
    assert rh[wing].mean() > 0.9
    assert rh[dist > 12].mean() < 0.05               # isolated hot pixels are not grown


class _Loader:
    def __init__(self, st):
        self.st, self.height = st, st.shape[1]

    def __call__(self, i, y0, y1):
        return self.st[i, y0:y1]


@pytest.mark.parametrize("grow", [0, 3])
def test_chunked_matches_whole(grow):
    st, *_ = _stack(9, seed=6, wings=0.01)
    whole = C.combine(st, "sigma", grow_px=grow)
    parts = C.combine_chunked(list(range(9)), _Loader(st), "sigma", rows=7, workers=3, grow_px=grow)
    for a, b in zip(whole, parts):
        assert np.allclose(a, b, atol=1e-6, equal_nan=True)


def test_memory_planning():
    assert C.available_memory() > 64 << 20
    per_row = 30 * 6000 * 3 * 4 * C.TEMP_FACTOR
    rows, workers = C.plan_bands(30, 6000 * 3, 4000, max_rows=256, workers=8, avail=2 << 30)
    assert 1 <= rows <= 256 and 1 <= workers <= 8
    assert rows * workers * per_row <= 0.5 * (2 << 30)
    assert C.plan_bands(30, 6000 * 3, 4000, max_rows=256, workers=8, avail=1 << 40) == (250, 8)
    rows, workers = C.plan_bands(30, 6000 * 3, 4000, max_rows=256, workers=8, avail=100 << 20)
    assert workers == 1 and rows >= 1
    assert C.plan_bands(5, 10, 3, max_rows=256, avail=1 << 40)[0] == 3


def test_fast_nanmedian():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(9, 40, 50)).astype(np.float32)
    a[rng.random(a.shape) < 0.3] = np.nan
    a[:, 0, 0] = np.nan
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        ref = np.nanmedian(a, axis=0)
    assert np.allclose(C.nanmedian(a), ref, atol=1e-6, equal_nan=True)


# ---- satellite trails

def _field(seed, n=512, nstar=900, shift=(0.0, 0.0), psf=1.3):
    rng = np.random.default_rng(seed)
    xs, ys = rng.uniform(0, n, nstar), rng.uniform(0, n, nstar)
    fl = np.minimum(30 * (rng.pareto(1.2, nstar) + 1), 3e5)
    big = np.zeros((4 * n, 4 * n), np.float32)
    np.add.at(big, (np.clip(((ys + shift[1]) * 4).astype(int), 0, 4 * n - 1),
                    np.clip(((xs + shift[0]) * 4).astype(int), 0, 4 * n - 1)), fl)
    big = cv2.GaussianBlur(big, (0, 0), psf * 4)
    return 100 + cv2.resize(big, (n, n), interpolation=cv2.INTER_AREA) * 16


def _frames(seed, shift=(0.35, -0.25)):
    rng = np.random.default_rng(seed + 100)
    ref = _field(seed) + rng.normal(0, 0.4, (512, 512)).astype(np.float32)      # median of ~8 frames
    frame = _field(seed, shift=shift, psf=1.4) + rng.normal(0, 1.0, (512, 512)).astype(np.float32)
    frame[rng.random(frame.shape) < 5e-4] += 50                                  # hot pixels
    return frame, ref


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_trail_mask_star_field_no_false_positives(seed):
    frame, ref = _frames(seed, shift=(0.8, -0.6) if seed == 3 else (0.35, -0.25))
    assert trail_mask(frame, ref, 1.0).mean() < 0.001


@pytest.mark.parametrize("seed", [1, 2])
def test_trail_mask_finds_faint_trail(seed):
    frame, ref = _frames(seed)
    dist = _line_dist(frame.shape, (40, 0), (470, 511))
    frame += (2.0 * np.exp(-dist ** 2 / (2 * 1.0 ** 2))).astype(np.float32)        # 2 sigma per pixel
    m, lines = trail_mask(frame, ref, 1.0, return_lines=True)
    assert len(lines) >= 1
    assert m[dist < 1.5].mean() > 0.9
    assert (m & (dist > 15)).mean() < 0.001


def test_trail_mask_rgb_and_nan():
    frame, ref = _frames(4)
    dist = _line_dist(frame.shape, (0, 300), (511, 120))
    rng = np.random.default_rng(9)
    f3 = np.stack([frame + rng.normal(0, 0.5, frame.shape) for _ in range(3)], -1)
    f3 += (40.0 * np.exp(-dist ** 2 / 4))[..., None]
    f3[:30] = np.nan                                                              # not covered
    m = trail_mask(f3, np.stack([ref] * 3, -1), [1.1, 1.1, 1.1])
    assert m.shape == frame.shape and m[(dist < 2) & (np.mgrid[0:512, 0:512][0] > 40)].mean() > 0.9
    assert (m & (dist > 20)).mean() < 0.001


def test_percentile_keeps_dark_sky_noise():
    # real M8 case: 5 calibrated frames, sky ~0.003 with ~10 % frame-to-frame noise. The -20 %/+10 % bounds
    # alone rejected half of the pixels; the noise floor keeps them while a satellite is still removed.
    rng = np.random.default_rng(4)
    s = rng.normal(0.003, 0.0003, (5, 64, 200, 3)).astype(np.float32)
    s[2, 10:14, 50:150] = 0.5
    out, rl, rh = C.combine(s, "percentile", 4, 3)
    assert 100 * float(rl.sum() + rh.sum()) / (rl.size * 5) < 5
    assert abs(float(out[12, 100, 0]) - 0.003) < 0.001
