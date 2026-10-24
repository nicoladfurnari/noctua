"""TGV, Richardson-Lucy and starlet on the GPU (CuPy) and their xp code on the CPU.

The xp code (the one that runs on the GPU) is run here with numpy and compared with frozen copies of the
previous OpenCV implementations; GPU tests run only when CuPy and a GPU are available.

Tolerances: the CPU paths are bit-identical (or ~1e-8) to the previous code. The xp paths use FFT convolutions
(RL) and a different summation order (resampling, starlet), float32 rounding gives ~3e-5 on RL (2e-4 across
bands), ~1e-7 on TGV; the wavelet thresholding is discontinuous, so a coefficient exactly at the threshold can
flip: a handful of pixels (< 1e-4 of them) may differ by up to ~2e-3, all the others within 1e-6.
"""
import cv2
import numpy as np
import pytest

from astroforge import deconv, denoise, gpu, tgv


# ------------------------------------------------------------------ frozen previous CPU implementations
def legacy_rl(img, psf, iterations=20, tv_weight=0.002, dering=0.0, protect=None):
    def tv_div(u, eps=1e-5):
        gx, gy = np.zeros_like(u), np.zeros_like(u)
        gx[:, :-1] = u[:, 1:] - u[:, :-1]
        gy[:-1] = u[1:] - u[:-1]
        n = np.sqrt(gx * gx + gy * gy + eps * eps)
        gx /= n
        gy /= n
        div = gx.copy()
        div[:, 1:] -= gx[:, :-1]
        div += gy
        div[1:] -= gy[:-1]
        return div

    eps = 1e-7
    obs = np.maximum(np.ascontiguousarray(img, np.float32), 0) + eps
    psf = np.asarray(psf, np.float32)
    psf_t = np.ascontiguousarray(psf[::-1, ::-1])
    est = obs.copy()
    lower = None
    if dering > 0:
        rc = max(2, int((psf >= psf.max() * 0.05).sum(axis=1).max()))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rc + 1, 2 * rc + 1))
        lower = cv2.GaussianBlur(cv2.erode(obs, ker), (0, 0), rc / 2)
    for _ in range(int(iterations)):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REFLECT)
        corr = cv2.filter2D(obs / np.maximum(conv, eps), -1, psf_t, borderType=cv2.BORDER_REFLECT)
        if tv_weight > 0:
            corr /= np.maximum(1 - np.float32(tv_weight) * tv_div(est), 0.5)
        est *= corr
        if lower is not None:
            est += np.float32(dering) * np.maximum(lower - est, 0)
    est -= eps
    if protect is not None:
        est = obs * protect + est * (1 - protect)
    return est.astype(np.float32)


def legacy_wavelet(img, mask=None, thresholds=(3.0, 2.0, 1.0, 0.5), amounts=(0.6, 0.5, 0.4, 0.3)):
    noise = [0.890, 0.201, 0.086, 0.042, 0.021, 0.010]
    out = np.empty_like(img)
    for ci in range(img.shape[-1]):
        c = img[..., ci]
        layers = []
        for j in range(len(thresholds)):
            k = denoise._atrous_kernel(j)
            s = cv2.sepFilter2D(c, -1, k, k, borderType=cv2.BORDER_REFLECT)
            layers.append(c - s)
            c = s
        layers.append(c)
        w1 = layers[0][::4, ::4]
        sigma = float(np.median(np.abs(w1 - np.median(w1)))) / 0.6745 / noise[0]
        rec = layers[-1].copy()
        for j, (t, a) in enumerate(zip(thresholds, amounts)):
            w = layers[j]
            rec += np.where(np.abs(w) < t * sigma * noise[j], w * (1 - a), w)
        out[..., ci] = rec
    if mask is not None:
        out = img * (1 - mask[..., None]) + out * mask[..., None]
    return np.clip(out, 0, 1)


def _legacy_cp(f, a1, a0, iterations, tol=0.0, state=None):
    def dx(u, out):
        np.subtract(u[:, 1:], u[:, :-1], out=out[:, :-1])
        out[:, -1] = 0

    def dy(u, out):
        np.subtract(u[1:], u[:-1], out=out[:-1])
        out[-1] = 0

    def divx(p, out):
        np.copyto(out[:, :-1], p[:, :-1])
        out[:, -1] = 0
        out[:, 1:] -= p[:, :-1]

    def divy_add(p, out):
        out[:-1] += p[:-1]
        out[1:] -= p[:-1]

    ia1, ia0 = 1 / np.maximum(a1, 1e-12), 1 / np.maximum(a0, 1e-12)
    tau = sig = np.float32(1 / np.sqrt(12))
    tf, k = tau * f, np.float32(1 / (1 + tau))
    z = lambda: np.zeros_like(f)  # noqa: E731
    if state is None:
        u, w1, w2, p1, p2, q1, q2, q3 = f.copy(), z(), z(), z(), z(), z(), z(), z()
    else:
        u, w1, w2, p1, p2, q1, q2, q3 = (np.ascontiguousarray(x, np.float32) for x in state)
    ub, wb1, wb2 = u.copy(), w1.copy(), w2.copy()
    t1, t2, t3, n = z(), z(), z(), z()
    check, it = u.copy(), 0
    for it in range(1, iterations + 1):
        dx(ub, t1); t1 -= wb1; t1 *= sig; p1 += t1                      # noqa: E702
        dy(ub, t2); t2 -= wb2; t2 *= sig; p2 += t2                      # noqa: E702
        np.hypot(p1, p2, out=n); n *= ia1; np.maximum(n, 1, out=n)      # noqa: E702
        p1 /= n; p2 /= n                                                # noqa: E702
        dx(wb1, t1); t1 *= sig; q1 += t1                                # noqa: E702
        dy(wb2, t2); t2 *= sig; q2 += t2                                # noqa: E702
        dy(wb1, t1); dx(wb2, t3); t1 += t3; t1 *= sig / 2; q3 += t1     # noqa: E702
        np.multiply(q1, q1, out=n); np.multiply(q2, q2, out=t1); n += t1  # noqa: E702
        np.multiply(q3, q3, out=t1); t1 *= 2; n += t1                   # noqa: E702
        np.sqrt(n, out=n); n *= ia0; np.maximum(n, 1, out=n)            # noqa: E702
        q1 /= n; q2 /= n; q3 /= n                                       # noqa: E702
        divx(p1, t1); divy_add(p2, t1)                                  # noqa: E702
        np.copyto(ub, u)
        t1 *= tau; t1 += tf; u += t1; u *= k                            # noqa: E702
        np.subtract(u, ub, out=ub); ub += u                             # noqa: E702
        divx(q1, t1); divy_add(q3, t1); t1 += p1; t1 *= tau             # noqa: E702
        divx(q3, t2); divy_add(q2, t2); t2 += p2; t2 *= tau             # noqa: E702
        w1 += t1; w2 += t2                                              # noqa: E702
        np.add(w1, t1, out=wb1); np.add(w2, t2, out=wb2)                # noqa: E702
        if tol > 0 and it % 10 == 0:
            np.subtract(u, check, out=t1)
            if float(np.abs(t1).mean()) / 10 < tol:
                break
            np.copyto(check, u)
    return (u, w1, w2, p1, p2, q1, q2, q3), it


def legacy_tgv_channel(f, strength, aspect=2.0, weight=None, iterations=100, tol=4e-3, sigma=None):
    T, O = 256, 16
    f = np.ascontiguousarray(f, np.float32)
    sigma = tgv.noise_sigma(f) if sigma is None else sigma
    a1 = np.float32(strength * sigma) * (weight if weight is not None else np.ones_like(f))
    h, w = f.shape
    out = np.empty_like(f)
    its = []
    for y in range(0, h, T):
        for x in range(0, w, T):
            y0, x0 = max(0, y - O), max(0, x - O)
            y1, x1 = min(h, y + T + O), min(w, x + T + O)
            ft = np.ascontiguousarray(f[y0:y1, x0:x1])
            aa = np.ascontiguousarray(a1[y0:y1, x0:x1], np.float32)
            a0 = aa * aspect
            th, tw = ft.shape
            state = None
            if min(th, tw) >= 32:
                ds = lambda a: cv2.resize(a, (tw // 2, th // 2), interpolation=cv2.INTER_AREA)  # noqa: E731
                us = lambda a: cv2.resize(a, (tw, th), interpolation=cv2.INTER_LINEAR)          # noqa: E731
                st, _ = _legacy_cp(ds(ft), ds(aa) / 2, ds(a0) / 4, iterations, tol * sigma * 2)
                state = [us(v) * k for v, k in zip(st, (1, 0.5, 0.5, 2, 2, 4, 4, 4))]
            st, n = _legacy_cp(ft, aa, a0, iterations, tol * sigma, state)
            out[y:y + T, x:x + T] = st[0][y - y0:y - y0 + T, x - x0:x - x0 + T]
            its.append(n)
    return out, max(its)


# ------------------------------------------------------------------ data
def _starfield(h=520, w=610, seed=3, n=250, fwhm=3.5):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 0.05, np.float32)
    img += np.linspace(0, 0.03, w, dtype=np.float32)[None]
    k = deconv.moffat_kernel(fwhm, 3.0)
    r = k.shape[0] // 2
    for y, x, a in zip(rng.integers(r, h - r, n), rng.integers(r, w - r, n), rng.uniform(0.05, 0.9, n)):
        img[y - r:y + r + 1, x - r:x + r + 1] += a * k / k.max()
    return np.clip(img + rng.normal(0, 0.01, img.shape).astype(np.float32), 0, 1)


@pytest.fixture(scope="module")
def star():
    return _starfield()


@pytest.fixture(scope="module")
def rgb(star):
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 0.008, star.shape + (3,)).astype(np.float32)
    return np.clip(np.stack([star, star * 0.8, star * 0.6], -1) + noise, 0, 1).astype(np.float32)


def _has_gpu():
    return gpu.cupy() is not None and bool(gpu.devices())


needs_gpu = pytest.mark.skipif(not _has_gpu(), reason="CuPy / GPU not available")


# ------------------------------------------------------------------ device selection
def test_select_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(gpu._STATE, "devices", [])
    assert gpu.select("auto") == []
    assert gpu.select(0) == []
    assert gpu.select(None, gpu=False) == []
    monkeypatch.setitem(gpu._STATE, "devices", [(0, "fake", 8.0), (1, "fake", 4.0)])
    monkeypatch.setitem(gpu._STATE, "enabled", True)
    assert gpu.select("auto") == [0, 1]
    assert gpu.select(1) == [1]
    assert gpu.select([0, 5]) == [0]
    assert gpu.select("cpu") == [] and gpu.select("auto", gpu=False) == []


# ------------------------------------------------------------------ Richardson-Lucy
@pytest.mark.parametrize("tv,dering", [(0.002, 0.8), (0.0, 0.0)])
def test_rl_cpu_equals_previous(star, tv, dering):
    psf = deconv.moffat_kernel(3.5, 3.0)
    ref = legacy_rl(star, psf, 12, tv, dering)
    out = deconv.richardson_lucy(star, psf, 12, tv, dering, device="cpu")
    assert np.abs(out - ref).max() <= 1e-6


def test_rl_xp_numpy_equals_previous(star):
    """The GPU code (FFT convolutions, reflect padding) with numpy."""
    for psf in (deconv.moffat_kernel(3.5, 3.0), deconv.gaussian_kernel(2.2, 0.3, 0.5)):
        ref = legacy_rl(star, psf, 12, 0.002, 0.8)
        out = deconv.richardson_lucy(star, psf, 12, 0.002, 0.8, device="numpy")
        assert np.abs(out - ref).max() < 1e-4


def test_fft_corr_equals_filter2d():
    rng = np.random.default_rng(1)
    a = rng.random((97, 130)).astype(np.float32)
    for k in (deconv.moffat_kernel(2.0), deconv.moffat_kernel(6.0, 2.5, 0.3, 1.0), rng.random((5, 9), np.float32)):
        ref = cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REFLECT)
        got = deconv._FFTCorr(np, a.shape, k)(a)
        assert np.abs(got - ref).max() < 1e-5


def test_rl_bands_equal_global(star):
    psf = deconv.moffat_kernel(3.5, 3.0)
    obs = np.maximum(star, 0) + deconv._EPS
    lower = deconv._dering_lower(obs, psf)
    whole = deconv._rl_bands(obs, lower, psf, 15, 0.002, 0.8, xp=np)
    for rows in (97, 200, 333):
        banded = deconv._rl_bands(obs, lower, psf, 15, 0.002, 0.8, rows=rows, xp=np)
        assert np.abs(banded - whole).max() < 3e-4


def test_deconvolve_tiles_equal_previous(star, monkeypatch):
    out = deconv.deconvolve(star, 8, tiles=2, dering=0.7, device="cpu")
    xp_out = deconv.deconvolve(star, 8, tiles=2, dering=0.7, device="numpy")
    monkeypatch.setattr(deconv, "richardson_lucy", lambda im, k, it, tv, dr, pr, *a: legacy_rl(im, k, it, tv, dr, pr))
    ref = deconv.deconvolve(star, 8, tiles=2, dering=0.7)
    assert np.abs(out - ref).max() <= 1e-6
    assert np.abs(xp_out - ref).max() < 1e-4


# ------------------------------------------------------------------ TGV
def test_tgv_cpu_equals_previous(star):
    s = tgv.noise_sigma(star)
    wgt = tgv.edge_weight(star, s)
    ref, nr = legacy_tgv_channel(star, 2.0, 2.0, wgt, 100, 4e-3, s)
    out, n = tgv.tgv_channel(star, 2.0, 2.0, wgt, 100, 4e-3, s, device="cpu")
    assert n == nr
    assert np.abs(out - ref).max() <= 1e-6


@pytest.mark.parametrize("tol", [4e-3, 0.0])
def test_tgv_batched_xp_equals_previous(star, tol):
    """The GPU code (tiles batched by shape, per-tile convergence, xp resampling) with numpy."""
    s = tgv.noise_sigma(star)
    wgt = tgv.edge_weight(star, s)
    ref, nr = legacy_tgv_channel(star, 2.0, 2.0, wgt, 60, tol, s)
    out, n = tgv.tgv_channel(star, 2.0, 2.0, wgt, 60, tol, s, device="numpy")
    assert n == nr
    assert np.abs(out - ref).max() < 1e-4


def test_tgv_batch_size_does_not_matter(star, monkeypatch):
    s = tgv.noise_sigma(star)
    a, _ = tgv.tgv_channel(star, 2.0, sigma=s, device="numpy")
    monkeypatch.setattr(tgv, "GPU_BYTES", 1 << 40)       # one tile per batch
    b, _ = tgv.tgv_channel(star, 2.0, sigma=s, device="numpy")
    assert np.abs(a - b).max() < 1e-6


def test_tgv_batch_per_tile_convergence():
    """Tiles converging at different iterations inside one batch keep their own result."""
    rng = np.random.default_rng(2)
    flat = np.full((80, 80), 0.1, np.float32) + rng.normal(0, 1e-4, (80, 80)).astype(np.float32)
    busy = (0.1 + 0.05 * rng.random((80, 80))).astype(np.float32)
    f = np.stack([flat, busy, flat * 2])
    a1 = np.full_like(f, 0.02)
    u, its = tgv.tgv_batch(f, a1, a1 * 2, 200, 1e-4)
    for i in range(3):
        ui, ni = tgv.tgv_tile(f[i], a1[i], a1[i] * 2, 200, 1e-4)
        assert its[i] == ni
        assert np.abs(u[i] - ui).max() < 1e-5
    assert len(set(its.tolist())) > 1


def test_resample_equals_cv2():
    rng = np.random.default_rng(4)
    for h, w in ((288, 288), (289, 273), (131, 40), (33, 35)):
        x = rng.random((2, h, w)).astype(np.float32)
        ref = np.stack([cv2.resize(v, (w // 2, h // 2), interpolation=cv2.INTER_AREA) for v in x])
        assert np.abs(tgv._resample(x, (h // 2, w // 2), "area") - ref).max() < 1e-6
        up = np.stack([cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) for v in ref])
        assert np.abs(tgv._resample(ref, (h, w), "linear") - up).max() < 1e-5


def test_tgv_tiled_vs_whole(star):
    """The 256 px tiles with 16 px overlap (unchanged design) against one whole-image solve: ~1e-3 sigma on
    average, a few tenths of sigma at worst on tile seams near bright stars."""
    s = tgv.noise_sigma(star)
    wgt = tgv.edge_weight(star, s)
    a1 = np.float32(2.0 * s) * wgt
    whole, _ = tgv.tgv_tile(star, a1, a1 * 2, 100, 0.0)
    tiled, _ = tgv.tgv_channel(star, 2.0, 2.0, wgt, 100, 0.0, s, device="numpy")
    d = np.abs(tiled - whole)
    assert d.mean() < 0.01 * s and d.max() < 0.5 * s


def test_tgv_denoise_rgb_xp(rgb):
    a = tgv.denoise(rgb, device="cpu")
    b = tgv.denoise(rgb, device="numpy")
    assert np.abs(a - b).max() < 1e-4


# ------------------------------------------------------------------ starlet / wavelet denoise
def test_starlet_xp_equals_cv2():
    rng = np.random.default_rng(0)
    x = rng.random((300, 411)).astype(np.float32)
    for a, b in zip(denoise.starlet(x, 5), denoise.starlet(x, 5, xp_code=True)):
        assert np.abs(a - b).max() < 1e-6


def _close_wavelet(a, b):
    d = np.abs(a - b)
    return d.max() < 5e-3 and (d > 1e-6).mean() < 1e-4


def test_wavelet_cpu_equals_previous(rgb):
    mask = np.linspace(0, 1, rgb.shape[1], dtype=np.float32)[None].repeat(rgb.shape[0], 0)
    ref = legacy_wavelet(rgb.copy(), mask)
    assert np.abs(denoise.denoise(rgb.copy(), mask, device="cpu") - ref).max() <= 1e-7
    assert _close_wavelet(denoise.denoise(rgb.copy(), mask, device="numpy"), ref)


def test_wavelet_bands_equal_whole(rgb):
    th, am = (3.0, 2.0, 1.0, 0.5), (0.6, 0.5, 0.4, 0.3)
    whole = denoise._denoise_xp(rgb, th, am, xp=np)
    for rows in (37, 100, 251):
        assert np.abs(denoise._denoise_xp(rgb, th, am, rows=rows, xp=np) - whole).max() < 1e-6


# ------------------------------------------------------------------ GPU (CuPy) vs CPU
@needs_gpu
def test_gpu_rl_equals_cpu(star):
    big = np.tile(star, (2, 2))          # above MIN_GPU_PIXELS
    psf = deconv.moffat_kernel(3.5, 3.0)
    cpu = deconv.richardson_lucy(big, psf, 15, 0.002, 0.8, device="cpu")
    out = deconv.richardson_lucy(big, psf, 15, 0.002, 0.8, device=0)
    # cuFFT and the CPU FFT round differently in float32; 15 iterations accumulate ~2e-4 at the star peaks
    # (measured on RTX 3070 Ti / 3060): relative bound on the peak, tight bound on the average
    diff = np.abs(out - cpu)
    assert diff.max() < 5e-3 * float(cpu.max())
    assert float(diff.mean()) < 1e-5


@needs_gpu
def test_gpu_rl_bands_and_tiles(star):
    psf = deconv.moffat_kernel(3.5, 3.0)
    obs = np.maximum(star, 0) + deconv._EPS
    lower = deconv._dering_lower(obs, psf)
    whole = deconv._rl_bands(obs, lower, psf, 15, 0.002, 0.8, xp=np)
    banded = deconv._rl_bands(obs, lower, psf, 15, 0.002, 0.8, devs=[gpu.devices()[0][0]], rows=150)
    assert np.abs(banded - whole).max() < 3e-4
    big = np.tile(star, (2, 2))
    cpu = deconv.deconvolve(big, 10, tiles=2, device="cpu")
    out = deconv.deconvolve(big, 10, tiles=2, device="auto")
    assert np.abs(out - cpu).max() < 1e-4


@needs_gpu
def test_gpu_tgv_equals_cpu(star, rgb):
    cp = gpu.cupy()
    s = tgv.noise_sigma(star)
    wgt = tgv.edge_weight(star, s)
    big, bw = np.tile(star, (2, 2)), np.tile(wgt, (2, 2))
    cpu, n_cpu = tgv.tgv_channel(big, 2.0, 2.0, bw, 60, 0.0, s, device="cpu")
    out, n_gpu = tgv.tgv_channel(big, 2.0, 2.0, bw, 60, 0.0, s, device=0)
    assert n_cpu == n_gpu and np.abs(out - cpu).max() < 1e-4
    cpu, _ = tgv.tgv_channel(big, 2.0, 2.0, bw, 100, 4e-3, s, device="cpu")
    out, _ = tgv.tgv_channel(big, 2.0, 2.0, bw, 100, 4e-3, s, device=0)
    assert np.abs(out - cpu).max() < 0.02 * s        # a tile may stop 10 iterations apart (float sums)
    f = np.stack([star[:288, :288], star[100:388, 200:488]])
    a1 = np.full_like(f, 2 * s)
    u_cpu, it_cpu = tgv.tgv_batch(f, a1, a1 * 2, 50, 0.0)
    u_gpu, it_gpu = tgv.tgv_batch(cp.asarray(f), cp.asarray(a1), cp.asarray(a1 * 2), 50, 0.0)
    assert np.abs(cp.asnumpy(u_gpu) - u_cpu).max() < 1e-5 and list(it_cpu) == list(it_gpu)
    big3 = np.tile(rgb, (2, 2, 1))
    assert np.abs(tgv.denoise(big3, device=0, iterations=60) - tgv.denoise(big3, device="cpu", iterations=60)).max() \
        < 0.02 * s + 1e-4


@needs_gpu
def test_gpu_wavelet_equals_cpu(rgb):
    cp = gpu.cupy()
    big = np.tile(rgb, (2, 2, 1))
    cpu = denoise.denoise(big.copy(), device="cpu")
    assert _close_wavelet(denoise.denoise(big.copy(), device=0), cpu)
    th, am = (3.0, 2.0, 1.0, 0.5), (0.6, 0.5, 0.4, 0.3)
    whole = denoise._denoise_xp(big, th, am, xp=np)
    banded = denoise._denoise_xp(big, th, am, devs=[gpu.devices()[0][0]], rows=200)
    assert _close_wavelet(banded, whole)
    x = big[..., 1].copy()
    for a, b in zip(denoise.starlet(x, 4), denoise.starlet(cp.asarray(x), 4)):
        assert np.abs(cp.asnumpy(b) - a).max() < 1e-6


@pytest.mark.skipif(not _has_gpu() or len(gpu.devices()) < 2, reason="needs two GPUs")
def test_multi_gpu_equals_cpu(star):
    big = np.tile(star, (3, 3))
    psf = deconv.moffat_kernel(3.5, 3.0)
    cpu = deconv.richardson_lucy(big, psf, 10, 0.002, 0.8, device="cpu")
    assert np.abs(deconv.richardson_lucy(big, psf, 10, 0.002, 0.8, device="auto") - cpu).max() < 3e-4
    s = tgv.noise_sigma(big)
    a, _ = tgv.tgv_channel(big, 2.0, sigma=s, iterations=50, tol=0.0, device="cpu")
    b, _ = tgv.tgv_channel(big, 2.0, sigma=s, iterations=50, tol=0.0, device="auto")
    assert np.abs(a - b).max() < 1e-4
