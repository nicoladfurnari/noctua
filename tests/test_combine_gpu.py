"""GPU rejection (combine_xp / combine_gpu) must give the CPU combine() result.

combine_xp is written against an array module: with xp=numpy it runs the GPU code path on the CPU
(pieces along the rows, host-side grow), so it is tested everywhere. The real-GPU tests run only
where CuPy and a CUDA device are available (home PC):

    .venv\\Scripts\\python.exe -m pytest -q tests/test_combine_gpu.py -rs
"""
import numpy as np
import pytest

from astroforge import gpu
from astroforge.stack import combine as C

METHODS = ("none", "median", "percentile", "sigma", "winsorized", "linear_fit", "esd", "auto")
H, W = 24, 31


def _stack(n, seed=0, rgb=False, nan=True):
    rng = np.random.default_rng(seed)
    shape = (n, H, W, 3) if rgb else (n, H, W)
    st = np.ones(shape, np.float32)
    st *= (1.0 + 0.05 * np.sin(np.arange(W) / 5.0)).astype(np.float32).reshape((W, 1) if rgb else (W,))
    if rgb:
        st *= np.array([1.0, 0.9, 1.1], np.float32)
    st += rng.normal(0, 0.01, shape).astype(np.float32)
    hot = rng.random(shape) < 0.02                                    # hot / cold outliers
    st[hot] += rng.choice([-0.3, 0.2, 0.8], hot.sum()).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    trail = np.abs(yy - 0.6 * xx - 3) < 1.0
    st[n // 2][trail] += 0.3                                          # satellite trail in one frame
    if nan:
        st[0, :, :8] = np.nan                                         # frame not covering the left side
        st[: max(1, n - 2), :3, -4:] = np.nan                         # only 2 frames cover this corner
        st[:, 20:22, 10:13] = np.nan                                  # nobody covers this
        st[rng.random(shape) < 0.03] = np.nan                         # sparse holes
    return st


def _same(a, b, tol=1e-5):
    for x, y in zip(a, b):
        assert x.shape == y.shape and x.dtype == y.dtype
        if x.dtype == np.float32:
            assert np.array_equal(np.isnan(x), np.isnan(y))
            assert np.nanmax(np.abs(x - y), initial=0) <= tol
        else:
            assert np.array_equal(x, y)


@pytest.mark.parametrize("n", [5, 12, 25])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("weighted", [False, True])
def test_xp_numpy_matches_cpu(method, n, weighted):
    st = _stack(n, seed=n)
    w = np.linspace(0.4, 1.6, n) if weighted else None
    ref = C.combine(st, method, weights=w)
    for rows in (None, 7, 1):                       # whole band and pieces (smaller GPU memory)
        _same(C.combine_xp(st, method, weights=w, xp=np, rows=rows), ref)
    if method not in ("none", "median"):
        assert ref[1].sum() + ref[2].sum() > 0      # the data really exercises the rejection


@pytest.mark.parametrize("method", ("sigma", "winsorized", "linear_fit", "esd"))
def test_xp_numpy_rgb_low_high_iterations(method):
    st = _stack(12, seed=3, rgb=True)
    kw = dict(low=2.5, high=2.0, iterations=3, weights=np.arange(1, 13))
    _same(C.combine_xp(st, method, xp=np, rows=5, **kw), C.combine(st, method, **kw))


@pytest.mark.parametrize("rgb", [False, True])
@pytest.mark.parametrize("method", ("sigma", "winsorized", "esd"))
def test_xp_numpy_grow_matches_cpu(method, rgb):
    st = _stack(12, seed=5, rgb=rgb)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.abs(yy - 0.6 * xx - 3)
    wing = (d >= 1.0) & (d < 3.0)
    st[6][wing] += 0.006                            # faint trail wings: only grow rejects them
    ref = C.combine(st, method, grow_px=3)
    assert ref[2].sum() > C.combine(st, method)[2].sum()
    _same(C.combine_xp(st, method, grow_px=3, xp=np, rows=4), ref)   # grown across the pieces


def test_esd_kwargs_and_1d():
    st = _stack(25, seed=8)
    kw = dict(esd_outliers=0.2, esd_alpha=0.01)
    _same(C.combine_xp(st, "esd", xp=np, rows=6, **kw), C.combine(st, "esd", **kw))
    v = st[:, 4, 9:12].copy()                       # (N, P) stacks without rows
    _same(C.combine_xp(v[:, 0], "sigma", xp=np), C.combine(v[:, 0], "sigma"))


def test_fit_rows_and_gpu_methods():
    per = C.gpu_bytes_per_row(30, 6000 * 3, "sigma")
    assert per == 30 * 6000 * 3 * C.GPU_BYTES
    assert gpu.fit_rows(per, 4000, fraction=0.6, free=8 << 30) == int(0.6 * (8 << 30) // per)
    assert gpu.fit_rows(per, 4000, free=1) == 1 and gpu.fit_rows(1, 50, free=1 << 30) == 50
    assert set(C.GPU_METHODS) == set(C.METHODS) - {"auto"}
    assert C.gpu_bytes_per_row(30, 10, "esd") > C.gpu_bytes_per_row(30, 10, "sigma")


# ---- real GPU (skipped without CuPy and a CUDA device)

def _cupy_device():
    cp = gpu.cupy()
    if cp is None:
        return None
    try:
        return cp if cp.cuda.runtime.getDeviceCount() > 0 else None
    except Exception:
        return None


needs_gpu = pytest.mark.skipif(_cupy_device() is None, reason="needs CuPy and a CUDA GPU")


def _close(a, b, tol=1e-5, max_frac=1e-3):
    # float sums differ in the last bits between GPU and CPU: a value exactly on a threshold may be
    # kept on one and rejected on the other, so allow a tiny fraction of differing pixels
    for x, y in zip(a, b):
        assert x.shape == y.shape and x.dtype == y.dtype
        if x.dtype == np.float32:
            assert np.array_equal(np.isnan(x), np.isnan(y))
            bad = np.abs(np.nan_to_num(x) - np.nan_to_num(y)) > tol
        else:
            bad = x != y
        assert bad.mean() <= max_frac, f"{bad.sum()} pixels differ"


@needs_gpu
@pytest.mark.parametrize("n", [5, 12, 25])
@pytest.mark.parametrize("method", METHODS)
def test_gpu_matches_cpu(method, n):
    cp = _cupy_device()
    st = _stack(n, seed=n, rgb=True)
    w = np.linspace(0.4, 1.6, n)
    ref = C.combine(st, method, weights=w)
    with cp.cuda.Device(0):
        _close(C.combine_gpu(st, method, weights=w), ref)
        _close(C.combine_xp(st, method, weights=w, xp=cp, rows=5), ref)
        _close(C.combine_gpu(st, method, weights=w, grow_px=2), C.combine(st, method, weights=w, grow_px=2))


@needs_gpu
def test_gpu_chunked_all_devices():
    st = _stack(12, seed=11)

    class Loader:
        height = st.shape[1]

        def __call__(self, i, y0, y1):
            return st[i, y0:y1]

    for method in ("winsorized", "linear_fit", "esd"):
        _close(C.combine_chunked(list(range(12)), Loader(), method), C.combine(st, method))
