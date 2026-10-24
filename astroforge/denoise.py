"""Starlet (a trous B3-spline) wavelet noise reduction, MLT-like.

CPU: OpenCV separable filters. GPU (CuPy): the same transform written against xp (shifted slices of a
reflect-padded array), whole channels when they fit, else row bands with an exact halo; channels / bands are
spread over the GPUs with gpu.run_tasks."""
import cv2
import numpy as np

from . import gpu as _gpu

# noise of each starlet layer for unit gaussian noise (Starck & Murtagh)
_LAYER_NOISE = [0.890, 0.201, 0.086, 0.042, 0.021, 0.010]
_B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16
MIN_GPU_PIXELS = 1 << 18      # smaller channels stay on the CPU


def _atrous_kernel(scale):
    step = 2 ** scale
    k = np.zeros(4 * step + 1, np.float32)
    k[::step] = _B3
    return k


def _smooth_xp(c, step):
    """cv2.sepFilter2D(c, -1, k, k, borderType=BORDER_REFLECT) with the a trous kernel of this step, for numpy
    or cupy: 5 shifted slices per axis of a reflect ("symmetric") padded array, horizontal pass first."""
    xp = _gpu.array_module(c)
    r = 2 * step
    h, w = c.shape
    p = xp.pad(c, ((0, 0), (r, r)), mode="symmetric")
    t = p[:, 0:w] * _B3[0]
    for i in range(1, 5):
        t += p[:, i * step:i * step + w] * _B3[i]
    p = xp.pad(t, ((r, r), (0, 0)), mode="symmetric")
    out = p[0:h] * _B3[0]
    for i in range(1, 5):
        out += p[i * step:i * step + h] * _B3[i]
    return out


def halo(layers):
    """Rows of context that make a band of a `layers`-level starlet exact: sum of the kernel radii."""
    return 2 * (2 ** int(layers) - 1)


def starlet(ch, layers, xp_code=None):
    """Return [w1..wn, residual] for one channel (numpy: OpenCV; cupy array: on its GPU).
    xp_code=True forces the xp code (the GPU one) also for numpy arrays."""
    xp = _gpu.array_module(ch)
    generic = xp is not np if xp_code is None else xp_code
    out = []
    c = ch
    for j in range(layers):
        if generic:
            smooth = _smooth_xp(c, 2 ** j)
        else:
            k = _atrous_kernel(j)
            smooth = cv2.sepFilter2D(c, -1, k, k, borderType=cv2.BORDER_REFLECT)
        out.append(c - smooth)
        c = smooth
    out.append(c)
    return out


def _sigma(w1):
    xp = _gpu.array_module(w1)
    return float(xp.median(xp.abs(w1 - xp.median(w1)))) / 0.6745 / _LAYER_NOISE[0]


def denoise_channel(ch, thresholds, amounts, sigma=None, xp_code=None):
    """Soft attenuation of the small starlet coefficients; ch numpy or cupy. sigma: noise of the channel (None =
    measured on the first layer)."""
    layers = starlet(ch, len(thresholds), xp_code)
    if sigma is None:
        sigma = _sigma(layers[0][::4, ::4])
    xp = _gpu.array_module(ch)
    rec = layers[-1].copy()
    for j, (t, a) in enumerate(zip(thresholds, amounts)):
        w = layers[j]
        small = xp.abs(w) < t * sigma * _LAYER_NOISE[j]
        w = xp.where(small, w * (1 - a), w)
        rec += w
    return rec


def _bands(h, rows):
    return [(y, min(h, y + rows)) for y in range(0, h, rows)]


def _denoise_xp(img, thresholds, amounts, devs=(), rows=None, xp=None):
    """Every channel with the xp code: on the GPUs devs (gpu.run_tasks) or with xp (numpy) here. rows: band
    height (None = whole channels if they fit in the GPU memory)."""
    h, w, nc = img.shape
    L = len(thresholds)
    hl = halo(L)
    if rows is None:
        rows = h
        if devs:
            fit = _gpu.fit_rows(4 * w * (L + 8), h, free=_gpu.min_free(devs))
            rows = h if fit >= h else max(16, fit - 2 * hl)
    rows = max(1, min(int(rows), h))
    out = np.empty(img.shape, np.float32)
    bands = _bands(h, rows)
    sig = [None] * nc

    def sigma_band(task, mod):            # first layer only, rows / columns on the global ::4 grid
        c, (y0, y1) = task
        a0, a1 = max(0, y0 - 2), min(h, y1 + 2)
        ch = mod.asarray(np.ascontiguousarray(img[a0:a1, :, c], np.float32))
        w1 = (ch - _smooth_xp(ch, 1))[y0 - a0:y1 - a0]
        return _gpu.to_host(w1[(-y0) % 4::4, ::4])

    def band(task, mod):
        c, (y0, y1) = task
        a0, a1 = max(0, y0 - hl), min(h, y1 + hl)
        ch = mod.asarray(np.ascontiguousarray(img[a0:a1, :, c], np.float32))
        r = denoise_channel(ch, thresholds, amounts, sig[c], xp_code=True)
        out[y0:y1, :, c] = _gpu.to_host(r[y0 - a0:y1 - a0])

    def run(fn, tasks):
        res = {}
        if devs:
            cp = _gpu.cupy()
            _gpu.run_tasks(tasks, lambda t, d: res.__setitem__(t, fn(t, cp)))
        else:
            for t in tasks:
                res[t] = fn(t, xp or np)
        return res

    if len(bands) > 1:                    # the noise is measured on the whole channel
        parts = run(sigma_band, [(c, b) for c in range(nc) for b in bands])
        for c in range(nc):
            sig[c] = _sigma(np.concatenate([parts[(c, b)] for b in bands]))
    run(band, [(c, b) for c in range(nc) for b in bands])
    if devs:
        _gpu.free_pool()
    return out


def denoise(img, mask=None, thresholds=(3.0, 2.0, 1.0, 0.5), amounts=(0.6, 0.5, 0.4, 0.3), device="auto",
            gpu=True, log=None):
    """Wavelet noise reduction. `mask` (H,W) in [0,1]: 1 = full denoise, 0 = untouched.
    device: "auto" (every GPU when CuPy works, else the CPU), GPU index or list, "cpu"; gpu=False forces the CPU;
    "numpy" runs the GPU code with numpy (tests)."""
    out = None
    devs = _gpu.select(device, gpu, "wavelet denoise", log) if img.shape[0] * img.shape[1] >= MIN_GPU_PIXELS else []
    if device == "numpy":
        out = _denoise_xp(img, thresholds, amounts, xp=np).astype(img.dtype)
    elif devs:
        try:
            out = _denoise_xp(img, thresholds, amounts, devs).astype(img.dtype)
        except Exception as e:      # out of memory, driver...: the CPU gives the same result
            _gpu.free_pool()
            _gpu.log_once(f"wavelet denoise: GPU failed ({type(e).__name__}: {e}), using the CPU", log)
    if out is None:
        out = np.empty_like(img)
        for c in range(img.shape[-1]):
            out[..., c] = denoise_channel(img[..., c], thresholds, amounts)
    if mask is not None:
        m = mask[..., None]
        out = img * (1 - m) + out * m
    np.clip(out, 0, 1, out=out)
    return out
