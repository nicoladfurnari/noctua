"""GPU support (NVIDIA via CuPy), with work spread over several, even different, GPUs.

Work is split into independent tasks (row bands, frames) pulled from a shared queue by one
worker per device: a faster GPU simply takes more tasks, so mixed cards (e.g. a 3070 Ti next to a
slower or older card) share the work in proportion to their speed without any calibration. The only
loss is the tail (the last task running alone on the slow card), so callers should create several
tasks per device (combine_chunked makes >= 4 bands per GPU). Memory is handled per device too: a
task larger than what fits in that card's free memory is split by the caller (see fit_rows), so a
smaller card never limits the band size of a bigger one.
Everything falls back to the CPU when CuPy or a GPU is missing.

GPU code is written against an array module ``xp`` (cupy or numpy, see array_module), so the very
same kernels run (and are tested) on the CPU with numpy.
"""
import os
import queue
import threading

_STATE = {"enabled": os.environ.get("ASTROFORGE_GPU", "1") != "0", "devices": None}


def cupy():
    """The cupy module, or None when it cannot be imported (the result is cached)."""
    if "cupy" not in _STATE:
        try:
            import warnings
            with warnings.catch_warnings():
                # cupy-cuda12x[ctk] ships its own CUDA: the "CUDA path could not be detected" warning is harmless
                warnings.filterwarnings("ignore", message="CUDA path could not be detected")
                import cupy as cp
        except Exception:
            cp = None
        _STATE["cupy"] = cp
    return _STATE["cupy"]


def array_module(a):
    """cupy for CuPy arrays, numpy otherwise (also when CuPy is not installed)."""
    cp = cupy()
    if cp is not None:
        return cp.get_array_module(a)
    import numpy
    return numpy


def mem_info():
    """(free, total) bytes of the current CuPy device; free includes blocks cached by CuPy's pool."""
    cp = cupy()
    free, total = cp.cuda.runtime.memGetInfo()
    try:
        free += cp.get_default_memory_pool().free_bytes()
    except Exception:
        pass
    return int(free), int(total)


def free_pool():
    """Give the blocks cached by CuPy's memory pools back to the device (e.g. after an out of memory)."""
    cp = cupy()
    if cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def fit_rows(bytes_per_row, height, fraction=0.6, free=None):
    """Rows of a band whose GPU temporaries (bytes_per_row each) fit in fraction of the free memory
    of the current device (free: bytes, None = ask the device). Always >= 1, at most height."""
    if free is None:
        try:
            free = mem_info()[0]
        except Exception:
            free = 2 << 30
    return int(max(1, min(height, fraction * free // max(1, bytes_per_row))))


def devices():
    """[(index, name, memory_GB)] of usable GPUs; empty when disabled or unavailable."""
    if not _STATE["enabled"]:
        return []
    if _STATE["devices"] is None:
        cp = cupy()
        found = []
        if cp is not None:
            try:
                for i in range(cp.cuda.runtime.getDeviceCount()):
                    p = cp.cuda.runtime.getDeviceProperties(i)
                    found.append((i, p["name"].decode(), round(p["totalGlobalMem"] / 2 ** 30, 1)))
            except Exception:
                found = []
        sel = os.environ.get("ASTROFORGE_GPUS")       # e.g. "0,1" to choose cards
        if sel:
            keep = {int(x) for x in sel.split(",") if x.strip()}
            found = [d for d in found if d[0] in keep]
        _STATE["devices"] = found
    return _STATE["devices"]


def set_enabled(flag):
    _STATE["enabled"] = bool(flag)
    _STATE["devices"] = None


def describe():
    ds = devices()
    return ", ".join(f"GPU{i} {n} {m}GB" for i, n, m in ds) if ds else "CPU only"


def run_tasks(tasks, gpu_fn, cpu_fn=None, cpu_workers=0):
    """Run every task once. gpu_fn(task, device_index) on GPU workers, cpu_fn(task) on CPU workers.

    With no GPU, cpu_fn runs on max(1, cpu_workers) threads. Exceptions are re-raised.
    """
    q = queue.Queue()
    for t in tasks:
        q.put(t)
    errors = []
    devs = devices()
    workers = []

    def loop(fn):
        while not errors:
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            try:
                fn(t)
            except Exception as e:  # surfaced to the caller below
                errors.append(e)

    def gpu_loop(dev):
        cp = cupy()
        with cp.cuda.Device(dev):
            loop(lambda t: gpu_fn(t, dev))

    for d in devs:
        workers.append(threading.Thread(target=gpu_loop, args=(d[0],), daemon=True))
    n_cpu = cpu_workers if devs else max(1, cpu_workers or 1)
    if cpu_fn is not None:
        for _ in range(n_cpu):
            workers.append(threading.Thread(target=loop, args=(cpu_fn,), daemon=True))
    if not workers:
        raise RuntimeError("no worker available (no GPU and no CPU function)")
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    if errors:
        raise errors[0]


# ------------------------------------------------------------------ helpers for the post-processing kernels
_ONCE = set()


def log_once(msg, log=None):
    """Log msg once per process (logging 'astroforge.gpu' at info level, plus log(msg) if given)."""
    if msg in _ONCE:
        return
    _ONCE.add(msg)
    import logging
    logging.getLogger("astroforge.gpu").info(msg)
    if log is not None:
        log(msg)


def select(device="auto", gpu=True, what="", log=None):
    """Device indices to use for a computation: [] = CPU (numpy).

    device: None / "auto" / True = every usable GPU, int or list of ints = those GPUs, "cpu" / False = CPU.
    gpu=False forces the CPU. A GPU that was asked for but is not usable falls back silently to the CPU
    (logged once)."""
    if not gpu or device is False or device == "cpu" or device == "numpy":
        return []
    avail = [d[0] for d in devices()]
    if device is None or device is True or (isinstance(device, str) and device == "auto"):
        if not avail:
            log_once(f"{what or 'GPU'}: no usable GPU ({'CuPy missing' if cupy() is None else 'none found'}), "
                     "using the CPU", log)
        return avail
    want = [device] if isinstance(device, int) else [int(d) for d in device]
    sel = [d for d in want if d in avail]
    if not sel:
        log_once(f"{what or 'GPU'}: GPU {want} not usable, using the CPU", log)
    return sel


def min_free(devs):
    """Smallest free memory (bytes, CuPy pool included) among the device indices devs."""
    cp = cupy()
    out = []
    for d in devs:
        try:
            with cp.cuda.Device(d):
                out.append(mem_info()[0])
        except Exception:
            out.append(2 << 30)
    return min(out) if out else 2 << 30


def to_host(a):
    """numpy array of a (cupy or numpy)."""
    cp = cupy()
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


def fft_module(xp):
    """scipy.fft for numpy (multi-threaded, float32-native), cupy.fft for cupy."""
    import numpy
    if xp is numpy:
        import scipy.fft
        return scipy.fft
    return xp.fft
