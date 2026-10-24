"""The GUI live preview runs on a downscaled image: pixel-sized params must be scaled to match."""
import cv2
import numpy as np

from astroforge import pipeline, plugins
from astroforge.plugins.base import Context


def _stretched_field(h=800, w=1200, seed=2):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w, 3), 0.12, np.float32)
    img += (0.25 * np.exp(-(((xx - 600) / 260) ** 2 + ((yy - 400) / 180) ** 2)))[..., None] * [1, 0.5, 0.6]
    for _ in range(250):
        x, y, a, s = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.2, 0.8), rng.uniform(1.5, 5)
        img += (a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s ** 2)))[..., None]
    return np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1).astype(np.float32)


STEPS = [{"plugin": "contrast.hdr", "params": {"scale": 96}},
         {"plugin": "contrast.dark_structure", "params": {"scale": 32, "amount": 0.8}},
         {"plugin": "stars.reduce", "params": {"size": 9, "amount": 0.9}},
         {"plugin": "cosmetic.chroma_noise", "params": {"sigma": 8}}]


def _small(img, f):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def test_scaled_params():
    pl = plugins.get("stars.remove_basic")
    p = pl.scaled(pl.resolve({}), 0.25)
    assert p["size"] % 2 == 1 and p["size"] >= 5 and p["scales"] == 2
    dbe = plugins.get("background.dbe")
    assert dbe.scaled(dbe.resolve({"samples": [[400, 800]]}), 0.25)["samples"] == [[100, 200]]


def test_preview_matches_full_result():
    full = _stretched_field()
    f = 0.25
    ref = _small(pipeline.run_steps(full, STEPS, Context(log=lambda m: None, linear=False)), f)
    small = _small(full, f)
    scaled = pipeline.run_steps(small, STEPS, Context(log=lambda m: None, linear=False, scale=f))
    unscaled = pipeline.run_steps(small, STEPS, Context(log=lambda m: None, linear=False))
    err_scaled = float(np.abs(scaled - ref).mean())
    err_unscaled = float(np.abs(unscaled - ref).mean())
    assert err_scaled < 0.6 * err_unscaled, (err_scaled, err_unscaled)
