"""External programs, run only where config.json points to an installed copy."""
from pathlib import Path

import numpy as np
import pytest

from astroforge import pipeline, plugins
from astroforge.plugins.base import Context

TOOLS = pipeline.load_config().get("tools", {})


def _needs(name):
    return pytest.mark.skipif(not (TOOLS.get(name) and Path(TOOLS[name]).exists()), reason=f"{name} not installed")


def _field(seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    img = np.full((256, 256, 3), 0.03, np.float32)
    stars = np.zeros((256, 256), bool)
    for _ in range(40):
        x, y = rng.uniform(10, 246, 2)
        img += (0.6 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 5.0))[..., None]
        stars |= (xx - x) ** 2 + (yy - y) ** 2 < 4
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32), stars


@_needs("starnet")
@pytest.mark.parametrize("linear", [True, False])
def test_starnet_removes_stars(linear):
    img, stars = _field()
    src = img if linear else np.sqrt(img)
    ctx = Context(log=lambda s: None, tools=TOOLS, linear=linear)
    logs = []
    ctx.log = logs.append
    out = plugins.get("stars.starnet").run(src, ctx, stride=256, upsample=False)
    assert out.shape == src.shape
    assert out[stars].mean() < 0.5 * src[stars].mean()
    assert "stars" in ctx.layers
    assert any("from StarNet (unscreen)" in m for m in logs)              # StarNet 2.6+ star layer used
    st = ctx.layers["stars"]
    back = out + st if linear else 1 - (1 - out) * (1 - st)               # how stars.recombine puts them back
    assert np.abs(back - src)[stars].mean() < 0.01


@_needs("deepsnr")
def test_deepsnr_lowers_noise():
    img, stars = _field()
    ctx = Context(log=lambda s: None, tools=TOOLS, linear=True)
    out = plugins.get("ext.deepsnr").run(img, ctx, amount=1.0, model=2, stride=256)
    assert out.shape == img.shape
    assert out[~stars].std() < img[~stars].std()


@_needs("cosmic_clarity")
@pytest.mark.parametrize("mode", ["sharpen", "denoise"])
def test_cosmic_clarity_runs(mode):
    img, stars = _field()
    src = np.sqrt(img)                      # Cosmic Clarity works on stretched data
    ctx = Context(log=lambda s: None, tools=TOOLS, linear=False)
    pl = plugins.get("ext.cosmic_clarity")
    out = pl.run(src, ctx, **pl.resolve({"mode": mode}))
    assert out.shape == src.shape and np.isfinite(out).all()
    assert np.abs(out - src).mean() > 1e-4          # the program really processed the image
    if mode == "denoise":
        assert out[~stars].std() < src[~stars].std()
