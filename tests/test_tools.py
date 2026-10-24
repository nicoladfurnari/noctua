"""Tests for the utility plugins. Run: .venv\\Scripts\\python.exe -m pytest tests"""
import numpy as np
import pytest

from astroforge import plugins
from astroforge.plugins.tools import pixelmath

CTX = plugins.Context(log=lambda m: None)
IMG = np.random.default_rng(0).random((60, 80, 3)).astype(np.float32)


def run(pid, img=IMG, **params):
    pl = plugins.get(pid)
    return pl.run(img, CTX, **pl.resolve(params))


def test_pixelmath_iif():
    out = run("tools.pixelmath", expression="iif(T > 0.5, T, T*0.5)")
    assert np.allclose(out, np.where(IMG > 0.5, IMG, IMG * 0.5))


def test_pixelmath_per_channel():
    out = run("tools.pixelmath", expr_r="0.5*R+0.5*B")
    assert np.allclose(out[..., 0], 0.5 * IMG[..., 0] + 0.5 * IMG[..., 2])
    assert np.allclose(out[..., 1], IMG[..., 1])


@pytest.mark.parametrize("bad", ['__import__("os")', "T.real", "[1,2]", 'open("x")', "(lambda: 1)()", "X + 1"])
def test_pixelmath_blocks_code(bad):
    with pytest.raises(ValueError):
        pixelmath(bad, {"T": IMG[..., 0]})


def test_statistical_stretch_hits_target():
    out = run("stretch.statistical", img=IMG * 0.01, target=0.25)
    assert abs(float(np.median(out)) - 0.25) < 0.05


def test_web_export_resize_and_text():
    out = run("export.web", long_side=40, text="AstroForge")
    assert max(out.shape[:2]) == 40 and out.shape[-1] == 3


@pytest.mark.parametrize("method", ["average_neutral", "maximum_neutral", "maximum_mask", "additive_mask"])
def test_scnr_removes_green_cast(method):
    img = np.full((20, 20, 3), [0.3, 0.5, 0.3], np.float32)   # strong green cast on a grey background
    out = run("color.scnr", img=img, amount=1.0, method=method)
    assert out[..., 1].mean() < 0.5 and np.allclose(out[..., 0], 0.3) and np.allclose(out[..., 2], 0.3)
    if method.endswith("neutral"):
        assert abs(out[..., 1].mean() - 0.3) < 1e-6


def test_scnr_keeps_neutral_and_lightness():
    grey = np.full((8, 8, 3), 0.4, np.float32)
    assert np.allclose(run("color.scnr", img=grey, amount=1.0), grey)
    from astroforge.stats import luminance
    img = np.full((8, 8, 3), [0.3, 0.5, 0.3], np.float32)
    out = run("color.scnr", img=img, amount=1.0, preserve_lightness=True)
    assert np.allclose(luminance(out), luminance(img), atol=1e-4) and out[..., 1].mean() <= out[..., 0].mean() + 1e-6


def test_mtf_never_nan_at_extreme_midtones():
    from astroforge import stretch
    x = np.linspace(0, 1, 11, dtype=np.float32)
    for m in (0.0, 1.0, 0.5, 1e-9):
        assert np.isfinite(stretch.mtf(m, x)).all()
    flat = np.full((8, 8, 3), 0.5, np.float32)        # no spread: auto params degenerate
    assert np.isfinite(stretch.auto_stretch(flat)[0]).all()
