"""Channel masks and painted masks (Dynamic Paint Brush style)."""
import numpy as np
import pytest

from astroforge import masks, stretch
from astroforge.plugins import Context


def _img(h=100, w=120, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (h, w, 3)).astype(np.float32)


def test_channel_mask_equals_stretched_channel():
    img = _img()
    m = masks.make_mask(img, {"type": "channel", "channel": "G", "black": 0.1, "mid": 0.3, "white": 0.9})
    x = np.clip((img[..., 1] - 0.1) / 0.8, 0, 1)
    assert np.allclose(m, stretch.mtf(0.3, x), atol=1e-6)
    # defaults: the plain channel; R / B / L
    assert np.allclose(masks.make_mask(img, {"type": "channel", "channel": "r"}), img[..., 0], atol=1e-6)
    assert np.allclose(masks.make_mask(img, {"type": "channel", "channel": "B"}), img[..., 2], atol=1e-6)
    from astroforge.stats import luminance
    assert np.allclose(masks.make_mask(img, {"type": "channel", "channel": "L"}), luminance(img), atol=1e-6)
    # linear image: auto-stretched first (like range / luminance masks)
    lin = (img * 0.05).astype(np.float32)
    ctx = Context(linear=True)
    st = stretch.auto_stretch(lin, 0.25, -2.8)[0]
    assert np.allclose(masks.make_mask(lin, {"type": "channel", "channel": "G"}, ctx), st[..., 1], atol=1e-5)
    # mono: its only channel; bad name: error
    mono = img[..., :1]
    assert np.allclose(masks.make_mask(mono, {"type": "channel", "channel": "B"}), mono[..., 0])
    with pytest.raises(ValueError):
        masks.make_mask(img, {"type": "channel", "channel": "X"})
    with pytest.raises(ValueError):
        masks.make_mask(img, {"type": "channel", "nope": 1})


def test_paint_adds_and_erases():
    img = _img()
    # black base, one hard white dab at (30, 40) r 10
    m = masks.make_mask(img, {"type": "paint", "base": None, "strokes": [[30, 40, 10, 1, 0]]})
    assert m.shape == (100, 120)
    assert m[40, 30] == 1 and m[40, 37] == 1 and m[40, 45] == 0 and m[10, 100] == 0
    assert abs(m.sum() - np.pi * 100) < 15             # area of the disc
    # white base (range lower 0 = everything) and an erase dab
    full = {"type": "range", "lower": 0.0, "upper": 1.0}
    m = masks.make_mask(img, {"type": "paint", "base": full, "strokes": [[60, 50, 8, 0, 0]]})
    assert m[50, 60] == 0 and m[50, 75] == 1 and m.mean() > 0.95
    # in order: paint then erase part of it
    m = masks.make_mask(img, {"type": "paint", "base": None,
                              "strokes": [[50, 50, 20, 1, 0], [50, 50, 5, 0, 0]]})
    assert m[50, 50] == 0 and m[50, 62] == 1 and m[50, 75] == 0
    # grey value
    m = masks.make_mask(img, {"type": "paint", "strokes": [[50, 50, 10, 0.4, 0]]})
    assert abs(m[50, 50] - 0.4) < 1e-6
    # the base keeps its own parameters, the outer invert works on the painted result
    m = masks.make_mask(img, {"type": "paint", "base": None, "strokes": [[30, 40, 10, 1, 0]], "invert": True})
    assert m[40, 30] == 0 and m[10, 100] == 1


def test_paint_softness():
    img = _img()
    hard = masks.make_mask(img, {"type": "paint", "strokes": [[60, 50, 20, 1, 0]]})
    soft = masks.make_mask(img, {"type": "paint", "strokes": [[60, 50, 20, 1, 1]]})
    half = masks.make_mask(img, {"type": "paint", "strokes": [[60, 50, 20, 1, 0.5]]})
    assert soft[50, 60] > 0.99 and hard[50, 60] == 1 and half[50, 60] == 1
    # profile along x: hard is a step at r, soft falls off smoothly from the center, half keeps a core of r/2
    assert hard[50, 60 + 15] == 1 and 0.2 < soft[50, 60 + 10] < 0.8 and half[50, 60 + 8] == 1
    assert 0 < half[50, 60 + 15] < 1 and soft[50, 60 + 25] == 0 and half[50, 60 + 25] == 0
    prof = soft[50, 60:85]
    assert np.all(np.diff(prof) <= 1e-6)                 # monotonic falloff
    assert soft.sum() < half.sum() < hard.sum()


def test_paint_stroke_groups_do_not_pile_up():
    img = _img()
    # a stroke = list of dabs joined by segments: no gaps between distant dabs, and a soft stroke of value 1
    # composited once (overlapping dabs inside it do not add up)
    stroke = [[20, 50, 6, 1, 0], [80, 50, 6, 1, 0]]
    m = masks.make_mask(img, {"type": "paint", "strokes": [stroke]})
    assert m[50, 20:80].min() == 1 and m[50, 95] == 0 and m[30, 50] == 0
    soft = [[50, 50, 10, 0.5, 1], [51, 50, 10, 0.5, 1], [52, 50, 10, 0.5, 1]]
    one = masks.make_mask(img, {"type": "paint", "strokes": [soft[:1]]})
    grp = masks.make_mask(img, {"type": "paint", "strokes": [soft]})
    assert grp.max() <= 0.5 + 1e-6 and abs(grp[50, 50] - one[50, 50]) < 0.05
    # flat dabs (not grouped) are composited one after the other
    flat = masks.make_mask(img, {"type": "paint", "strokes": soft})
    assert flat[50, 51] > grp[50, 51] - 1e-6


def test_paint_scales_with_ctx_scale():
    big = _img(200, 240)
    small = big[::2, ::2].copy()
    spec = {"type": "paint", "base": None, "strokes": [[100, 80, 30, 1, 0.5], [[160, 150, 10, 0, 0]]]}
    full = masks.make_mask(big, spec)
    prev = masks.make_mask(small, spec, Context(scale=0.5))
    # same mask, downscaled: compare with 2x2 means
    ref = full.reshape(100, 2, 120, 2).mean((1, 3))
    assert np.abs(ref - prev).mean() < 0.01
    assert prev[40, 50] == 1 and prev[40, 70] == 0       # dab center (100, 80) -> (50, 40), radius 15
    # radius never below half a pixel
    tiny = masks.make_mask(small, {"type": "paint", "strokes": [[100, 80, 0.2, 1, 0]]}, Context(scale=0.5))
    assert tiny.max() > 0
    # paint on a channel base, scaled blur
    m = masks.make_mask(small, {"type": "paint", "base": {"type": "channel", "channel": "G"},
                                "strokes": [[100, 80, 20, 1, 0]], "blur": 2}, Context(scale=0.5))
    assert m.shape == small.shape[:2] and m[40, 50] > 0.95
