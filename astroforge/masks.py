"""Masks shared by several plugins. All return float32 (H, W) in [0, 1]."""
import cv2
import numpy as np

from . import stars, stretch
from .stats import luminance


def lum_mask(img, linear):
    """Luminance mask; auto-stretched first when the image is still linear."""
    lum = luminance(img)
    if linear:
        lum, _ = stretch.auto_stretch(lum[..., None], 0.25, -2.8)
        lum = lum[..., 0]
    return np.clip(lum, 0, 1).astype(np.float32)


def star_mask(img, linear, grow=2, blur=1.5, scale=1.0):
    """grow/blur are full-resolution pixels; scale < 1 for a downscaled preview (ctx.scale)."""
    if linear:
        img, _ = stretch.auto_stretch(img, 0.25, -2.8)
    return stars.star_mask(img, grow=max(1, round(grow * scale)), blur=max(0.5, blur * scale))


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def blur(mask, sigma):
    return cv2.GaussianBlur(mask, (0, 0), sigma) if sigma > 0 else mask


# ------------------------------------------------------------------ colour spaces (sRGB primaries, D65)
_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                 [0.0193339, 0.1191920, 0.9503041]], np.float32)
_RGB = np.linalg.inv(_XYZ).astype(np.float32)
_WHITE = _XYZ.sum(1)
_E = (6 / 29) ** 3
HUES = {"red": 0.0, "yellow": 60.0, "green": 120.0, "cyan": 180.0, "blue": 240.0, "magenta": 300.0}


def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def _f(t):
    return np.where(t > _E, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)


def _finv(t):
    return np.where(t > 6 / 29, t ** 3, 3 * (6 / 29) ** 2 * (t - 4 / 29))


def rgb_to_lab(img):
    """CIE L*a*b* (L 0..100) of an RGB image whose values are sRGB-encoded (PixInsight default RGBWS)."""
    if img.shape[-1] != 3:
        y = srgb_to_linear(img[..., 0])
        z = np.zeros_like(y)
        return (116 * _f(y) - 16).astype(np.float32), z, z
    xyz = srgb_to_linear(img) @ _XYZ.T / _WHITE
    fx, fy, fz = (_f(xyz[..., i]) for i in range(3))
    return ((116 * fy - 16).astype(np.float32), (500 * (fx - fy)).astype(np.float32),
            (200 * (fy - fz)).astype(np.float32))


def lab_to_rgb(L, a, b):
    fy = (L + 16) / 116
    xyz = np.stack([_finv(fy + a / 500), _finv(fy), _finv(fy - b / 200)], -1) * _WHITE
    return np.clip(linear_to_srgb(xyz @ _RGB.T), 0, 1).astype(np.float32)


def cie_lightness(img):
    """CIE L* / 100, (H, W)."""
    return rgb_to_lab(img)[0] / 100


def hsv(img):
    """Hue in degrees, saturation, value (cv2 HSV on float RGB)."""
    return cv2.cvtColor(np.ascontiguousarray(np.clip(img, 0, 1), np.float32), cv2.COLOR_RGB2HSV)


def hue_window(hue, center, width=60.0, feather=None):
    """1 within +-width/2 of `center` (degrees, circular), cosine falloff to 0 over `feather` (default width/2)."""
    feather = width / 2 if feather is None else feather
    d = np.abs((hue - center + 180) % 360 - 180)
    t = np.clip((d - width / 2) / max(feather, 1e-6), 0, 1)
    return (0.5 * (1 + np.cos(np.pi * t))).astype(np.float32)


# ------------------------------------------------------------------ PixInsight-style masks
def range_mask(img, lower=0.5, upper=1.0, fuzziness=0.0, smoothness=0.0, lightness=True):
    """RangeSelection: 1 where the value is in [lower, upper]. fuzziness 0..1 softens the limits (ramps inside
    the range), smoothness = gaussian blur in pixels. RGB: CIE L* if lightness, else per channel (max)."""
    if img.shape[-1] == 3 and lightness:
        vals = [cie_lightness(img)]
    else:
        vals = [img[..., c] for c in range(img.shape[-1])]
    fz = max(0.0, fuzziness) * (upper - lower) / 2
    out = None
    for v in vals:
        if fz > 0:
            m = smoothstep(v, lower, lower + fz) * (1 - smoothstep(v, upper - fz, upper) if upper < 1 else 1)
        else:
            m = ((v >= lower) & (v <= upper)).astype(np.float32)
        out = m if out is None else np.maximum(out, m)
    return blur(out.astype(np.float32), smoothness)


def color_mask(img, color="red", width_deg=60.0, smooth=0.0):
    """ColorMask script: hue window around a primary/secondary colour, weighted by chroma (normalised to its
    99.5th percentile) so grey pixels and noise stay out. color: name in HUES, a hue in degrees or a list."""
    if img.shape[-1] != 3:
        return np.zeros(img.shape[:2], np.float32)
    h = hsv(img)
    chroma = img.max(-1) - img.min(-1)
    ref = float(np.percentile(chroma[::4, ::4], 99.5))
    w = np.clip(chroma / max(ref, 1e-6), 0, 1)
    out = 0
    for c in (color if isinstance(color, (list, tuple)) else [color]):
        out = np.maximum(out, hue_window(h[..., 0], HUES.get(c, c) if isinstance(c, str) else float(c), width_deg))
    return blur((out * w).astype(np.float32), smooth)


def luminance_mask(img, black=0.0, mid=0.5, white=1.0):
    """CIE L* (value for mono) through a histogram transform (shadows, midtones, highlights)."""
    v = cie_lightness(img) if img.shape[-1] == 3 else img[..., 0]
    x = np.clip((v - black) / max(white - black, 1e-6), 0, 1)
    return np.clip(stretch.mtf(mid, x), 0, 1).astype(np.float32)


def channel_mask(img, channel="G", black=0.0, mid=0.5, white=1.0):
    """One channel (R, G, B or L = luminance) through a histogram transform, like ChannelExtraction + a
    stretched channel used as a mask. A mono image uses its only channel for every name."""
    c = str(channel).upper()
    if img.shape[-1] == 1:
        v = img[..., 0]
    elif c == "L":
        v = luminance(img)
    elif c in "RGB" and len(c) == 1:
        v = img[..., "RGB".index(c)]
    else:
        raise ValueError(f"mask 'channel': channel must be R, G, B or L, not {channel!r}")
    x = np.clip((v - black) / max(white - black, 1e-6), 0, 1)
    return np.clip(stretch.mtf(mid, x), 0, 1).astype(np.float32)


def _is_dab(s):
    return len(s) >= 3 and all(isinstance(v, (int, float)) for v in s)


def paint_strokes(m, strokes, scale=1.0):
    """Paint soft round brush strokes on mask m (H, W), in place, like the Dynamic Paint Brush script.

    strokes: list of dabs [x, y, radius, value, softness] or of strokes = lists of dabs (a drag of the brush:
    its dabs are joined by segments and the stroke is composited once, so overlapping dabs do not pile up).
    x, y, radius are full-resolution pixels (continuous coordinates: pixel i covers [i, i + 1)) multiplied by
    scale. value 0..1 is the grey painted (1 = white = processed, 0 = protected); softness 0..1 is the soft
    part of the radius (0 = hard edge, 1 = fully soft). Applied in order: out = m * (1 - a) + value * a."""
    h, w = m.shape
    for st in strokes or []:
        dabs = [st] if _is_dab(st) else list(st)
        dabs = [[float(v) for v in d] + [1.0, 0.5][len(d) - 3:] if len(d) < 5 else [float(v) for v in d[:5]]
                for d in dabs if len(d) >= 3]
        if not dabs:
            continue
        value = float(np.clip(dabs[0][3], 0, 1))
        alpha = None
        for k, (x, y, r, _, soft) in enumerate(dabs):
            x0, y0 = (dabs[k - 1][0], dabs[k - 1][1]) if k else (x, y)
            x, y, x0, y0 = x * scale, y * scale, x0 * scale, y0 * scale
            r = max(float(r) * scale, 0.5)
            soft = float(np.clip(soft, 0, 1))
            ia, ib = int(max(0, np.floor(min(x, x0) - r - 1))), int(min(w, np.ceil(max(x, x0) + r + 1)))
            ja, jb = int(max(0, np.floor(min(y, y0) - r - 1))), int(min(h, np.ceil(max(y, y0) + r + 1)))
            if ia >= ib or ja >= jb:
                continue
            yy, xx = np.mgrid[ja:jb, ia:ib].astype(np.float32) + 0.5
            dx, dy = x - x0, y - y0
            ll = dx * dx + dy * dy
            t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / ll, 0, 1) if ll > 0 else 0.0
            d = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
            core = r * (1 - soft)
            if r - core < 1:           # hard edge: one pixel of antialiasing
                a = 1 - smoothstep(d, r - 0.5, r + 0.5)
            else:
                a = 1 - smoothstep(d, core, r)
            if alpha is None:
                alpha = np.zeros((h, w), np.float32)
            np.maximum(alpha[ja:jb, ia:ib], a, out=alpha[ja:jb, ia:ib])
        if alpha is not None:
            m *= 1 - alpha
            m += value * alpha
    return m


_SPEC_KEYS = {"range": {"lower", "upper", "fuzziness", "smoothness", "lightness"},
              "color": {"color", "width", "smooth"}, "luminance": {"black", "mid", "white"},
              "channel": {"channel", "black", "mid", "white"},
              "stars": {"grow", "soft"}, "lum": set(), "layer": {"name"}, "paint": {"base", "strokes"}}


def make_mask(img, spec, ctx=None):
    """Mask (H, W) in [0, 1] for a per-step mask: out = img * (1 - m) + processed * m.

    spec = {"type": ..., <type params>, "invert": False, "blur": 0 (px), "strength": 1.0, "stretch": auto}
    (for "paint", invert/blur/strength act on the painted result; the base has its own)
      "range":     lower 0.5, upper 1.0, fuzziness 0, smoothness 0 (px), lightness True   (RangeSelection)
      "color":     color "red"|"yellow"|"green"|"cyan"|"blue"|"magenta" (or hue deg, or a list),
                   width 60 (deg), smooth 0 (px)                                           (ColorMask)
      "luminance": black 0, mid 0.5, white 1                       (CIE L* + histogram transform)
      "stars":     grow 2, soft 1.5 (px)                           (star mask)
      "lum":       auto-stretched luminance
      "layer":     name -> ctx.layers[name] (e.g. "hdr_mask_1"), resized to the image
      "channel":   channel "R"|"G"|"B"|"L" (L = luminance), black 0, mid 0.5, white 1
                   (one channel, e.g. the green one, through a histogram transform)
      "paint":     base = another mask spec (or None = black), strokes = [[x, y, radius, value, softness], ...]
                   or a list of strokes, each a list of such dabs (one drag of the brush), drawn in order on
                   the base with a soft round brush (Dynamic Paint Brush): value 1 = paint white (process),
                   0 = erase (protect); softness 0 = hard edge .. 1 = fully soft; x, y, radius in
                   full-resolution pixels. See paint_strokes.
    "stretch" (default = ctx.linear) auto-stretches a linear image before range/color/luminance/channel masks.
    Pixel sizes are full resolution and are multiplied by ctx.scale (GUI preview)."""
    spec = dict(spec or {})
    t = spec.pop("type", "lum")
    invert, sig, strength = spec.pop("invert", False), spec.pop("blur", 0), spec.pop("strength", 1.0)
    scale = getattr(ctx, "scale", 1.0) if ctx is not None else 1.0
    linear = getattr(ctx, "linear", False) if ctx is not None else False
    do_stretch = spec.pop("stretch", linear)
    if t not in _SPEC_KEYS:
        raise ValueError(f"unknown mask type '{t}' ({', '.join(_SPEC_KEYS)})")
    bad = set(spec) - _SPEC_KEYS[t]
    if bad:
        raise ValueError(f"mask '{t}': unknown keys {sorted(bad)}")
    if t == "paint":
        base = spec.get("base")
        m = (make_mask(img, base, ctx) if base else np.zeros(img.shape[:2], np.float32)).copy()
        m = paint_strokes(m, spec.get("strokes") or [], scale)
        return _finish(m, sig * scale, invert, strength)
    src = (stretch.auto_stretch(img, 0.25, -2.8)[0]
           if do_stretch and t in ("range", "color", "luminance", "channel") else img)
    g = spec.get
    if t == "range":
        m = range_mask(src, g("lower", 0.5), g("upper", 1.0), g("fuzziness", 0.0), g("smoothness", 0.0) * scale,
                       g("lightness", True))
    elif t == "color":
        m = color_mask(src, g("color", "red"), g("width", 60.0), g("smooth", 0.0) * scale)
    elif t == "luminance":
        m = luminance_mask(src, g("black", 0.0), g("mid", 0.5), g("white", 1.0))
    elif t == "channel":
        m = channel_mask(src, g("channel", "G"), g("black", 0.0), g("mid", 0.5), g("white", 1.0))
    elif t == "stars":
        m = star_mask(img, linear, g("grow", 2), g("soft", 1.5), scale)
    elif t == "lum":
        m = lum_mask(img, linear)
    else:
        m = np.asarray(ctx.layers[g("name")], np.float32)
        m = m[..., 0] if m.ndim == 3 else m
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    return _finish(m, sig * scale, invert, strength)


def _finish(m, sigma, invert, strength):
    m = blur(np.ascontiguousarray(m, np.float32), sigma)
    if invert:
        m = 1 - m
    return np.clip(m * strength, 0, 1).astype(np.float32)
