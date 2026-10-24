"""More PixInsight processes: STF (linked or separate channels), MaskedStretch, AdaptiveStretch, IntegerResample, LarsonSekanina, Convolution,
Invert, Binarize, ConvertToGrayscale, ChannelExtraction, LRGBCombination, ChannelCombination."""
import cv2
import numpy as np

from .. import masks, stretch
from ..stats import luminance, median_mad
from .base import Param, Plugin, register
from .stretch_extra import _preserve_color
from .workflow import align_to, load_image


def _gray(img, mode="luminance"):
    if img.shape[-1] == 1:
        return img[..., 0]
    if mode == "lightness":
        return masks.cie_lightness(img)
    if mode == "average":
        return img.mean(-1)
    return luminance(img)


def _fit_to(ext, img, log, what):
    """External file brought to the current geometry: preview scale, registration, size, channels."""
    if ext.shape[:2] != img.shape[:2]:
        ext = align_to(ext, img, log)
    if ext.shape[:2] != img.shape[:2]:
        log(f"{what}: size {ext.shape[1]}x{ext.shape[0]} resized to {img.shape[1]}x{img.shape[0]}")
        ext = cv2.resize(ext, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        ext = ext if ext.ndim == 3 else ext[..., None]
    return ext.astype(np.float32)


# ------------------------------------------------------------------ stretches
@register
class STF(Plugin):
    id = "stretch.stf"
    name = "STF (ScreenTransferFunction)"
    category = "stretch"
    stage = "linear"
    description = ("PixInsight AutoSTF applied with HistogramTransformation: shadows at median + clip x MAD, "
                   "midtones bringing the background to the target. Channels linked (same curve on R, G, B, "
                   "colors kept) or separate (one curve per channel: also neutralizes the background tint, "
                   "the 'unlinked' STF). Boost = PixInsight's 'Boost' button (brighter, harder).")
    params = {"channels": Param("linked", choices=["linked", "separate"], help="one curve for all channels or "
                                                                               "one per channel"),
              "target": Param(0.25, 0.02, 0.6, "target background (PixInsight: 0.25)"),
              "shadows_clip": Param(-2.8, -10.0, 0.0, "black point in MAD units below the median (PixInsight: "
                                                       "-2.8)"),
              "boost": Param(False, help="like the Boost button: target x1.5, shadows clipping x0.6")}

    def run(self, img, ctx, channels, target, shadows_clip, boost=False):
        if boost:
            target, shadows_clip = min(0.6, target * 1.5), shadows_clip * 0.6
        if channels == "separate" and img.shape[-1] > 1:
            out, logs = np.empty_like(img, dtype=np.float32), []
            for c in range(img.shape[-1]):
                ch = img[..., c:c + 1]
                c0, m = stretch.auto_params(ch, target, shadows_clip)
                out[..., c] = stretch.histogram_transform(ch, c0, m)[..., 0]
                logs.append(f"{'RGB'[c] if img.shape[-1] == 3 else c} c0={c0:.5f} m={m:.5f}")
            ctx.log("STF separate channels: " + ", ".join(logs))
        else:
            c0, m = stretch.auto_params(img, target, shadows_clip)
            out = stretch.histogram_transform(img, c0, m)
            ctx.log(f"STF linked: c0={c0:.5f} m={m:.5f}")
        ctx.linear = False
        return np.clip(out, 0, 1).astype(np.float32)


@register
class MaskedStretch(Plugin):
    id = "stretch.masked"
    name = "MaskedStretch"
    category = "stretch"
    stage = "linear"
    description = ("PixInsight MaskedStretch: many small histogram stretches, each masked by the image itself, so "
                   "the background reaches the target while bright stars and cores are protected (no bloated "
                   "stars, colors kept). Good before StarNet or for star fields.")
    params = {"target": Param(0.125, 0.02, 0.5, "target background median"),
              "iterations": Param(50, 5, 200, "more iterations = smoother protection"),
              "clipping": Param(0.0005, 0.0, 0.05, "fraction of the darkest pixels clipped to black"),
              "color": Param("luminance", choices=["luminance", "rgb"], help="preserve colors or stretch channels")}

    def run(self, img, ctx, target, iterations, clipping, color="luminance"):
        lum = _gray(img)
        c0 = float(np.quantile(lum[::2, ::2], clipping)) if clipping > 0 else 0.0
        x = np.clip((lum if color == "luminance" else img) - c0, 0, None) / max(1 - c0, 1e-6)
        x = np.clip(x, 0, 1).astype(np.float32)
        sub = (slice(None, None, 3), slice(None, None, 3))
        b0 = float(np.median(x[sub])) if x.ndim == 2 else float(np.median(_gray(x)[sub]))
        b0 = max(b0, 1e-6)
        for k in range(1, int(iterations) + 1):
            m_img = x if x.ndim == 2 else _gray(x)
            b = max(float(np.median(m_img[sub])), 1e-7)
            want = b0 ** (1 - k / iterations) * target ** (k / iterations)       # geometric path to the target
            y = np.clip((want - b * b) / max(1 - b, 1e-6), b, 0.999)             # bg after mtf, before the mask
            m = float(stretch.mtf(y, b))                                          # mtf(m, b) == y
            mask = m_img if x.ndim == 2 else m_img[..., None]
            x = (stretch.mtf(m, x) * (1 - mask) + x * mask).astype(np.float32)
        if color == "luminance" and img.shape[-1] == 3:
            base = np.clip((img - c0) / max(1 - c0, 1e-6), 0, None)
            out = _preserve_color(base, np.clip((lum - c0) / max(1 - c0, 1e-6), 0, None), x)
        else:
            out = x if x.ndim == 3 else x[..., None]
        ctx.log(f"MaskedStretch: {iterations} iterations, background {b0:.5f} -> "
                f"{float(np.median(_gray(out)[sub])):.3f}")
        ctx.linear = False
        return np.clip(out, 0, 1).astype(np.float32)


@register
class AdaptiveStretch(Plugin):
    id = "stretch.adaptive"
    name = "AdaptiveStretch"
    category = "stretch"
    stage = "linear"
    description = ("PixInsight AdaptiveStretch: the curve is built from the image itself, giving contrast to the "
                   "brightness levels where neighbouring pixels really differ (signal) and none to those where "
                   "they differ only by noise. Contrast protection keeps the background from going grey; a "
                   "black point, a background ceiling and a share of plain stretch for the highlights keep "
                   "real photos (Milky Way fields, bright cores) under control.")
    params = {"noise_threshold": Param(None, 0.0, 0.1, "differences below this are noise; None = 1 sigma of "
                                                         "the measured noise"),
              "contrast_protection": Param(0.3, 0.0, 1.0, "weight of the noise against the signal"),
              "black_sigma": Param(-2.8, -10.0, 0.0, "black point in noise units from the median"),
              "highlights": Param(0.15, 0.0, 4.0, "share of a plain stretch mixed in (gradation in bright cores)"),
              "color": Param("rgb", choices=["rgb", "luminance"], help="rgb = same curve on every channel like "
                             "PixInsight (bright cores go to white); luminance = colors kept (cores stay saturated)"),
              "max_background": Param(0.12, 0.0, 0.5, "background median ceiling after the curve (0 = free)"),
              "points": Param(4096, 256, 16384, "curve resolution")}

    def run(self, img, ctx, noise_threshold, contrast_protection, black_sigma=-2.8, highlights=0.15,
            max_background=0.12, points=4096, color="rgb"):
        lum0 = np.clip(_gray(img), 0, 1)
        med, mad = median_mad(lum0)
        c0 = max(0.0, med + black_sigma * 1.4826 * mad)
        lum = np.clip((lum0 - c0) / (1 - c0), 0, 1)
        base = np.clip((img - c0) / (1 - c0), 0, None)
        s = lum[::2, ::2] if lum.size > 4e6 else lum
        n = int(points)
        if noise_threshold is None:
            _, d = median_mad(s[:, 1:] - s[:, :-1], 1)
            noise_threshold = 1.4826 * d
        u = lambda v: np.sqrt(np.sqrt(v))         # noqa: E731  levels binned on x^(1/4): linear data get resolution
        pos, neg = np.zeros(n + 1), np.zeros(n + 1)
        for a, b in ((s[:, 1:], s[:, :-1]), (s[1:, :], s[:-1, :])):
            lo = np.minimum(a, b).ravel()
            hi = np.maximum(a, b).ravel()
            sig = (hi - lo) > noise_threshold
            il, ih = (u(lo) * n).astype(np.int64), (u(hi) * n).astype(np.int64)
            for acc, keep in ((pos, sig), (neg, ~sig)):     # +1 from lo to hi: transitions across each level
                np.add.at(acc, il[keep], 1)
                np.add.at(acc, ih[keep], -1)
        pos, neg = np.cumsum(pos)[:n], np.cumsum(neg)[:n]
        noise_share = neg / np.maximum(pos + neg, 1)          # levels made mostly of noise get less contrast
        f = pos * (1 - contrast_protection * noise_share)
        f = f + (highlights + 1e-3) * f.mean()
        curve = np.concatenate([[0], np.cumsum(f)])
        curve /= curve[-1]
        new = np.interp(u(lum), np.linspace(0, 1, n + 1), curve).astype(np.float32)
        bg = float(np.median(new[::2, ::2]))
        grid = np.linspace(0, 1, n + 1)
        m_bg = float(stretch.mtf(max_background, bg)) if max_background and bg > max_background else None
        if m_bg is not None:                                    # Milky Way fields: lots of "signal" at sky level
            new = stretch.mtf(m_bg, new).astype(np.float32)     # midtones, no clip
        if color == "rgb" and img.shape[-1] == 3:
            curve_x = curve if m_bg is None else stretch.mtf(m_bg, curve)
            out = np.stack([np.interp(u(np.clip(base[..., c], 0, 1)), grid, curve_x) for c in range(3)], -1)
        else:
            out = _preserve_color(base, lum, new) if img.shape[-1] == 3 else new[..., None]
        ctx.log(f"AdaptiveStretch: black point {c0:.5f}, noise threshold {noise_threshold:.5f}, background "
                f"{med:.4f} -> {float(np.median(new[::2, ::2])):.3f}" + (f" (ceiling from {bg:.3f})" if bg > max_background > 0 else ""))
        ctx.linear = False
        return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ geometry
@register
class IntegerResample(Plugin):
    id = "geometry.integer_resample"
    name = "IntegerResample"
    category = "geometry"
    description = ("PixInsight IntegerResample: binning by an integer factor (average, median, min or max of each "
                   "block: less noise, smaller file) or enlargement by pixel repetition. Star layers follow.")
    params = {"factor": Param(2, 2, 8, "integer factor"),
              "direction": Param("down", choices=["down", "up"], help="down = bin, up = enlarge"),
              "mode": Param("average", choices=["average", "median", "min", "max"], help="block combination")}

    @staticmethod
    def _one(a, f, direction, mode):
        if direction == "up":
            return np.repeat(np.repeat(a, f, 0), f, 1)
        h, w = a.shape[0] // f * f, a.shape[1] // f * f
        blocks = a[:h, :w].reshape(h // f, f, w // f, f, *a.shape[2:])
        op = {"average": np.mean, "median": np.median, "min": np.min, "max": np.max}[mode]
        return op(blocks, axis=(1, 3)).astype(np.float32)

    def run(self, img, ctx, factor, direction, mode):
        f = int(factor)
        for k, v in list(ctx.layers.items()):     # e.g. the star layer, to be recombined later
            if isinstance(v, np.ndarray) and v.shape[:2] == img.shape[:2]:
                ctx.layers[k] = self._one(v, f, direction, mode)
        out = self._one(img, f, direction, mode)
        ctx.log(f"IntegerResample {'/' if direction == 'down' else 'x'}{f} ({mode}): "
                f"{img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]}")
        return out


# ------------------------------------------------------------------ comets
def _polar_sample(img, cx, cy, dphi, dr):
    """img sampled at the same radius + dr and angle + dphi around (cx, cy)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy) - dr
    phi = np.arctan2(dy, dx) - dphi
    mx = (cx + r * np.cos(phi)).astype(np.float32)
    my = (cy + r * np.sin(phi)).astype(np.float32)
    chans = [cv2.remap(np.ascontiguousarray(img[..., c]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
             for c in range(img.shape[-1])]
    return np.stack(chans, -1)


@register
class LarsonSekanina(Plugin):
    id = "sharpen.larson_sekanina"
    name = "LarsonSekanina"
    category = "sharpen"
    description = ("PixInsight LarsonSekanina: rotational gradient around the comet nucleus (2I minus the image "
                   "rotated by +/- angle and shifted radially) to show jets and shells in the coma.")
    params = {"center": Param(None, help="[[x, y]] of the nucleus in pixels; None = brightest point", px="xy"),
              "angle": Param(10.0, 0.5, 90.0, "rotation, degrees"),
              "radial": Param(0.0, 0.0, 50.0, "radial shift, pixels", px=True),
              "amount": Param(0.8, 0.0, 5.0, "how much of the filter is added"),
              "output": Param("add", choices=["add", "filter"], help="add to the image, or the filter only")}

    def run(self, img, ctx, center, angle, radial, amount, output):
        if center:
            pt = center[0] if isinstance(center[0], (list, tuple)) else center
            cx, cy = float(pt[0]), float(pt[1])
        else:
            sm = cv2.GaussianBlur(np.ascontiguousarray(_gray(img), np.float32), (0, 0), max(2.0, 4 * ctx.scale))
            cy, cx = np.unravel_index(int(np.argmax(sm)), sm.shape)
        a = np.deg2rad(angle)
        f = 2 * img - _polar_sample(img, cx, cy, a, radial) - _polar_sample(img, cx, cy, -a, radial)
        out = 0.5 + amount * f if output == "filter" else img + amount * f
        ctx.log(f"LarsonSekanina around ({cx:.0f}, {cy:.0f}), {angle} deg, radial {radial} px")
        return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ small tools
@register
class Convolution(Plugin):
    id = "tools.convolution"
    name = "Convolution"
    category = "tools"
    description = "PixInsight Convolution: Gaussian, box or Moffat blur (soften a mask, smooth a background)."
    params = {"kind": Param("gaussian", choices=["gaussian", "box", "moffat"], help="filter shape"),
              "sigma": Param(2.0, 0.3, 100.0, "size (Gaussian sigma, box half width), pixels", px=True),
              "beta": Param(4.0, 1.0, 10.0, "Moffat beta (low = longer wings)")}

    def run(self, img, ctx, kind, sigma, beta=4.0):
        x = np.ascontiguousarray(img, np.float32)
        if kind == "gaussian":
            out = cv2.GaussianBlur(x, (0, 0), sigma)
        elif kind == "box":
            k = int(round(sigma)) * 2 + 1
            out = cv2.blur(x, (k, k))
        else:
            r = int(np.ceil(sigma * 4))
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            alpha = sigma / np.sqrt(2 ** (1 / beta) - 1) / 1.1774            # same FWHM as the Gaussian
            ker = (1 + (xx ** 2 + yy ** 2) / alpha ** 2) ** -beta
            out = cv2.filter2D(x, -1, (ker / ker.sum()).astype(np.float32), borderType=cv2.BORDER_REFLECT)
        out = out if out.ndim == 3 else out[..., None]
        ctx.log(f"Convolution {kind} {sigma:.2f} px")
        return out.astype(np.float32)


@register
class Invert(Plugin):
    id = "tools.invert"
    name = "Invert"
    category = "tools"
    description = "PixInsight Invert: negative image (1 - x). Twice gives the original back."
    params = {}

    def run(self, img, ctx):
        return (1 - np.clip(img, 0, 1)).astype(np.float32)


@register
class Binarize(Plugin):
    id = "tools.binarize"
    name = "Binarize"
    category = "tools"
    description = "PixInsight Binarize: white above the threshold, black below (masks of stars or bright areas)."
    params = {"threshold": Param(0.5, 0.0, 1.0, "level"),
              "per_channel": Param(False, help="each channel on its own, else on the luminance")}

    def run(self, img, ctx, threshold, per_channel):
        if per_channel or img.shape[-1] == 1:
            out = (img > threshold).astype(np.float32)
        else:
            out = np.repeat((_gray(img) > threshold)[..., None], img.shape[-1], -1).astype(np.float32)
        ctx.log(f"Binarize at {threshold}: {float(out.mean()) * 100:.1f}% white")
        return out


@register
class Grayscale(Plugin):
    id = "tools.grayscale"
    name = "ConvertToGrayscale"
    category = "tools"
    description = "PixInsight ConvertToGrayscale: one grey channel from the color image."
    params = {"mode": Param("luminance", choices=["luminance", "lightness", "average"],
                            help="luminance (RGB weights), CIE L* or simple average"),
              "keep_rgb": Param(False, help="three equal channels instead of one")}

    def run(self, img, ctx, mode, keep_rgb):
        g = _gray(img, mode)[..., None].astype(np.float32)
        return np.repeat(g, 3, -1) if keep_rgb else g


_CHANNELS = ["R", "G", "B", "L", "L*", "a*", "b*", "H", "S", "V"]


@register
class ChannelExtraction(Plugin):
    id = "tools.channel_extract"
    name = "ChannelExtraction"
    category = "tools"
    description = ("PixInsight ChannelExtraction: keeps one channel (R, G, B, luminance, CIE L*a*b* or HSV) as a "
                   "grey image; the other channels can be saved as layers for PixelMath or masks.")
    params = {"channel": Param("L", choices=_CHANNELS, help="channel to keep"),
              "save_layers": Param(False, help="also save every channel in the layers (channel_R, channel_a*...)")}

    @staticmethod
    def _all(img):
        if img.shape[-1] == 1:
            v = img[..., 0]
            return {c: v for c in _CHANNELS}
        L, a, b = masks.rgb_to_lab(img)
        hsv = masks.hsv(img)
        return {"R": img[..., 0], "G": img[..., 1], "B": img[..., 2], "L": luminance(img), "L*": L / 100,
                "a*": a / 256 + 0.5, "b*": b / 256 + 0.5, "H": hsv[..., 0] / 360, "S": hsv[..., 1], "V": hsv[..., 2]}

    def run(self, img, ctx, channel, save_layers):
        ch = self._all(img)
        if save_layers:
            for k, v in ch.items():
                ctx.layers[f"channel_{k}"] = np.clip(v, 0, 1).astype(np.float32)[..., None]
        ctx.log(f"ChannelExtraction: {channel}")
        return np.clip(ch[channel], 0, 1).astype(np.float32)[..., None]


_NB_CACHE = {}          # (path, mtime, shape, scale) -> narrowband registered on the image (live preview)


def _nb_load(path, img, ctx, what):
    """Narrowband file as one channel registered on img (other camera: other scale/rotation), cached."""
    from pathlib import Path
    p = Path(path)
    key = (str(p), p.stat().st_mtime, img.shape[:2], round(ctx.scale, 4))
    if key in _NB_CACHE:
        return _NB_CACHE[key]
    ext = load_image(path)
    ext = _gray(ext)[..., None] if ext.shape[-1] == 3 else ext
    if ctx.scale != 1:
        ext = cv2.resize(ext, (round(ext.shape[1] * ctx.scale), round(ext.shape[0] * ctx.scale)),
                         interpolation=cv2.INTER_AREA)
        ext = ext if ext.ndim == 3 else ext[..., None]
    out = np.nan_to_num(align_to(ext, img, ctx.log), nan=0.0)
    if out.shape[:2] != img.shape[:2]:
        out = _fit_to(out, img, ctx.log, what)
    out = out[..., 0].astype(np.float32)
    if len(_NB_CACHE) > 8:
        _NB_CACHE.clear()
    _NB_CACHE[key] = out
    return out


def _stretched_signal(nb, chan, flatten=True, stretch_first=True):
    """Narrowband signal to blend into a stretched colour channel: sky gradient removed (DBE-like samples, only
    where brighter than the sky), levels matched to the channel on the sky and the highlights, stars removed by
    a morphological opening (extended nebulosity stays), noise floor removed. Returns (signal, star kernel px)."""
    h, w = nb.shape
    from ..pipeline import looks_stretched
    if stretch_first and not looks_stretched(nb[..., None]):   # a linear master next to a finished colour image
        nb = stretch.auto_stretch(nb[..., None], 0.2, -2.8)[0][..., 0]
    if flatten:
        from .. import background
        try:
            model = background.extract_rbf(nb[..., None], n=24, size=max(7, round(max(h, w) / 70)))[1][..., 0]
        except Exception:        # too few samples on tiny previews
            model = background.extract(nb[..., None], degree=2)[1][..., 0]
        nb = nb - np.maximum(model, float(np.median(model[::8, ::8]))) + float(np.median(nb[::8, ::8]))
    sn, sr = float(np.median(nb[::4, ::4])), float(np.median(chan[::4, ::4]))
    hn, hr = float(np.percentile(nb[::4, ::4], 99.5)), float(np.percentile(chan[::4, ::4], 99.5))
    sig = np.clip((hr - sr) / max(hn - sn, 1e-6) * (nb - sn), 0, None).astype(np.float32)
    px = max(2, round(max(h, w) / 275))           # 4 px on a 1100 px preview, ~18 px on a 5000 px image
    if not stretch_first:     # linear stars are far brighter than the nebula: their wings outlast a small opening
        px *= 2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    sig = cv2.GaussianBlur(cv2.morphologyEx(sig, cv2.MORPH_OPEN, k), (0, 0), max(1.0, px / 4))
    dark = chan < np.percentile(chan[::4, ::4], 50)
    d = sig[dark]
    floor = float(np.median(d)) + 1.4826 * float(np.median(np.abs(d - np.median(d))))
    return np.clip(sig - floor, 0, None), px


def _continuum_ratio(nb, ref):
    """nb / ref on the stars (pure continuum): how much of the broadband channel the narrowband one sees."""
    from ..stack import analyze
    st = analyze.detect(np.ascontiguousarray(ref, np.float32), 300, saturation=2.0)
    if len(st["x"]) < 10:
        return None
    xi = np.clip(st["x"].round().astype(int), 1, ref.shape[1] - 2)
    yi = np.clip(st["y"].round().astype(int), 1, ref.shape[0] - 2)
    win = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    r = np.max([ref[yi + dy, xi + dx] for dy, dx in win], 0)
    n = np.max([nb[yi + dy, xi + dx] for dy, dx in win], 0)
    ok = (r > 5 * max(float(np.median(np.abs(ref))), 1e-6)) & (r < 0.9) & (n > 0)
    if ok.sum() < 8:
        return None
    return float(np.median(n[ok] / r[ok]))


@register
class NBRGBCombination(Plugin):
    id = "color.nb_rgb"
    name = "NBRGBCombination (RGB + banda stretta)"
    category = "color"
    description = ("PixInsight NBRGBCombination for the hybrid mode: adds Ha (and SII) to red and OIII to green and "
                   "blue of a colour image. Continuum subtraction (from the stars, which are pure continuum) keeps "
                   "only the emission, so the narrowband stars do not bloat the colour stars. Files from another "
                   "camera are registered on the image (scale, rotation). Modes: nbrgb (continuum-subtracted "
                   "emission added, the PixInsight process), max (PixelMath max(R, Ha)), screen. Best on linear "
                   "data, right after the background extraction.")
    params = {"ha": Param("", help="Ha file (integrated, linear)"),
              "ha_amount": Param(1.0, 0.0, 5.0, "how much Ha shows (0 = none)"),
              "oiii": Param("", help="OIII file"),
              "oiii_amount": Param(1.0, 0.0, 5.0, "how much OIII shows (green and blue)"),
              "sii": Param("", help="SII file"),
              "sii_amount": Param(0.5, 0.0, 5.0, "how much SII shows (red)"),
              "ha_blue": Param(0.0, 0.0, 1.0, "share of the Ha emission also in blue (H-beta: pinker nebula)"),
              "mode": Param("nbrgb", choices=["nbrgb", "max", "screen"], help="how the emission goes in"),
              "nb_bandwidth": Param(7.0, 2.0, 30.0, "narrowband filter width, nm"),
              "rgb_bandwidth": Param(100.0, 30.0, 200.0, "colour filter width (R, G, B of the camera), nm"),
              "continuum": Param(None, 0.0, 10.0, "nb/rgb ratio of the stars; None = measured on the stars"),
              "flatten": Param(True, help="remove the narrowband's own sky gradient (vignetting, light pollution) "
                                          "before adding it, or it tints the whole sky")}

    def run(self, img, ctx, ha, ha_amount, oiii, oiii_amount, sii, sii_amount, ha_blue, mode, nb_bandwidth,
            rgb_bandwidth, continuum, flatten):
        if img.shape[-1] != 3:
            raise ValueError("NBRGBCombination needs a colour image")
        if not (ha or oiii or sii):
            raise ValueError("NBRGBCombination: choose at least one narrowband file (Ha, OIII or SII)")
        out = img.astype(np.float32).copy()
        bg = [float(np.median(img[::4, ::4, c])) for c in range(3)]
        q = nb_bandwidth / rgb_bandwidth
        plan = [(ha, ha_amount, [(0, 1.0), (2, ha_blue)], 0, "Ha"), (sii, sii_amount, [(0, 1.0)], 0, "SII"),
                (oiii, oiii_amount, [(1, 1.0), (2, 1.0)], 1, "OIII")]
        for path, amount, targets, cref, what in plan:
            if not path or amount <= 0:
                continue
            nb = _nb_load(path, img, ctx, what)
            chan = img[..., cref] if what != "OIII" else img[..., 1:].mean(-1)
            if not ctx.linear:
                # finished (stretched) files: the two images were stretched differently, so "narrowband = k x
                # channel + emission" does not hold (on M8 + M20 Ha it gave no emission on the nebula and noise
                # on the sky). Instead, as by hand on stretched images: the narrowband without its gradient and
                # without stars, its levels matched to the channel, blended in
                sig, px = _stretched_signal(nb, chan, flatten)
                for c, share in targets:
                    if share <= 0:
                        continue
                    add = np.clip(amount * share * sig, 0, 1)
                    if mode == "max":      # lighten: the channel where the narrowband is brighter
                        out[..., c] = np.maximum(out[..., c], bg[c] + add)
                    elif mode == "screen":
                        out[..., c] = 1 - (1 - out[..., c]) * (1 - add)
                    else:
                        out[..., c] = out[..., c] + add
                ctx.log(f"NBRGB {what} x{amount:g}: stretched data, levels matched to the channel, stars removed "
                        f"({px} px), signal p99 {float(np.percentile(sig, 99)):.3f}, mode {mode}")
                continue
            if flatten:      # Ha files can keep corner gradients that would read as emission everywhere
                from .. import background
                model = background.extract(nb[..., None], degree=2)[1][..., 0]
                # only the parts brighter than the median sky are removed: they make false emission; a model
                # that dips (corners of a field full of nebula) would add some instead
                nb0 = nb - np.maximum(model, float(np.median(model[::8, ::8])))
            else:
                nb0 = nb - float(np.median(nb[::4, ::4]))
            ref = chan - (bg[cref] if what != "OIII" else np.mean(bg[1:]))
            k = continuum or _continuum_ratio(nb0, ref)
            if not k:
                k = q
                ctx.log(f"NBRGB {what}: not enough stars for the continuum, using the filter widths ({q:.3f})")
            # emission in narrowband units, then in the colour channel's units (q / k: same flux per nm)
            line = np.clip(nb0 - k * ref, 0, None) * (q / k)
            span = float(np.percentile(ref[::4, ::4], 99)) - float(np.median(ref[::4, ::4]))
            if float(np.percentile(line, 99)) < 0.1 * max(span, 1e-9):
                # the colour image already holds this line (dual-band filter, or a channel the line dominates): the
                # stars give the same ratio as the nebula, so the subtraction leaves ~nothing and the slider did
                # nothing (Rosetta light-BINNING_1 + Ha: emission 0.4% of the red). Add the narrowband itself instead,
                # without stars and levels matched to the channel, as for finished files
                emis = float(np.percentile(line, 99))
                line, px = _stretched_signal(nb, chan, flatten, stretch_first=False)
                ctx.log(f"NBRGB {what}: the colour image already contains this line (continuum {k:.3f} on the "
                        f"stars, emission only {emis:.5f}): using the {what} without stars ({px} px), levels "
                        f"matched to the channel")
            if mode == "max":       # PixelMath max(R, Ha) after LinearFit of the narrowband on the channel
                from ..stack.sources import fit_levels
                a, b = fit_levels(nb, img[..., cref])
                fitted = a + b * nb
            for c, share in targets:
                if share <= 0:
                    continue
                add = amount * share * line
                if mode == "max":       # amount 1 = max(R, Ha); never darker, never lifts the sky below Ha
                    out[..., c] = out[..., c] + amount * share * np.clip(fitted - bg[cref] + bg[c] - out[..., c], 0, None)
                elif mode == "screen":
                    out[..., c] = 1 - (1 - out[..., c]) * (1 - np.clip(add, 0, 1))
                else:
                    out[..., c] = out[..., c] + add
            ctx.log(f"NBRGB {what} x{amount:g}: continuum {k:.4f} (filters {q:.3f}), emission "
                    f"p99 {float(np.percentile(line, 99)):.5f}, mode {mode}")
        return np.clip(out, 0, 1).astype(np.float32)


@register
class LRGBCombination(Plugin):
    id = "color.lrgb"
    name = "LRGBCombination"
    category = "color"
    stage = "nonlinear"
    description = ("PixInsight LRGBCombination: puts a separate luminance (L filter or a stretched Ha) under the "
                   "color image: detail from L, color from RGB. The L file is registered on the image if needed.")
    params = {"luminance": Param("", help="stretched L file (FITS/XISF/TIFF)"),
              "weight": Param(1.0, 0.0, 1.0, "how much of the new luminance (0 = none)"),
              "match": Param(True, help="give L the tone distribution of the image first (no washed-out colors)"),
              "saturation": Param(1.0, 0.0, 2.0, "chroma factor"),
              "chroma_blur": Param(0.0, 0.0, 10.0, "blur of the color only, pixels (hides color noise)", px=True)}

    def run(self, img, ctx, luminance, weight, match, saturation, chroma_blur):
        if not luminance:
            raise ValueError("LRGBCombination: choose the luminance file")
        ext = load_image(luminance)
        if ctx.scale != 1:
            ext = cv2.resize(ext, (round(ext.shape[1] * ctx.scale), round(ext.shape[0] * ctx.scale)),
                             interpolation=cv2.INTER_AREA)
            ext = ext if ext.ndim == 3 else ext[..., None]
        ext = _fit_to(ext, img, ctx.log, "LRGB")
        newL = np.clip(_gray(ext, "lightness") if ext.shape[-1] == 3 else ext[..., 0], 0, 1) * 100
        L, a, b = masks.rgb_to_lab(img)
        if match:          # same tone distribution as the image's own L*: only detail and lower noise come from L
            q = np.linspace(0, 1, 513)
            src = np.quantile(newL[::2, ::2], q)
            dst = np.quantile(L[::2, ::2], q)
            newL = np.interp(newL, np.maximum.accumulate(src + q * 1e-6), dst).astype(np.float32)
        if chroma_blur > 0:
            a, b = cv2.GaussianBlur(a, (0, 0), chroma_blur), cv2.GaussianBlur(b, (0, 0), chroma_blur)
        out = masks.lab_to_rgb(L + weight * (newL - L), a * saturation, b * saturation)
        ctx.log(f"LRGB: luminance {luminance} weight {weight}" + (" (levels matched)" if match else ""))
        return out


@register
class ChannelCombination(Plugin):
    id = "color.channel_combine"
    name = "ChannelCombination"
    category = "color"
    description = ("PixInsight ChannelCombination: builds the color image from mono files, one per channel "
                   "(e.g. R, G, B or Ha, OIII, SII masters). An empty channel keeps the current image's channel; "
                   "files are registered on the current image.")
    params = {"R": Param("", help="file for red"), "G": Param("", help="file for green"),
              "B": Param("", help="file for blue")}

    def run(self, img, ctx, R, G, B):
        base = img if img.shape[-1] == 3 else np.repeat(img[..., :1], 3, -1)
        out = base.astype(np.float32).copy()
        for c, path in enumerate((R, G, B)):
            if not path:
                continue
            ext = load_image(path)
            if ctx.scale != 1:
                ext = cv2.resize(ext, (round(ext.shape[1] * ctx.scale), round(ext.shape[0] * ctx.scale)),
                                 interpolation=cv2.INTER_AREA)
                ext = ext if ext.ndim == 3 else ext[..., None]
            out[..., c] = _fit_to(ext, base, ctx.log, "ChannelCombination")[..., 0]
            ctx.log(f"ChannelCombination: {'RGB'[c]} <- {path}")
        return out
