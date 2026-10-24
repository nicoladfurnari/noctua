"""Tone and color finishing: curves, saturation, SCNR, local contrast, dark structures, HDR."""
import cv2
import numpy as np

from .. import denoise, enhance, masks
from ..stats import luminance
from .base import Param, Plugin, register


def _apply_lum(img, new_lum):
    """Replace luminance keeping color ratios."""
    lum = luminance(img)
    ratio = (new_lum / np.maximum(lum, 1e-6))[..., None]
    return np.clip(img * ratio, 0, 1).astype(np.float32)


# order the per-channel curves are applied in (Photoshop / PixInsight: the composite curve, then the channels)
CURVE_ORDER = ["rgb", "r", "g", "b", "l", "a", "b_star", "c", "s", "h"]


def _identity(pts):
    return all(abs(float(x) - float(y)) < 1e-6 for x, y in pts)


@register
class Curves(Plugin):
    id = "contrast.curves"
    name = "Curves"
    category = "contrast"
    stage = "nonlinear"
    description = ("Monotone spline curve like PixInsight CurvesTransformation: RGB/K, R, G, B, CIE L*, a*, b*, "
                   "c* (chroma), HSV saturation or hue (hue curve: out hue / 360 as a function of hue / 360). "
                   "'curves' holds one curve per channel, applied in the order RGB, R, G, B, L*, a*, b*, c*, S, H "
                   "(like the channel tabs of Photoshop); without it, 'points' on 'channel' (older profiles).")
    params = {"points": Param([[0, 0], [0.15, 0.14], [0.5, 0.52], [1, 1]], help="list of [in, out] pairs"),
              "channel": Param("rgb", choices=CURVE_ORDER,
                               help="rgb = all channels (K); a/b_star: 0.5 = neutral; c: 1 = chroma 128"),
              "curves": Param({}, help="{channel: [[in, out], ...]} one curve per channel; replaces points/channel")}

    def run(self, img, ctx, points, channel="rgb", curves=None):
        todo = {k: v for k, v in (curves or {}).items() if k in CURVE_ORDER and v and len(v) >= 2}
        if not todo:
            todo = {channel: points}
        for ch in CURVE_ORDER:
            if ch in todo and not _identity(todo[ch]):
                img = self._one(img, [tuple(p) for p in sorted(todo[ch])], ch)
        return img

    @staticmethod
    def _one(img, pts, channel):
        if channel == "rgb" or (img.shape[-1] != 3 and channel == "l"):
            return enhance.apply_curve(img, pts)
        if img.shape[-1] != 3:
            return img
        if channel in ("r", "g", "b"):
            out = img.copy()
            k = "rgb".index(channel)
            out[..., k] = enhance.apply_curve(img[..., k], pts)
            return out
        if channel in ("s", "h"):
            hsv = masks.hsv(img)
            if channel == "s":
                hsv[..., 1] = enhance.apply_curve(hsv[..., 1], pts)
            else:
                hsv[..., 0] = (enhance.apply_curve(hsv[..., 0] / 360, pts) * 360) % 360
            return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1).astype(np.float32)
        L, a, b = masks.rgb_to_lab(img)
        if channel == "l":
            L = enhance.apply_curve(L / 100, pts) * 100
        elif channel == "a":
            a = (enhance.apply_curve(a / 256 + 0.5, pts) - 0.5) * 256
        elif channel == "b_star":
            b = (enhance.apply_curve(b / 256 + 0.5, pts) - 0.5) * 256
        else:
            c = np.hypot(a, b)
            f = np.where(c < 128, enhance.apply_curve(c / 128, pts) * 128 / np.maximum(c, 1e-6), 1)
            a, b = a * f, b * f
        return masks.lab_to_rgb(L, a, b)


@register
class Saturation(Plugin):
    id = "contrast.saturation"
    name = "Saturation"
    category = "contrast"
    stage = "nonlinear"
    description = "Boosts color, weighted by brightness so the background noise is not colored."
    params = {"amount": Param(0.35, -1.0, 2.0, "saturation change"),
              "masked": Param(True, help="apply mostly on bright structures")}

    def run(self, img, ctx, amount, masked):
        if img.shape[-1] != 3:
            return img
        return enhance.saturation(img, amount, mask=masks.lum_mask(img, False) if masked else None)


@register
class SCNR(Plugin):
    id = "color.scnr"
    name = "SCNR (green removal)"
    category = "color"
    description = ("Removes a colour cast like PixInsight SCNR: average/maximum neutral or maximum/additive mask, "
                   "green (default), red or blue, optionally keeping the lightness.")
    params = {"amount": Param(0.8, 0.0, 1.0, "strength"),
              "method": Param("average_neutral", choices=list(enhance.SCNR_METHODS)),
              "color": Param("green", choices=["green", "red", "blue"]),
              "preserve_lightness": Param(False, help="restore the luminance lost by removing the cast")}

    def run(self, img, ctx, amount, method, color, preserve_lightness):
        return enhance.scnr(img, amount, method, color, preserve_lightness) if img.shape[-1] == 3 else img


@register
class HubblePalette(Plugin):
    id = "color.sho_gold"
    name = "Hubble palette (SHO gold/blue)"
    category = "color"
    stage = "nonlinear"
    description = ("Turns the raw green of SHO into the classic gold/blue look: excess green (Ha) moves "
                   "to red, OIII areas get bluer.")
    params = {"amount": Param(0.8, 0.0, 1.0, "how much of the excess green is converted"),
              "gold": Param(0.7, 0.0, 1.0, "share of the removed green added to red"),
              "blue": Param(0.3, 0.0, 1.0, "extra blue in OIII-dominated areas")}

    def run(self, img, ctx, amount, gold, blue):
        if img.shape[-1] != 3:
            return img
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        excess = np.maximum(g - (r + b) / 2, 0) * amount
        out = img.copy()
        out[..., 1] = g - excess
        out[..., 0] = r + gold * excess
        oiii = np.maximum(b - r, 0)
        out[..., 2] = b + blue * oiii
        return np.clip(out, 0, 1).astype(np.float32)


@register
class LocalContrast(Plugin):
    id = "contrast.local"
    name = "Local contrast (CLAHE)"
    category = "contrast"
    stage = "nonlinear"
    description = "Contrast-limited adaptive histogram equalization on luminance (LHE-like)."
    params = {"amount": Param(0.3, 0.0, 1.0, "blend with the original"),
              "clip": Param(2.0, 1.0, 6.0, "contrast limit"),
              "tiles": Param(8, 2, 32, "tile grid size")}

    def run(self, img, ctx, amount, clip, tiles):
        lum = luminance(img)
        u16 = (np.clip(lum, 0, 1) * 65535).astype(np.uint16)
        eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(u16).astype(np.float32) / 65535
        # keep stars and background as they are, enhance the nebula
        m = masks.smoothstep(lum, 0.15, 0.45) * (1 - masks.star_mask(img, False, scale=ctx.scale))
        new = lum + amount * m * (eq - lum)
        return _apply_lum(img, new)


@register
class DarkStructure(Plugin):
    id = "contrast.dark_structure"
    name = "Dark structure enhance"
    category = "contrast"
    stage = "nonlinear"
    description = "Darkens dust lanes and dark nebulae relative to their surroundings."
    params = {"amount": Param(0.5, 0.0, 1.5, "strength"),
              "scale": Param(16.0, 4.0, 64.0, "size of the structures, pixels", px=True)}

    def run(self, img, ctx, amount, scale):
        lum = luminance(img)
        fine = cv2.GaussianBlur(lum, (0, 0), max(0.5, 2 * ctx.scale))
        coarse = cv2.GaussianBlur(lum, (0, 0), scale)
        dark = np.clip(coarse - fine, 0, None)
        dark *= 1 - masks.star_mask(img, False, grow=4, blur=3, scale=ctx.scale)
        return np.clip(img - amount * dark[..., None], 0, 1).astype(np.float32)


@register
class HDRCompress(Plugin):
    id = "contrast.hdr"
    name = "HDR compression"
    category = "contrast"
    stage = "nonlinear"
    description = "Recovers detail in bright cores (e.g. M42, M8) by compressing large scales only there."
    params = {"amount": Param(0.5, 0.0, 1.0, "compression of bright large scales"),
              "threshold": Param(0.5, 0.1, 0.9, "brightness where compression starts"),
              "scale": Param(48.0, 8.0, 256.0, "size of the large scale, pixels", px=True)}

    def run(self, img, ctx, amount, threshold, scale):
        lum = luminance(img)
        base = cv2.GaussianBlur(lum, (0, 0), scale)
        detail = lum - base
        w = masks.smoothstep(base, threshold, 1.0)
        target = threshold + (base - threshold) * (1 - amount)
        new_base = base * (1 - w) + np.where(base > threshold, target, base) * w
        return _apply_lum(img, np.clip(new_base + detail, 0, 1))


@register
class WaveletSharpen(Plugin):
    id = "sharpen.wavelet"
    name = "Wavelet sharpening"
    category = "sharpen"
    stage = "nonlinear"
    description = "Boosts small/medium wavelet scales on the nebula, not on stars or background."
    params = {"amount": Param(0.5, 0.0, 2.0, "boost"),
              "layers": Param([2, 3], help="scales boosted (1 = finest, mostly noise)", px="octave")}

    def run(self, img, ctx, amount, layers):
        lum = luminance(img)
        w = denoise.starlet(np.ascontiguousarray(lum), max(layers))
        boost = sum(w[j - 1] for j in layers)
        m = masks.smoothstep(lum, 0.12, 0.35) * (1 - masks.star_mask(img, False, grow=3, scale=ctx.scale))
        return _apply_lum(img, np.clip(lum + amount * boost * m, 0, 1))


@register
class MultiscaleLinearSharpen(Plugin):
    id = "sharpen.mlt"
    name = "MultiscaleLinearTransform (sharpen)"
    category = "sharpen"
    stage = "any"
    description = ("PixInsight MultiscaleLinearTransform used for sharpening: starlet layers 1..n (1, 2, 4, 8, 16 px), "
                   "each multiplied by 1 + bias, with an optional noise threshold per layer (coefficients below "
                   "threshold x layer noise are not boosted), on the lightness through a luminance mask so the "
                   "background is left alone. Best on the starless image (no star halos to boost).")
    params = {"bias": Param([0.0, 0.1, 0.075, 0.05, 0.025], help="per layer, 0 = unchanged, 0.1 = +10% (default: the user's PixInsight setting)"),
              "threshold": Param([3.0, 2.0, 1.0, 0.5, 0.0], help="per layer, noise sigmas below which nothing is boosted"),
              "mask_low": Param(0.1, 0.0, 1.0, "luminance mask: no sharpening below this"),
              "mask_high": Param(0.4, 0.0, 1.0, "luminance mask: full sharpening above this"),
              "protect_stars": Param(False, help="also exclude stars (when they are still in the image)")}

    def run(self, img, ctx, bias, threshold, mask_low, mask_high, protect_stars):
        bias = list(bias) if isinstance(bias, (list, tuple)) else [float(bias)] * 5
        n = len(bias)
        th = (list(threshold) if isinstance(threshold, (list, tuple)) else [float(threshold)] * n) + [0.0] * n
        # in the live preview the finest full-resolution layers are sub-pixel: start further down
        skip = min(n - 1, max(0, -round(np.log2(ctx.scale)))) if ctx.scale < 1 else 0
        lum = np.ascontiguousarray(luminance(img).astype(np.float32))
        ws = denoise.starlet(lum, n - skip)
        out = ws[-1].copy()
        for j, w in enumerate(ws[:-1]):
            b, t = float(bias[j + skip]), float(th[j + skip])
            if b and t > 0:
                sig = float(np.median(np.abs(w))) * 1.4826 or 1e-9
                gain = 1 + b * np.clip((np.abs(w) - t * sig) / max(t * sig, 1e-9), 0, 1)
                out += w * gain
            else:
                out += w * (1 + b)
        m = masks.smoothstep(masks.lum_mask(img, ctx.linear), mask_low, mask_high)   # stretched copy if linear
        if protect_stars:
            m = m * (1 - masks.star_mask(img, ctx.linear, grow=3, scale=ctx.scale))
        new = lum + (out - lum) * m
        ctx.log("MLT bias " + ", ".join(f"{b:+.2f}" for b in bias))
        return _apply_lum(img, np.clip(new, 0, 1))
