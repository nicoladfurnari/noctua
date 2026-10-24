"""PixInsight workflow processes (Papotti/Airola, GAEEB): TGV/MMT denoise, HistogramTransformation, linked MTF,
selective colour, LinearFit, HDRComposition, healing, rotation, manual crop, first-layer subtraction."""
import math
from pathlib import Path

import cv2
import numpy as np

from .. import denoise, io, masks, stretch, tgv
from ..stats import channel_stats, luminance, median_mad
from .base import Param, Plugin, register


def _lst(v, n):
    v = list(v) if isinstance(v, (list, tuple)) else [v]
    return [float(v[min(i, len(v) - 1)]) for i in range(n)]


# ------------------------------------------------------------------ noise
@register
class TGVDenoise(Plugin):
    id = "noise.tgv"
    name = "TGV denoise"
    category = "noise"
    description = ("Total Generalized Variation (2nd order) like PixInsight TGVDenoise: luminance and chrominance "
                   "strengths in noise sigmas, edge protection from the measured noise, local support from a "
                   "luminance mask, automatic convergence. Linear or stretched.")
    params = {"strength_l": Param(1.5, 0.0, 20.0, "luminance regularisation, in noise sigmas"),
              "strength_c": Param(3.0, 0.0, 30.0, "chrominance regularisation, in noise sigmas"),
              "aspect": Param(2.0, 0.1, 10.0, "TGV aspect ratio a0/a1 (higher = flatter gradients)"),
              "edge_protection": Param(3.0, 0.0, 20.0, "gradients above this many noise sigmas are protected (0 off)"),
              "support": Param(0.7, 0.0, 1.0, "local support: less smoothing on bright structures (0 off)"),
              "support_mid": Param(0.5, 0.01, 0.99, "midtones of the support mask (lower = protects more)"),
              "iterations": Param(100, 10, 1000, "maximum iterations"),
              "convergence": Param(0.004, 0.0, 0.1, "stop when the mean change per iteration < this x sigma"),
              "chroma_scale": Param(2, 1, 4, "chrominance denoised at 1/N resolution (faster)")}

    def run(self, img, ctx, strength_l, strength_c, aspect, edge_protection, support, support_mid, iterations,
            convergence, chroma_scale):
        sup = None
        if support > 0:      # smooth mask: computed at 1/4 resolution
            h, w = img.shape[:2]
            small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
            small = small if small.ndim == 3 else small[..., None]
            src = stretch.auto_stretch(small, 0.25, -2.8)[0] if ctx.linear else small
            sup = cv2.resize(1 - support * masks.luminance_mask(src, 0.0, support_mid, 1.0), (w, h),
                             interpolation=cv2.INTER_LINEAR)
        return tgv.denoise(img, strength_l, strength_c, aspect, edge_protection, sup, int(iterations), convergence,
                           int(chroma_scale), ctx.log)


def _median_dilated(ch, size, d):
    """size x size median (3 or 5) on a grid dilated by d (a trous), via the d x d polyphase sub-images."""
    if d == 1:
        return cv2.medianBlur(ch, size)
    out = np.empty_like(ch)
    for a in range(d):
        for b in range(d):
            sub = np.ascontiguousarray(ch[a::d, b::d])
            out[a::d, b::d] = cv2.medianBlur(sub, size) if min(sub.shape) >= size else sub
    return out


def mmt(ch, layers):
    """Multiscale median transform: [w1..wn, residual]; scales 3, 5, 9, 17, 33... px."""
    out, c = [], ch
    for j in range(layers):
        s = _median_dilated(c, 3, 1) if j == 0 else _median_dilated(c, 5, 2 ** (j - 1))
        out.append(c - s)
        c = s
    out.append(c)
    return out


@register
class MMTDenoise(Plugin):
    id = "noise.mmt"
    name = "MMT noise reduction"
    category = "noise"
    description = ("Multiscale Median Transform noise reduction (EZ Denoise MMT step): median layers of 3, 5, 9, "
                   "17... px, coefficients below threshold x layer noise attenuated by the layer amount.")
    params = {"thresholds": Param([3.0, 2.5, 2.0, 1.5], help="per layer, in noise sigmas of that layer"),
              "amounts": Param([0.7, 0.6, 0.5, 0.4], help="per layer attenuation 0..1"),
              "protect": Param(0.5, 0.0, 1.0, "protect bright structures with a luminance mask")}

    def run(self, img, ctx, thresholds, amounts, protect):
        n = max(len(thresholds), len(amounts))
        th, am = _lst(thresholds, n), _lst(amounts, n)
        # at preview scale the finest full-resolution layers are sub-pixel: drop them
        skip = min(n - 1, max(0, -round(math.log2(ctx.scale)))) if ctx.scale < 1 else 0
        th, am = th[skip:], am[skip:]
        out = np.empty_like(img)
        for c in range(img.shape[-1]):
            ws = mmt(np.ascontiguousarray(img[..., c]), len(th))
            rec = ws[-1].copy()
            for w, t, a in zip(ws[:-1], th, am):
                _, mad = median_mad(w, 2)
                rec += np.where(np.abs(w) < t * mad * 1.4826, w * (1 - a), w)
            out[..., c] = rec
        if protect > 0:
            m = (1 - protect * masks.lum_mask(img, ctx.linear))[..., None]
            out = img * (1 - m) + out * m
        return np.clip(out, 0, 1).astype(np.float32)


@register
class FirstLayerSubtract(Plugin):
    id = "noise.first_layer_subtract"
    name = "First layer subtraction"
    category = "noise"
    description = "First noise reduction of the workflow: subtracts a fraction of the first a trous layer (B - R)."
    params = {"amount": Param(0.5, 0.0, 1.0, "fraction of the first starlet layer removed")}

    def run(self, img, ctx, amount):
        out = np.empty_like(img)
        for c in range(img.shape[-1]):
            ch = np.ascontiguousarray(img[..., c])
            out[..., c] = ch - amount * denoise.starlet(ch, 1)[0]
        return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ stretch
@register
class HistogramTransform(Plugin):
    id = "stretch.histogram"
    name = "Histogram transformation"
    category = "stretch"
    description = ("Manual HistogramTransformation like PixInsight HT: shadows / midtones / highlights, "
                   "one value for linked channels or [r, g, b].")
    params = {"shadows": Param(0.0, 0.0, 1.0, "shadows clipping (number or [r, g, b])"),
              "midtones": Param(0.5, 0.0001, 0.9999, "midtones balance (number or [r, g, b])"),
              "highlights": Param(1.0, 0.0, 1.0, "highlights clipping (number or [r, g, b])")}

    def run(self, img, ctx, shadows, midtones, highlights):
        n = img.shape[-1]
        s, m, h = _lst(shadows, n), _lst(midtones, n), _lst(highlights, n)
        out = np.empty_like(img)
        for c in range(n):
            out[..., c] = stretch.histogram_transform(img[..., c], s[c], m[c], max(h[c], s[c] + 1e-6))
        if any(abs(v - 0.5) > 1e-9 for v in m):
            ctx.linear = False
        return np.clip(out, 0, 1).astype(np.float32)


@register
class LinkedMTF(Plugin):
    id = "stretch.linked_mtf"
    name = "HT stretch linked RGB"
    category = "stretch"
    stage = "linear"
    description = ("Bill Blanshan / Juan Conejero 'HT Stretch Linked RGB' PixelMath: shadows at median + C x "
                   "1.4826 MAD, background brought to B with the same curve on every channel.")
    params = {"C": Param(-2.8, -10.0, 0.0, "shadows clipping in MAD units"),
              "B": Param(0.20, 0.01, 0.9, "target background")}

    def run(self, img, ctx, C, B):
        st = channel_stats(img)
        m = float(np.mean([a for a, _ in st]))
        d = float(np.mean([b for _, b in st]))
        c = min(max(0.0, m + C * 1.4826 * d), 1.0)
        mid = float(stretch.mtf(B, m - c))
        x = np.maximum(0, (img - c) / max(1 - c, 1e-6))
        ctx.log(f"linked MTF: shadows {c:.5f}, midtones {mid:.5f}")
        ctx.linear = False
        return np.clip(stretch.mtf(mid, x), 0, 1).astype(np.float32)


# ------------------------------------------------------------------ colour
@register
class SelectiveColor(Plugin):
    id = "color.selective"
    name = "Selective colour correction"
    category = "color"
    stage = "nonlinear"
    description = ("SelectiveColorCorrection: changes hue, saturation and lightness of one hue range only "
                   "(soft hue window, grey pixels untouched).")
    params = {"color": Param("red", choices=list(masks.HUES)),
              "hue_shift": Param(0.0, -60.0, 60.0, "hue rotation, degrees"),
              "saturation": Param(0.0, -1.0, 2.0, "relative chroma change"),
              "lightness": Param(0.0, -1.0, 1.0, "relative lightness change"),
              "width": Param(60.0, 10.0, 180.0, "full-strength hue range, degrees (soft edges of width/2)")}

    def run(self, img, ctx, color, hue_shift, saturation, lightness, width):
        if img.shape[-1] != 3:
            return img
        h = masks.hsv(img)
        chroma = img.max(-1) - img.min(-1)
        w = masks.hue_window(h[..., 0], masks.HUES[color], width) * masks.smoothstep(chroma, 0.0, 0.03)
        L, a, b = masks.rgb_to_lab(img)
        c, ang = np.hypot(a, b), np.arctan2(b, a) + np.deg2rad(hue_shift) * w
        c = c * np.maximum(0, 1 + saturation * w)
        L = L * np.maximum(0, 1 + lightness * w)
        out = masks.lab_to_rgb(L, c * np.cos(ang), c * np.sin(ang))
        wm = (w > 0)[..., None]
        return np.where(wm, out, img).astype(np.float32)


@register
class LinearFit(Plugin):
    id = "color.linear_fit"
    name = "Linear fit"
    category = "color"
    stage = "linear"
    description = ("PixInsight LinearFit: fits the other channels onto the reference (G) with a robust line, "
                   "rejecting pixels outside [reject_low, reject_high] and outliers (stars).")
    params = {"reference": Param("g", choices=["r", "g", "b"]),
              "reject_low": Param(0.0, 0.0, 1.0, "ignore pixels at or below this value"),
              "reject_high": Param(0.92, 0.0, 1.0, "ignore pixels at or above this value")}

    def run(self, img, ctx, reference, reject_low, reject_high):
        if img.shape[-1] != 3:
            return img
        k = "rgb".index(reference)
        s = img[::2, ::2].reshape(-1, 3)
        ok = np.all((s > reject_low) & (s < reject_high), 1)
        out = img.copy()
        for c in range(3):
            if c == k:
                continue
            x, y = s[ok, c].astype(np.float64), s[ok, k].astype(np.float64)
            keep = np.ones(len(x), bool)
            a, b = 0.0, 1.0
            for _ in range(5):
                if keep.sum() < 10:
                    break
                b, a = np.polyfit(x[keep], y[keep], 1)
                r = y - (a + b * x)
                keep = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r[keep])) + 1e-12
            out[..., c] = a + b * img[..., c]
            ctx.log(f"linear fit {'rgb'[c]} -> {reference}: {a:+.5f} + {b:.4f} x")
        return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ HDR composition
def load_image(path):
    if str(path).lower().endswith(".xisf"):
        from ..stack import xisf
        img = xisf.read(path)[0]
        return img if img.ndim == 3 else img[..., None]
    return io.load(path)[0]


def align_to(img, ref, log=print):
    """Register img on ref by star matching unless they already line up."""
    from ..stack import analyze, register as reg
    from scipy.spatial import cKDTree
    a = analyze.detect(np.ascontiguousarray(luminance(img), np.float32), 300, saturation=2.0)
    b = analyze.detect(np.ascontiguousarray(luminance(ref), np.float32), 300, saturation=2.0)
    xa, xb = np.stack([a["x"], a["y"]], 1), np.stack([b["x"], b["y"]], 1)
    if len(xa) < 5 or len(xb) < 5:
        return img
    if img.shape[:2] == ref.shape[:2]:
        d, _ = cKDTree(xb).query(xa)
        if np.mean(d < 0.75) > 0.5:
            return img
    M, n, rms = reg.match(xa, xb)
    log(f"HDR: registered ({n} stars, rms {rms:.2f} px)")
    out = reg.warp(img, M, ref.shape[:2])
    return np.nan_to_num(out, nan=0.0)


@register
class HDRComposition(Plugin):
    id = "tools.hdr_composition"
    name = "HDR composition"
    category = "tools"
    stage = "linear"
    description = ("HDRComposition: replaces saturated areas of the current (longest) integration with shorter "
                   "exposures of the same field (registered if needed), scaled by a robust fit on the unsaturated "
                   "overlap, through smooth masks (saved as hdr_mask_N layers). Result rescaled into [0, 1].")
    params = {"images": Param([], help="shorter integrations, longest first (FITS/XISF paths)"),
              "threshold": Param(0.8, 0.1, 1.0, "binarizing threshold for saturated areas"),
              "smoothness": Param(7.0, 0.0, 50.0, "mask smoothness, pixels", px=True),
              "growth": Param(1, 0, 20, "mask growth, pixels", px=True)}

    def run(self, img, ctx, images, threshold, smoothness, growth):
        out = img.astype(np.float32).copy()
        prev, A, B = img, 0.0, 1.0          # previous (longer) source and its mapping to the output units
        for i, path in enumerate(images or [], 1):
            ext = load_image(path)
            if ctx.scale != 1:
                ext = cv2.resize(ext, (round(ext.shape[1] * ctx.scale), round(ext.shape[0] * ctx.scale)),
                                 interpolation=cv2.INTER_AREA)
                ext = ext if ext.ndim == 3 else ext[..., None]
            ext = align_to(ext, img, ctx.log)
            if ext.shape[:2] != img.shape[:2]:
                ext = cv2.resize(ext, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
                ext = ext if ext.ndim == 3 else ext[..., None]
            if ext.shape[-1] != img.shape[-1]:
                ext = np.repeat(ext[..., :1], img.shape[-1], -1)
            pmx = prev.max(-1)
            # robust fit prev = a + b ext on pixels unsaturated in both and with real signal
            lp = luminance(prev)
            med, mad = median_mad(lp)
            sel = (pmx < threshold * 0.9) & (ext.max(-1) < threshold * 0.9) & (lp > med + 10 * 1.4826 * mad)
            ys, xs = np.nonzero(sel)
            if len(ys) < 50:
                ctx.log(f"HDR: {Path(path).name}: not enough overlap, skipped")
                continue
            pick = np.random.default_rng(0).choice(len(ys), min(len(ys), 200000), replace=False)
            yi, xi = ys[pick], xs[pick]
            x, y = ext[yi, xi].ravel().astype(np.float64), prev[yi, xi].ravel().astype(np.float64)
            keep = np.ones(len(x), bool)
            a, b = 0.0, 1.0
            for _ in range(5):
                b, a = np.polyfit(x[keep], y[keep], 1)
                r = y - (a + b * x)
                keep = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r[keep])) + 1e-12
            A, B = A + B * a, B * b
            sat = (pmx >= threshold).astype(np.float32)
            m = sat
            if growth > 0:
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * growth + 1, 2 * growth + 1)))
            m = np.maximum(masks.blur(m, smoothness), sat)[..., None]   # saturated pixels always replaced
            out = out * (1 - m) + (A + B * ext) * m
            prev = ext
            ctx.layers[f"hdr_mask_{i}"] = m[..., 0]
            ctx.log(f"HDR: {Path(path).name}: x{B:.3f} {A:+.5f}, {100 * float(m.mean()):.2f}% of the image replaced")
        top = float(out.max())
        if top > 1:
            out /= top
        return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ cosmetic / geometry
@register
class Heal(Plugin):
    id = "cosmetic.heal"
    name = "Heal (clone / blemish)"
    category = "cosmetic"
    description = ("Blemish blaster for dust donuts and leftovers: each [x, y, radius] spot is filled with the "
                   "smooth surrounding background plus matched noise (or OpenCV Telea inpainting), feathered edge.")
    params = {"spots": Param([], help="list of [x, y, radius] in full-resolution pixels", px="xy"),
              "method": Param("smooth", choices=["smooth", "telea"]),
              "feather": Param(0.2, 0.0, 1.0, "soft edge, fraction of the radius"),
              "noise": Param(True, help="add noise matched to the surroundings")}

    def run(self, img, ctx, spots, method, feather, noise):
        out = img.copy()
        h, w = img.shape[:2]
        rng = np.random.default_rng(0)
        for x, y, r in spots or []:
            r = max(1.0, float(r))
            R = int(math.ceil(r * (2.5 + feather)))
            x0, y0, x1, y1 = max(0, int(x) - R), max(0, int(y) - R), min(w, int(x) + R + 1), min(h, int(y) + R + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = out[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d = np.hypot(xx - x, yy - y).astype(np.float32)
            hole = d <= r * (1 + feather)
            alpha = 1 - masks.smoothstep(d, r, r * (1 + feather) + 1e-3) if feather > 0 else (d <= r).astype(np.float32)
            ring = (d > r * (1 + feather)) & (d <= r * (2.2 + feather))
            fill = np.empty_like(roi)
            for c in range(img.shape[-1]):
                ch = np.ascontiguousarray(roi[..., c])
                if method == "telea":
                    fill[..., c] = cv2.inpaint(ch, hole.astype(np.uint8), max(3, round(r / 3)), cv2.INPAINT_TELEA)
                else:
                    k = (~hole).astype(np.float32)
                    s = max(1.0, r / 1.5)
                    fill[..., c] = cv2.GaussianBlur(ch * k, (0, 0), s) / np.maximum(cv2.GaussianBlur(k, (0, 0), s), 1e-6)
                if noise and ring.sum() > 8:
                    hp = ch - cv2.GaussianBlur(ch, (0, 0), 2.0)
                    sd = 1.4826 * float(np.median(np.abs(hp[ring])))
                    fill[..., c] += rng.normal(0, sd * 1.2, ch.shape).astype(np.float32) * hole
            out[y0:y1, x0:x1] = roi * (1 - alpha[..., None]) + fill * alpha[..., None]
        if spots:
            ctx.log(f"healed {len(spots)} spots")
        return np.clip(out, 0, 1).astype(np.float32)


def _inscribed(w, h, angle):
    """Largest axis-aligned rectangle inside a w x h rectangle rotated by angle (radians)."""
    if w <= 0 or h <= 0:
        return 0, 0
    long_, short = max(w, h), min(w, h)
    s, c = abs(math.sin(angle)), abs(math.cos(angle))
    if short <= 2 * s * c * long_ or abs(s - c) < 1e-10:
        x = 0.5 * short
        wr, hr = (x / s, x / c) if w >= h else (x / c, x / s)
    else:
        cos2 = c * c - s * s
        wr, hr = (w * c - h * s) / cos2, (h * c - w * s) / cos2
    return int(wr), int(hr)


@register
class Rotate(Plugin):
    id = "geometry.rotate"
    name = "Rotate / flip"
    category = "geometry"
    description = ("Rotation by 90/180/270 (exact) or any angle (Lanczos, counter-clockwise, optional crop to the "
                   "largest rectangle without empty corners) and horizontal/vertical flip.")
    params = {"angle": Param(0.0, -360.0, 360.0, "degrees, counter-clockwise"),
              "flip": Param("none", choices=["none", "horizontal", "vertical"]),
              "crop": Param(True, help="arbitrary angle: crop away the empty corners")}

    def run(self, img, ctx, angle, flip, crop):
        out = img
        a = float(angle) % 360
        if abs(a - round(a / 90) * 90) < 1e-9:
            out = np.rot90(out, int(round(a / 90)) % 4)
        else:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), a, 1.0)
            cs, sn = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(round(h * sn + w * cs)), int(round(h * cs + w * sn))
            M[0, 2] += (nw - w) / 2
            M[1, 2] += (nh - h) / 2
            out = np.stack([cv2.warpAffine(np.ascontiguousarray(img[..., c]), M, (nw, nh), flags=cv2.INTER_LANCZOS4,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            for c in range(img.shape[-1])], -1)
            if crop:
                cw, ch = _inscribed(w, h, math.radians(a))
                cw, ch = max(1, cw - 2), max(1, ch - 2)
                x0, y0 = (nw - cw) // 2, (nh - ch) // 2
                out = out[y0:y0 + ch, x0:x0 + cw]
        if flip == "horizontal":
            out = out[:, ::-1]
        elif flip == "vertical":
            out = out[::-1]
        return np.ascontiguousarray(np.clip(out, 0, 1), np.float32)


@register
class ManualCrop(Plugin):
    id = "crop.manual"
    name = "Crop"
    category = "geometry"
    description = "Crops to the rectangle x0, y0 - x1, y1 (full-resolution pixels; None = image edge)."
    params = {"x0": Param(0, 0, None, "left", px=True), "y0": Param(0, 0, None, "top", px=True),
              "x1": Param(None, 0, None, "right (exclusive)", px=True),
              "y1": Param(None, 0, None, "bottom (exclusive)", px=True)}

    def run(self, img, ctx, x0, y0, x1, y1):
        h, w = img.shape[:2]
        x0, y0 = min(max(0, int(x0 or 0)), w - 1), min(max(0, int(y0 or 0)), h - 1)
        x1 = w if x1 is None else min(max(x0 + 1, int(x1)), w)
        y1 = h if y1 is None else min(max(y0 + 1, int(y1)), h)
        return np.ascontiguousarray(img[y0:y1, x0:x1])
