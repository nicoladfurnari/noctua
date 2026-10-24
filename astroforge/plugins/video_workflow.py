"""Processes from a PixInsight nebula workflow (Trifid, Askar 180): Seti Astro StarStretch, star size by
midtones, Correct Magenta Stars, UnsharpMask, Bill Blanshan's NarrowbandNormalization, automatic colour
gradient fit (HLP Astro Image Premiere style) and a text annotation box for exports."""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import enhance, masks, stretch
from ..stats import LUMA, luminance, median_mad
from .annotate import _font, _rgb
from .base import Param, Plugin, register
from .stretch_extra import _preserve_color


def _by_max(x, f):
    """Colour-preserving curve: every channel scaled by f(max)/max (hue and ratios kept, never above 1)."""
    m = x.max(axis=-1, keepdims=True)
    return np.clip(x * (f(m) / np.maximum(m, 1e-7)), 0, 1).astype(np.float32)


def _erode_blend(x, amount):
    if amount <= 0:
        return x
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    er = np.stack([cv2.erode(np.ascontiguousarray(x[..., c]), ker) for c in range(x.shape[-1])], -1)
    return (x * (1 - amount) + er * amount).astype(np.float32)


# ------------------------------------------------------------------ stars
@register
class StarStretch(Plugin):
    id = "stars.stretch"
    name = "Star stretch"
    category = "stars"
    description = ("Seti Astro StarStretch: stretches the star layer saved by a star removal (StarNet/StarX) or, "
                   "without a layer, the image. Rational stretch (3^amount x) / ((3^amount - 1) x + 1) applied "
                   "colour-preserving, colour boost, optional green removal and a slight star reduction. On the "
                   "layer the image passes through unchanged; add the stars back with 'stars.recombine'.")
    params = {"amount": Param(5.0, 0.0, 8.0, "stretch amount (0 = none)"),
              "color_boost": Param(1.0, 0.0, 3.0, "star saturation: chroma x (1 + 0.5 boost), 0 = unchanged"),
              "reduce": Param(0.15, 0.0, 1.0, "slight star reduction (3x3 erosion blend)"),
              "remove_green": Param(False, help="SCNR green on the stars")}

    def run(self, img, ctx, amount, color_boost, reduce, remove_green):
        st = ctx.layers.get("stars")
        x = np.clip(st if st is not None else img, 0, 1).astype(np.float32)
        k = 3.0 ** amount
        y = _by_max(x, lambda m: k * m / ((k - 1) * m + 1))
        if color_boost and y.shape[-1] == 3:
            lum = luminance(y)[..., None]
            y = lum + (y - lum) * (1 + 0.5 * color_boost)
            y = np.clip(y / np.maximum(y.max(axis=-1, keepdims=True), 1), 0, 1)
        if remove_green and y.shape[-1] == 3:
            y = enhance.scnr(y, 1.0, "average_neutral", "green", preserve_lightness=False)
        y = _erode_blend(y.astype(np.float32), reduce)
        ctx.log(f"star stretch {amount} (x{k:.0f}), colour boost {color_boost}, "
                + ("star layer" if st is not None else "image"))
        if st is None:
            ctx.linear = False
            return y
        ctx.layers["stars"] = y
        ctx.layers["stars_mode"] = "screen"     # stretched stars are screened onto the stretched image
        return img


@register
class StarSize(Plugin):
    id = "stars.size"
    name = "Star size (midtones)"
    category = "stars"
    stage = "nonlinear"
    description = ("Changes the apparent star diameter with a midtones transfer on the star layer (as moving the "
                   "HistogramTransformation midtones of the star image): + enlarges (brighter wings), - shrinks. "
                   "Colour-preserving. Without a star layer it works on the image under a star mask.")
    params = {"amount": Param(0.3, -1.0, 1.0, "-1 shrink .. +1 enlarge (midtones 0.5 - 0.45 amount)")}

    def run(self, img, ctx, amount):
        m = 0.5 - 0.45 * float(amount)
        st = ctx.layers.get("stars")
        ctx.log(f"star size {amount:+.2f}: midtones {m:.3f} on the " + ("star layer" if st is not None else "image"))
        if st is not None:
            ctx.layers["stars"] = _by_max(np.clip(st, 0, 1), lambda v: stretch.mtf(m, v))
            return img
        mask = masks.star_mask(img, ctx.linear, scale=ctx.scale)[..., None]
        new = _by_max(np.clip(img, 0, 1), lambda v: stretch.mtf(m, v))
        return (img + (new - img) * mask).astype(np.float32)


# ------------------------------------------------------------------ colour
@register
class FixMagenta(Plugin):
    id = "color.fix_magenta"
    name = "Correct magenta stars"
    category = "color"
    description = ("Correct Magenta Stars (SHO/HOO): SCNR green on the inverted image, inverted back, so magenta "
                   "(high R and B, low G) becomes neutral while green/teal areas are untouched. invert=False is a "
                   "plain SCNR green. Optionally only inside a star mask.")
    params = {"amount": Param(1.0, 0.0, 1.0, "strength"),
              "method": Param("average_neutral", choices=list(enhance.SCNR_METHODS), help="SCNR protection method"),
              "invert": Param(True, help="invert before SCNR (removes magenta); off = plain SCNR green"),
              "stars_only": Param(False, help="apply only inside a star mask")}

    def run(self, img, ctx, amount, method, invert, stars_only):
        if img.shape[-1] != 3:
            ctx.log("fix magenta: mono image, skipped")
            return img
        x = np.clip(img, 0, 1)
        y = enhance.scnr(1 - x if invert else x, amount, method, "green", preserve_lightness=False)
        out = 1 - y if invert else y
        if stars_only:
            m = masks.star_mask(img, ctx.linear, scale=ctx.scale)
            ctx.log(f"fix magenta: star mask coverage {m.mean():.1%}")
            out = x + (out - x) * m[..., None]
        return np.clip(out, 0, 1).astype(np.float32)


_NB = {"HOO": {"H": 0, "O": (1, 2)}, "SHO": {"S": 0, "H": 1, "O": 2}, "HSO": {"H": 0, "S": 1, "O": 2},
       "HOS": {"H": 0, "O": 1, "S": 2}}


def _spread(x, q=0.995):
    s = x[::2, ::2]
    med = float(np.median(s))
    return med, max(float(np.quantile(s, q)) - med, 1e-6)


@register
class NBNormalization(Plugin):
    id = "color.nb_normalization"
    name = "Narrowband normalization"
    category = "color"
    stage = "nonlinear"
    description = ("NarrowbandNormalization (after Bill Blanshan's NBN script) for HOO/SHO/HSO/HOS composites: "
                   "OIII and SII are normalised to the Ha background and signal range, boosted, remapped to the "
                   "palette, lightness taken from the original or from Ha, highlights reduced, blended with the "
                   "original. A reconstruction of the method, not the script's PixelMath.")
    params = {"mode": Param("HOO", choices=list(_NB), help="channel order of the image (and of the output)"),
              "boost": Param(1.0, 0.0, 3.0, "OIII boost after normalisation"),
              "sii_boost": Param(1.0, 0.0, 3.0, "SII boost after normalisation (SHO/HSO/HOS)"),
              "blackpoint": Param(0.0, 0.0, 1.0, "fraction of every channel's background removed first"),
              "lightness": Param("original", choices=["original", "ha", "off"],
                                 help="lightness of the result: the original image, Ha, or as mapped"),
              "highlight_reduction": Param(0.0, 0.0, 1.0, "compress the highlights (1 = brightest to 50%)"),
              "blend": Param(1.0, 0.0, 1.0, "blend with the original")}

    def run(self, img, ctx, mode, boost, sii_boost, blackpoint, lightness, highlight_reduction, blend):
        if img.shape[-1] != 3:
            ctx.log("NBN: mono image, skipped")
            return img
        x = np.clip(img, 0, 1).astype(np.float32)
        if blackpoint:
            bg = np.array([median_mad(x[..., c])[0] for c in range(3)], np.float32) * blackpoint
            x = np.clip((x - bg) / np.maximum(1 - bg, 1e-6), 0, 1)
        lay = _NB[mode]
        H = x[..., lay["H"]]
        mh, sh = _spread(H)
        ch = {"H": H}
        for key, gain in (("O", boost), ("S", sii_boost)):
            if key not in lay:
                continue
            idx = lay[key]
            X = x[..., list(idx)].mean(-1) if isinstance(idx, tuple) else x[..., idx]
            mx, sx = _spread(X)
            k = float(np.clip(sh / sx, 0.1, 10.0)) * gain
            ch[key] = np.clip(mh + (X - mx) * k, 0, None)
            ctx.log(f"NBN {mode}: {key} scaled x{k:.2f} to Ha (bg {mx:.4f} -> {mh:.4f})")
        if mode == "HOO":
            out = np.stack([ch["H"], ch["O"], ch["O"]], -1)
        else:
            out = np.stack([ch[c] for c in mode], -1)
        out = out / np.maximum(out.max(axis=-1, keepdims=True), 1)
        if lightness != "off":
            target = luminance(np.clip(img, 0, 1)) if lightness == "original" else H
            out = _preserve_color(out, luminance(out), target)
        if highlight_reduction:
            out = _by_max(out, lambda m: m - 0.5 * highlight_reduction * m * m)
        out = img * (1 - blend) + out * blend
        return np.clip(out / np.maximum(out.max(axis=-1, keepdims=True), 1), 0, 1).astype(np.float32)


def _poly_terms(u, v, deg):
    return np.stack([u ** i * v ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


_GRAD = {"low": (1, 8), "medium": (2, 12), "high": (3, 16)}      # polynomial degree, sample grid


@register
class GradientFit(Plugin):
    id = "color.gradient_fit"
    name = "Colour gradient fit"
    category = "background"
    description = ("Automatic colour gradient removal (like HLP Astro Image Premiere): the large-scale R-G and B-G "
                   "differences (or R/G, B/G ratios) are fitted with a low-order polynomial on star-free "
                   "background boxes and removed, then the luminance is restored: blue/purple casts from low "
                   "altitude or light pollution go away, brightness gradients and nebula are kept.")
    params = {"strength": Param("medium", choices=list(_GRAD), help="low: plane, medium: degree 2, high: degree 3"),
              "method": Param("difference", choices=["difference", "ratio"],
                              help="additive (light pollution) or multiplicative model"),
              "keep": Param(0.7, 0.2, 1.0, "fraction of darkest sample boxes used as background"),
              "amount": Param(1.0, 0.0, 1.0, "strength of the correction")}

    def run(self, img, ctx, strength, method, keep, amount):
        if img.shape[-1] != 3:
            ctx.log("gradient fit: mono image, skipped")
            return img
        deg, n = _GRAD[strength]
        h, w = img.shape[:2]
        ys, xs = np.linspace(0, h, n + 1).astype(int), np.linspace(0, w, n + 1).astype(int)
        med, uv = [], []
        for i in range(n):
            for j in range(n):
                box = img[ys[i]:ys[i + 1]:2, xs[j]:xs[j + 1]:2].reshape(-1, 3)
                if len(box):
                    med.append(np.median(box, axis=0))
                    uv.append(((xs[j] + xs[j + 1]) / w - 1, (ys[i] + ys[i + 1]) / h - 1))
        med, uv = np.array(med, np.float64), np.array(uv)
        lum = med @ LUMA
        sel = lum <= np.quantile(lum, keep) + 1e-12
        A = _poly_terms(uv[sel, 0], uv[sel, 1], deg)
        G = np.maximum(med[sel, 1], 1e-6)
        sh, sw = max(2, h // 8), max(2, w // 8)
        gu, gv = np.meshgrid((np.arange(sw) + 0.5) / sw * 2 - 1, (np.arange(sh) + 0.5) / sh * 2 - 1)
        At = _poly_terms(gu.ravel(), gv.ravel(), deg)
        surf = []
        for c in (0, 2):
            d = med[sel, c] - med[sel, 1] if method == "difference" else med[sel, c] / G
            ok = np.ones(len(d), bool)
            for _ in range(3):           # robust fit: reject boxes with nebula / star clusters
                coef = np.linalg.lstsq(A[ok], d[ok], rcond=None)[0]
                r = d - A @ coef
                s = 1.4826 * np.median(np.abs(r[ok])) + 1e-9
                ok = np.abs(r) < 2.5 * s
                if ok.sum() < A.shape[1] + 2:
                    ok[:] = True
                    break
            s = (At @ coef).reshape(sh, sw).astype(np.float32)
            s = s - np.median(s) if method == "difference" else s / max(float(np.median(s)), 1e-6)
            surf.append(cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR))
        out = img.astype(np.float32).copy()
        if method == "difference":
            sR, sB = surf[0] * amount, surf[1] * amount
            out[..., 0] -= sR
            out[..., 2] -= sB
            out += (LUMA[0] * sR + LUMA[2] * sB)[..., None]       # same luminance as before
        else:
            out[..., 0] /= np.maximum(surf[0], 1e-3) ** amount
            out[..., 2] /= np.maximum(surf[1], 1e-3) ** amount
            out *= (luminance(img) / np.maximum(luminance(out), 1e-7))[..., None]
        ctx.log(f"gradient fit {strength} ({method}): {int(sel.sum())} boxes, R/G range "
                f"{np.ptp(surf[0]):.4f}, B/G range {np.ptp(surf[1]):.4f}")
        return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ sharpen
@register
class UnsharpMask(Plugin):
    id = "sharpen.unsharp"
    name = "Unsharp mask"
    category = "sharpen"
    stage = "nonlinear"
    description = ("PixInsight UnsharpMask: (x - k blur) / (1 - k) with a Gaussian of sigma px. Deringing clamps "
                   "the result toward the local min/max of the original (dark = no black rings, bright = no "
                   "white halos). Luminance only keeps the colours.")
    params = {"sigma": Param(2.0, 0.3, 50.0, "Gaussian standard deviation in pixels", px=True),
              "amount": Param(0.6, 0.0, 0.95, "k: 0.5 doubles the detail"),
              "dering_dark": Param(0.1, 0.0, 1.0, "dark deringing (1 = never below the local minimum)"),
              "dering_bright": Param(0.0, 0.0, 1.0, "bright deringing (1 = never above the local maximum)"),
              "luminance": Param(True, help="sharpen the luminance only")}

    def run(self, img, ctx, sigma, amount, dering_dark, dering_bright, luminance):
        size = 2 * int(np.ceil(sigma)) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        def usm(ch):
            ch = np.ascontiguousarray(ch, dtype=np.float32)
            o = (ch - amount * cv2.GaussianBlur(ch, (0, 0), sigma)) / (1 - amount)
            if dering_dark:
                lo = cv2.erode(ch, ker)
                o = np.where(o < lo, o + (lo - o) * dering_dark, o)
            if dering_bright:
                hi = cv2.dilate(ch, ker)
                o = np.where(o > hi, o - (o - hi) * dering_bright, o)
            return o
        ctx.log(f"unsharp mask sigma {sigma:.2f}px amount {amount}")
        if luminance and img.shape[-1] == 3:
            lum = img @ LUMA
            out = img + (usm(lum) - lum)[..., None]
        else:
            out = np.stack([usm(img[..., c]) for c in range(img.shape[-1])], -1)
        return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ annotation
_CORNERS = ["top-left", "top-right", "bottom-left", "bottom-right", "top", "bottom", "center"]


@register
class AnnotateText(Plugin):
    id = "annotate.text"
    name = "Text box"
    category = "annotate"
    description = ("Text overlay for exports (PixInsight Annotation): multi-line text in a corner or at x,y "
                   "(fractions of the image), font size relative to the image height, colour, opacity and an "
                   "optional background box. Drawn in float, the rest of the image is untouched.")
    params = {"text": Param("", help="text; new lines with \\n"),
              "position": Param("bottom-right", choices=_CORNERS, help="corner, or [x, y] fractions (box top-left)"),
              "font_size": Param(0.03, 0.005, 0.3, "text line height as a fraction of the image height"),
              "color": Param("#ffffff", help="text colour (hex)"),
              "opacity": Param(1.0, 0.0, 1.0, "text opacity"),
              "align": Param("left", choices=["left", "center", "right"], help="line alignment"),
              "box": Param(True, help="draw a background box"),
              "box_color": Param("#000000", help="box colour (hex)"),
              "box_opacity": Param(0.5, 0.0, 1.0, "box opacity"),
              "margin": Param(0.02, 0.0, 0.3, "distance from the edges, fraction of the short side"),
              "apply": Param(True, help="return the image with the text (else only kept in ctx.layers)")}

    def run(self, img, ctx, text, position, font_size, color, opacity, align, box, box_color, box_opacity, margin,
            apply):
        text = str(text or "").replace("\\n", "\n")
        if not text.strip():
            return img
        h, w = img.shape[:2]
        px = max(6, int(round(font_size * h)))
        font = _font(px)
        spacing = max(1, px // 4)
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        x0, y0, x1, y1 = probe.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align=align)
        pad = max(2, int(round(px * 0.4)))
        bw, bh = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
        mg = int(round(margin * min(h, w)))
        if isinstance(position, (list, tuple)):
            bx, by = int(round(float(position[0]) * w)), int(round(float(position[1]) * h))
        else:
            bx = {"left": mg, "right": w - mg - bw}.get(position.split("-")[-1], (w - bw) // 2)
            by = {"top": mg, "bottom": h - mg - bh}.get(position.split("-")[0], (h - bh) // 2)
        tm = Image.new("L", (bw, bh))
        ImageDraw.Draw(tm).multiline_text((pad - x0, pad - y0), text, fill=255, font=font, spacing=spacing,
                                          align=align)
        tm = np.asarray(tm, np.float32) / 255
        # clip the box to the image
        cx0, cy0, cx1, cy1 = max(bx, 0), max(by, 0), min(bx + bw, w), min(by + bh, h)
        if cx0 >= cx1 or cy0 >= cy1:
            ctx.log("text box outside the image, skipped")
            return img
        tm = tm[cy0 - by:cy1 - by, cx0 - bx:cx1 - bx][..., None]
        out = np.clip(img, 0, 1).astype(np.float32).copy()
        reg = out[cy0:cy1, cx0:cx1]
        col = lambda hx: np.array(_rgb(hx), np.float32) / 255  # noqa: E731
        tc, bc = col(color), col(box_color)
        if img.shape[-1] == 1:
            tc, bc = np.array([tc @ LUMA]), np.array([bc @ LUMA])
        if box:
            reg[:] = reg * (1 - box_opacity) + bc * box_opacity
        reg[:] = reg * (1 - tm * opacity) + tc * tm * opacity
        ctx.log(f"text box {bw}x{bh}px at ({bx}, {by}), font {px}px")
        if not apply:
            ctx.layers["text_image"] = out
            return img
        return out
