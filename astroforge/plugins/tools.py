"""Utility plugins: PixelMath, statistical stretch, web/social export."""
import ast
import operator

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import stretch
from ..stats import luminance
from .base import Param, Plugin, register

# ------------------------------------------------------------------ PixelMath
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUNCS = {
    "min": np.minimum, "max": np.maximum, "abs": np.abs, "sqrt": lambda x: np.sqrt(np.maximum(x, 0)),
    "log": lambda x: np.log(np.maximum(x, 1e-9)), "exp": np.exp,
    "clip": lambda x, a=0.0, b=1.0: np.clip(x, a, b),
    "iif": lambda c, a, b: np.where(c, a, b),
    "mtf": lambda m, x: stretch.mtf(m, x),
    "mean": lambda *a: sum(a) / len(a),
    "med": lambda x: np.median(x), "avg": lambda x: np.mean(x),
}


def pixelmath(expr, env):
    """Evaluate an arithmetic expression on arrays without eval(): only whitelisted nodes."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise ValueError(f"unknown name '{n.id}' (use T, R, G, B, L)")
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP:
            return _CMP[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"not allowed in PixelMath: {ast.dump(n)[:60]}")
    return ev(ast.parse(expr.replace("$T", "T"), mode="eval"))


@register
class PixelMath(Plugin):
    id = "tools.pixelmath"
    name = "PixelMath"
    category = "tools"
    description = ("Formulas on pixels. Names: T (current channel), R, G, B, L (luminance). Functions: min, max, "
                   "abs, sqrt, log, exp, clip, iif(cond,a,b), mtf(m,x), mean, med, avg. "
                   "Example: iif(T > 0.5, T, T*0.8) or, per channel, R: 0.5*R+0.5*B")
    params = {"expression": Param("T", help="used for every channel when the per-channel ones are empty"),
              "expr_r": Param("", help="red channel (optional)"),
              "expr_g": Param("", help="green channel (optional)"),
              "expr_b": Param("", help="blue channel (optional)")}

    def run(self, img, ctx, expression, expr_r, expr_g, expr_b):
        env = {"L": luminance(img)}
        if img.shape[-1] == 3:
            env.update(R=img[..., 0], G=img[..., 1], B=img[..., 2])
        out = np.empty_like(img)
        per = [expr_r, expr_g, expr_b]
        for c in range(img.shape[-1]):
            e = (per[c] if img.shape[-1] == 3 else "") or expression
            v = pixelmath(e, {**env, "T": img[..., c]})
            out[..., c] = np.broadcast_to(np.asarray(v, np.float32), img.shape[:2])
        return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ statistical stretch
@register
class StatisticalStretch(Plugin):
    id = "stretch.statistical"
    name = "Statistical stretch"
    category = "stretch"
    description = ("Stretch to a target median from the image statistics (idea of Seti Astro). Unlinked mode "
                   "stretches each channel on its own, which also neutralizes a color cast.")
    params = {"target": Param(0.25, 0.05, 0.6, "target median (background brightness)"),
              "black_sigma": Param(-2.8, -10.0, 0.0, "black point in noise units"),
              "linked": Param(False, help="same stretch on all channels (keeps the original colors)"),
              "boost": Param(0.0, 0.0, 1.0, "extra contrast curve after the stretch")}

    def run(self, img, ctx, target, black_sigma, linked, boost):
        if linked or img.shape[-1] == 1:
            out, _ = stretch.auto_stretch(img, target, black_sigma)
        else:
            out = np.empty_like(img)
            for c in range(img.shape[-1]):
                out[..., c:c + 1], _ = stretch.auto_stretch(img[..., c:c + 1], target, black_sigma)
        if boost:
            # contrast above the target only: nebula brighter, background left alone
            x = out
            out = np.clip(x + boost * (x - target) * (1 - x) * np.clip((x - target) * 4, 0, 1), 0, 1)
        ctx.linear = False
        return out.astype(np.float32)


# ------------------------------------------------------------------ web / social export
@register
class WebExport(Plugin):
    id = "export.web"
    name = "Web/social resize + watermark"
    category = "export"
    stage = "nonlinear"
    description = "Last step for sharing: resize to the long side, light sharpening, optional signature."
    params = {"long_side": Param(2048, 512, 8192, "long side in pixels (Instagram 1080, web 2048)", px=True),
              "sharpen": Param(0.3, 0.0, 1.0, "unsharp mask after the resize"),
              "text": Param("", help="watermark, e.g. your name and the object"),
              "position": Param("bottom-right", choices=["bottom-right", "bottom-left", "top-right", "top-left"]),
              "opacity": Param(0.6, 0.1, 1.0)}

    def run(self, img, ctx, long_side, sharpen, text, position, opacity):
        h, w = img.shape[:2]
        f = long_side / max(h, w)
        if f < 1:
            img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
            if img.ndim == 2:
                img = img[..., None]
        if sharpen:
            blur = cv2.GaussianBlur(img, (0, 0), 1.0)
            if blur.ndim == 2:
                blur = blur[..., None]
            img = np.clip(img + sharpen * (img - blur), 0, 1)
        if text:
            h, w = img.shape[:2]
            base = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8).squeeze()).convert("RGBA")
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            size = max(14, w // 60)
            try:
                font = ImageFont.truetype("segoeui.ttf", size)
            except OSError:
                font = ImageFont.load_default()
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            x = size if "left" in position else w - tw - size
            y = size if "top" in position else h - th - size * 1.4
            draw.text((x, y), text, font=font, fill=(255, 255, 255, int(255 * opacity)))
            out = Image.alpha_composite(base, layer).convert("RGB" if img.shape[-1] == 3 else "L")
            img = np.asarray(out, np.float32) / 255
            if img.ndim == 2:
                img = img[..., None]
        return img.astype(np.float32)
