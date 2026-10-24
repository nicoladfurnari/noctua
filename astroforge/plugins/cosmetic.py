"""Cosmetic plugins: hot/cold pixels, chroma noise, star halos, star color."""
import cv2
import numpy as np

from .. import denoise, masks
from ..stats import luminance
from .base import Param, Plugin, register


@register
class HotPixels(Plugin):
    id = "cosmetic.hot_pixels"
    name = "Hot/cold pixel removal"
    category = "cosmetic"
    description = "Replaces isolated pixels that deviate strongly from their 3x3 median (CosmeticCorrection-like)."
    params = {"sigma": Param(8.0, 3.0, 30.0, "deviation threshold in noise units"),
              "cold": Param(True, help="also fix dark (cold) pixels")}

    def run(self, img, ctx, sigma, cold):
        out = img.copy()
        total = 0
        for c in range(img.shape[-1]):
            ch = np.ascontiguousarray(img[..., c])
            med = cv2.medianBlur(ch, 3)
            diff = ch - med
            d = diff[::4, ::4]
            s = float(np.median(np.abs(d - np.median(d)))) * 1.4826
            bad = diff > sigma * s
            if cold:
                bad |= diff < -sigma * s
            # a real star has bright neighbours: keep pixels whose median is also high
            bad &= med < np.median(ch[::8, ::8]) + 3 * sigma * s
            out[..., c][bad] = med[bad]
            total += int(bad.sum())
        ctx.log(f"fixed {total} hot/cold pixels")
        return out


@register
class ChromaNoise(Plugin):
    id = "cosmetic.chroma_noise"
    name = "Color noise reduction"
    category = "cosmetic"
    description = "Smooths color (not detail) in the background, removing colored speckles."
    params = {"sigma": Param(2.5, 0.5, 10.0, "blur radius of the color components", px=True),
              "amount": Param(0.9, 0.0, 1.0, "strength"),
              "protect": Param(0.8, 0.0, 1.0, "protection of bright areas and stars")}

    def run(self, img, ctx, sigma, amount, protect):
        if img.shape[-1] != 3:
            return img
        lum = luminance(img)[..., None]
        chroma = img - lum
        smooth = cv2.GaussianBlur(chroma, (0, 0), sigma)
        m = (amount * (1 - protect * masks.lum_mask(img, ctx.linear)))[..., None]
        out = lum + chroma * (1 - m) + smooth * m
        return np.clip(out, 0, 1).astype(np.float32)


@register
class StarHalo(Plugin):
    id = "stars.halo_reduce"
    name = "Star halo reduction"
    category = "stars"
    stage = "nonlinear"
    description = "Attenuates the medium wavelet scales around stars (halos, bloated glow) keeping the cores."
    params = {"amount": Param(0.6, 0.0, 1.0, "halo attenuation"),
              "first_layer": Param(2, 1, 4, "first scale treated as halo (1-2 = core)", px="octave"),
              "layers": Param(3, 1, 4, "how many scales")}

    def run(self, img, ctx, amount, first_layer, layers):
        m = masks.star_mask(img, ctx.linear, grow=6, blur=4, scale=ctx.scale)[..., None]
        n = first_layer + layers
        out = np.empty_like(img)
        for c in range(img.shape[-1]):
            w = denoise.starlet(np.ascontiguousarray(img[..., c]), n)
            halo = sum(np.maximum(w[j], 0) for j in range(first_layer, n))
            out[..., c] = img[..., c] - amount * halo * m[..., 0]
        return np.clip(out, 0, 1).astype(np.float32)


@register
class StarColor(Plugin):
    id = "stars.color_boost"
    name = "Star color boost"
    category = "stars"
    stage = "nonlinear"
    description = "Increases saturation only inside the stars (Astronomy Tools 'increase star color')."
    params = {"amount": Param(0.6, 0.0, 2.0, "saturation increase")}

    def run(self, img, ctx, amount):
        if img.shape[-1] != 3:
            return img
        m = masks.star_mask(img, ctx.linear, grow=3, blur=1.5, scale=ctx.scale)[..., None]
        lum = luminance(img)[..., None]
        out = lum + (img - lum) * (1 + amount * m)
        return np.clip(out, 0, 1).astype(np.float32)
