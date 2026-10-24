"""Plugins wrapping the core engine: crop, background, color, noise, stretch."""
import cv2
import numpy as np

from .. import background, color, crop, denoise, masks, stretch
from ..stats import channel_stats, describe
from .base import Param, Plugin, register


@register
class AutoCrop(Plugin):
    id = "crop.auto"
    name = "Auto crop"
    category = "geometry"
    description = "Removes the ragged borders left by registration and integration."
    params = {"pad": Param(12, 0, 200, "extra pixels removed on each side", px=True),
              "max_frac": Param(0.06, 0.0, 0.3, "maximum crop per side, fraction of the short side")}

    def run(self, img, ctx, pad, max_frac):
        out, m = crop.auto_crop(img, pad=pad, max_frac=max_frac)
        ctx.log(f"crop L/R/T/B = {m} -> {out.shape[1]}x{out.shape[0]}")
        return out


@register
class BackgroundPoly(Plugin):
    id = "background.poly"
    name = "Background extraction"
    category = "background"
    stage = "linear"
    description = "Fits a robust polynomial to background tiles and subtracts it (ABE-like)."
    params = {"degree": Param(2, 1, 4, "polynomial degree; 1-2 for fields full of nebulosity"),
              "tiles": Param(40, 10, 120, "number of tiles along the long side"),
              "reject_sigma": Param(1.5, 0.5, 5.0, "how aggressively bright tiles are rejected")}

    def run(self, img, ctx, degree, tiles, reject_sigma):
        out, model, kept = background.extract(img, degree=degree, tiles=tiles, reject_sigma=reject_sigma)
        ctx.layers["background_model"] = model
        ctx.log(f"background removed, {kept:.0%} tiles used")
        return out


@register
class NeutralizeBackground(Plugin):
    id = "color.neutralize_background"
    name = "Background neutralization"
    category = "color"
    description = "Makes the sky background gray. Works on linear and stretched images."

    def run(self, img, ctx):
        if img.shape[-1] != 3:
            return img
        out, bg = color.neutralize_background(img)
        ctx.log(f"background levels were {np.round(bg, 5)}")
        return out


@register
class StarWhiteBalance(Plugin):
    id = "color.white_balance_stars"
    name = "Star white balance"
    category = "color"
    stage = "linear"
    description = "Scales channels so that the average star is white (ColorCalibration-like)."
    params = {"max_stars": Param(3000, 100, 20000, "number of stars measured"),
              "max_correction": Param(2.0, 1.0, 8.0, "largest factor between channels (stars behind dust "
                                                     "or low on the horizon are really red)")}

    def run(self, img, ctx, max_stars, max_correction):
        if img.shape[-1] != 3:
            return img
        out, k, n = color.white_balance_stars(img, max_stars, max_correction)
        ctx.log(f"white balance {np.round(k, 3)} from {n} stars")
        return out


@register
class SaturatedCores(Plugin):
    id = "color.fix_saturated"
    name = "Saturated cores fix"
    category = "color"
    description = ("Bright cores clipped in the camera (galaxy nuclei, bright stars) clip at a different level in "
                   "each color after flat and white balance, and turn magenta. They are made neutral white.")
    params = {"start": Param(0.6, 0.2, 0.95, "fraction of the clip level where the fix starts")}

    def run(self, img, ctx, start):
        if img.shape[-1] != 3:
            return img
        # each channel clips at its own level after flat and white balance: detect per channel
        w = np.zeros(img.shape[:2], np.float32)
        for c in range(3):
            ch = img[::2, ::2, c]
            clip = float(np.percentile(ch, 99.995))
            # a clipped channel has a plateau at its top: many pixels within 1% of the level, against the ones
            # just below it (a star profile has ~10x fewer at the top than in the 9% under it). Without a plateau
            # nothing is saturated and the channel is left alone (bright cores were greyed for nothing)
            top = int((ch >= 0.99 * clip).sum())
            below = int(((ch >= 0.9 * clip) & (ch < 0.99 * clip)).sum())
            if top < 8 or top < 0.5 * below or clip <= 0:      # (8 pixels of the 2x2 subsample: a real plateau)
                continue
            w = np.maximum(w, masks.smoothstep(img[..., c], start * clip, 0.95 * clip))
        w = cv2.GaussianBlur(w, (0, 0), 2)[..., None]
        mx = img.max(axis=-1)
        if not w.any():
            return img
        ctx.log(f"neutralized {int((w[..., 0] > 0.5).sum())} clipped pixels")
        return (img * (1 - w) + mx[..., None] * w).astype(np.float32)


@register
class WaveletDenoise(Plugin):
    id = "noise.wavelet"
    name = "Wavelet noise reduction"
    category = "noise"
    description = "Starlet noise reduction (MLT-like), stronger in the dark areas."
    params = {"strength": Param(1.0, 0.0, 2.0, "scales the default amounts"),
              "protect": Param(1.0, 0.0, 1.0, "1 = full protection of bright areas via luminance mask")}

    def run(self, img, ctx, strength, protect):
        amounts = tuple(min(1.0, a * strength) for a in (0.6, 0.5, 0.4, 0.3))
        mask = 1 - protect * masks.lum_mask(img, ctx.linear)
        return denoise.denoise(img, mask, amounts=amounts)


@register
class AutoStretch(Plugin):
    id = "stretch.auto"
    name = "Auto stretch"
    category = "stretch"
    description = "STF-like linked stretch. Run it twice with a small target for a gentler result."
    params = {"target": Param(0.2, 0.02, 0.5, "target background brightness"),
              "shadows_clip": Param(-2.8, -10.0, 0.0, "black point in MAD units below the median")}

    def run(self, img, ctx, target, shadows_clip):
        out, (c0, m) = stretch.auto_stretch(img, target, shadows_clip)
        ctx.log(f"stretch c0={c0:.5f} m={m:.5f}")
        ctx.linear = False
        return out


@register
class Stats(Plugin):
    id = "analysis.stats"
    name = "Statistics"
    category = "analysis"
    description = "Logs per-channel median, MAD, maximum and a noise estimate. Image unchanged."

    def run(self, img, ctx):
        ctx.log(describe(img))
        w1 = denoise.starlet(np.ascontiguousarray(img[::2, ::2, 0]), 1)[0]
        sigma = float(np.median(np.abs(w1 - np.median(w1)))) / 0.6745 / 0.890
        snr = channel_stats(img)[0][0] / max(sigma, 1e-9)
        ctx.log(f"noise sigma (ch0) ~ {sigma:.6f}, median/noise ~ {snr:.1f}")
        return img
