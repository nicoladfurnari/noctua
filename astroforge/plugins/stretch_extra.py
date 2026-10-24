"""Additional stretches (GHS, arcsinh) and linear deconvolution."""
import cv2
import numpy as np

from .. import deconv, masks, stretch
from ..stats import channel_stats, luminance
from .base import Param, Plugin, register


def _ghs(x, D, b, SP):
    """Generalized hyperbolic stretch core, normalized to [0, 1]."""
    def t(v):
        d = v - SP
        s = np.sign(d)
        a = np.abs(d)
        if b == 0:
            y = 1 - np.exp(-D * a)
        elif b == -1:
            y = np.log1p(D * a)
        elif b < 0:
            y = (1 - (1 - b * D * a) ** ((b + 1) / b)) / (D * (b + 1))
        else:
            y = 1 - (1 + b * D * a) ** (-1 / b)
        return s * y
    lo, hi = t(np.float32(0)), t(np.float32(1))
    return ((t(x) - lo) / (hi - lo)).astype(np.float32)


def _preserve_color(img, lum, new):
    """Scale RGB by the luminance stretch ratio; where a channel would exceed 1, desaturate toward white at the
    same luminance instead of clipping (no hue shift, no magenta/yellow star cores)."""
    x = np.clip(img, 0, None)
    out = x * (new / np.maximum(lum, 1e-7))[..., None]
    new = np.clip(new, 0, 1)[..., None]
    mx = out.max(axis=-1, keepdims=True)
    s = np.where(mx > 1, (1 - new) / np.maximum(mx - new, 1e-7), 1)
    return np.clip(new + (out - new) * s, 0, 1)


@register
class GHS(Plugin):
    id = "stretch.ghs"
    name = "Generalized hyperbolic stretch"
    category = "stretch"
    description = ("GHS (Payne & Cranfield, PixInsight GeneralizedHyperbolicStretch): strong stretch around a "
                   "symmetry point SP, b sets where contrast goes, LP/HP keep shadows/highlights linear. Modes: "
                   "stretch, inverse (undo a stretch), lin (linear stretch from a black point).")
    params = {"D": Param(8.0, 0.0, 20.0, "stretch strength (ln(D+1) in the original GHS), 0 = identity"),
              "b": Param(4.0, -5.0, 15.0, "local intensity: higher = more focused around SP, -1 log, 0 exp"),
              "SP": Param(None, 0.0, 1.0, "symmetry point; None or 'auto' = background median"),
              "LP": Param(0.0, 0.0, 1.0, "shadow protection: linear below LP (LP <= SP)"),
              "HP": Param(1.0, 0.0, 1.0, "highlight protection: linear above HP (HP >= SP)"),
              "mode": Param("stretch", choices=["stretch", "inverse", "lin"],
                            help="stretch, inverse transformation, or linear stretch from BP"),
              "BP": Param(0.0, 0.0, 0.99, "black point of the 'lin' mode"),
              "color": Param("luminance", choices=["luminance", "rgb"], help="preserve colors or stretch channels")}

    def run(self, img, ctx, D, b, SP, color, LP=0.0, HP=1.0, mode="stretch", BP=0.0):
        if SP is None or SP == "auto":
            SP = float(np.mean([m for m, _ in channel_stats(img)]))
        if mode == "lin":
            f = lambda v: np.clip((v - BP) / (1 - BP), 0, 1).astype(np.float32)  # noqa: E731
        else:
            f = lambda v: stretch.ghs(v, np.expm1(D), b, SP, LP, HP, mode == "inverse")  # noqa: E731
        if color == "luminance" and img.shape[-1] == 3:
            lum = luminance(img)
            out = _preserve_color(img, lum, f(np.clip(lum, 0, 1)))
        else:
            out = f(np.clip(img, 0, 1))
        ctx.log(f"GHS {mode} D={D} b={b} SP={SP:.5f} LP={LP} HP={HP}" + (f" BP={BP}" if mode == "lin" else ""))
        if mode != "lin":      # a linear stretch keeps the data linear
            ctx.linear = False
        return out.astype(np.float32)


@register
class Arcsinh(Plugin):
    id = "stretch.arcsinh"
    name = "Arcsinh stretch"
    category = "stretch"
    description = ("Color-preserving arcsinh stretch (PixInsight ArcsinhStretch): keeps star colors instead of "
                   "washing them to white.")
    params = {"beta": Param(300.0, 1.0, 5000.0, "stretch factor"),
              "black_sigma": Param(-2.8, -10.0, 0.0, "black point in noise units from the median"),
              "protect_highlights": Param(False, help="scale the whole image so no pixel clips (else per pixel)"),
              "use_rgb_working_space": Param(True, help="luminance with RGB working space weights, else R=G=B"),
              "estimate_black_point": Param(False, help="automatic black point (0.2% darkest pixels clipped) "
                                                        "instead of black_sigma")}

    def run(self, img, ctx, beta, black_sigma, protect_highlights=False, use_rgb_working_space=True,
            estimate_black_point=False):
        if estimate_black_point:
            low = img[::2, ::2].reshape(-1, img.shape[-1])
            bp = max(0.0, float(np.min(np.quantile(low, 0.002, axis=0))))
        else:
            st = channel_stats(img)
            bp = max(0.0, float(np.mean([m + black_sigma * d * 1.4826 for m, d in st])))
        x = np.clip((img - bp) / (1 - bp), 0, 1)
        if x.shape[-1] != 3:
            lum = x[..., 0]
        else:
            lum = luminance(x) if use_rgb_working_space else x.mean(-1)
        new = np.arcsinh(beta * lum) / np.arcsinh(beta)
        out = x * (new / np.maximum(lum, 1e-7))[..., None]
        if protect_highlights:
            out = out / max(float(out.max()), 1.0)
        else:
            # bring clipped colors back into range without hue shift
            out = out / np.maximum(out.max(axis=-1, keepdims=True), 1)
        ctx.log(f"arcsinh beta {beta}, black point {bp:.5f}")
        ctx.linear = False
        return np.clip(out, 0, 1).astype(np.float32)


@register
class Deconvolution(Plugin):
    id = "sharpen.deconvolution"
    name = "Deconvolution (Richardson-Lucy TV)"
    category = "sharpen"
    stage = "linear"
    description = ("Richardson-Lucy with total variation regularisation and a Moffat PSF fitted on the stars "
                   "(optionally per tile). Deringing keeps dark halos out; the background is protected.")
    params = {"psf": Param("auto", choices=["auto", "gaussian"], help="auto = elliptical Moffat fitted on stars"),
              "fwhm": Param(None, 0.8, 15.0, "PSF FWHM in pixels; None = measured from the stars", px=True),
              "iterations": Param(15, 1, 100, "RL iterations"),
              "tv": Param(0.002, 0.0, 0.02, "total variation regularisation (noise and ringing control)"),
              "deringing": Param(0.8, 0.0, 1.0, "local clamping against dark rings around stars"),
              "tiles": Param(1, 1, 6, "1 = one global PSF; N = PSF measured on an N x N grid of tiles"),
              "channels": Param("luminance", choices=["luminance", "rgb"],
                                help="deconvolve the luminance only (colours kept) or each channel"),
              "amount": Param(0.8, 0.0, 1.0, "blend with the original")}

    def run(self, img, ctx, psf, fwhm, iterations, tv, deringing, tiles, channels, amount):
        sat = img.max(axis=-1) >= 0.95
        protect = None
        if sat.any():
            protect = cv2.GaussianBlur(cv2.dilate(sat.astype(np.float32), np.ones((9, 9), np.uint8)), (0, 0), 2)
        kw = dict(iterations=iterations, psf=psf, fwhm=fwhm, tv_weight=tv, dering=deringing, tiles=int(tiles),
                  protect=protect)
        ctx.log(f"deconvolution {iterations} it., TV {tv}, deringing {deringing}, tiles {tiles}, {channels}")
        if channels == "luminance" and img.shape[-1] == 3:
            lum = luminance(img)
            new = deconv.deconvolve(lum, log=ctx.log, **kw)
            e = 1e-5 + 0.05 * float(np.median(lum[::4, ::4]))
            out = img * ((np.maximum(new, 0) + e) / (np.maximum(lum, 0) + e))[..., None]
        else:
            out = np.empty_like(img)
            for c in range(img.shape[-1]):
                out[..., c] = deconv.deconvolve(img[..., c], log=ctx.log if c == 0 else None, **kw)
        m = (amount * masks.smoothstep(masks.lum_mask(img, True), 0.2, 0.45))[..., None]
        return np.clip(img * (1 - m) + out * m, 0, 1).astype(np.float32)
