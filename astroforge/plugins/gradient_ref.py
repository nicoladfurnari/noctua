"""Gradient correction against an all-sky survey reference (MultiscaleGradientCorrection-like)."""
import numpy as np

from .. import allsky, astrometry
from .base import Param, Plugin, register


def _scaled_wcs(wcs, scale):
    """WCS of the full image adapted to an image `scale` times smaller (GUI preview)."""
    if scale == 1:
        return wcs
    w = wcs.celestial.deepcopy()
    w.sip = None
    for i in range(2):
        w.wcs.ctype[i] = w.wcs.ctype[i].replace("-SIP", "")
    w.wcs.crpix = (np.array(w.wcs.crpix) - 0.5) * scale + 0.5
    if w.wcs.has_cd():
        w.wcs.cd = np.array(w.wcs.cd) / scale
    else:
        w.wcs.cdelt = np.array(w.wcs.cdelt) / scale
    w.wcs.set()
    return w


def _tools(ctx):
    if ctx.tools:
        return ctx.tools
    from ..pipeline import load_config
    return load_config().get("tools", {})


@register
class BackgroundAllSky(Plugin):
    id = "background.allsky"
    name = "Gradient correction (all-sky reference)"
    category = "background"
    stage = "linear"
    description = ("MultiscaleGradientCorrection-like: downloads a survey image of the field (DSS2 / Pan-STARRS "
                   "via CDS hips2fits, plate solved with ASTAP if needed), fits it to the image at the scales "
                   "below `scale` and removes only what the survey does not show at large scale: gradients go, "
                   "extended faint nebulosity stays. Without WCS or network falls back to background.dbe.")
    params = {"survey": Param("auto", choices=allsky.SURVEY_CHOICES,
                              help="reference survey: auto = DSS2 red+blue plates (whole sky); dss2_color = 8-bit "
                                   "colour composite; panstarrs = PS1 g/r (dec > -30)"),
              "scale": Param(0.0, 0.0, 600.0, "gradient scale in arcmin: larger structures not in the survey are "
                                              "gradient; 0 = auto (1024 px, at most half the image)"),
              "correction": Param("subtraction", choices=["subtraction", "division"],
                                  help="subtraction for gradients (light pollution), division for vignetting"),
              "protect": Param(0.0, 0.0, 1.0, "inside nebulae seen by the survey, blend the model towards a "
                                              "quadratic fitted on the dark sky (when the survey saturates)"),
              "fallback": Param("dbe", choices=["dbe", "none"],
                                help="when there is no WCS / ASTAP or the download fails: run background.dbe, "
                                     "or leave the image unchanged")}

    def _fallback(self, img, ctx, fallback, correction, why):
        if fallback == "none":
            ctx.log(f"allsky: {why}; image unchanged")
            return img
        ctx.log(f"allsky: {why}; falling back to background.dbe")
        from .base import get
        dbe = get("background.dbe")
        p = dbe.scaled(dbe.resolve({"correction": correction}), ctx.scale)
        return dbe.run(img, ctx, **p)

    def run(self, img, ctx, survey, scale, correction, protect, fallback):
        wcs = ctx.layers.get("wcs")
        if wcs is not None:
            wcs = _scaled_wcs(wcs, ctx.scale)
        else:
            wcs = astrometry.solve(img, _tools(ctx), header=ctx.header, log=ctx.log)
            if wcs is None:
                return self._fallback(img, ctx, fallback, correction, "no plate solution")
            if ctx.scale == 1:
                ctx.layers["wcs"] = wcs
        try:
            pxs = astrometry.pixel_scale_arcsec(wcs)
        except Exception as e:  # malformed WCS
            return self._fallback(img, ctx, fallback, correction, f"unusable WCS ({e})")
        scale_px = scale * 60.0 / pxs if scale > 0 else allsky.default_scale_px(img.shape)
        ref = allsky.reference_background(wcs, img.shape, survey, scale_px=scale_px, log=ctx.log)
        if ref is None:
            return self._fallback(img, ctx, fallback, correction, "survey reference unavailable")
        model, info = allsky.gradient_model(img, ref, scale_px=scale_px, protect=protect, log=ctx.log)
        out = allsky.apply(img, model, correction)
        ctx.layers["background_model"] = model
        ctx.layers["allsky_reference"] = ref
        ctx.layers["allsky"] = {**info, "survey": allsky.pick_survey(survey, wcs, img.shape),
                                "scale_arcmin": scale_px * pxs / 60.0}
        ctx.log(f"allsky: {correction}, scale {scale_px * pxs / 60:.1f}' ({scale_px:.0f} px), "
                f"reference factors {' '.join(f'{v:.3g}' for v in info['a'])} corr {np.round(info['corr'], 2).tolist()}")
        return out.astype(np.float32, copy=False)
