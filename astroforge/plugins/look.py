"""Final look matching: brings the result towards the look of reference final images (see astroforge.look)."""
from .. import look
from .base import Param, Plugin, register


@register
class LookMatch(Plugin):
    id = "look.match"
    name = "Look match (reference finals)"
    category = "contrast"
    stage = "nonlinear"
    description = ("Last step of the automatic profiles: sky to a neutral grey level, nebula brightness, nebula "
                   "saturation and highlights moved towards target numbers measured on finished images "
                   "('astroforge look <final.jpg>' prints them, the GUI makes a profile from an image).")
    params = {"sky": Param(look.DEFAULT["sky"], 0.0, 0.3, "sky (background) level"),
              "faint": Param(look.DEFAULT["faint"], 0.0, 0.3, "faint background (dust, outer glow) above the sky"),
              "nebula": Param(look.DEFAULT["nebula"], 0.05, 0.9, "brightness of the nebula (brightest quarter)"),
              "saturation": Param(look.DEFAULT["saturation"], 0.0, 1.0, "mean colour saturation of the nebula"),
              "white": Param(look.DEFAULT["white"], 0.5, 1.0, "highlight ceiling (stars and cores roll off below it)"),
              "strength": Param(1.0, 0.0, 1.0, "0 = unchanged, 1 = reach the target"),
              "neutral_sky": Param(True, help="grey sky (removes a colour cast left in the background)")}

    def run(self, img, ctx, sky, faint, nebula, saturation, white, strength, neutral_sky):
        return look.match(img, sky, nebula, saturation, white, strength, neutral_sky, log=ctx.log, faint=faint)


@register
class BackgroundAuto(Plugin):
    id = "background.auto"
    name = "Automatic gradient removal"
    category = "background"
    stage = "linear"
    description = ("Chooses the gradient tool from the data: when the object fills a small part of the frame DBE "
                   "or GraXpert's AI model, which also remove complex gradients; when nebulosity fills the frame a "
                   "plane / low-degree surface, which cannot eat the nebula (AI and sample-based models take wide "
                   "faint nebulosity for gradient). On real data GraXpert ate the faint outer nebula of narrowband "
                   "targets (Sh2-252), DBE did not: DBE is the default for small objects.")
    params = {"small": Param("dbe", choices=["dbe", "graxpert"], help="tool for a small object (GraXpert -> DBE if missing)"),
              "max_area": Param(0.3, 0.0, 1.0, "nebula share of the frame above which only the simple model is used"),
              "degree": Param(1, 0, 3, "degree of the simple model"),
              "smoothing": Param(0.1, 0.0, 1.0, "GraXpert smoothing")}

    def run(self, img, ctx, small, max_area, degree, smoothing):
        from .. import stretch
        from .base import get
        st, _ = stretch.auto_stretch(img, 0.15, -2.8)
        area = look.measure(st)["nebula_area"]
        if area <= max_area:
            gx = get("ext.graxpert_background")
            if small == "graxpert" and ctx.tools.get("graxpert"):
                ctx.log(f"background.auto: nebula {area:.0%} of the frame -> GraXpert AI")
                return gx.run(img, ctx, **gx.resolve({"smoothing": smoothing}))
            ctx.log(f"background.auto: nebula {area:.0%} of the frame -> DBE")
            dbe = get("background.dbe")
            return dbe.run(img, ctx, **dbe.resolve({}))
        ctx.log(f"background.auto: nebula {area:.0%} of the frame -> degree-{degree} surface")
        poly = get("background.poly")
        return poly.run(img, ctx, **poly.resolve({"degree": degree}))
