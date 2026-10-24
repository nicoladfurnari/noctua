"""DBE-like background extraction: thin-plate RBF through sample boxes."""
from .. import background
from .base import Param, Plugin, register


@register
class BackgroundDBE(Plugin):
    id = "background.dbe"
    name = "Dynamic background extraction"
    category = "background"
    stage = "linear"
    description = ("DynamicBackgroundExtraction-like: smooth thin-plate surface through sample boxes placed "
                   "automatically on the sky (stars and nebulae rejected) or given by hand.")
    params = {"samples": Param([], help="list of [x, y] sample centers in pixels; empty = automatic", px="xy"),
              "samples_per_side": Param(20, 5, 60, "automatic samples along the long side"),
              "box": Param(25, 5, 201, "sample box size in pixels", px="odd"),
              "smoothing": Param(0.02, 0.0, 10.0, "surface smoothing; 0 = passes exactly through the samples"),
              "reject_sigma": Param(2.5, 1.0, 6.0, "automatic samples brighter than the trend are rejected"),
              "correction": Param("subtraction", choices=["subtraction", "division"],
                                  help="subtraction for gradients (light pollution), division for vignetting")}

    def run(self, img, ctx, samples, samples_per_side, box, smoothing, reject_sigma, correction):
        out, model, pts = background.extract_rbf(img, samples=samples or None, smoothing=smoothing, size=box,
                                                 n=samples_per_side, correction=correction,
                                                 reject_sigma=reject_sigma)
        ctx.layers["background_model"] = model
        ctx.layers["background_samples"] = pts.tolist()
        ctx.log(f"background ({correction}) from {len(pts)} {'user' if samples else 'automatic'} samples")
        return out
