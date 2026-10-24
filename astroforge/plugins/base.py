"""Plugin base class, parameter schema and registry."""
import math
from dataclasses import dataclass, field

REGISTRY = {}


@dataclass
class Param:
    default: object
    min: float | None = None
    max: float | None = None
    help: str = ""
    choices: list | None = None
    px: object = False   # length in full-resolution pixels: True, "odd", "octave" (wavelet layer) or "xy" (points)


@dataclass
class Context:
    """State shared along a pipeline run."""
    log: object = print
    header: object = None
    out_dir: object = None
    linear: bool = True
    tools: dict = field(default_factory=dict)    # paths of external programs
    scale: float = 1.0                           # image size / full size (GUI live preview < 1)
    layers: dict = field(default_factory=dict)   # e.g. "stars" saved by star removal


class Plugin:
    id = ""
    name = ""
    category = ""
    stage = "any"          # "linear", "nonlinear" or "any"
    description = ""
    params: dict = {}

    def run(self, img, ctx, **p):
        raise NotImplementedError

    def resolve(self, given):
        unknown = set(given) - set(self.params)
        if unknown:
            raise ValueError(f"{self.id}: unknown parameters {sorted(unknown)}")
        out = {k: v.default for k, v in self.params.items()}
        out.update(given)
        return out

    def scaled(self, params, scale):
        """Pixel-sized params adapted to an image `scale` times the full resolution (live preview),
        so the preview looks like the full-size result."""
        if scale == 1:
            return params
        out = dict(params)
        shift = round(math.log2(scale))
        for k, p in self.params.items():
            v = out.get(k)
            if not p.px or v is None or isinstance(v, bool):
                continue
            if p.px == "xy":                  # [[x, y], ...], or one flat [x, y]
                v = [[c * scale for c in pt] for pt in v] if v and isinstance(v[0], (list, tuple)) else [c * scale for c in v]
            elif p.px == "octave":            # distinct layers: two that collapse on the same scale would count twice
                v = sorted({max(1, int(x) + shift) for x in v}) if isinstance(v, list) else max(1, int(v) + shift)
            else:
                v = v * scale
                if p.px == "odd":
                    v = int(round(v)) // 2 * 2 + 1
                elif isinstance(p.default, int):
                    v = int(round(v))
                if p.min is not None:
                    v = max(type(v)(p.min), v)
            out[k] = v
        return out


def register(cls):
    if cls.id in REGISTRY:
        raise ValueError(f"duplicate plugin id {cls.id}")
    REGISTRY[cls.id] = cls()
    return cls


def get(plugin_id):
    try:
        return REGISTRY[plugin_id]
    except KeyError:
        raise KeyError(f"unknown plugin '{plugin_id}'. Available: {', '.join(sorted(REGISTRY))}") from None
