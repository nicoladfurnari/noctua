"""Frame weighting and approval with PixInsight SubframeSelector-style expressions.

The variables are the per-frame measurements (FWHM, Eccentricity, SNRWeight, PSFFlux, PSFSignal, Stars,
Noise) and their Min / Max over the session (FWHMMin, FWHMMax, ...). The presets are the formulas of the
GAEEB PixInsight workflow (Papotti / Airola): "standard", "nebula", "galaxy", "cluster"; a custom
expression can be given instead. Results are in the same units as in PixInsight (e.g. 50..100), so an
approval threshold such as 70 ("Weight > 70") means the same thing.
"""
import ast
import operator

FORMULAS = {
    "standard": "(20*(1-(FWHM-FWHMMin)/(FWHMMax-FWHMMin)))+(20*(1-(Eccentricity-EccentricityMin)/(EccentricityMax-EccentricityMin)))"
                "+(20*(PSFFlux-PSFFluxMin)/(PSFFluxMax-PSFFluxMin))+50",
    "nebula": "(10*(1-(FWHM-FWHMMin)/(FWHMMax-FWHMMin)))+(10*(1-(Eccentricity-EccentricityMin)/(EccentricityMax-EccentricityMin)))"
              "+(30*(SNRWeight-SNRWeightMin)/(SNRWeightMax-SNRWeightMin))+50",
    "galaxy": "(15*(1-(FWHM-FWHMMin)/(FWHMMax-FWHMMin)))+(15*(1-(Eccentricity-EccentricityMin)/(EccentricityMax-EccentricityMin)))"
              "+(20*(SNRWeight-SNRWeightMin)/(SNRWeightMax-SNRWeightMin))+50",
    "cluster": "(20*(1-(FWHM-FWHMMin)/(FWHMMax-FWHMMin)))+(20*(1-(Eccentricity-EccentricityMin)/(EccentricityMax-EccentricityMin)))"
               "+(10*(SNRWeight-SNRWeightMin)/(SNRWeightMax-SNRWeightMin))+50",
}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"min": min, "max": max, "abs": abs, "sqrt": lambda x: x ** 0.5}


def evaluate(expr, names):
    """Arithmetic only (+ - * / ** and min/max/abs/sqrt): no attribute access, no other calls."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in names:
                raise ValueError(f"unknown variable {n.id}")
            return names[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Div) and b == 0:
                return 0.0          # all frames equal in this measurement: the term does not discriminate
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"not allowed in a weighting expression: {ast.dump(n)[:60]}")
    return float(ev(ast.parse(expr, mode="eval")))


def measurements(q):
    return {"FWHM": q.fwhm, "Eccentricity": q.eccentricity, "SNRWeight": q.snr,
            "PSFFlux": q.psf_signal * q.noise, "PSFSignal": q.psf_signal, "Stars": q.stars, "Noise": q.noise}


def apply(quals, formula, approval=None):
    """Weight every frame with `formula` (preset name or expression), in place. Frames below `approval`
    are rejected. Returns the list of formula values (None for frames without stars)."""
    expr = FORMULAS.get(formula, formula)
    ok = [q for q in quals if q.stars > 0]
    if not ok:
        return [None] * len(quals)
    per = [measurements(q) for q in ok]
    names_all = {}
    for k in per[0]:
        vals = [m[k] for m in per]
        names_all[k + "Min"], names_all[k + "Max"] = min(vals), max(vals)
    values = []
    for q in quals:
        if q.stars == 0:
            values.append(None)
            continue
        v = evaluate(expr, {**names_all, **measurements(q)})
        values.append(v)
        if approval is not None and v < approval and q.accepted:
            q.accepted, q.reason = False, f"weight {v:.1f} < {approval:g}"
    vmax = max(v for v in values if v is not None) or 1.0
    for q, v in zip(quals, values):
        if v is not None:
            q.weight = max(v, 0.0) / vmax
    return values
