"""Mono camera with filters: align per-filter integrations and combine them into color.

Palettes: RGB, LRGB, HOO, SHO (Hubble), HaRGB, and mono when a single filter is present.
"""
import re

import numpy as np

from .. import io
from ..stats import luminance
from . import analyze, register, xisf

_ALIASES = {
    "L": ("l", "lum", "luminance", "clear", "c"),
    "R": ("r", "red"),
    "G": ("g", "green"),
    "B": ("b", "blue"),
    "Ha": ("ha", "h-alpha", "halpha", "h_alpha", "h", "hydrogen"),
    "OIII": ("oiii", "o3", "o-iii", "o", "oxygen"),
    "SII": ("sii", "s2", "s-ii", "s", "sulfur", "sulphur"),
}
PALETTES = {
    "RGB": ("R", "G", "B"), "LRGB": ("L", "R", "G", "B"), "HOO": ("Ha", "OIII"),
    "SHO": ("SII", "Ha", "OIII"), "HaRGB": ("Ha", "R", "G", "B"), "MONO": (),
}


def canonical(name):
    """Canonical filter name. Bandwidths and separators written by capture programs are ignored:
    'Ha 7nm', 'H-Alpha_7nm', 'OIII 6.5nm', 'SII-3nm (Antlia)' -> Ha, OIII, SII."""
    n = (name or "").strip().lower()
    for canon, names in _ALIASES.items():
        if n in names:
            return canon
    s = re.sub(r"\(.*?\)", " ", n)                        # brand in brackets
    s = re.sub(r"\d+(?:[.,]\d+)?\s*nm\b", " ", s)          # bandwidth
    s = re.sub(r"[\s_]+", " ", s).strip(" -")
    for cand in (s, s.replace(" ", ""), s.split(" ")[0] if s else ""):
        for canon, names in _ALIASES.items():
            if cand in names:
                return canon
    return name or "L"


def best_palette(filters):
    have = set(filters)
    for p in ("LRGB", "HaRGB", "SHO", "RGB", "HOO"):
        if set(PALETTES[p]) <= have:
            return p
    return "MONO"


def _read(path):
    return xisf.read(path)[0] if str(path).lower().endswith(".xisf") else io.load(path)[0]


def align(paths):
    """paths: {filter: file}. Returns {filter: (H, W) float32} aligned on the first filter."""
    items = list(paths.items())
    ref_name, ref_path = items[0]
    ref = _read(ref_path)[..., 0]
    st = analyze.detect(np.ascontiguousarray(ref))
    ref_xy = np.stack([st["x"], st["y"]], 1)
    out = {ref_name: ref}
    # area covered by every channel: each integration has its own empty borders
    covered = np.isfinite(ref) & (ref > 0)
    for name, p in items[1:]:
        img = _read(p)[..., 0]
        s2 = analyze.detect(np.ascontiguousarray(img))
        M, n, rms = register.match(np.stack([s2["x"], s2["y"]], 1), ref_xy)
        w = register.warp(img, M, ref.shape)
        w = w[..., 0] if w.ndim == 3 else w
        covered &= np.isfinite(w) & (w > 0)
        out[name] = np.nan_to_num(w)
    rows = np.where(covered.mean(1) > 0.98)[0]
    cols = np.where(covered.mean(0) > 0.98)[0]
    if len(rows) and len(cols):
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        out = {k: v[y0:y1, x0:x1] for k, v in out.items()}
    return out


def _match(ch, ref):
    """Linear fit of one channel's background and spread onto a reference channel."""
    m1, m2 = float(np.median(ch[::4, ::4])), float(np.median(ref[::4, ::4]))
    d1 = float(np.median(np.abs(ch[::4, ::4] - m1))) or 1e-9
    d2 = float(np.median(np.abs(ref[::4, ::4] - m2))) or 1e-9
    return np.clip((ch - m1) * (d2 / d1) + m2, 0, 1)


def combine(chans, palette="auto", ha_blend=0.7):
    """chans: {filter: (H, W)} aligned, linear. Returns float32 (H, W, 3) or (H, W, 1)."""
    if palette == "auto":
        palette = best_palette(chans)
    names = {p.upper(): p for p in list(PALETTES) + ["MONO", "HOO", "SHO", "HaRGB", "RGB", "LRGB"]}
    palette = names.get(str(palette).upper(), palette)       # "sho" from the command line is SHO
    c = dict(chans)
    if palette == "MONO":
        k = next(iter(c))
        return c[k][..., None].astype(np.float32), palette
    if palette == "HOO":
        ha, o = c["Ha"], _match(c["OIII"], c["Ha"])
        return np.stack([ha, o, o], -1).astype(np.float32), palette
    if palette == "SHO":
        ref = c["Ha"]
        return np.stack([_match(c["SII"], ref), ref, _match(c["OIII"], ref)], -1).astype(np.float32), palette
    rgb = np.stack([c["R"], _match(c["G"], c["R"]), _match(c["B"], c["R"])], -1)
    if palette == "HaRGB":
        ha = _match(c["Ha"], c["R"])
        rgb[..., 0] = np.maximum(rgb[..., 0], ha_blend * ha + (1 - ha_blend) * rgb[..., 0])
    if palette == "LRGB":
        lum = luminance(rgb)
        L = _match(c["L"], lum)
        rgb = rgb * (L / np.maximum(lum, 1e-6))[..., None]
    return np.clip(rgb, 0, 1).astype(np.float32), palette
