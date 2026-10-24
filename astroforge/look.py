"""The "look" of a finished image and how to move an image towards a target look.

measure() reads the few numbers that make two finished astro images look alike: sky level and tint,
nebula brightness, colour saturation of the nebula, highlight level. It works on a reduced, star-free copy
(median filter), so JPEG finals downloaded from the web and AstroForge results are measured the same way.
match() moves an image towards a target look (from a reference final image or a preset): sky to a neutral
grey level, nebula brightness with a midtones curve that keeps the sky and white fixed, saturation of the
nebula only (the sky is left grey), soft highlight roll-off.
"""
import numpy as np
from scipy import ndimage as ndi

from .stretch import mtf

_W = np.array([0.2126, 0.7152, 0.0722], np.float32)
HUES = ("red", "yellow", "green", "cyan", "blue", "magenta")
DEFAULT = {"sky": 0.07, "faint": 0.03, "nebula": 0.30, "saturation": 0.40, "white": 0.95}


def _small(img, side=1200):
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        a = a[..., None]
    step = max(1, int(np.ceil(max(a.shape[:2]) / side)))
    if step > 1:
        h, w = a.shape[0] // step * step, a.shape[1] // step * step
        a = a[:h, :w].reshape(h // step, step, w // step, step, a.shape[2]).mean((1, 3))
    if a.shape[2] == 1:
        a = np.repeat(a, 3, 2)
    a = np.clip(np.nan_to_num(a), 0, 1)
    # painted frames / empty stacking borders: drop outer rows and columns that are (almost) black
    L = a @ _W
    rows, cols = np.where(L.mean(1) > 0.01)[0], np.where(L.mean(0) > 0.01)[0]
    if len(rows) > L.shape[0] // 2 and len(cols) > L.shape[1] // 2:
        a = a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        m = 3                                    # anti-aliased frame edge
        a = a[m:-m, m:-m] if min(a.shape[:2]) > 8 * m else a
    return a


def _lum(a):
    return a @ _W


def _regions(a):
    """Star-free copy, its smoothed luminance and the sky / nebula pixel masks."""
    s = ndi.median_filter(a, size=(5, 5, 1))
    ls = ndi.uniform_filter(_lum(s), 9)
    sky = ls <= np.percentile(ls, 15)
    # nebula = clearly above the sky (a quarter of the range up to the bright end): does not depend on how
    # much of the frame the object fills, unlike a fixed percentile
    s0, top = float(np.median(ls[sky])), float(np.percentile(ls, 99.5))
    neb = ls >= s0 + 0.25 * (top - s0)
    if neb.mean() < 0.01:
        neb = ls >= np.percentile(ls, 99)
    return s, ls, sky, neb


def _sat(a):
    mx, mn = a.max(-1), a.min(-1)
    return np.where(mx > 0.02, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def measure(img):
    """Look numbers of an image (H, W, C) in display space, values 0..1."""
    a = _small(img)
    s, ls, sky, neb = _regions(a)
    L = _lum(a)
    nb = s[neb]
    h = np.degrees(np.arctan2(np.sqrt(3) * (nb[:, 1] - nb[:, 2]), 2 * nb[:, 0] - nb[:, 1] - nb[:, 2])) % 360
    hist, _ = np.histogram(h, bins=6, range=(-30, 330), weights=_sat(nb))
    hist = hist / max(hist.sum(), 1e-9)
    return {
        "sky": round(float(np.median(ls[sky])), 4),
        "sky_rgb": [round(float(v), 4) for v in np.median(s[sky], 0)],
        # faint = what is neither sky nor nebula (dust, outer glow, Milky Way): its median above the sky
        "faint": round(float(np.median(ls[~neb & ~sky]) - np.median(ls[sky])) if (~neb & ~sky).any() else 0.0, 4),
        "nebula": round(float(np.median(ls[neb])), 4),
        "nebula_area": round(float(neb.mean()), 3),
        "saturation": round(float(_sat(nb).mean()), 4),
        "white": round(float(np.percentile(L, 99.9)), 4),
        "hues": {k: round(float(v), 3) for k, v in zip(HUES, hist)},
    }


def target_from(stats):
    """match() parameters reproducing measured look numbers."""
    return {k: round(float(stats[k]), 3) for k in DEFAULT}


def distance(a, b):
    """How far two looks are (0 = same); each term is about 1 for a clearly visible difference."""
    d = {"sky": abs(a["sky"] - b["sky"]) / 0.03,
         "nebula": abs(a["nebula"] - b["nebula"]) / 0.08,
         "saturation": abs(a["saturation"] - b["saturation"]) / 0.10,
         "tint": float(np.ptp(np.subtract(a["sky_rgb"], a["sky"])) - np.ptp(np.subtract(b["sky_rgb"], b["sky"]))) / 0.02}
    d["tint"] = abs(d["tint"])
    d["total"] = float(np.sqrt(np.mean([v * v for v in d.values()])))
    return {k: round(v, 2) for k, v in d.items()}


def _curve(x, t, a, b):
    """Tone curve with fixed points t -> t and 1 -> 1 moving a -> b (MTF on the range above t)."""
    if a <= t + 1e-4 or b <= t + 1e-4:
        return x
    u, v = (a - t) / (1 - t), (b - t) / (1 - t)
    m = float(mtf(v, u))                       # mtf(m, u) = v
    above = t + (1 - t) * mtf(m, (x - t) / (1 - t))
    return np.where(x > t, above, x).astype(np.float32)


def _curve_pts(x, pts):
    """Monotone (PCHIP) tone curve through (0,0), pts..., (1,1); as a 4096-entry LUT."""
    from scipy.interpolate import PchipInterpolator
    xs, ys = [0.0], [0.0]
    for a, b in sorted(pts) + [(1.0, 1.0)]:
        if a > xs[-1] + 1e-3 and b > ys[-1]:
            xs.append(float(a)); ys.append(float(b))
    if xs[-1] < 1.0:
        xs.append(1.0); ys.append(1.0)
    g = np.linspace(0, 1, 4096)
    lut = np.clip(PchipInterpolator(xs, ys)(g), 0, 1).astype(np.float32)
    return np.interp(x, g, lut).astype(np.float32)


def match(img, sky=0.07, nebula=0.30, saturation=0.40, white=0.95, strength=1.0, neutral_sky=True, log=None,
          faint=None):
    """Move img (display space) towards the target look. Each adjustment is limited so that a strange
    measurement (frame full of nebula, almost no sky) cannot wreck the image."""
    out = np.clip(np.asarray(img, np.float32), 0, 1)
    mono = out.shape[-1] == 1
    sky0 = measure(out)["sky"]
    sky = float(np.clip(sky, sky0 - 0.06, sky0 + 0.06))     # never more than this from where it started
    for it in range(2):                         # second pass corrects what the first one moved
        m = measure(out)
        k = strength
        # 1. sky level and colour: black point per channel, white kept
        t = m["sky"] + k * (sky - m["sky"])
        src = m["sky_rgb"] if (neutral_sky and not mono) else [m["sky"]] * 3
        for c in range(out.shape[-1]):
            s = float(src[c] if not mono else m["sky"])
            if s < 0.98:
                out[..., c] = t + (out[..., c] - s) * (1 - t) / (1 - s)
        out = np.clip(out, 0, 1)
        # 2. nebula brightness (sky and white fixed)
        neb_now = (m["nebula"] - m["sky"]) * (1 - t) / max(1 - m["sky"], 1e-6) + t
        goal = float(np.clip(nebula, t + 0.02, 0.9))
        goal = neb_now + k * (goal - neb_now)
        goal = float(np.clip(goal, t + 0.5 * (neb_now - t), t + 2.0 * (neb_now - t)))
        if faint is None:
            out = _curve(out, t, neb_now, goal)
        else:
            # faint background (dust, glow) brought to its own level, between sky and nebula
            f_now = t + m["faint"] * (1 - t) / max(1 - m["sky"], 1e-6)
            f_goal = f_now + k * (t + faint - f_now)
            f_goal = float(np.clip(f_goal, t + 0.4 * (f_now - t), t + 1.5 * (f_now - t)))
            if t + 0.005 < f_now < neb_now - 0.02:
                f_goal = min(f_goal, goal - 0.02)
                out = _curve_pts(out, [(t, t), (f_now, f_goal), (neb_now, goal)])
            else:
                out = _curve(out, t, neb_now, goal)
        # 3. saturation of the nebula, sky left alone
        if not mono:
            m2 = measure(out)
            gain = float(np.clip(1 + k * (saturation / max(m2["saturation"], 1e-3) - 1), 0.6, 2.0))
            L = _lum(out)[..., None]
            ls = ndi.uniform_filter(ndi.median_filter(L[..., 0], 3), 9)
            w = np.clip((ls - m2["sky"]) / max(m2["nebula"] - m2["sky"], 1e-3), 0, 1)[..., None]
            out = np.clip(L + (out - L) * (1 + (gain - 1) * w), 0, 1).astype(np.float32)
        if log:
            log(f"look.match pass {it + 1}: sky {m['sky']:.3f}->{t:.3f}, faint {m['faint']:.3f}, "
                f"nebula {neb_now:.3f}->{goal:.3f}")
    # 4. highlights: soft roll-off above the knee so cores keep detail
    if white < 0.999 and float(np.percentile(_lum(_small(out)), 99.9)) > white:
        knee = white - 0.15
        x = out
        y = knee + (white - knee) * (1 - np.exp(-(x - knee) / max(white - knee, 1e-3)))
        out = np.where(x > knee, np.minimum(y, x), x).astype(np.float32)
    return out
