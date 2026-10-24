"""Satellite / airplane trail detection in a single registered frame against a reference (integration).

trail_mask() works on the difference frame - reference in noise units: stars of the reference are masked,
the residual is matched-filtered with short oriented line kernels (a 2 sigma per pixel trail becomes a
>10 sigma ridge), lines are found with HoughLinesP and each one is verified along its whole length
(median of the cross-section must be positive, total significance high) before being rasterised.
Typical use: pixels of the mask set to NaN in that frame before integrating (works with few frames,
where sigma clipping cannot reject anything).
"""
import warnings

import cv2
import numpy as np


def _blocks(a, size, fn):
    """fn statistic of size x size blocks, interpolated back to a.shape."""
    h, w = a.shape
    by, bx = max(1, h // size), max(1, w // size)
    t = a[:by * size, :bx * size].reshape(by, size, bx, size).transpose(0, 2, 1, 3).reshape(by, bx, -1)
    v = fn(t).astype(np.float32)
    return cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)


def _significance(frame, reference, noise):
    f = np.asarray(frame, np.float32)
    r = np.asarray(reference, np.float32)
    if f.ndim == 2:
        f, r = f[..., None], r[..., None]
    nz = np.broadcast_to(np.asarray(noise, np.float32).reshape(-1), (f.shape[-1],))
    nz = np.maximum(nz, 1e-12)
    sq = np.sqrt(f.shape[-1])
    d = ((f - r) / nz).mean(-1) * sq
    ref = (r / nz).mean(-1) * sq
    bad = ~np.isfinite(d) | ~np.isfinite(ref)
    return np.where(bad, 0, d).astype(np.float32), np.where(bad, 0, ref).astype(np.float32), bad


def _star_mask(ref, max_star_px):
    """Compact sources of the reference (stars, small galaxies): misregistration leaves residuals there."""
    bg = _blocks(ref, 32, lambda t: np.median(t, -1))
    hp = ref - bg
    sd = _blocks(hp, 32, lambda t: 1.4826 * np.median(np.abs(t - np.median(t, -1, keepdims=True)), -1))
    m = (hp > 3 * np.maximum(sd, 1e-6)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = np.zeros(nl, np.uint8)
    small[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) <= max_star_px
    # elongated structures (a trail left in a mean reference, nebula filaments) are not stars
    m = small[lab]
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(m, grow).astype(bool)


def _line_kernels(length, n_angles):
    ks = []
    c = length // 2
    for a in np.arange(n_angles) * np.pi / n_angles:
        big = np.zeros((length * 4, length * 4), np.float32)
        dx, dy = np.cos(a) * c * 4, np.sin(a) * c * 4
        o = length * 2
        cv2.line(big, (int(o - dx), int(o - dy)), (int(o + dx), int(o + dy)), 1.0, 4)
        k = cv2.resize(big, (length, length), interpolation=cv2.INTER_AREA)
        k /= np.sqrt((k * k).sum())         # unit noise gain: response in sigma of white noise
        ks.append(k)
    return ks


def _sample(d, p0, p1, offsets, step=1.0):
    """d sampled along p0 -> p1 at perpendicular offsets: (len(offsets), n_points)."""
    p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
    v = p1 - p0
    ln = float(np.hypot(*v))
    u = v / max(ln, 1e-6)
    nrm = np.array([-u[1], u[0]], np.float32)
    t = np.arange(0, ln, step, dtype=np.float32)
    pts = p0[None] + t[:, None] * u[None]
    out = []
    for o in offsets:
        q = (pts + np.float32(o) * nrm).astype(np.float32)
        out.append(cv2.remap(d, q[:, 0].reshape(1, -1), q[:, 1].reshape(1, -1), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)[0])
    return np.array(out)


def _nzmean(a, axis):
    """Mean ignoring NaN (outside the image) and exact zeros (masked stars)."""
    ok = np.isfinite(a) & (a != 0)
    n = ok.sum(axis)
    return np.where(n > 0, np.where(ok, a, 0).sum(axis) / np.maximum(n, 1), np.nan)


def _nzmedian(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.where(np.isfinite(a) & (a != 0), a, np.nan), axis)


def _verify(d, dw, p0, p1, b, k, min_len):
    """Line refined on the ridge with its statistics and width, or None when it is not a real trail.

    d: residual clipped at +-5 sigma (statistics), dw: unclipped (ridge position and width, by medians).
    """
    p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
    v = p1 - p0
    ln = float(np.hypot(*v))
    u = v / max(ln, 1e-6)
    nrm = np.array([-u[1], u[0]], np.float32)
    # Hough on binned data is +-b px off: best perpendicular offset on the first and last third
    offs = np.arange(-2 * b - 2, 2 * b + 2.5, 0.5)
    ends = []
    for t0, t1 in ((0, ln / 3), (2 * ln / 3, ln)):
        m = _nzmedian(_sample(dw, p0 + u * t0, p0 + u * t1, offs), 1)
        if not np.isfinite(m).any():
            return None
        ends.append(offs[np.nanargmax(m)])
    oa, ob = ends
    q0 = p0 + nrm * (oa - (ob - oa) * 0.25)         # offsets measured at 1/6 and 5/6 of the length
    q1 = p1 + nrm * (ob + (ob - oa) * 0.25)
    cs = _nzmean(_sample(d, q0, q1, np.array([-1.0, 0.0, 1.0])), 0)
    cs = cs[np.isfinite(cs)]                         # masked stars and borders are skipped
    if cs.size < 0.5 * min_len:
        return None
    sig = float(cs.mean() * np.sqrt(cs.size * 3 / 1.5))       # +-1 px samples are correlated
    med = float(np.median(cs))
    if med < 0.5 or sig < max(10.0, 3 * k):
        return None
    # width from the averaged cross profile: out to where the wings drop below 0.5 sigma per pixel
    wo = np.arange(-25, 26, 1.0)
    wp = _nzmedian(_sample(dw, q0, q1, wo), 1)
    on = np.nan_to_num(wp) > max(0.5, 4.0 / np.sqrt(cs.size))
    i = j = int(np.nanargmax(wp)) if np.isfinite(wp).any() else 25
    while i > 0 and on[i - 1]:
        i -= 1
    while j < len(wo) - 1 and on[j + 1]:
        j += 1
    q0, q1 = q0 + nrm * (wo[i] + wo[j]) / 2, q1 + nrm * (wo[i] + wo[j]) / 2    # centre on the whole profile
    wid = int(np.clip(wo[j] - wo[i] + 5, 5, 51))
    return dict(p0=q0.tolist(), p1=q1.tolist(), significance=sig, median=med, width=wid, n=int(cs.size))


def _merge(lines):
    """Join collinear pieces of the same trail (split by masked stars, or parallel edges of a wide one)."""
    out = []
    for ln in sorted(lines, key=lambda l: -l["n"]):
        p0, p1 = np.array(ln["p0"]), np.array(ln["p1"])
        u = (p1 - p0) / max(np.hypot(*(p1 - p0)), 1e-6)
        for o in out:
            q0, q1 = np.array(o["p0"]), np.array(o["p1"])
            uo = (q1 - q0) / max(np.hypot(*(q1 - q0)), 1e-6)
            nrm = np.array([-uo[1], uo[0]])
            if abs(np.dot(u, uo)) > np.cos(np.radians(1.5)) and \
                    max(abs(np.dot(p0 - q0, nrm)), abs(np.dot(p1 - q0, nrm))) < max(4, o["width"] / 2):
                t = [np.dot(p - q0, uo) for p in (p0, p1, q0, q1)]
                o["p0"], o["p1"] = (q0 + uo * min(t)).tolist(), (q0 + uo * max(t)).tolist()
                o["significance"] = float(np.hypot(o["significance"], ln["significance"]))
                o["width"] = max(o["width"], ln["width"])
                o["n"] += ln["n"]
                break
        else:
            out.append(dict(ln))
    return out


def trail_mask(frame, reference, noise, k=3.0, min_length_px=None, width_px=None, return_lines=False):
    """Bool (H, W) mask of satellite / airplane trails in frame (H, W) or (H, W, C), registered on reference.

    noise: per-pixel noise sigma of one frame (scalar or per channel). k: threshold in sigma of the
    matched-filtered residual. min_length_px: shortest trail (default 8% of the short side, >= 64).
    width_px: rasterised width (default measured on each trail's profile, plus margin).
    return_lines: also return [dict(p0, p1, width, significance, median)].
    """
    d, ref, bad = _significance(frame, reference, noise)
    h, w = d.shape
    min_len = int(min_length_px or max(64, 0.08 * min(h, w)))
    # local background (gradients, clouds) and noise (nebula shot noise, reference noise) per block
    d -= _blocks(d, 64, lambda t: np.median(t, -1))
    loc = _blocks(d, 64, lambda t: 1.4826 * np.median(np.abs(t - np.median(t, -1, keepdims=True)), -1))
    d /= np.maximum(loc, 1e-3 * max(float(np.median(loc)), 1e-6))
    d[_star_mask(ref, 40) | bad] = 0
    dw = d.copy()
    d = np.clip(d, -5, 5)           # hot pixels / cosmic rays must not light up the line filters
    # bin (the mean keeps the line SNR for trails narrower than b), matched filter with oriented lines
    b = int(np.clip(min(h, w) // 512, 1, 4))
    db = cv2.resize(d, (w // b, h // b), interpolation=cv2.INTER_AREA) * b if b > 1 else d
    L = int(max(9, min(41, min_len // (3 * b)))) | 1
    resp = np.full(db.shape, -np.inf, np.float32)
    for kk in _line_kernels(L, int(np.clip(np.pi * L / 2, 12, 64))):
        np.maximum(resp, cv2.filter2D(db, -1, kk, borderType=cv2.BORDER_CONSTANT), out=resp)
    binary = (resp > k).astype(np.uint8) * 255
    minb = max(8, min_len // b)
    segs = cv2.HoughLinesP(binary, 1, np.pi / 720, threshold=max(10, minb // 2), minLineLength=minb,
                           maxLineGap=max(4, minb // 6))
    lines = []
    if segs is not None:
        for x0, y0, x1, y1 in (segs.reshape(-1, 4).astype(np.float32) * b + (b - 1) / 2).tolist()[:200]:
            r = _verify(d, dw, (x0, y0), (x1, y1), b, k, min_len)
            if r is not None:
                lines.append(r)
    lines = _merge(lines)
    mask = np.zeros((h, w), np.uint8)
    for ln in lines:
        p0, p1 = np.array(ln["p0"], np.float32), np.array(ln["p1"], np.float32)
        u = (p1 - p0) / max(float(np.hypot(*(p1 - p0))), 1e-6)
        if width_px:
            ln["width"] = int(width_px)
        e = u * ln["width"]
        cv2.line(mask, tuple(int(round(c)) for c in p0 - e), tuple(int(round(c)) for c in p1 + e), 1, ln["width"])
        del ln["n"]
    out = mask.astype(bool)
    return (out, lines) if return_lines else out
