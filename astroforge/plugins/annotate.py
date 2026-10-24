"""Object annotation (like PixInsight AnnotateImage): catalogue objects, equatorial grid, compass, scale bar.

The annotation is an output: it is drawn on a copy and saved as <name>_annotated.jpg; the processed image
passes through unchanged unless apply=True.
"""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import astrometry, catalogs, io
from .base import Param, Plugin, register

SHIFT = 4                     # cv2 sub-pixel bits
FONTS = ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")


def _rgb(hexstr, default=(255, 255, 255)):
    s = str(hexstr).lstrip("#")
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return default


def _font(size):
    for name in FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _fx(v):
    return int(round(v * (1 << SHIFT)))


def _nice(value, steps):
    """Largest step <= value (or the smallest step)."""
    ok = [s for s in steps if s <= value]
    return ok[-1] if ok else steps[0]


class _Labels:
    """Greedy label placement that avoids overlaps when possible."""

    def __init__(self, draw, w, h, font, stroke):
        self.draw, self.w, self.h, self.font, self.stroke = draw, w, h, font, stroke
        self.boxes = []

    def size(self, text):
        x0, y0, x1, y1 = self.draw.textbbox((0, 0), text, font=self.font, stroke_width=self.stroke)
        return x1 - x0, y1 - y0, x0, y0

    def place(self, text, x, y, r, color, force=False):
        tw, th, ox, oy = self.size(text)
        d = r * 0.72 + 3
        cands = [(x + d, y - d - th), (x + d, y + d), (x - d - tw, y - d - th), (x - d - tw, y + d),
                 (x - tw / 2, y - r - th - 2), (x - tw / 2, y + r + 2)]
        best = None
        for cx, cy in cands:
            cx, cy = min(max(cx, 2), self.w - tw - 2), min(max(cy, 2), self.h - th - 2)
            box = (cx, cy, cx + tw, cy + th)
            if not _overlap(box, self.boxes):
                best = box
                break
        if best is None:
            if not force:
                return False
            cx, cy = cands[0]
            cx, cy = min(max(cx, 2), self.w - tw - 2), min(max(cy, 2), self.h - th - 2)
            best = (cx, cy, cx + tw, cy + th)
        self.boxes.append(best)
        self.draw.text((best[0] - ox, best[1] - oy), text, font=self.font, fill=color,
                       stroke_width=self.stroke, stroke_fill=(0, 0, 0))
        return True


def _label_text(o, mode):
    if o["kind"] == "star" or mode == "designation":
        return o["names"][0] if mode != "full" or o["kind"] != "star" else o["label"]
    if mode == "common":
        return o["common"][0] if o["common"] else o["names"][0]
    name = o["names"][0]
    alt = [n for n in o["names"][1:] if not n.startswith("Sh2")][:1]
    text = name + (f" ({alt[0]})" if alt and o["messier"] else "")
    return text + (f"  {o['common'][0]}" if o["common"] else "")


def _grid_lines(wcs, w, h):
    """Polylines (list of (N,2) pixel arrays) and labels for an RA/Dec grid covering the image."""
    t = np.linspace(0, 1, 60)
    edge = np.concatenate([np.c_[t * (w - 1), 0 * t], np.c_[t * (w - 1), 0 * t + h - 1],
                           np.c_[0 * t, t * (h - 1)], np.c_[0 * t + w - 1, t * (h - 1)],
                           [[(w - 1) / 2, (h - 1) / 2]]])
    rd = wcs.all_pix2world(edge, 0)
    ra_c, dec_c = rd[-1]
    pole = None
    for pd in (90.0, -90.0):
        px, py = catalogs._world2pix(wcs, np.array([ra_c]), np.array([pd]))
        if np.isfinite(px[0]) and 0 <= px[0] < w and 0 <= py[0] < h:
            pole = pd
    dra = (rd[:, 0] - ra_c + 180) % 360 - 180
    dec_lo, dec_hi = float(np.nanmin(rd[:, 1])), float(np.nanmax(rd[:, 1]))
    if pole is not None:
        ra_lo, ra_hi = -180.0, 180.0
        dec_lo, dec_hi = (dec_lo, 90.0) if pole > 0 else (-90.0, dec_hi)
    else:
        ra_lo, ra_hi = float(np.nanmin(dra)), float(np.nanmax(dra))
    span_dec = max(dec_hi - dec_lo, 1e-4)
    dec_steps = [s / 3600 for s in (10, 20, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 18000, 36000)]
    ddec = _nice(span_dec / 4, dec_steps)
    span_ra = max(ra_hi - ra_lo, 1e-4)
    ra_steps = [s * 15 / 3600 for s in (1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600,
                                        7200, 10800)]
    dra_step = _nice(span_ra / 4, ra_steps)
    lines = []
    n = 200
    for dec in np.arange(math.ceil(dec_lo / ddec) * ddec, dec_hi + 1e-9, ddec):
        if abs(dec) >= 90:
            continue
        ras = ra_c + np.linspace(ra_lo, ra_hi, n)
        x, y = catalogs._world2pix(wcs, ras % 360, np.full(n, dec))
        lines.append(("dec", float(dec), np.c_[x, y]))
    for r in np.arange(math.ceil((ra_c + ra_lo) / dra_step) * dra_step, ra_c + ra_hi + 1e-9, dra_step):
        decs = np.linspace(max(dec_lo, -89.9), min(dec_hi, 89.9), n)
        x, y = catalogs._world2pix(wcs, np.full(n, r % 360), decs)
        lines.append(("ra", float(r % 360), np.c_[x, y]))
    return lines, ddec, dra_step


def _fmt_dec(d, step):
    s = "-" if d < 0 else "+"
    a = abs(d) + 1e-9
    deg, mins = int(a), int((a - int(a)) * 60)
    secs = (a * 3600) % 60
    if step >= 1:
        return f"{s}{deg}°"
    if step >= 1 / 60:
        return f"{s}{deg}°{mins:02d}'"
    return f"{s}{deg}°{mins:02d}'{int(secs):02d}\""


def _fmt_ra(r, step):
    hrs = (r / 15 + 1e-9) % 24
    hh, mm = int(hrs), int((hrs - int(hrs)) * 60)
    ss = (hrs * 3600) % 60
    if step >= 15:
        return f"{hh}h"
    if step >= 0.25:
        return f"{hh}h{mm:02d}m"
    return f"{hh}h{mm:02d}m{int(ss):02d}s"


def _fmt_angle(arcsec):
    if arcsec >= 3600:
        return f"{arcsec / 3600:g}°"
    if arcsec >= 60:
        return f"{arcsec / 60:g}'"
    return f"{arcsec:g}\""


def _overlap(box, boxes):
    return any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in boxes)


def _split_visible(pts, w, h):
    """Split a polyline where it becomes undefined or runs far outside the image."""
    segs, seg = [], []
    for p in pts:
        if np.isfinite(p).all() and -w < p[0] < 2 * w and -h < p[1] < 2 * h:
            seg.append(p)
            continue
        if len(seg) > 1:
            segs.append(np.array(seg))
        seg = []
    if len(seg) > 1:
        segs.append(np.array(seg))
    return segs


def render(img, wcs, objects, wcs_shape=None, labels="full", font_size=0, line_width=0, opacity=0.9,
           colors=None, grid=False, compass=True, scale_bar=True):
    """Annotated uint8 RGB copy of img (float (H,W,C) in 0..1). Lines are drawn anti-aliased with OpenCV,
    text with a TrueType font through Pillow; everything is sized from the image dimensions."""
    h, w = img.shape[:2]
    colors = colors or {}
    base = np.clip(img, 0, 1)
    if base.shape[-1] == 1:
        base = np.repeat(base, 3, axis=-1)
    base8 = (base * 255 + 0.5).astype(np.uint8)
    canvas = base8.copy()
    u = min(h, w) / 1000.0
    lw = int(line_width) if line_width else max(1, int(round(1.6 * u)))
    fs = int(font_size) if font_size else max(10, int(round(min(h, w) / 55)))
    col = {k: _rgb(colors.get(k, d)) for k, d in
           (("galaxy", "#ffd25a"), ("nebula", "#ff7a7a"), ("cluster", "#7ad8ff"), ("dark", "#c8a2ff"),
            ("star", "#ffffff"), ("other", "#a0ffa0"), ("grid", "#6f8fbf"), ("compass", "#ffffff"))}
    wh, ww = (wcs_shape or (h, w))[:2]
    sx, sy = w / ww, h / wh            # WCS image -> this image
    texts = []                         # centred texts placed after the line work
    reserved = []                      # boxes object labels must avoid

    # ---- RA/Dec grid
    grid_segs = []
    ddec = dra = None
    if grid:
        lines, ddec, dra = _grid_lines(wcs, ww, wh)
        for kind, val, pts in lines:
            for seg in _split_visible(pts * [sx, sy], w, h):
                grid_segs.append((kind, val, seg))
                cv2.polylines(canvas, [np.round(seg * (1 << SHIFT)).astype(np.int32)], False, col["grid"],
                              max(1, lw - 1), cv2.LINE_AA, SHIFT)

    # ---- objects
    point_r = 12 * u + 4
    star_r = 9 * u + 3
    for o in objects:
        c = col.get(o["kind"], col["other"])
        x, y = o["x"], o["y"]
        if o["kind"] == "star":
            cv2.circle(canvas, (_fx(x), _fx(y)), _fx(star_r), c, lw, cv2.LINE_AA, SHIFT)
        elif o["a"] >= 4 * u + 2:
            cv2.ellipse(canvas, (_fx(x), _fx(y)), (_fx(o["a"]), _fx(max(o["b"], 1))), o["angle"], 0, 360, c, lw,
                        cv2.LINE_AA, SHIFT)
        else:   # point-like: four ticks around the position
            r0 = 5 * u + 2
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(canvas, (_fx(x + dx * r0), _fx(y + dy * r0)), (_fx(x + dx * point_r), _fx(y + dy * point_r)),
                         c, lw, cv2.LINE_AA, SHIFT)

    # ---- compass (top right)
    if compass:
        L = 3 * fs
        cx, cy = w - L - 2.5 * fs, L + 2.5 * fs
        ra0, dec0 = wcs.all_pix2world([[cx / sx, cy / sy]], 0)[0]
        jac = catalogs.local_jacobian(wcs, ra0, dec0)[0] * np.array([[sx], [sy]])
        for k, letter in ((1, "N"), (0, "E")):
            v = jac[:, k] / max(np.hypot(*jac[:, k]), 1e-9)
            cv2.arrowedLine(canvas, (_fx(cx), _fx(cy)), (_fx(cx + v[0] * L), _fx(cy + v[1] * L)), col["compass"],
                            lw + 1, cv2.LINE_AA, SHIFT, 0.22)
            texts.append((letter, cx + v[0] * (L + 0.75 * fs), cy + v[1] * (L + 0.75 * fs)))
        reserved.append((cx - L - 1.5 * fs, cy - L - 1.5 * fs, cx + L + 1.5 * fs, cy + L + 1.5 * fs))

    # ---- scale bar (bottom left)
    if scale_bar:
        scale = astrometry.pixel_scale_arcsec(wcs) / math.sqrt(sx * sy)   # arcsec per output pixel
        steps = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 18000, 36000]
        length = _nice(w / 6 * scale, steps)
        px = length / scale
        x0, y0 = 2 * fs, h - 2 * fs
        cv2.line(canvas, (_fx(x0), _fx(y0)), (_fx(x0 + px), _fx(y0)), col["compass"], lw + 1, cv2.LINE_AA, SHIFT)
        for xe in (x0, x0 + px):
            cv2.line(canvas, (_fx(xe), _fx(y0 - 0.35 * fs)), (_fx(xe), _fx(y0 + 0.35 * fs)), col["compass"],
                     lw + 1, cv2.LINE_AA, SHIFT)
        texts.append((_fmt_angle(length), x0 + px / 2, y0 - 1.1 * fs))
        reserved.append((x0 - fs, y0 - 2 * fs, x0 + px + fs, y0 + fs))

    # ---- text
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font, small = _font(fs), _font(max(9, int(fs * 0.8)))
    stroke = max(1, fs // 12)
    lab = _Labels(draw, w, h, font, stroke)
    for text, x, y in texts:
        tw, th, ox, oy = lab.size(text)
        draw.text((x - tw / 2 - ox, y - th / 2 - oy), text, font=font, fill=col["compass"], stroke_width=stroke,
                  stroke_fill=(0, 0, 0))
    lab.boxes.extend(reserved)

    if labels != "none":
        # Messier first, then stars and larger objects: they get the best label positions
        order = sorted(objects, key=lambda o: (-(o["messier"] > 0), -(o["kind"] == "star"), -o["a"]))
        for o in order:
            xs, ys = min(max(o["x"], 0), w - 1), min(max(o["y"], 0), h - 1)
            if o["kind"] == "star":
                r = star_r
            elif o["a"] >= 4 * u + 2:
                r = min(o["a"], 0.25 * min(w, h))
            else:
                r = point_r
            lab.place(_label_text(o, labels), xs, ys, r, col.get(o["kind"], col["other"]),
                      force=o["messier"] > 0 or o["kind"] == "star" or o["a"] > 20 * u)

    if grid:
        glab = _Labels(draw, w, h, small, stroke)
        glab.boxes = lab.boxes
        done = set()
        for kind, val, seg in grid_segs:
            inside = seg[(seg[:, 0] >= 0) & (seg[:, 0] < w) & (seg[:, 1] >= 0) & (seg[:, 1] < h)]
            key = (kind, round(val, 6))
            if not len(inside) or key in done:
                continue
            done.add(key)
            p = inside[np.argmin(inside[:, 0])] if kind == "dec" else inside[np.argmax(inside[:, 1])]
            text = _fmt_dec(val, ddec) if kind == "dec" else _fmt_ra(val, dra)
            tw, th, ox, oy = glab.size(text)
            tx, ty = min(max(p[0] + 4, 2), w - tw - 2), min(max(p[1] - th - 4, 2), h - th - 2)
            box = (tx, ty, tx + tw, ty + th)
            if _overlap(box, glab.boxes):
                continue
            glab.boxes.append(box)
            draw.text((tx - ox, ty - oy), text, font=small, fill=col["grid"], stroke_width=stroke,
                      stroke_fill=(0, 0, 0))

    out = np.asarray(pil, np.float32)
    a = float(np.clip(opacity, 0, 1))
    return np.clip(base8 + a * (out - base8) + 0.5, 0, 255).astype(np.uint8)


@register
class AnnotateObjects(Plugin):
    id = "annotate.objects"
    name = "Annotate objects (catalogue overlay)"
    category = "export"
    stage = "any"
    description = ("Marks Messier / NGC / IC / Caldwell / Sharpless objects and bright stars from the offline "
                   "OpenNGC catalogue, with optional RA/Dec grid, compass and scale bar. Needs a plate solve "
                   "(SPCC stores it; otherwise ASTAP is run here). Saves <name>_annotated.jpg; the image "
                   "itself is unchanged unless 'apply' is on.")
    params = {
        "messier": Param(True, help="Messier objects"),
        "ngc": Param(True, help="NGC objects"),
        "ic": Param(True, help="IC objects"),
        "other": Param(True, help="Barnard, Melotte, Caldwell-only and other addendum objects"),
        "sharpless": Param(False, help="Sharpless HII regions (coarse positions)"),
        "stars": Param(True, help="names of stars brighter than magnitude 3"),
        "mag_limit": Param(0.0, 0.0, 25.0, "hide objects fainter than this (0 = no limit)"),
        "min_size": Param(8.0, 0.0, 500.0, "hide objects smaller than this many pixels (Messier always shown)", px=True),
        "labels": Param("full", choices=["full", "designation", "common", "none"]),
        "grid": Param(False, help="equatorial RA/Dec grid"),
        "compass": Param(True, help="north / east arrows"),
        "scale_bar": Param(True, help="angular scale bar"),
        "font_size": Param(0, 0, 400, "label size in pixels (0 = automatic)", px=True),
        "line_width": Param(0, 0, 50, "line width in pixels (0 = automatic)", px=True),
        "opacity": Param(0.9, 0.1, 1.0),
        "color_galaxy": Param("#ffd25a"), "color_nebula": Param("#ff7a7a"), "color_cluster": Param("#7ad8ff"),
        "color_star": Param("#ffffff"), "color_grid": Param("#6f8fbf"),
        "filename": Param("", help="output name without extension (default: object name + _annotated)"),
        "quality": Param(92, 50, 100, "JPEG quality"),
        "apply": Param(False, help="also return the annotated image to the next steps"),
    }

    def run(self, img, ctx, messier, ngc, ic, other, sharpless, stars, mag_limit, min_size, labels, grid, compass,
            scale_bar, font_size, line_width, opacity, color_galaxy, color_nebula, color_cluster, color_star,
            color_grid, filename, quality, apply):
        wcs = ctx.layers.get("wcs")
        wcs_shape = ctx.layers.get("wcs_shape")
        if wcs is None:
            wcs = astrometry.solve(img, ctx.tools, header=ctx.header, log=ctx.log)
            if wcs is None:
                ctx.log("annotate: no plate solution (run SPCC or configure ASTAP), skipped")
                return img
            ctx.layers["wcs"] = wcs
            wcs_shape = img.shape[:2]
        if wcs_shape is None and wcs.pixel_shape is not None:
            ww, wh = wcs.pixel_shape
            # accept the WCS image size only if it has the same aspect ratio (scaled copy of this image)
            if abs(ww / wh - img.shape[1] / img.shape[0]) < 0.01:
                wcs_shape = (wh, ww)
        cats = [c for c, on in (("messier", messier), ("ngc", ngc), ("ic", ic), ("other", other),
                                ("sh2", sharpless)) if on]
        objs = catalogs.objects_in_field(wcs, img.shape, mag_limit=mag_limit or None, min_size_px=min_size,
                                         catalogs=cats, stars=stars, wcs_shape=wcs_shape)
        ctx.layers["annotations"] = objs
        ctx.log(f"annotate: {len(objs)} objects in the field"
                + (": " + ", ".join(o["names"][0] for o in objs[:12]) + ("..." if len(objs) > 12 else "")
                   if objs else ""))
        colors = {"galaxy": color_galaxy, "nebula": color_nebula, "cluster": color_cluster, "star": color_star,
                  "grid": color_grid}
        out = render(img, wcs, objs, wcs_shape=wcs_shape, labels=labels, font_size=font_size,
                     line_width=line_width, opacity=opacity, colors=colors, grid=grid, compass=compass,
                     scale_bar=scale_bar)
        if ctx.out_dir is not None and ctx.scale == 1:
            name = filename or _default_name(ctx)
            path = Path(ctx.out_dir) / f"{name}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            io.save_jpeg(out.astype(np.float32) / 255, path, quality=int(quality))
            ctx.layers["annotated_path"] = str(path)
            ctx.log(f"annotate: saved {path}")
        if apply:   # RGB even for mono input: the overlay is coloured
            return out.astype(np.float32) / 255
        return img


def _default_name(ctx):
    stem = ctx.layers.get("stem") or ""
    if not stem and ctx.header is not None:
        stem = str(ctx.header.get("OBJECT", "") or "").strip()
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem) or "image"
    return f"{stem}_annotated"
