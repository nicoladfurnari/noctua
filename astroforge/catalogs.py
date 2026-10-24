"""Offline deep-sky and bright-star catalogues for object annotation.

Data (astroforge/data, see ATTRIBUTION.txt): OpenNGC (NGC/IC/Messier/Caldwell + addendum, CC-BY-SA-4.0),
Sharpless HII regions (VizieR VII/20), Yale Bright Star Catalogue stars brighter than V=3 (VizieR V/50).
Pixel coordinates follow astrometry.py: x = column, y = row, 0-based.
"""
import csv
import gzip
import re
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"

TYPE_NAMES = {
    "*": "star", "**": "double star", "*Ass": "stellar association", "OCl": "open cluster",
    "GCl": "globular cluster", "Cl+N": "cluster with nebula", "G": "galaxy", "GPair": "galaxy pair",
    "GTrpl": "galaxy triplet", "GGroup": "galaxy group", "PN": "planetary nebula", "HII": "HII region",
    "DrkN": "dark nebula", "EmN": "emission nebula", "Neb": "nebula", "RfN": "reflection nebula",
    "SNR": "supernova remnant", "Nova": "nova", "Other": "other",
}
# drawing classes (one colour each in the annotate plugin)
TYPE_CLASS = {"G": "galaxy", "GPair": "galaxy", "GTrpl": "galaxy", "GGroup": "galaxy",
              "OCl": "cluster", "GCl": "cluster", "*Ass": "cluster", "Cl+N": "nebula",
              "PN": "nebula", "HII": "nebula", "EmN": "nebula", "Neb": "nebula", "RfN": "nebula",
              "SNR": "nebula", "DrkN": "dark", "*": "star", "**": "star", "Nova": "star"}
CATALOGS = ("messier", "ngc", "ic", "other", "sh2")


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


@lru_cache(maxsize=1)
def load():
    """Deep-sky catalogue as a dict of numpy arrays (one entry per object)."""
    with gzip.open(DATA / "deepsky.csv.gz", "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cat = {
        "id": np.array([r["id"] for r in rows], dtype=object),
        "cat": np.array([r["cat"] for r in rows], dtype=object),
        "type": np.array([r["type"] for r in rows], dtype=object),
        "ra": np.array([_num(r["ra"]) for r in rows]),
        "dec": np.array([_num(r["dec"]) for r in rows]),
        "maj": np.array([_num(r["maj"]) for r in rows]),     # arcmin
        "min": np.array([_num(r["min"]) for r in rows]),     # arcmin
        "pa": np.array([_num(r["pa"]) for r in rows]),       # deg, north through east
        "mag": np.array([_num(r["mag"]) for r in rows]),
        "messier": np.array([int(r["messier"]) if r["messier"] else 0 for r in rows]),
        "caldwell": np.array([int(r["caldwell"]) if r["caldwell"] else 0 for r in rows]),
        "sh2": np.array([r["sh2"] for r in rows], dtype=object),
        "common": np.empty(len(rows), dtype=object),
    }
    for i, r in enumerate(rows):
        cat["common"][i] = r["common"].split("|") if r["common"] else []
    return cat


@lru_cache(maxsize=1)
def bright_stars():
    """Stars brighter than V=3: dict name, bayer, ra, dec, vmag."""
    with open(DATA / "brightstars.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {"name": np.array([r["name"] for r in rows], dtype=object),
            "bayer": np.array([r["bayer"] for r in rows], dtype=object),
            "ra": np.array([float(r["ra"]) for r in rows]), "dec": np.array([float(r["dec"]) for r in rows]),
            "vmag": np.array([_num(r["vmag"]) for r in rows])}


def designations(i, cat=None):
    """All names of catalogue row i: primary first (Messier > NGC/IC > other), then Caldwell/Sh2."""
    cat = cat or load()
    names = []
    if cat["messier"][i]:
        names.append(f"M{cat['messier'][i]}")
    names.append(cat["id"][i])
    if cat["caldwell"][i] and cat["id"][i] != f"C{cat['caldwell'][i]}":
        names.append(f"C{cat['caldwell'][i]}")
    if cat["sh2"][i]:
        names.append(cat["sh2"][i])
    return names


def _norm(name):
    s = re.sub(r"[\s_-]+", "", str(name).upper())
    s = re.sub(r"^(MESSIER)", "M", s)
    s = re.sub(r"^(CALDWELL)", "C", s)
    s = re.sub(r"^SH2", "SH2-", s)
    return re.sub(r"(?<=[A-Z-])0+(?=\d)", "", s)


@lru_cache(maxsize=1)
def _index():
    cat = load()
    idx = {}
    for i in range(len(cat["id"])):
        for n in designations(i, cat) + list(cat["common"][i]):
            idx.setdefault(_norm(n), i)
    return idx


def find(name):
    """Look up an object by name ('M42', 'NGC 1976', 'IC434', 'C20', 'Sh2-155', 'Lagoon Nebula').
    Returns a dict or None."""
    i = _index().get(_norm(name))
    if i is None:
        return None
    cat = load()
    return {"id": cat["id"][i], "names": designations(i, cat), "common": list(cat["common"][i]),
            "type": cat["type"][i], "ra": float(cat["ra"][i]), "dec": float(cat["dec"][i]),
            "maj": float(cat["maj"][i]), "min": float(cat["min"][i]), "pa": float(cat["pa"][i]),
            "mag": float(cat["mag"][i])}


def _sep_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _field(wcs, shape):
    h, w = shape[:2]
    ra0, dec0 = wcs.all_pix2world([[(w - 1) / 2, (h - 1) / 2]], 0)[0]
    t = np.linspace(0, 1, 9)
    edge = np.concatenate([np.c_[t * (w - 1), t * 0], np.c_[t * (w - 1), t * 0 + h - 1],
                           np.c_[t * 0, t * (h - 1)], np.c_[t * 0 + w - 1, t * (h - 1)]])
    rd = wcs.all_pix2world(edge, 0)
    return float(ra0), float(dec0), float(np.nanmax(_sep_deg(ra0, dec0, rd[:, 0], rd[:, 1])))


def _world2pix(wcs, ra, dec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            x, y = wcs.all_world2pix(ra, dec, 0, quiet=True)
        except Exception:  # SIP inversion failed far from the field
            x, y = wcs.wcs_world2pix(ra, dec, 0)
    return np.asarray(x, float), np.asarray(y, float)


def local_jacobian(wcs, ra, dec, step_arcmin=1.0):
    """Pixel offset per arcmin towards east (column 0) and north (column 1) at each (ra, dec): (N, 2, 2)."""
    ra, dec = np.atleast_1d(ra).astype(float), np.atleast_1d(dec).astype(float)
    d = step_arcmin / 60.0
    cosd = np.maximum(np.cos(np.radians(dec)), 1e-6)
    x0, y0 = _world2pix(wcs, ra, dec)
    xe, ye = _world2pix(wcs, ra + d / cosd, dec)
    dn = np.where(dec + d > 90, -d, d)
    xn, yn = _world2pix(wcs, ra, dec + dn)
    j = np.empty((len(ra), 2, 2))
    j[:, 0, 0], j[:, 1, 0] = (xe - x0) / step_arcmin, (ye - y0) / step_arcmin
    j[:, 0, 1], j[:, 1, 1] = (xn - x0) / step_arcmin * np.sign(dn / d), (yn - y0) / step_arcmin * np.sign(dn / d)
    return j


def sky_ellipse_to_pixels(jac, maj, mnr, pa):
    """Semi-axes (pixels) and angle (deg, from +x towards +y) of sky ellipses (arcmin, PA north->east)."""
    p = np.radians(np.nan_to_num(pa))
    u = np.stack([np.sin(p), np.cos(p)], -1)       # major axis direction (east, north)
    v = np.stack([np.cos(p), -np.sin(p)], -1)
    m = np.stack([u * (maj / 2)[:, None], v * (mnr / 2)[:, None]], -1)   # (N, 2, 2): columns = semi-axes
    a = jac @ m
    uu, s, _ = np.linalg.svd(a)
    ang = np.degrees(np.arctan2(uu[:, 1, 0], uu[:, 0, 0]))
    return s[:, 0], s[:, 1], ang


def objects_in_field(wcs, shape, mag_limit=None, min_size_px=0.0, catalogs=("messier", "ngc", "ic", "other"),
                     stars=True, star_mag_limit=3.0, wcs_shape=None):
    """Catalogue objects visible in an image with the given WCS.

    shape: (H, W) of the image the results refer to. wcs_shape: (H, W) of the image the WCS was solved on,
    when different (e.g. a scaled preview); pixel results are rescaled to `shape`.
    mag_limit: drop objects fainter than this (objects without magnitude are kept). min_size_px: drop objects
    whose major axis is smaller than this many pixels (Messier objects and stars are always kept).
    Returns a list of dicts sorted by size (largest first): id, names, common, label, type, kind, ra, dec,
    x, y, a, b (semi-axes px, 0 for point sources), angle (deg), size_arcmin, mag, messier.
    """
    h, w = shape[:2]
    wh, ww = (wcs_shape or shape)[:2]
    sx, sy = w / ww, h / wh
    ra0, dec0, rad = _field(wcs, (wh, ww))
    cat = load()
    sel = {"messier": cat["messier"] > 0, "ngc": cat["cat"] == "NGC", "ic": cat["cat"] == "IC",
           "other": cat["cat"] == "other", "sh2": cat["cat"] == "Sh2"}
    want = np.zeros(len(cat["id"]), bool)
    for c in catalogs:
        want |= sel[c]
    if "messier" not in catalogs:
        want &= ~sel["messier"]
    maj = np.nan_to_num(cat["maj"])
    sep = _sep_deg(ra0, dec0, cat["ra"], cat["dec"])
    idx = np.nonzero(want & (sep <= rad + maj / 120.0 + 0.01))[0]
    out = []
    if len(idx):
        mnr = np.where(np.isfinite(cat["min"][idx]), cat["min"][idx], maj[idx])
        jac = local_jacobian(wcs, cat["ra"][idx], cat["dec"][idx])
        ok = np.isfinite(jac).all(axis=(1, 2))      # unprojectable positions gave NaN and an SVD error
        idx, jac, mnr = idx[ok], jac[ok], mnr[ok]
        a, b, ang = sky_ellipse_to_pixels(jac, maj[idx], mnr, cat["pa"][idx])
        x, y = _world2pix(wcs, cat["ra"][idx], cat["dec"][idx])
        # anisotropic rescale (wcs image -> output image): exact for sx == sy, which is the normal case
        k = np.sqrt(sx * sy)
        x, y, a, b = x * sx, y * sy, a * k, b * k
        for j, i in enumerate(idx):
            if not (np.isfinite(x[j]) and np.isfinite(y[j])):
                continue
            r = max(a[j], 1.0)
            if x[j] < -r or y[j] < -r or x[j] > w - 1 + r or y[j] > h - 1 + r:
                continue
            m = cat["mag"][i]
            is_m = cat["messier"][i] > 0
            if not is_m and mag_limit is not None and np.isfinite(m) and m > mag_limit:
                continue
            if not is_m and min_size_px and 2 * a[j] < min_size_px:
                continue
            names = designations(i, cat)
            common = list(cat["common"][i])
            out.append({"id": cat["id"][i], "names": names, "common": common,
                        "label": names[0] + (f" {common[0]}" if common else ""),
                        "type": cat["type"][i], "kind": TYPE_CLASS.get(cat["type"][i], "other"),
                        "ra": float(cat["ra"][i]), "dec": float(cat["dec"][i]),
                        "x": float(x[j]), "y": float(y[j]), "a": float(a[j]), "b": float(b[j]),
                        "angle": float(ang[j]), "size_arcmin": float(maj[i]),
                        "mag": None if not np.isfinite(m) else float(m), "messier": int(cat["messier"][i])})
    if stars:
        st = bright_stars()
        ok = (_sep_deg(ra0, dec0, st["ra"], st["dec"]) <= rad) & (np.nan_to_num(st["vmag"], nan=99) <= star_mag_limit)
        if ok.any():
            x, y = _world2pix(wcs, st["ra"][ok], st["dec"][ok])
            for n, bay, ra, dec, m, xx, yy in zip(st["name"][ok], st["bayer"][ok], st["ra"][ok], st["dec"][ok],
                                                  st["vmag"][ok], x * sx, y * sy):
                if np.isfinite(xx) and np.isfinite(yy) and 0 <= xx <= w - 1 and 0 <= yy <= h - 1:
                    out.append({"id": n, "names": [n, bay], "common": [], "label": n, "type": "*", "kind": "star",
                                "ra": float(ra), "dec": float(dec), "x": float(xx), "y": float(yy), "a": 0.0,
                                "b": 0.0, "angle": 0.0, "size_arcmin": 0.0, "mag": float(m), "messier": 0})
    out.sort(key=lambda o: -o["a"])
    return out
