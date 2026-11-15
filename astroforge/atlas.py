"""The atlas: every target photographed so far, placed in space.

A small register (``~/.astroforge/atlas/atlas.json`` + one thumbnail per result) filled when a stacking or an export
ends, and the scene the GUI draws in 3D: the catalogue objects with a known distance and the registered targets, in
galactic cartesian coordinates centred on the Sun (x towards the galactic centre, y towards l = 90 deg, z towards
the north galactic pole), light years.

Distances come from ``data/distances.csv`` (approximate literature values, a few hundred popular targets). An object
without one gets a typical distance for its type and is marked ``known: False``: the map says so.
"""
import csv
import hashlib
import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import catalogs

DATA = Path(__file__).parent / "data"
_LOCK = threading.Lock()

# equatorial J2000 -> galactic rotation (IAU 1958 system, Hipparcos Vol. 1 eq. 1.5.11)
_EQ2GAL = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                    [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
                    [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669]])

# OpenNGC type -> (kind shown on the map, typical distance in light years when unknown)
_KINDS = {"G": ("galaxy", 6e7), "GPair": ("galaxy", 1e8), "GTrpl": ("galaxy", 1e8), "GGroup": ("galaxy", 2e8),
          "GCl": ("globular", 3e4), "OCl": ("cluster", 3e3), "*Ass": ("cluster", 3e3), "Cl+N": ("nebula", 4e3),
          "Neb": ("nebula", 3e3), "HII": ("nebula", 4e3), "EmN": ("nebula", 3e3), "RfN": ("nebula", 1.5e3),
          "DrkN": ("dark", 1e3), "SNR": ("remnant", 3e3), "PN": ("planetary", 3e3), "Nova": ("star", 5e3),
          "*": ("star", 1e3), "**": ("star", 1e3)}


def folder():
    """Where the register lives (ASTROFORGE_ATLAS overrides it: the tests use a temporary folder)."""
    d = Path(os.environ.get("ASTROFORGE_ATLAS") or Path.home() / ".astroforge" / "atlas")
    d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=1)
def distances():
    """Catalogue id -> distance in light years, from data/distances.csv (names resolved through the catalogue)."""
    out = {}
    with open(DATA / "distances.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            o = catalogs.find(row["name"])
            try:
                ly = float(row["ly"])
            except (TypeError, ValueError):
                continue
            if o and ly > 0:
                out[o["id"]] = ly
    return out


def kind_of(type_code):
    return _KINDS.get(str(type_code or ""), ("other", 3e3))


def galactic_xyz(ra, dec, ly):
    """(x, y, z) in light years, Sun at the origin, galactic axes."""
    r, d = np.radians(float(ra)), np.radians(float(dec))
    v = _EQ2GAL @ np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
    return [float(c) * float(ly) for c in v]


def galactic_lb(ra, dec):
    x, y, z = galactic_xyz(ra, dec, 1.0)
    return float(np.degrees(np.arctan2(y, x)) % 360), float(np.degrees(np.arcsin(np.clip(z, -1, 1))))


def _describe(o, ly=None, known=True):
    """One object of the scene from a catalogue entry (catalogs.find)."""
    kind, typical = kind_of(o["type"])
    if ly is None:
        ly, known = distances().get(o["id"]), True
        if ly is None:
            ly, known = typical, False
    names = o.get("names") or [o["id"]]
    l, b = galactic_lb(o["ra"], o["dec"])
    return {"id": o["id"], "name": names[0], "names": names, "common": (o.get("common") or [""])[0],
            "type": o["type"], "kind": kind, "ra": o["ra"], "dec": o["dec"], "l": round(l, 3), "b": round(b, 3),
            "ly": ly, "known": known, "xyz": [round(c, 3) for c in galactic_xyz(o["ra"], o["dec"], ly)],
            "size_arcmin": None if not np.isfinite(o.get("maj", np.nan)) else float(o["maj"])}


# ------------------------------------------------------------------ the register
def _path():
    return folder() / "atlas.json"


def load():
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("targets"), dict) else {"targets": {}}
    except (OSError, ValueError):
        return {"targets": {}}


def _save(data):
    tmp = _path().with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _path())


def thumb_name(file):
    return hashlib.sha1(os.path.normcase(os.path.abspath(str(file))).encode()).hexdigest()[:16] + ".jpg"


def record(name, ra=None, dec=None, *, file, kind="stack", exposure_s=None, frames=None, date=None, thumb=None,
           filters=None):
    """Register a result (an integration or an exported final) under its target. name: what the image shows
    ('M 42', 'NGC 7000', the OBJECT of the header); ra, dec: field centre in degrees when known (plate solve).
    thumb: JPEG bytes. The same file registered again replaces its entry. Returns the target, or None when the
    image cannot be placed (no catalogue object by that name and no coordinates)."""
    o = catalogs.find(name) if name else None
    if o is not None:
        key, shown = o["id"], (o.get("names") or [o["id"]])[0]
        ra, dec = o["ra"], o["dec"]
    elif ra is not None and dec is not None and np.isfinite([ra, dec]).all():
        shown = str(name).strip() or f"campo {float(ra) / 15:.2f}h {float(dec):+.1f}°"
        key = "pos:" + shown.lower()
    else:
        return None
    session = {"file": str(file), "kind": kind, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "date": date or "",
               "exposure_s": float(exposure_s) if exposure_s else None, "frames": int(frames) if frames else None,
               "filters": list(filters or []), "thumb": ""}
    with _LOCK:
        if thumb:
            session["thumb"] = thumb_name(file)
            (folder() / session["thumb"]).write_bytes(thumb)
        data = load()
        t = data["targets"].setdefault(key, {"key": key, "name": shown, "ra": float(ra), "dec": float(dec),
                                             "catalog": o["id"] if o else "", "sessions": []})
        t["name"], t["ra"], t["dec"] = shown, float(ra), float(dec)
        same = os.path.normcase(os.path.abspath(str(file)))
        t["sessions"] = [s for s in t["sessions"] if os.path.normcase(os.path.abspath(s["file"])) != same]
        t["sessions"].append(session)
        _save(data)
        return t


def remove(key, file=None):
    """Forget a target, or one of its results (file). Returns True when something was removed."""
    with _LOCK:
        data = load()
        t = data["targets"].get(key)
        if t is None:
            return False
        gone = t["sessions"] if file is None else [s for s in t["sessions"] if s["file"] == file]
        if file is not None and not gone:
            return False
        t["sessions"] = [s for s in t["sessions"] if s not in gone]
        if not t["sessions"]:
            data["targets"].pop(key)
        used = {s["thumb"] for x in data["targets"].values() for s in x["sessions"]}
        for s in gone:
            if s.get("thumb") and s["thumb"] not in used:
                (folder() / s["thumb"]).unlink(missing_ok=True)
        _save(data)
        return True


# ------------------------------------------------------------------ the scene
def scene():
    """What the 3D map draws: {objects, targets, stats}. objects: catalogue objects with a known distance
    (the backdrop) plus every registered target; each has xyz in light years and captured True/False."""
    data = load()
    objects, seen = {}, set()
    for key, t in data["targets"].items():
        o = catalogs.find(t.get("catalog") or t["name"])
        if o is not None:
            d = _describe(o)
        else:
            l, b = galactic_lb(t["ra"], t["dec"])
            d = {"id": key, "name": t["name"], "names": [t["name"]], "common": "", "type": "", "kind": "other",
                 "ra": t["ra"], "dec": t["dec"], "l": round(l, 3), "b": round(b, 3), "ly": 3e3, "known": False,
                 "xyz": [round(c, 3) for c in galactic_xyz(t["ra"], t["dec"], 3e3)], "size_arcmin": None}
        ss = sorted(t["sessions"], key=lambda s: s.get("when", ""))
        finals = [s for s in ss if s["kind"] == "final" and s.get("thumb")]
        best = (finals or [s for s in ss if s.get("thumb")] or [None])[-1]
        # integration time: every result that carries one (an export registered by Noctua has none: its time is
        # the one of the integration it comes from, already counted)
        counted = ss
        d.update(captured=True, key=key, sessions=ss, thumb=best["thumb"] if best else "",
                 exposure_s=sum(s["exposure_s"] or 0 for s in counted),
                 frames=sum(s["frames"] or 0 for s in counted),
                 first=min((s.get("date") or s.get("when", ""))[:10] for s in ss) if ss else "")
        objects[d["id"]] = d
        seen.add(d["id"])
    for oid in distances():
        if oid in seen:
            continue
        o = catalogs.find(oid)
        if o is not None:
            objects[oid] = {**_describe(o), "captured": False}
    got = [o for o in objects.values() if o["captured"]]
    stats = {"targets": len(got), "results": sum(len(o["sessions"]) for o in got),
             "exposure_s": sum(o["exposure_s"] for o in got),
             "nearest": min(got, key=lambda o: o["ly"])["id"] if got else "",
             "farthest": max(got, key=lambda o: o["ly"])["id"] if got else "",
             "catalogue": len(objects) - len(got)}
    return {"objects": sorted(objects.values(), key=lambda o: o["ly"]), "stats": stats}
