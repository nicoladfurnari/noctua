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
            "size_arcmin": None if not np.isfinite(o.get("maj", np.nan)) else float(o["maj"]),
            "size2_arcmin": None if not np.isfinite(o.get("min", np.nan)) else float(o["min"]),
            "mag": None if not np.isfinite(o.get("mag", np.nan)) else float(o["mag"])}


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


# ------------------------------------------------------------------ observing site and sky previews
SITE_DEFAULT = {"name": "Rieti, Italia", "lat": 42.404, "lon": 12.857}


def site():
    """Where the user observes from (the visibility of the targets is worked out for it): site.json in the atlas
    folder, Rieti by default."""
    try:
        s = json.loads((folder() / "site.json").read_text(encoding="utf-8"))
        lat, lon = float(s["lat"]), float(s["lon"])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return {"name": str(s.get("name") or "")[:80], "lat": lat, "lon": lon}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return dict(SITE_DEFAULT)


def set_site(name, lat, lon):
    lat, lon = float(lat), float(lon)
    if not (np.isfinite([lat, lon]).all() and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("latitudine tra -90 e 90, longitudine tra -180 e 180")
    s = {"name": str(name or "").strip()[:80] or f"{lat:.3f}, {lon:.3f}", "lat": lat, "lon": lon}
    (folder() / "site.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    return s


SKY_HIPS = "CDS/P/DSS2/color"
HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"     # as in allsky.py


def sky_fov(o):
    """Field of the preview in degrees: the object with room around it."""
    maj = o.get("maj", np.nan)
    return float(np.clip((maj / 60.0) * 1.8 if np.isfinite(maj) and maj > 0 else 0.5, 0.25, 8.0))


SKYVIEW = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"


def _get(url, timeout):
    import urllib.request
    from . import net
    req = urllib.request.Request(url, headers={"User-Agent": "AstroForge"})
    with net.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    if raw[:2] != bytes([0xFF, 0xD8]):
        raise OSError("la risposta non è un'immagine JPEG")
    return raw


def _sky_hips2fits(o, fov, timeout):
    """DSS2 colour from CDS hips2fits (the main server, then its mirror)."""
    import urllib.parse
    q = urllib.parse.urlencode({"hips": SKY_HIPS, "width": 640, "height": 440, "fov": round(fov, 4),
                                "projection": "TAN", "ra": o["ra"], "dec": o["dec"], "format": "jpg"})
    err = None
    for base in (HIPS2FITS, HIPS2FITS.replace("//alasky.", "//alaskybis.")):
        try:
            return _get(base + "?" + q, timeout)
        except OSError as e:
            err = e
    raise err


def _sky_skyview(o, fov, timeout):
    """NASA SkyView fallback: DSS2 blue, red and infrared plates (one grey JPEG each) put together in colour,
    red = DSS2 red (it holds H-alpha) lifted by the infrared, blue = DSS2 blue, green between the two."""
    import io
    import urllib.parse
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    def plate(survey):
        q = urllib.parse.urlencode({"Position": f"{o['ra']:.5f},{o['dec']:.5f}", "Survey": survey,
                                    "Pixels": "640,440", "Size": f"{fov:.4f},{fov * 440 / 640:.4f}",
                                    "Return": "JPEG", "Scaling": "Linear"})
        return np.asarray(Image.open(io.BytesIO(_get(SKYVIEW + "?" + q, timeout))).convert("L"), np.float32) / 255

    with ThreadPoolExecutor(3) as ex:
        ir, r, b = ex.map(plate, ("DSS2 IR", "DSS2 Red", "DSS2 Blue"))
    if r.shape != (440, 640) or float(r.std()) < 1e-3:
        raise OSError("SkyView non ha dati per questo campo")

    def level(c):
        lo, hi = np.percentile(c, [30, 99.9])
        return np.clip((c - lo) / max(hi - lo, 1e-3), 0, 1)

    def gap(c):                                  # a strip outside the plate: black in this band only
        return (c < 0.008) & (float(np.median(c)) > 0.03)

    lr, lb, li = level(r), level(b), level(ir)
    lr, lb, li = np.where(gap(r), lb, lr), np.where(gap(b), lr, lb), np.where(gap(ir), lr, li)
    red = np.maximum(lr, 0.6 * li)
    blue = lb * 0.78
    rgb = np.clip(np.dstack([red * 1.05, 0.6 * red + 0.4 * blue, blue]), 0, 1) ** 1.25
    buf = io.BytesIO()
    Image.fromarray((rgb * 255).astype(np.uint8)).save(buf, "JPEG", quality=88)
    return buf.getvalue()


_CDS_DOWN = {"until": 0.0}          # CDS failed: skip it for a while (its mirror can hang 20-40 s before failing)
_CDS_PAUSE = 1800
_SKYVIEW_MAX_FOV = 3.5              # SkyView mosaics many plates for wide fields: 5 deg takes ~13 s, 3.5 ~5 s
_SKY_LOCKS = {}
_SKY_LOCKS_LOCK = threading.Lock()


def _sky_file(o):
    return folder() / "sky" / (hashlib.sha1(o["id"].encode()).hexdigest()[:16] + ".jpg")


def sky_preview(oid, timeout=6):
    """Path of a DSS2 colour picture of a catalogue object, downloaded once and kept in <atlas>/sky: CDS
    hips2fits (short timeout; after a failure it is left alone for half an hour), else NASA SkyView. None when
    the object is unknown; raises OSError when no service answers."""
    o = catalogs.find(oid)
    if o is None:
        return None
    f = _sky_file(o)
    if f.is_file() and f.stat().st_size > 0:
        return f
    with _SKY_LOCKS_LOCK:
        lock = _SKY_LOCKS.setdefault(o["id"], threading.Lock())
    with lock:                                    # a click and the prefetch on the same object: one download
        if f.is_file() and f.stat().st_size > 0:
            return f
        f.parent.mkdir(exist_ok=True)
        fov = sky_fov(o)
        raw = None
        if time.time() >= _CDS_DOWN["until"]:
            try:
                raw = _sky_hips2fits(o, fov, timeout)
            except OSError:
                _CDS_DOWN["until"] = time.time() + _CDS_PAUSE
        if raw is None:
            raw = _sky_skyview(o, min(fov, _SKYVIEW_MAX_FOV), 60)
        tmp = f.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, f)
        return f


_PREFETCH = {"started": False}


def prefetch_sky(ids, pause=0.5):
    """Download in the background, one at a time, the previews still missing (Messier first): after one pass
    every card opens at once. Started once per process; stops at the first object whose download fails twice
    in a row (no Internet)."""
    if _PREFETCH["started"] or os.environ.get("ASTROFORGE_ATLAS_PREFETCH", "1") == "0":
        return False
    _PREFETCH["started"] = True

    def run():
        fails = 0
        for oid in ids:
            o = catalogs.find(oid)
            if o is None or _sky_file(o).is_file():
                continue
            try:
                sky_preview(oid)
                fails = 0
            except Exception:
                fails += 1
                if fails >= 2:
                    _PREFETCH["started"] = False           # try again on the next visit of the atlas page
                    return
            time.sleep(pause)

    threading.Thread(target=run, daemon=True, name="atlas-sky-prefetch").start()
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
                 "xyz": [round(c, 3) for c in galactic_xyz(t["ra"], t["dec"], 3e3)], "size_arcmin": None,
                 "size2_arcmin": None, "mag": None}
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
    return {"objects": sorted(objects.values(), key=lambda o: o["ly"]), "stats": stats, "site": site()}
