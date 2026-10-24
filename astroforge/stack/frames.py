"""Scan folders, classify frames (light/dark/flat/bias/darkflat) and group them."""
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from astropy.io import fits

FITS_EXT = {".fit", ".fits", ".fts"}
TYPES = ("light", "dark", "flat", "bias", "darkflat")

_TYPE_WORDS = [
    ("darkflat", r"dark[\s_-]*flat|flat[\s_-]*dark"),
    ("bias", r"bias|offset"),
    ("dark", r"dark"),
    ("flat", r"flat"),
    ("light", r"light|object"),
]
# file name prefixes written by APT / NINA / SharpCap (L_, D_, F_, B_)
_PREFIX = {"L": "light", "D": "dark", "F": "flat", "B": "bias"}


@dataclass
class Frame:
    path: str
    type: str
    width: int = 0
    height: int = 0
    exptime: float = 0.0
    gain: float | None = None
    iso: float | None = None        # DSLR / mirrorless sensitivity (ISOSPEED), shown on the Sorgenti page
    offset: float | None = None
    temp: float | None = None
    binning: int = 1
    filter: str = ""
    bayer: str = ""
    object: str = ""
    date: str = ""
    camera: str = ""
    focal: float | None = None
    pixel_size: float | None = None
    notes: list = field(default_factory=list)
    keywords: dict = field(default_factory=dict)    # grouping keywords asked by the settings (WBPP)

    def to_dict(self):
        return asdict(self)


def _classify(path, header):
    imtyp = str(header.get("IMAGETYP", header.get("FRAME", ""))).lower()
    for t, pat in _TYPE_WORDS:
        if re.search(pat, imtyp):
            return t
    m = re.match(r"([LDFB])_", path.name)
    if m:
        return _PREFIX[m.group(1)]
    for t, pat in _TYPE_WORDS:                       # the file name
        if re.search(pat, path.name.lower()):
            return t
    for folder in reversed(path.parts[-3:-1]):       # the folders: "#Dark", "Flat 10k ADU gain 121 -30C"...
        t = _calib_folder(folder)
        if t:
            return t
    return "light"


# words allowed beside the type word in the name of a calibration folder: "Dark 300s gain 121 -30C" is a dark
# folder, "Dark Doodad" or "Dark Horse" is a target whose lights must not be taken for darks
_CAL_OK = {"gain", "bin", "iso", "offset", "ofs", "master", "masters", "frame", "frames", "exp", "exposure", "s",
           "sec", "c", "g", "k", "adu", "e", "x", "min", "ms", "temp", "cal", "calib", "calibration"}


def _calib_folder(name):
    low = name.lower()
    for t, pat in _TYPE_WORDS[:4]:
        if re.search(pat, low):
            others = [w for w in re.findall(r"[a-z]+", low) if not re.search(pat, w) and w not in _CAL_OK]
            return t if not others else None
    return None


def _num(h, *keys):
    for k in keys:
        if k in h:
            try:
                return float(h[k])
            except (TypeError, ValueError):
                pass
    return None


def read_frame(path, keywords=()):
    """keywords: FITS keywords to keep in f.keywords for grouping (NIGHT = observing night, computed)."""
    path = Path(path)
    h = fits.getheader(path)
    f = Frame(
        path=str(path), type=_classify(path, h),
        width=int(h.get("NAXIS1", 0)), height=int(h.get("NAXIS2", 0)),
        exptime=_num(h, "EXPTIME", "EXPOSURE") or 0.0,
        gain=_num(h, "GAIN"), iso=_num(h, "ISOSPEED", "ISO"), offset=_num(h, "OFFSET", "BLKLEVEL"),
        # setpoint first: the measured CCD-TEMP wobbles around it and would split one session in two groups
        temp=_num(h, "SET-TEMP", "SET_TEMP", "CCD-TEMP"),
        binning=int(_num(h, "XBINNING") or 1),
        filter=str(h.get("FILTER", "")).strip(),
        bayer=str(h.get("BAYERPAT", "")).strip().upper(),
        object=str(h.get("OBJECT", "")).strip(),
        date=str(h.get("DATE-OBS", "")),
        camera=str(h.get("INSTRUME", "")).split("-")[0].strip(),
        focal=_num(h, "FOCALLEN"), pixel_size=_num(h, "XPIXSZ"),
    )
    if not f.filter and not f.bayer:
        # mono frames without FILTER: try the folder name, then a token in the file name (L_ha_...)
        from .channels import _ALIASES
        known = {n for names in _ALIASES.values() for n in names if len(n) > 1}
        tokens = [path.parent.name.lower()] + re.split(r"[_\s-]+", path.stem.lower())[1:3]
        f.filter = next((t for t in tokens if t in known), "")
    if f.filter:
        from .channels import canonical
        f.filter = canonical(f.filter)
    if int(h.get("NAXIS", 2)) == 3:
        f.notes.append("already color (3 planes)")
    # a ready master (or an integrated stack) is not a raw frame: never integrated again, used as it is
    imtyp = str(h.get("IMAGETYP", h.get("FRAME", ""))).lower()
    if "master" in imtyp or (_num(h, "NCOMBINE") or 0) > 1 or path.stem.lower().startswith("master"):
        f.notes.append(f"master {f.type}")
        f.type = "master"
    for k in keywords or ():
        k = str(k).strip().upper()
        if k == "NIGHT":
            f.keywords[k] = night(f)
        elif k:
            v = h.get(k)
            f.keywords[k] = "" if v is None else str(v).strip()
    return f


# output folders of PixInsight WBPP, Siril, DSS, APT and AstroForge itself: never raw frames
SKIP_DIRS = {"calibrated", "registered", "master", "masters", "debayered", "logs", "apt_thumbs", "process",
             "astroforge", "work", "edit", "cosmetized", "measured", "fastintegration"}


def scan(folder, recursive=True, keywords=()):
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    frames = []
    for p in sorted(it):
        rel = p.relative_to(folder).parts[:-1]
        if any(part.lower() in SKIP_DIRS for part in rel):
            continue
        if p.suffix.lower() in FITS_EXT and p.is_file():
            try:
                frames.append(read_frame(p, keywords))
            except Exception as e:  # unreadable file: report it, keep scanning
                frames.append(Frame(path=str(p), type="unknown", notes=[f"unreadable: {e}"]))
    return frames


def night(f):
    """Observing night of a frame (date of DATE-OBS minus 12 h, so a session past midnight stays one night)."""
    from datetime import datetime, timedelta
    s = f.date.strip().replace("Z", "")[:19]
    try:
        t = datetime.fromisoformat(s)
    except ValueError:
        return ""
    if len(s) <= 10:                     # date only: nothing to shift (midnight minus 12 h was the day before)
        return t.date().isoformat()
    return (t - timedelta(hours=12)).date().isoformat()


def master_exptime(path):
    """EXPTIME / EXPOSURE of a master file (FITS or XISF), None if unknown."""
    try:
        p = Path(path)
        if p.suffix.lower() == ".xisf":
            from . import xisf
            kw = xisf.read(p)[1] or {}
            v = kw.get("EXPTIME", kw.get("EXPOSURE"))
        else:
            from astropy.io import fits
            with fits.open(p, memmap=True) as hd:
                v = hd[0].header.get("EXPTIME", hd[0].header.get("EXPOSURE"))
        return float(v) if v not in (None, "") else None
    except Exception:
        return None


def setup_key(f):
    """Frames that can be calibrated together share this key."""
    temp = None if f.temp is None else round(f.temp / 2) * 2   # 2 degree tolerance
    return (f.camera, f.width, f.height, f.binning, f.gain, f.offset, temp)


def group(frames, keys=()):
    """Return {type: {key: [frames]}}; key includes exposure and filter where they matter. keys: grouping
    keywords that split the lights into separate integrations (WBPP post-calibration grouping); their values
    go at the end of the light key as ((KEY, value), ...), after the positions the rest of the code reads."""
    keys = [str(k).strip().upper() for k in keys or () if str(k).strip()]
    out = {t: {} for t in TYPES}
    for f in frames:
        if f.type not in out:
            continue
        key = setup_key(f)
        # bias too: a folder like #MASTERS may hold "Bias 0.001s" and "Bias 0.3s", which must not be mixed
        if f.type in ("light", "dark", "darkflat", "bias"):
            key = key + (round(f.exptime or 0.0, 3),)
        if f.type in ("light", "flat"):
            key = key + (f.filter,)
        if f.type == "light" and keys:
            key = key + (tuple((k, f.keywords.get(k, "")) for k in keys),)
        out[f.type].setdefault(key, []).append(f)
    for t in out:
        out[t] = _merge_temps(out[t])
    return out


def _merge_temps(groups, tol=2.0):
    """Groups equal but for the temperature (index 6 of the key), within `tol` degrees, become one group with the
    temperature of the larger: setup_key rounds to 2 degree bins, and a sensor wobbling around a bin edge
    (-9.4 / -8.9 C) split a session in two, of which only the larger was stacked."""
    merged = {}
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        rest = key[:6] + key[7:]
        hit = next((k for k in merged if k[:6] + k[7:] == rest and k[6] is not None and key[6] is not None
                    and abs(k[6] - key[6]) <= tol), None)
        if hit is None:
            merged[key] = list(groups[key])
        else:
            merged[hit] += groups[key]
    return merged


def describe_key(key):
    cam, w, h, b, gain, off, temp, *rest = key
    s = f"{cam} {w}x{h} bin{b}"
    if gain is not None:
        s += f" gain {gain:g}"
    if off is not None:
        s += f" offset {off:g}"
    if temp is not None:
        s += f" {temp:+.0f}C"
    for r in rest:
        if isinstance(r, tuple):
            s += "".join(f" {k}={v or '-'}" for k, v in r)
        elif isinstance(r, float):
            s += f" {r:g}s"
        elif r:
            s += f" filter {r}"
    return s


def outlier_exposures(frames, tol=0.05):
    """Frames whose exposure differs from the group median (e.g. test flats)."""
    if not frames:
        return []
    exps = sorted(f.exptime for f in frames)
    med = exps[len(exps) // 2]
    return [f for f in frames if med > 0 and abs(f.exptime - med) / med > tol]
