"""Reusable master library (master bias/dark/flat/darkflat reuse across projects, like DSS / WBPP).

A library is a folder with ``index.json`` and the masters it built (XISF, one sub-folder per kind),
plus any number of read-only external folders (e.g. the user's ``#MASTERS`` folder with PixInsight,
APT or NINA masters) indexed from their FITS/XISF headers by ``scan_external``.

Every entry carries the acquisition setup (camera, size, binning, gain, offset, set-temperature,
exposure, filter, night), provenance (frames, source hash, rejection, creation date) and quality
statistics with a sanity verdict: a broken master (e.g. a master flat with a median of 37 ADU) is
flagged and ``find`` skips it unless asked, so it is never reused silently.

Array convention: masters are float32 (H, W) in [0, 1] of 65535 ADU, as ``calibrate.load_raw``;
flats are returned normalized to unit median by ``load``.
"""
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import calibrate, frames, xisf

KINDS = ("bias", "dark", "flat", "darkflat")
DEFAULT_ROOT = Path.home() / ".astroforge" / "masters"
# ASTROFORGE_MASTER_LIBRARY overrides the default (the tests point it to a temporary folder)
DEFAULT_ROOT = Path(os.environ["ASTROFORGE_MASTER_LIBRARY"]) if os.environ.get("ASTROFORGE_MASTER_LIBRARY") else DEFAULT_ROOT
MASTER_EXT = {".fit", ".fits", ".fts", ".xisf"}

# sanity thresholds
FLAT_MIN_ADU = 1000.0          # raw (not normalized) master flat median below this: implausible exposure
FLAT_CFA_MAX_RATIO = 4.0       # max/min of the four Bayer site medians of a flat
FLAT_LOW_FRACTION = 0.05       # fraction of pixels below 5% of the median (vignetting does not do this)
BIAS_MAX_ADU = 20000.0         # bias/dark median above this: not a bias/dark (or saturated)


@dataclass
class Match:
    """Result of MasterLibrary.find: the entry, its score (higher is better) and why."""
    entry: dict
    score: float
    reasons: list = field(default_factory=list)
    dark_scale: float = 1.0     # light exposure / dark exposure when a dark of another exposure is used

    @property
    def path(self):
        return self.entry["path"]


# ---------------------------------------------------------------- helpers

def _num(v):
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return float(str(v).strip().strip("'").strip())
    except (TypeError, ValueError):
        return None


def _str(v):
    return "" if v is None else str(v).strip().strip("'").strip()


def _camera(v):
    return _str(v).split("-")[0].strip()


def _setup(light):
    """Light setup as a dict from a Frame, a dict or a frames.setup_key / group key tuple."""
    if isinstance(light, frames.Frame):
        d = {"camera": light.camera, "width": light.width, "height": light.height, "binning": light.binning,
             "gain": light.gain, "offset": light.offset, "temp": light.temp, "exptime": light.exptime,
             "filter": light.filter, "night": frames.night(light)}
    elif isinstance(light, (tuple, list)):
        cam, w, h, b, gain, off, temp, *rest = light
        d = {"camera": cam, "width": w, "height": h, "binning": b, "gain": gain, "offset": off, "temp": temp}
        for r in rest:
            if isinstance(r, (int, float)) and not isinstance(r, bool):
                d["exptime"] = float(r)
            elif isinstance(r, str):
                d["filter"] = r
    else:
        d = dict(light)
        if "temp" not in d:
            d["temp"] = d.get("set_temp", d.get("temperature"))
    return d


def _days(a, b):
    try:
        return abs((date.fromisoformat(str(a)[:10]) - date.fromisoformat(str(b)[:10])).days)
    except (TypeError, ValueError):
        return None


def _hash(paths, depends=()):
    ident = "|".join(sorted(str(p) for p in paths))
    if depends:
        ident += "|built-with|" + "|".join(str(d) for d in depends)
    return hashlib.sha1(ident.encode()).hexdigest()[:16]


def read_master(path):
    """(array float32 (H, W) in [0, 1] of 65535 ADU, header dict) of a master FITS/XISF.

    Unlike load_raw, float data are rescaled only when they are clearly in ADU (median > 2), so a
    normalized flat with a few hot pixels above 2 is not divided by 65535."""
    path = Path(path)
    if path.suffix.lower() == ".xisf":
        a, kw = xisf.read(path)
        a = a[..., 0] if a.shape[-1] == 1 else a.mean(-1)
        hdr = {k: _str(v) for k, v in kw.items()}
    else:
        with fits.open(path, memmap=False) as hdul:
            h = hdul[0].header
            a = hdul[0].data
        hdr = {k: h[k] for k in h.keys() if k}
        if a.ndim == 3:
            a = a[0]
        if np.issubdtype(a.dtype, np.integer):
            a = a.astype(np.float32) / 65535.0
        else:
            a = a.astype(np.float32)
            if float(np.nanmedian(a[::7, ::7])) > 2:
                a /= 65535.0
        if str(h.get("ROWORDER", "TOP-DOWN")).upper() == "BOTTOM-UP":
            a = a[::-1]
    return np.ascontiguousarray(a, np.float32), hdr


def _kind_of(imagetyp, name):
    """Master kind from IMAGETYP ('Master Dark', 'MasterDark', 'MASTER FLAT', 'Dark Flat') or the file name."""
    for text in (imagetyp, name):
        t = re.sub(r"[\s_\-]+", "", str(text).lower())
        if not t:
            continue
        if "darkflat" in t or "flatdark" in t:
            return "darkflat"
        if "bias" in t or "offset" in t:
            return "bias"
        if "dark" in t:
            return "dark"
        if "flat" in t:
            return "flat"
    return None


# ---------------------------------------------------------------- quality

def quality(kind, master, normalized=None, level_adu=None, bayer=None):
    """Statistics and sanity flags of a master. Returns (stats dict, list of flag reasons).

    normalized: True if the flat is already divided by its median (built by AstroForge); None = guess.
    level_adu: median of the raw flats in ADU before normalization, when known (built flats).
    """
    a = np.asarray(master, np.float32)
    sub = a[::3, ::3]               # odd stride: all four Bayer sites
    finite = np.isfinite(sub)
    flags = []
    if not finite.all():
        flags.append(f"{100 * (1 - finite.mean()):.1f}% non-finite pixels")
    sub = np.where(finite, sub, 0)
    med = float(np.median(sub))
    hp = a[::2, ::2] - np.roll(a[::2, ::2], 1, axis=1)
    noise = float(np.median(np.abs(hp[np.isfinite(hp)] - np.median(hp[np.isfinite(hp)])))) * 1.4826 / math.sqrt(2)
    st = {"median": med * 65535, "noise": noise * 65535}
    if kind in ("bias", "dark", "darkflat"):
        if float(np.std(sub)) == 0:
            flags.append("constant image (no noise): not a real master")
        if med * 65535 > BIAS_MAX_ADU:
            flags.append(f"median {med * 65535:.0f} ADU too high for a {kind}")
        return st, flags
    # ---- flats
    if normalized is None:
        normalized = 0.5 < med < 2.0
    st["normalized"] = bool(normalized)
    if not normalized:
        st["level_adu"] = med * 65535
    elif level_adu is not None:
        st["level_adu"] = float(level_adu)
    if st.get("level_adu") is not None and st["level_adu"] < FLAT_MIN_ADU:
        flags.append(f"median {st['level_adu']:.0f} ADU of 65535 implausibly low (under-exposed or broken flat)")
    if med <= 0:
        flags.append("median <= 0")
        return st, flags
    n = a / med
    nsub = n[::3, ::3]
    low = float(np.mean(np.where(np.isfinite(nsub), nsub, 0) < 0.05))
    if low > FLAT_LOW_FRACTION:
        flags.append(f"{100 * low:.0f}% of pixels below 5% of the median")
    # vignetting: smoothed level at the darkest / brightest of a coarse grid
    import cv2
    small = cv2.resize(np.where(np.isfinite(n), n, 0).astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    st["vignetting"] = [float(small.min()), float(small.max())]
    st["vignetting_pct"] = float(100 * (1 - small.min() / max(small.max(), 1e-9)))
    if a.ndim == 2 and min(a.shape) >= 4:
        h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
        sites = [float(np.median(a[y:h:2, x:w:2][::4, ::4])) for y in (0, 1) for x in (0, 1)]
        st["cfa_sites"] = [s / med for s in sites]
        is_cfa = bool(bayer) or calibrate.looks_like_cfa(a)
        if is_cfa:
            lo = min(sites)
            ratio = max(sites) / lo if lo > 0 else float("inf")
            st["cfa_ratio"] = ratio
            if ratio > FLAT_CFA_MAX_RATIO:
                flags.append(f"strong CFA imbalance (site ratio {ratio:.1f}): wrong light source, "
                             f"saturation or broken flat")
    return st, flags


# ---------------------------------------------------------------- library

class MasterLibrary:
    """Master library in ``root`` (default ~/.astroforge/masters) plus read-only ``extra`` folders."""

    def __init__(self, root=None, extra=(), log=print, strict_extra=()):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.log = log
        self.entries = {}
        if self.index_path.exists():
            try:
                self.entries = json.loads(self.index_path.read_text(encoding="utf-8")).get("masters", {})
            except (OSError, ValueError) as e:
                log(f"master library: index unreadable ({e}), starting empty")
        for folder in extra or ():
            self.scan_external(folder)
        # calibration folders of a run: they hold raw frames too, so only files that say "master" themselves
        # (IMAGETYP, NCOMBINE, file name) count, not files that merely sit in a folder called #MASTERS
        for folder in strict_extra or ():
            if folder not in (extra or ()):
                self.scan_external(folder, strict=True)

    # ---------------------------------------------------------- persistence
    def save(self):
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "masters": self.entries}, indent=1), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries.values())

    def get(self, entry_id):
        return self.entries.get(entry_id)

    def load(self, entry, cfa_normalize=False):
        """Master array of an entry (or id). Flats come back normalized (per Bayer site if cfa_normalize)."""
        if isinstance(entry, str):
            entry = self.entries[entry]
        a, _ = read_master(entry["path"])
        if entry["kind"] == "flat":
            a = calibrate.normalize_flat(a, "cfa" if cfa_normalize else "global")
        return a

    def remove(self, entry_id):
        """Forget an entry; files the library built are deleted, external files are never touched."""
        e = self.entries.pop(entry_id, None)
        if e and not e.get("external"):
            try:
                Path(e["path"]).unlink()
            except OSError:
                pass
        self.save()
        return e

    # ---------------------------------------------------------- add
    def lookup(self, kind, paths, depends=()):
        """Entry built from exactly these frames (and these bias/darkflat dependencies), if any."""
        h = _hash(paths, depends)
        e = self.entries.get(f"{kind}_{h}")
        return e if e and Path(e["path"]).exists() else None

    def add(self, kind, master, frames_, settings=None):
        """Store a master built from ``frames_`` (Frame objects or paths) and index it.

        settings (all optional): rejection, depends (ids/paths of the bias/darkflat used), cfa_normalize,
        flat_level (raw flat median in ADU before normalization), bayer, plus any metadata key of the entry
        (camera, gain, offset, temp, exptime, filter, night, ...) overriding what is read from the frames.
        Returns the entry dict.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown master kind {kind!r}")
        s = dict(settings or {})
        fs = [f if isinstance(f, frames.Frame) else frames.read_frame(f) for f in frames_]
        paths = [f.path for f in fs]
        depends = tuple(str(d) for d in s.pop("depends", ()) or () if d is not None)
        h = _hash(paths, depends)
        eid = f"{kind}_{h}"
        out = self.root / kind / f"master_{eid}.xisf"
        out.parent.mkdir(parents=True, exist_ok=True)
        master = np.asarray(master, np.float32)
        f0 = fs[0] if fs else frames.Frame(path="", type=kind)
        exps = sorted(f.exptime for f in fs) or [0.0]
        nights = [frames.night(f) for f in fs if frames.night(f)]
        meta = {
            "camera": f0.camera, "width": int(master.shape[1]), "height": int(master.shape[0]),
            "binning": f0.binning, "gain": f0.gain, "offset": f0.offset, "temp": f0.temp,
            "exptime": float(exps[len(exps) // 2]) if kind != "bias" else 0.0,
            "filter": f0.filter if kind == "flat" else "",
            "bayer": s.pop("bayer", None) or f0.bayer,
            "night": max(set(nights), key=nights.count) if nights and kind == "flat" else "",
            "date_obs": min((f.date for f in fs if f.date), default=""),
        }
        rejection = s.pop("rejection", "auto")
        cfa_norm = bool(s.pop("cfa_normalize", False))
        level = s.pop("flat_level", None)
        meta.update({k: v for k, v in s.items() if k in meta or k in ("night", "filter")})
        stats, flags = quality(kind, master, normalized=True if kind == "flat" else None,
                               level_adu=level, bayer=meta["bayer"])
        kw = {"IMAGETYP": f"Master {kind.capitalize() if kind != 'darkflat' else 'DarkFlat'}",
              "NCOMBINE": len(fs), "EXPTIME": meta["exptime"], "INSTRUME": meta["camera"],
              "XBINNING": meta["binning"], "FILTER": meta["filter"], "BAYERPAT": meta["bayer"] or "",
              "DATE-OBS": meta["date_obs"]}
        for k, v in (("GAIN", meta["gain"]), ("OFFSET", meta["offset"]), ("SET-TEMP", meta["temp"])):
            if v is not None:
                kw[k] = v
        xisf.write(master[..., None], out, {k: str(v).replace('"', "'") for k, v in kw.items()})
        entry = {"id": eid, "kind": kind, "path": str(out), "external": False, **meta,
                 "nframes": len(fs), "created": datetime.now().isoformat(timespec="seconds"),
                 "source_hash": h, "depends": list(depends), "rejection": rejection,
                 "cfa_normalize": cfa_norm, "stats": stats, "flags": flags}
        self.entries[eid] = entry
        self.save()
        if flags:
            self.log(f"master library: {kind} {eid} FLAGGED: " + "; ".join(flags))
        return entry

    # ---------------------------------------------------------- external masters
    def scan_external(self, folder, recursive=True, strict=False):
        """Index existing master files (PixInsight/APT/NINA/Siril) in a read-only folder from their headers.

        A file is taken as a master when IMAGETYP, its name or its folder say "master" (or NCOMBINE > 1).
        Files already indexed with the same size and date are not re-read. Returns the entries found."""
        folder = Path(folder)
        if not folder.exists():
            self.log(f"master library: {folder} not found")
            return []
        it = folder.rglob("*") if recursive else folder.glob("*")
        found, changed = [], False
        known = {e["path"]: e for e in self.entries.values() if e.get("external")}
        for p in sorted(it):
            if p.suffix.lower() not in MASTER_EXT or not p.is_file():
                continue
            st = p.stat()
            sig = f"{st.st_size}:{int(st.st_mtime)}"
            old = known.get(str(p))
            if old and old.get("file_sig") == sig:
                found.append(old)
                continue
            try:
                e = self._external_entry(p, sig, strict)
            except Exception as ex:   # unreadable file: report, keep scanning
                self.log(f"master library: skipping {p.name}: {ex}")
                continue
            if e is None:
                continue
            if old:
                self.entries.pop(old["id"], None)
            self.entries[e["id"]] = e
            found.append(e)
            changed = True
            msg = f"master library: indexed {e['kind']} {p.name}"
            if e["flags"]:
                msg += " FLAGGED: " + "; ".join(e["flags"])
            self.log(msg)
        if changed:
            self.save()
        return found

    def _external_entry(self, p, sig, strict=False):
        a, h = read_master(p)
        imagetyp = _str(h.get("IMAGETYP", h.get("FRAME", "")))
        ncomb = _num(h.get("NCOMBINE")) or 0
        is_master = ("master" in imagetyp.lower() or "master" in p.name.lower()
                     or (not strict and "master" in p.parent.name.lower()) or ncomb > 1)
        if not is_master:
            return None
        kind = _kind_of(imagetyp, p.stem)
        if kind is None:
            return None
        temp = next((_num(h.get(k)) for k in ("SET-TEMP", "SET_TEMP", "CCD-TEMP") if _num(h.get(k)) is not None),
                    None)
        filt = _str(h.get("FILTER", ""))
        if filt:
            from .channels import canonical
            filt = canonical(filt)
        bayer = _str(h.get("BAYERPAT", "")).upper()
        date_obs = _str(h.get("DATE-OBS", ""))
        nf = frames.Frame(path=str(p), type=kind, date=date_obs)
        created = _str(h.get("DATE", "")) or datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        stats, flags = quality(kind, a, bayer=bayer)
        eid = "ext_" + hashlib.sha1(str(p).encode()).hexdigest()[:16]
        return {"id": eid, "kind": kind, "path": str(p), "external": True, "file_sig": sig,
                "camera": _camera(h.get("INSTRUME", "")), "width": int(a.shape[1]), "height": int(a.shape[0]),
                "binning": int(_num(h.get("XBINNING")) or 1), "gain": _num(h.get("GAIN")),
                "offset": _num(h.get("OFFSET", h.get("BLKLEVEL"))), "temp": temp,
                "exptime": (_num(h.get("EXPTIME", h.get("EXPOSURE"))) or 0.0) if kind != "bias" else 0.0,
                "filter": filt if kind == "flat" else "", "bayer": bayer,
                "night": frames.night(nf) if kind == "flat" else "", "date_obs": date_obs,
                "nframes": int(ncomb), "created": created, "source_hash": "", "depends": [],
                "rejection": _str(h.get("REJECT", h.get("PIREJECT", ""))) or "unknown",
                "cfa_normalize": False, "stats": stats, "flags": flags}

    # ---------------------------------------------------------- find
    def candidates(self, kind, light, exptime=None, filter=None, night=None, max_age_days=180,
                   allow_flagged=False, scalable=False, temp_tol=2.0, now=None):
        """All compatible entries as Match objects, best first (see find)."""
        L = _setup(light)
        if exptime is None and kind == "dark":     # darkflats: pass the FLATS' exposure
            exptime = L.get("exptime")
        if filter is None and kind == "flat":
            filter = L.get("filter")
        if night is None and kind == "flat":
            night = L.get("night")
        today = (now or datetime.now()).date().isoformat()
        out = []
        for e in self.entries.values():
            if e["kind"] != kind or not Path(e["path"]).exists():
                continue
            m = self._score(e, L, kind, exptime, filter, night, max_age_days, allow_flagged, scalable,
                            temp_tol, today)
            if m is not None:
                out.append(m)
        out.sort(key=lambda m: -m.score)
        return out

    def find(self, kind, light, exptime=None, filter=None, night=None, max_age_days=180,
             allow_flagged=False, scalable=False, temp_tol=2.0, now=None):
        """Best master of ``kind`` for a light setup (Frame, dict or frames.setup_key/group key).

        Rules: size and binning must match; camera, gain and offset must match when both are known
        (offset is ignored for flats, gain too). Darks/darkflats/bias: set-temperature within temp_tol
        (exact match scores higher; up to 2x temp_tol with a warning). Darks of another exposure only
        when ``scalable`` (a master bias is available), with Match.dark_scale. Flats: same filter
        (unfiltered flats only as a fallback), same night preferred. Flagged masters are skipped unless
        allow_flagged. Masters older than max_age_days (bias/darks) get a warning in the reasons.
        Returns a Match (entry, score, reasons, dark_scale) or None.
        """
        c = self.candidates(kind, light, exptime, filter, night, max_age_days, allow_flagged, scalable,
                            temp_tol, now)
        return c[0] if c else None

    @staticmethod
    def _score(e, L, kind, exptime, filt, night, max_age_days, allow_flagged, scalable, temp_tol, today):
        reasons, score, scale = [], 100.0, 1.0
        if e.get("flags"):
            if not allow_flagged:
                return None
            score -= 60
            reasons.append("FLAGGED: " + "; ".join(e["flags"]))
        if L.get("width") and L.get("height") and (int(L["width"]), int(L["height"])) != (e["width"], e["height"]):
            return None
        if int(L.get("binning") or 1) != int(e.get("binning") or 1):
            return None
        cam_l, cam_e = _str(L.get("camera")), _str(e.get("camera"))
        if cam_l and cam_e and cam_l.lower() != cam_e.lower():
            return None
        if not (cam_l and cam_e):
            score -= 5
            reasons.append("camera unknown")
        checks = (("gain", 5), ("offset", 3)) if kind != "flat" else ()
        for key, pen in checks:
            a, b = L.get(key), e.get(key)
            if a is not None and b is not None:
                if abs(float(a) - float(b)) > 1e-6:
                    return None
                reasons.append(f"{key} {float(b):g} matches")
            else:
                score -= pen
                reasons.append(f"{key} not known")
        if kind != "flat":
            a, b = L.get("temp"), e.get("temp")
            if a is not None and b is not None:
                dt = abs(float(a) - float(b))
                tol = temp_tol * (2.5 if kind == "bias" else 1)
                if dt > 2 * tol:
                    return None
                if dt <= 0.5:
                    reasons.append(f"temperature {float(b):+g}C matches")
                elif dt <= tol:
                    score -= 3 * dt
                    reasons.append(f"temperature {float(b):+g}C ({dt:.1f}C off)")
                else:
                    score -= 10 * dt
                    reasons.append(f"WARNING: temperature {float(b):+g}C is {dt:.1f}C off the lights")
            else:
                score -= 5
                reasons.append("temperature not known")
        if kind in ("dark", "darkflat") and exptime:
            ee = float(e.get("exptime") or 0)
            if ee > 0 and abs(ee - exptime) <= 0.01 * exptime:
                reasons.append(f"exposure {ee:g}s matches")
            elif ee > 0 and scalable:
                scale = exptime / ee
                # scaling a longer dark down is less noisy than scaling a short one up
                score -= 15 + 10 * abs(math.log(scale)) + (5 if scale > 1 else 0)
                reasons.append(f"exposure {ee:g}s scaled x{scale:.3f} (needs the bias)")
            else:
                return None
        if kind == "flat":
            fe, fl = _str(e.get("filter")), _str(filt)
            if fe.lower() != fl.lower():
                if fe and fl:
                    return None
                score -= 30
                reasons.append(f"WARNING: filter '{fe or 'none'}' vs lights '{fl or 'none'}'")
            elif fe:
                reasons.append(f"filter {fe} matches")
            if night and e.get("night"):
                d = _days(night, e["night"])
                if d == 0:
                    score += 10
                    reasons.append("same night")
                elif d is not None:
                    score -= min(40, 5 + 2 * d)
                    reasons.append(f"WARNING: flat from another night ({d} days apart): dust may have moved")
            else:
                score -= 10
                reasons.append("night unknown: cannot check that dust did not move")
        else:
            d = _days(today, e.get("created") or e.get("date_obs"))
            if d is not None and d > max_age_days:
                score -= min(20, 5 + d / 60)
                reasons.append(f"WARNING: master is {d} days old (> {max_age_days}): sensor may have changed")
        n = int(e.get("nframes") or 0)
        score += min(5.0, math.log2(n + 1)) if n else 0
        if e.get("external"):
            reasons.append(f"external master {Path(e['path']).name}")
        return Match(e, round(score, 2), reasons, scale)
