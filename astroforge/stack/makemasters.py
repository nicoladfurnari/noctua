"""Build master bias / dark / dark flat / flat files on their own (no lights needed), with options to improve them.

Like the masters built during stacking, but for a whole calibration library at once (e.g. a camera's #MASTERS
with one sub folder per set): every compatible group of raw frames gives one master, named and tagged so that
AstroForge (and PixInsight / NINA) recognise it later, with a quality report.

Options (MASTER_DEFAULTS), and per type in options["bias"|"dark"|"darkflat"|"flat"] (TYPE_DEFAULTS, the
PixInsight ImageIntegration settings of each master: a per-type value wins over the common one):
- combination (average | median), rejection / low / high / iterations / esd_outliers / esd_alpha;
- flat: calibrate_with (auto = dark flat, else bias | darkflat | bias | none), normalize (auto | cfa | global),
  per_night, drop_odd_exposures (frames whose exposure differs from the set);
- rejection / low / high: pixel rejection of the integration and its sigmas;
- superbias: PixInsight-like Superbias, the master bias without its random noise (column and row pattern plus the
  large-scale shape kept): a clean bias even from few frames;
- defect_map: map of hot pixels (from the darks), dead pixels (from the flats) and bad columns, saved as FITS and
  usable by the stacking (setting "defect_map");
- flat_per_night: one master flat per filter and night (dust moves); flat_cfa: per Bayer site normalisation for
  colour cameras ("auto" = when the flats are CFA);
- min_frames: groups with fewer frames are skipped; format: xisf or fits; library: also store in the master library.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import calibrate, frames, masterlib, xisf

MASTER_DEFAULTS = {
    "rejection": "auto", "low": 4.0, "high": 3.0,
    "superbias": False, "superbias_scale": 32.0,
    "defect_map": True, "hot_sigma": 10.0, "cold_frac": 0.5, "columns_sigma": 8.0,
    "flat_per_night": True, "flat_cfa": "auto",
    "min_frames": 3, "format": "xisf", "library": True,
}


_BASE = {"combination": "average", "rejection": "auto", "low": 4.0, "high": 3.0, "iterations": 5,
         "esd_outliers": 0.3, "esd_alpha": 0.05, "min_frames": 3}
TYPE_DEFAULTS = {
    "bias": {**_BASE, "superbias": False, "superbias_scale": 32.0},
    "dark": dict(_BASE),
    "darkflat": dict(_BASE),
    "flat": {**_BASE, "calibrate_with": "auto", "normalize": "auto", "per_night": True, "drop_odd_exposures": True},
}
# options of the old single set that map onto every type (kept so saved settings and the CLI keep working)
_LEGACY = {"rejection": "rejection", "low": "low", "high": "high", "min_frames": "min_frames",
           "superbias": "superbias", "superbias_scale": "superbias_scale", "flat_per_night": "per_night"}


def type_options(o, kind):
    """Settings of one master type: TYPE_DEFAULTS < common options of `o` < o[kind]."""
    t = dict(TYPE_DEFAULTS[kind])
    for old, new in _LEGACY.items():
        if old in (o or {}) and new in t:
            t[new] = o[old]
    if kind == "flat" and "flat_cfa" in (o or {}):
        t["normalize"] = {True: "cfa", False: "global"}.get(o["flat_cfa"], "auto")
    t.update({k: v for k, v in ((o or {}).get(kind) or {}).items() if k in t})
    return t


def combine_kw(t):
    """build_master keywords from type_options()."""
    method = "median" if t["combination"] == "median" else t["rejection"]
    return {"method": method, "low": float(t["low"]), "high": float(t["high"]), "iterations": int(t["iterations"]),
            "esd_outliers": float(t["esd_outliers"]), "esd_alpha": float(t["esd_alpha"])}


def describe(t):
    if t["combination"] == "median":
        return "median"
    r = t["rejection"]
    if r in ("none", "median", "percentile"):
        return f"average, rejection {r}"
    if r == "esd":
        return f"average, ESD (outliers {t['esd_outliers']:g}, significance {t['esd_alpha']:g})"
    return f"average, rejection {r} {t['low']:g}/{t['high']:g} sigma"


def superbias(bias, scale=32.0):
    """Master bias without its random noise: column profile + row profile + large-scale residual (sigma `scale`).
    What remains of a bias after many frames is exactly that structure, so few frames give the result of many."""
    b = np.asarray(bias, np.float32)
    med = float(np.median(b))
    col = np.median(b, axis=0)                       # column pattern (CMOS / CCD columns)
    row = np.median(b - col[None, :], axis=1)        # row pattern left after the columns
    model = col[None, :] + row[:, None]
    resid = b - model
    smooth = cv2.GaussianBlur(resid, (0, 0), float(scale))   # amp glow, gradients of the bias level
    out = (model + smooth).astype(np.float32)
    return out if np.isfinite(out).all() else np.nan_to_num(out, nan=med)


def _name(kind, f0, n, night="", exptime=None):
    """master_dark_300s_-30C_g121_QHYCCD_n18.xisf ... (readable, unique per setup)."""
    parts = [f"master_{kind}"]
    if kind == "flat" and f0.filter:
        parts.append(f0.filter)
    # bias too: "Bias 0.001s" and "Bias 0.3s" of one camera must not get the same file name
    if exptime is not None and (kind != "bias" or exptime > 0):
        parts.append(f"{exptime:g}s")
    if f0.temp is not None:
        parts.append(f"{f0.temp:+.0f}C")
    if f0.gain is not None:
        parts.append(f"g{f0.gain:g}")
    if f0.binning and f0.binning != 1:
        parts.append(f"bin{f0.binning}")
    if night:
        parts.append(night)
    parts.append(f"n{n}")
    return "_".join(p.replace(" ", "") for p in parts)


def _keywords(kind, fs, master, night=""):
    f0 = fs[0]
    exps = sorted(f.exptime for f in fs)
    kw = {"IMAGETYP": f"Master {kind.capitalize() if kind != 'darkflat' else 'DarkFlat'}",
          "NCOMBINE": len(fs), "EXPTIME": 0.0 if kind == "bias" else exps[len(exps) // 2],
          "INSTRUME": f0.camera, "XBINNING": f0.binning, "FILTER": f0.filter if kind == "flat" else "",
          "BAYERPAT": f0.bayer or "", "DATE-OBS": min((f.date for f in fs if f.date), default=""),
          "SWCREATE": "Noctua masters"}
    for k, v in (("GAIN", f0.gain), ("OFFSET", f0.offset), ("SET-TEMP", f0.temp)):
        if v is not None:
            kw[k] = v
    if night:
        kw["NIGHT"] = night
    return kw


def _save(master, path, kw, fmt):
    if fmt == "fits":
        from astropy.io import fits
        h = fits.Header()
        for k, v in kw.items():
            h[k] = v
        fits.PrimaryHDU(np.asarray(master, np.float32), h).writeto(path, overwrite=True)
    else:
        xisf.write(master[..., None] if master.ndim == 2 else master, path,
                   {k: str(v).replace('"', "'") for k, v in kw.items()})


def make_masters(folders, out_dir, options=None, log=print, progress=None, cancel=None):
    """Build every master found in `folders` into `out_dir`. Returns the report (list of masters, defect maps)."""
    o = {**MASTER_DEFAULTS, **(options or {})}
    types = {k: type_options(options or {}, k) for k in TYPE_DEFAULTS}
    for k, t in types.items():
        log(f"{k}: {describe(t)}, min {t['min_frames']} frames"
            + (", superbias" if t.get("superbias") else "")
            + (f", calibrated with {t['calibrate_with']}, normalisation {t['normalize']}" if k == "flat" else ""))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    found = [f for d in folders for f in frames.scan(d)]
    raw = [f for f in found if f.type in ("bias", "dark", "flat", "darkflat")]
    ready = [f for f in found if f.type == "master"]
    log(f"{len(raw)} raw calibration frames in {len(folders)} folders"
        + (f" ({len(ready)} ready masters ignored)" if ready else ""))
    groups = frames.group(raw)
    lib = masterlib.MasterLibrary(None, (), log=log) if o["library"] else None
    report = {"out_dir": str(out), "options": o, "types": types, "masters": [], "skipped": [], "defect_maps": []}
    jobs = [(k, key, fs) for k in ("bias", "darkflat", "dark", "flat") for key, fs in groups[k].items()]
    biases = {}          # setup -> (exptime, master): the shortest bias of each setup calibrates darks and flats
    darkflats = {}       # (setup, exptime) -> master
    darks = {}           # setup -> [(exptime, master)]
    for i, (kind, key, fs) in enumerate(jobs, 1):
        if cancel and cancel():
            from .project import Cancelled          # (imported here: project imports this module)
            raise Cancelled()
        if progress:
            progress(i / max(len(jobs), 1), f"{kind} {i}/{len(jobs)}")
        setup = key[:7]
        t = types[kind]
        sets = [("", fs)]
        if kind == "flat" and t["per_night"]:
            by_night = {}
            for f in fs:
                by_night.setdefault(frames.night(f), []).append(f)
            sets = sorted(by_night.items())
        for night, gfs in sets:
            if kind == "flat" and t["drop_odd_exposures"]:
                bad = frames.outlier_exposures(gfs)
                if bad:
                    log(f"flat {night}: ignoring {len(bad)} frames with a different exposure")
                gfs = [f for f in gfs if f not in bad]
            label = frames.describe_key(key) + (f" {night}" if night else "")
            if len(gfs) < t["min_frames"]:
                log(f"{kind} {label}: only {len(gfs)} frames (min {t['min_frames']}), skipped")
                report["skipped"].append({"kind": kind, "group": label, "frames": len(gfs)})
                continue
            exp = sorted(f.exptime for f in gfs)[len(gfs) // 2]
            kw_build = {**combine_kw(t), "log": log}
            bias = biases.get(setup, (None, None))[1]
            if kind == "flat":
                df = darkflats.get((setup, round(exp, 3)))
                cw = t["calibrate_with"]
                if cw == "darkflat" and df is None:
                    log(f"flat {label}: no dark flat with {exp:g} s, using the bias")
                if cw == "bias" or cw == "none":
                    df = None
                if cw == "none":
                    bias = None
                cfa = t["normalize"] == "cfa" or (t["normalize"] == "auto" and bool(
                    gfs[0].bayer or calibrate.looks_like_cfa(calibrate.load_raw(gfs[0].path)[0])))
                kw_build.update(bias=bias, dark=df, cfa_normalize=cfa)
                how = "dark flat" if df is not None else ("bias" if bias is not None else "nothing (no bias / dark flat)")
                log(f"flat {label}: calibrated with {how}" + (", CFA normalisation" if cfa else ""))
            master = calibrate.build_master([f.path for f in gfs], kind, **kw_build)
            if kind == "bias" and t["superbias"]:
                master = superbias(master, t["superbias_scale"])
                log(f"bias {label}: superbias (columns + rows + large scale, sigma {t['superbias_scale']:g} px)")
            name = _name(kind, gfs[0], len(gfs), night if kind == "flat" else "", exp)
            path = out / f"{name}.{'fits' if o['format'] == 'fits' else 'xisf'}"
            kw = _keywords(kind, gfs, master, night)
            _save(master, path, kw, o["format"])
            stats, flags = masterlib.quality(kind, master, normalized=True if kind == "flat" else None,
                                             bayer=gfs[0].bayer)
            entry = {"kind": kind, "file": str(path), "group": label, "frames": len(gfs), "exptime": exp,
                     "filter": gfs[0].filter if kind == "flat" else "", "night": night, "stats": stats, "flags": flags}
            if kind == "dark" and bias is not None:
                sig = master - bias
                entry["stats"]["hot_pixels"] = int((sig - np.median(sig) > 0.01).sum())   # > ~650 ADU over the level
            report["masters"].append(entry)
            log(f"{kind} {label}: {len(gfs)} frames -> {path.name}" + (f"  FLAG: {'; '.join(flags)}" if flags else ""))
            if lib is not None:
                try:
                    lib.add(kind, master, gfs, {"rejection": describe(t), "cfa_normalize": kw_build.get("cfa_normalize", False)})
                except Exception as e:
                    log(f"master library: not saved ({e})")
            if kind == "bias":
                if setup not in biases or exp < biases[setup][0]:
                    biases[setup] = (exp, master)
            elif kind == "darkflat":
                darkflats[(setup, round(exp, 3))] = master
            elif kind == "dark":
                darks.setdefault(setup, []).append((exp, master, gfs[0]))
    if o["defect_map"]:
        report["defect_maps"] = _defect_maps(darks, biases, report, out, o, log)
    report["seconds"] = round(time.time() - t0, 1)
    (out / "masters_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "masters_report.txt").write_text(_text_report(report), encoding="utf-8")
    log(f"{len(report['masters'])} masters in {out} ({report['seconds']} s), report: masters_report.txt")
    return report


def _defect_maps(darks, biases, report, out, o, log):
    """One map per camera setup: hot pixels and bad columns from its longest dark, dead pixels from its flats."""
    maps = []
    flats = {}
    for e in report["masters"]:
        if e["kind"] == "flat":
            flats.setdefault(e["group"].split(" filter")[0], []).append(e["file"])
    for setup, ds in darks.items():
        exp, dark, f0 = max(ds, key=lambda d: d[0])
        bias = biases.get(setup, (None, None))[1]
        sig = dark - bias if bias is not None else dark - float(np.median(dark))
        flat = None
        for files in flats.values():
            try:
                cand = calibrate.load_master(files[0])
            except Exception:
                continue
            if cand.shape == dark.shape:
                flat = cand
                break
        bad = calibrate.defect_map(sig, flat, hot_sigma=o["hot_sigma"], cold_frac=o["cold_frac"],
                                   columns_sigma=o["columns_sigma"])
        if bad is None:
            continue
        name = _name("defects", f0, len(ds), exptime=exp)
        path = out / f"{name}.fits"
        from astropy.io import fits
        h = fits.Header({"IMAGETYP": "Defect Map", "INSTRUME": f0.camera, "EXPTIME": exp, "XBINNING": f0.binning})
        if f0.gain is not None:
            h["GAIN"] = f0.gain
        fits.PrimaryHDU(bad.astype(np.uint8), h).writeto(path, overwrite=True)
        n = int(bad.sum())
        cols = int(bad.all(axis=0).sum())
        log(f"defect map {path.name}: {n} pixels ({100 * n / bad.size:.3f}%), {cols} whole columns")
        maps.append({"file": str(path), "pixels": n, "columns": cols, "from_dark_s": exp,
                     "with_flat": flat is not None})
    return maps


def _text_report(r):
    lines = [f"Noctua - master creati in {r['out_dir']}", ""]
    for e in r["masters"]:
        st = e["stats"]
        extra = ""
        if e["kind"] == "flat" and "vignetting_pct" in st:
            extra = f"  vignettatura {st['vignetting_pct']:.0f}%"
        if "hot_pixels" in st:
            extra += f"  pixel caldi {st['hot_pixels']}"
        if e["kind"] == "flat":      # normalised to 1: level in ADU means nothing, noise as a share of the level
            level = f"normalizzato, rumore {100 * st.get('noise', 0) / 65535:.2f}%"
        else:
            level = f"livello {st.get('median', 0):.0f} ADU, rumore {st.get('noise', 0):.1f} ADU"
        lines.append(f"{e['kind']:8s} {Path(e['file']).name}: {e['frames']} frame, {level}{extra}"
                     + (f"  ATTENZIONE: {'; '.join(e['flags'])}" if e["flags"] else ""))
    for s in r["skipped"]:
        lines.append(f"saltato  {s['kind']} {s['group']}: solo {s['frames']} frame")
    for d in r["defect_maps"]:
        lines.append(f"difetti  {Path(d['file']).name}: {d['pixels']} pixel, {d['columns']} colonne intere")
    return "\n".join(lines) + "\n"
