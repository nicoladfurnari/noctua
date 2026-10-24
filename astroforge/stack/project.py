"""Full raw-to-integration workflow.

Project settings are a plain dict (JSON) so the GUI wizard and the CLI share them.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

from .. import gpu, io, stretch
from . import (analyze, calibrate, channels, combine, comet, darkopt, drizzle, dualband, frames, localnorm,
               register, satellites, subframes, xisf)

DEFAULTS = {
    "lights": [],              # folders or files
    "calibration": [],         # folders searched for bias/dark/flat/darkflat
    "masters": {},             # optional explicit master files: {"bias": path, "dark": path, "flat": path}
    "work_dir": "work",
    "output": "stack",
    "light_group": None,       # index into the light groups; None = largest
    "exclude": [],             # light paths excluded by the user
    "debayer": "rcd",          # rcd (best on undersampled stars) | bilinear | ea
    "bayer_pattern": None,     # None = header, verified on the data; or RGGB/BGGR/GRBG/GBRG
    "cosmetic": True,
    "auto_reject": True,
    "fwhm_sigma": 2.0,
    "ecc_max": 0.75,
    "rejection": "auto",       # auto | none | median | percentile | sigma | winsorized | linear_fit | esd
    "reject_grow": 2,          # px: grow high rejections that form structures (satellite trail wings)
    "satellites": "auto",      # trail detection per frame (True/False; auto = with fewer than 20 frames)
    "reject_low": 4.0,
    "reject_high": 3.0,
    "weights": True,
    "weighting": "psf",        # psf (PSF signal) | standard | nebula | galaxy | cluster | SubframeSelector expression
    "approval": None,          # e.g. 70: frames whose weighting formula is below it are rejected ("Weight > 70")
    "exposure_tolerance": 0,   # s: lights whose exposures differ by at most this are integrated together
    "drizzle": 1,              # 1 = off, 2, 3
    "pixfrac": 0.9,
    "bayer_drizzle": True,
    "drizzle_reject": True,
    "drizzle_kernel": "square",  # square | gaussian | lanczos3 (lanczos3: pixfrac 1, scale ~1, many frames)
    "flat_per_session": True,  # one master flat per night when lights and flats span several nights
    "dark_optimize": "auto",   # scale the dark to minimise noise (needs bias), also amp glow; auto = only when
                               # no dark has the lights' exposure (as the GAEEB workflow advises); True = always
    "dark_glow": True,
    "local_norm": False,       # local normalization (gradients that change between frames: Moon, LP)
    "local_norm_scale": 512,
    "dualband": False,         # OSC + dual-band filter: integrate, then split Ha (R) / OIII (G+B) -> HOO
    "dualband_ha_leak_g": 0.0,
    "dualband_b_gain": 1.0,
    "distortion": True,        # fit a polynomial distortion model per frame, used only when it helps
    "mosaic_wcs": True,        # mosaic: place the panels by plate solving (ASTAP) when every panel solves
    "registered_float16": False,  # store registered frames as float16: half the disk, ~1e-5 rounding
    "master_reuse": True,      # master library: masters are saved there and reused when a session lacks frames
    "master_library": None,    # folder (None = ~/.astroforge/masters)
    "defect_map": None,        # FITS from "Crea master" (hot/dead pixels, bad columns), used with the computed map
    "master_dirs": [],         # extra read-only folders with existing masters (e.g. the camera's #MASTERS)
    "master_options": {},      # per type {"bias": {...}, "dark": {...}, "darkflat": {...}, "flat": {...}}: how the
                               # masters built during stacking are integrated (makemasters.TYPE_DEFAULTS)
    "master_max_age_days": 180,
    "master_allow_flagged": False,  # also use masters flagged as suspicious (e.g. a flat at ~37 ADU)
    "flat_cfa_normalize": "auto",   # flats normalized per Bayer site (keeps colour balance); auto = colour cameras
    "banding": False,          # row/column banding reduction on the calibrated frames
    "banding_axis": "rows",
    "banding_amount": 1.0,
    "comet": False,            # comet stacking: comet_first_xy / comet_last_xy in reference-frame pixels
    "comet_first_xy": None,
    "comet_last_xy": None,
    "comet_radius": 40,
    "grouping_keywords": [],   # WBPP post-calibration grouping: FITS keywords (or NIGHT) that split the lights
                               # into separate groups/integrations, e.g. ["TELESCOP"] or ["NIGHT"]
    "calibration_keywords": [],  # WBPP pre-calibration grouping: calibration frames must share these values with
                                 # the lights when some of them do (e.g. ["ROTATOR"] for flats per rotation)
    "group_settings": {},      # per light group overrides {group label or index: {setting: value}}
    "all_groups": False,       # integrate every light group (each with its own settings), not only the chosen one
    "pedestal": 0,             # output pedestal in ADU added to the calibrated lights; "auto" = just enough so that
                               # no more than 0.01% of the pixels go below zero
    "large_scale_rejection": None,  # {"high": layers, "low": layers, "growth": px}: PixInsight LSPR (planes, glints)
    "plate_solve": "auto",     # solve the integration (ASTAP) and write the WCS in its FITS header; auto = when
                               # ASTAP is configured and the headers have RA/DEC hints
    "local_norm_reference": "integration",  # integration = reference made of the best frames (WBPP) | single
    "local_norm_ref_frames": 8,
    "mode": "single",          # single | multi (several cameras merged) | hybrid (colour + narrowband)
    "sources": [],             # multi / hybrid: [{"name", "role", "lights", "calibration", "master_dirs", "file"}]
}


# seconds per frame and megapixel of each stage when no run on this PC was timed yet (rough, 8-core CPU)
_DEFAULT_RATES = {"masters": 0.012, "calibrate": 0.05, "register": 0.06, "normalize": 0.08,
                  "integrate": 0.03, "drizzle": 0.25, "solve": 25.0}
_GPU_FACTOR = {"integrate": 0.08, "drizzle": 0.1}


def _timings_path():
    """~/.astroforge/timings.json (ASTROFORGE_TIMINGS overrides it: the tests use a temporary file)."""
    return Path(os.environ.get("ASTROFORGE_TIMINGS") or Path.home() / ".astroforge" / "timings.json")


def _rates(gpu_on):
    """Per frame-megapixel stage rates, from ~/.astroforge/timings.json (median of the runs with the same GPU
    setting) when there are any, else the defaults."""
    rates = {k: v * (_GPU_FACTOR.get(k, 1.0) if gpu_on else 1.0) for k, v in _DEFAULT_RATES.items()}
    source = "valori indicativi (nessuno stacking misurato su questo PC)"
    try:
        hist = json.loads(_timings_path().read_text(encoding="utf-8"))
        hist = [h for h in hist if h.get("gpu") == gpu_on and h.get("frames") and h.get("mpx")]
        if hist:
            for k in rates:
                if k == "solve":
                    vals = [h["timings"][k] for h in hist if k in h.get("timings", {})]
                else:
                    vals = [h["timings"][k] / (h["frames"] * h["mpx"] * ((h.get("drizzle") or 1) ** 2 / 4 if k == "drizzle" else 1))
                            for h in hist if k in h.get("timings", {})]
                if vals:
                    rates[k] = float(np.median(vals))
            source = f"tempi misurati in {len(hist)} stacking precedenti su questo PC"
    except Exception:
        pass
    rates["source"] = source
    return rates


class Cancelled(Exception):
    pass


def cfa_noise(cfa, pattern):
    """Per-channel (R, G, B) noise measured on the raw CFA sites."""
    idx = {"R": 0, "G": 1, "B": 2}
    acc = [[], [], []]
    for i, ch in enumerate(pattern):
        plane = np.ascontiguousarray(cfa[i // 2::2, i % 2::2])
        hp = plane - cv2.GaussianBlur(plane, (0, 0), 2)
        acc[idx[ch]].append(float(np.median(np.abs(hp[::2, ::2]))) * 1.4826 * 1.25)
    return [float(np.mean(a)) for a in acc]


class Runner:
    def __init__(self, settings, progress=None, log=print, cancel=None):
        self.s = {**DEFAULTS, **settings}
        self.progress = progress or (lambda stage, frac, msg="": None)
        self.log = log
        self.cancel = cancel or (lambda: False)
        if "gpu" in settings:
            gpu.set_enabled(settings["gpu"])
        self.work = Path(self.s["work_dir"])
        self.work.mkdir(parents=True, exist_ok=True)
        self.report = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def _check(self):
        if self.cancel():
            raise Cancelled()

    def _step(self, stage, frac, msg=""):
        self._check()
        now = time.time()
        last = getattr(self, "_stage", None)
        if last is None or last[0] != stage:
            if last is not None:
                t = self.report.setdefault("timings", {})
                t[last[0]] = round(t.get(last[0], 0.0) + now - last[1], 2)
            self._stage = (stage, now)
        self.progress(stage, frac, msg)
        if msg:
            self.log(msg)

    # ------------------------------------------------------------ scan
    def scan(self):
        found = []
        kws = list(self.s.get("grouping_keywords") or []) + list(self.s.get("calibration_keywords") or [])
        for src in self.s["lights"] + self.s["calibration"]:
            p = Path(src)
            found += [frames.read_frame(p, kws)] if p.is_file() else frames.scan(p, keywords=kws)
        light_roots = [str(Path(x)) for x in self.s["lights"]]
        under = lambda p, r: p == r or p.startswith(r.rstrip("\\/") + os.sep) or p.startswith(r.rstrip("\\/") + "/")
        lights = [f for f in found if f.type == "light" and any(under(f.path, r) for r in light_roots)]
        # bias/dark/flat unticked on the Sorgenti page never reach a master (lights are filtered per group later)
        excluded = set(self.s.get("exclude") or ())
        calib = [f for f in found if f.type != "light" and f.path not in excluded]
        groups = frames.group(lights + calib, self.s.get("grouping_keywords") or ())
        tol = self.s.get("exposure_tolerance") or 0
        if tol > 0:
            # WBPP "exposure tolerance": same setup and filter, exposures within tol seconds -> one integration
            # (each exposure is still calibrated with its own dark)
            merged = {}
            for k, fs in sorted(groups["light"].items(), key=lambda kv: -len(kv[1])):
                home = next((m for m in merged if m[:7] == k[:7] and m[8:] == k[8:] and abs(m[7] - k[7]) <= tol), None)
                merged.setdefault(home or k, []).extend(fs)
            groups["light"] = merged
        light_groups = sorted(groups["light"].items(), key=lambda kv: -len(kv[1]))
        return {"groups": groups, "light_groups": light_groups}

    # ------------------------------------------------------------ masters
    def _master_path(self, kind, paths, depends=()):
        """Cache file of a master: keyed on its frames AND on what it was built with (e.g. the flat's
        bias/darkflat), so adding darkflats later does not silently reuse a flat built without them."""
        ident = "|".join(sorted(paths))
        if depends:
            ident += "|built-with|" + "|".join(str(d) for d in depends)
        h = hashlib.sha1(ident.encode()).hexdigest()[:10]
        return self.work / "masters" / f"master_{kind}_{h}.xisf"

    def _kw_filter(self, fs):
        """WBPP pre-calibration grouping: keep the calibration frames with the lights' keyword values."""
        want = getattr(self, "_light_kw", None)
        if not want:
            return fs
        same = [f for f in fs if all(f.keywords.get(k, "") == v for k, v in want.items() if f.keywords.get(k, ""))]
        return same

    def _pick(self, groups, kind, key, exptime=None):
        """Calibration group compatible with the lights (same setup, exposure for darks/darkflats)."""
        cands = []
        for k, fs in groups[kind].items():
            fs = self._kw_filter(fs)
            if not fs:
                continue
            same = k[:7] == key[:7] or (k[0] == key[0] and k[1:4] == key[1:4] and k[4:6] == key[4:6])
            if not same:
                continue
            if kind in ("dark", "darkflat") and exptime and abs(k[7] - exptime) > 0.01 * exptime:
                continue
            if kind == "flat" and len(key) > 8 and k[7] != key[8]:   # flats must match the filter
                continue
            cands.append((len(fs), k, fs))
        if not cands and kind == "dark":
            # no exact exposure: use the largest dark group with the same setup (scaled later)
            cands = [(len(fs), k, fs) for k, fs in groups[kind].items() if k[1:6] == key[1:6]]
        if not cands and kind == "flat" and len(key) > 8:
            # flats without a FILTER keyword (or lights with one the flats lack, e.g. an OSC with a
            # clip-in L-eNhance): better than no flat at all, but say so
            cands = [(len(fs), k, fs) for k, fs in groups[kind].items()
                     if k[1:6] == key[1:6] and (not k[7] or not key[8])]
            if cands:
                self.log(f"WARNING: no flat with filter '{key[8]}': using flats with filter "
                         f"'{max(cands, key=lambda c: c[0])[1][7] or 'none'}'")
        if kind == "bias" and len(cands) > 1:
            # several bias exposures (e.g. 0.001 s and 0.3 s): the shortest is the real bias (no dark current)
            def exp(c):
                return c[1][7] if len(c[1]) > 7 and c[1][7] is not None else 0.0
            cands.sort(key=lambda c: (exp(c), -c[0]))
            if len({exp(c) for c in cands}) > 1:
                self.log(f"bias: {len(cands)} groups ({', '.join(f'{exp(c):g} s x{c[0]}' for c in cands)}), "
                         f"using the shortest ({exp(cands[0]):g} s)")
            return cands[0][2]
        # compare only the frame counts: keys may hold None (no GAIN/temperature) and are not orderable
        return max(cands, key=lambda c: c[0])[2] if cands else []

    @property
    def library(self):
        if not self.s["master_reuse"]:
            return None
        if getattr(self, "_lib", None) is None:
            from . import masterlib
            # ready masters may sit in the calibration folders next to raw frames: they are indexed too
            self._lib = masterlib.MasterLibrary(self.s["master_library"], self.s["master_dirs"], log=self.log,
                                                strict_extra=[c for c in self.s.get("calibration") or []])
        return self._lib

    def _mopts(self, kind):
        from .makemasters import type_options
        return type_options(self.s.get("master_options") or {}, kind)

    def _build_or_load(self, kind, fs, depends=(), **kw):
        from .makemasters import TYPE_DEFAULTS, combine_kw, describe, superbias
        paths = [f.path for f in fs]
        t = self._mopts(kind)
        ckw = combine_kw(t)
        if t != TYPE_DEFAULTS[kind]:          # built another way: its own cache file
            depends = tuple(depends) + (describe(t) + (f" superbias {t['superbias_scale']}" if t.get("superbias") else ""),)
        kw = {**ckw, **kw}
        mp = self._master_path(kind, paths, depends)
        if mp.exists():
            self.log(f"master {kind}: reused {mp.name}")
            return calibrate.load_master(mp), mp
        self._step("masters", 0.2 if kind != "flat" else 0.6, f"building master {kind} from {len(paths)} frames")
        self.log(f"master {kind}: {describe(t)}")
        master = calibrate.build_master(paths, kind, log=self.log, **kw)
        if kind == "bias" and t.get("superbias"):
            master = superbias(master, t["superbias_scale"])
            self.log(f"master bias: superbias (sigma {t['superbias_scale']:g} px)")
        mp.parent.mkdir(parents=True, exist_ok=True)
        calibrate.save_master(master, mp)
        if self.library is not None:
            try:      # the library is a convenience: never let it break a run
                self.library.add(kind, master, fs, {"depends": [str(d) for d in depends if d],
                                                    "cfa_normalize": bool(kw.get("cfa_normalize"))})
            except Exception as e:
                self.log(f"master library: not saved ({e})")
        return master, mp

    def _from_library(self, kind, light_key, **kw):
        """A master from the library / #MASTERS folders for a session without that kind of frames."""
        lib = self.library
        if lib is None:
            return None
        s = self.s
        match = lib.find(kind, light_key, max_age_days=s["master_max_age_days"],
                         allow_flagged=s["master_allow_flagged"], **kw)
        if match is None:
            return None
        self.log(f"master {kind} from the library: {Path(match.path).name}"
                 + (f" ({'; '.join(match.reasons)})" if match.reasons else ""))
        self.report.setdefault("library_masters", {})[kind] = {"path": match.path, "reasons": match.reasons}
        return match

    def masters(self, groups, light_key, light_exp, lights=None):
        s, m, ids = self.s, {}, {}
        explicit = s["masters"]
        for kind in ("bias", "dark"):
            if explicit.get(kind):
                m[kind], ids[kind] = calibrate.load_master(explicit[kind]), explicit[kind]
                self.log(f"master {kind}: {explicit[kind]}")
                if kind == "dark" and not self.report.get("dark_exptime"):
                    dexp = frames.master_exptime(explicit[kind])        # so a 120 s dark is scaled to 300 s lights
                    if dexp:
                        self.report["dark_exptime"] = dexp
                continue
            fs = self._pick(groups, kind, light_key, light_exp if kind == "dark" else None)
            if not fs:
                match = self._from_library(kind, light_key, **({"exptime": light_exp, "scalable": "bias" in m}
                                                               if kind == "dark" else {}))
                if match:
                    m[kind], ids[kind] = self.library.load(match.entry), match.path
                    if kind == "dark":
                        self.report["dark_exptime"] = match.entry.get("exptime") or light_exp
                else:
                    self.log(f"no {kind} frames found")
                continue
            m[kind], ids[kind] = self._build_or_load(kind, fs)
            if kind == "dark":
                self.report["dark_exptime"] = fs[0].exptime
        if explicit.get("flat"):
            m["flat"] = calibrate.load_master(explicit["flat"])
            return m
        flats = self._pick(groups, "flat", light_key)
        cfa = s["flat_cfa_normalize"] is True or (s["flat_cfa_normalize"] == "auto" and bool(
            (lights and lights[0].bayer) or s.get("bayer_pattern")))
        if not flats:
            match = self._from_library("flat", light_key, filter=light_key[8] if len(light_key) > 8 else None,
                                       night=frames.night(lights[0]) if lights else None)
            if match:
                m["flat"] = self.library.load(match.entry, cfa_normalize=cfa)
            else:
                self.log("no flat frames found")
            return m
        # dust moves between nights: with lights and flats from several nights, one master flat per night
        by_night = {}
        for f in flats:
            by_night.setdefault(frames.night(f), []).append(f)
        light_nights = {frames.night(f) for f in lights or []}
        per_session = (self.s.get("flat_per_session", True) and len(by_night) > 1 and len(light_nights) > 1
                       and "" not in by_night)
        sets = by_night if per_session else {"": flats}
        if (not per_session and len(by_night) > 1 and "" not in by_night and light_nights
                and "" not in light_nights):
            # lights of one night, flats of several (a whole #MASTERS folder): the flats of the closest night,
            # never a mix (and never the outlier-exposure filter picking the wrong night's set)
            from datetime import date
            ln = date.fromisoformat(sorted(light_nights)[len(light_nights) // 2])
            closest = min(by_night, key=lambda n: (abs((date.fromisoformat(n) - ln).days), -len(by_night[n])))
            self.log("flat: flats of " + ", ".join(f"{n} ({len(v)})" for n, v in sorted(by_night.items()))
                     + f": using {closest}, the night closest to the lights ({ln.isoformat()})")
            sets = {closest: by_night[closest]}
        built = {}
        tf = self._mopts("flat")
        if tf["normalize"] != "auto":
            cfa = tf["normalize"] == "cfa"
        for night_, fs in sets.items():
            bad = frames.outlier_exposures(fs) if tf["drop_odd_exposures"] else []
            if bad:
                self.log(f"flat {night_}: ignoring {len(bad)} frames with a different exposure: "
                         + ", ".join(Path(f.path).name for f in bad))
            fs = [f for f in fs if f not in bad]
            # darkflats must match the exposure of THESE flats (per-filter flats often differ: 0.5 s Ha, 3 s OIII)
            darkflat, df_id = None, None
            if tf["calibrate_with"] in ("bias", "none"):
                pass                          # flats calibrated with the bias only (or nothing)
            elif explicit.get("darkflat"):
                darkflat, df_id = calibrate.load_master(explicit["darkflat"]), explicit["darkflat"]
            else:
                flat_exp = sorted(f.exptime for f in fs)[len(fs) // 2]
                dfs = self._pick(groups, "darkflat", light_key, flat_exp)
                if dfs:
                    darkflat, df_id = self._build_or_load("darkflat", dfs)
                elif groups["darkflat"]:
                    self.log(f"darkflat: none with the flats' exposure ({flat_exp:g} s), using bias for the flats")
            if darkflat is not None:
                m["darkflat"] = darkflat
            fbias = None if tf["calibrate_with"] == "none" else m.get("bias")
            built[night_], _ = self._build_or_load("flat", fs, (ids.get("bias") if fbias is not None else None, df_id)
                                                   + (("cfa",) if cfa else ()),
                                                   bias=fbias, dark=darkflat, cfa_normalize=cfa)
        main = max(sets, key=lambda n: len(sets[n]))
        m["flat"] = built[main]
        if per_session:
            m["flat_sessions"] = built
            self.log("flat per session: " + ", ".join(f"{n} ({len(sets[n])} flats)" for n in sorted(sets)))
        return m

    def _dark_terms(self, groups, light_key, lights, m, area):
        """What to subtract from each light, per exposure: bias + k * (dark - bias) [+ amp glow term].
        k = exposure ratio, or the noise-optimal scale (dark optimization)."""
        s, terms, info = self.s, {}, {}
        crop = (lambda a: a[area[0]:area[1], area[2]:area[3]]) if area else (lambda a: a)
        bias = m.get("bias")
        for exp in sorted({f.exptime for f in lights}):
            if exp == lights[0].exptime or s["masters"].get("dark"):
                dark, dexp = m.get("dark"), self.report.get("dark_exptime")
            else:
                fs = self._pick(groups, "dark", light_key[:7] + (exp,) + light_key[8:], exp)
                dark, dexp = (crop(self._build_or_load("dark", fs)[0]), fs[0].exptime) if fs else (None, None)
            if dark is None:
                terms[exp] = bias
                if groups["dark"] or s["masters"].get("dark"):
                    self.log(f"WARNING: no dark for the {exp:g} s lights")
                continue
            ratio = exp / dexp if dexp else 1.0
            if bias is None:
                if abs(ratio - 1) > 0.01:
                    # the dark holds the bias too: scaling it would scale the bias, so it is used as is
                    self.log(f"WARNING: darks are {dexp:g} s, lights {exp:g} s and there is no bias: the dark "
                             f"cannot be scaled and is subtracted unscaled. Add bias frames or matching darks.")
                terms[exp] = dark
                continue
            opt = s["dark_optimize"] is True or (s["dark_optimize"] == "auto" and abs(ratio - 1) > 0.01)
            k, k_glow, glow = ratio, None, None
            if opt:
                raw0 = crop(calibrate.load_raw(next(f for f in lights if f.exptime == exp).path)[0])
                res = darkopt.optimal_scale(raw0, dark, bias, glow=s["dark_glow"])
                k, k_glow, glow = res if s["dark_glow"] else (res, None, None)
            term = bias + k * (dark - bias)
            if glow is not None and glow.any() and k_glow != k:
                term = term + (k_glow - k) * darkopt.glow_component(dark, bias, glow)
            terms[exp] = term.astype(np.float32)
            info[f"{exp:g}"] = {"dark_exptime": dexp, "k": k, "k_glow": k_glow, "optimized": opt,
                                "glow_area": float(glow.mean()) if glow is not None else 0.0}
            self.log(f"dark for {exp:g} s lights: {dexp:g} s master, "
                     + (f"optimized scale {k:.3f} (exposure ratio {ratio:.3f})" if opt else f"scale {k:.3f}")
                     + (f", amp glow {glow.mean():.1%} of the frame, scale {k_glow:.3f}" if glow is not None and glow.any() else ""))
        main = info.get(f"{lights[0].exptime:g}")
        if main:
            self.report["dark_scale"] = main["k"]
            if main["optimized"]:
                self.report["dark_optimize"] = {"k": main["k"], "k_glow": main["k_glow"], "glow_area": main["glow_area"]}
        self.report["darks"] = info
        return terms

    @staticmethod
    def _session_flat(m, f):
        """Master flat for a light: same night, else the nearest night with flats."""
        sessions = m.get("flat_sessions")
        if not sessions:
            return m.get("flat")
        n = frames.night(f)
        if n in sessions:
            return sessions[n]
        from datetime import date
        d = date.fromisoformat(n) if n else None
        return sessions[min(sessions, key=lambda k: abs((date.fromisoformat(k) - d).days) if d else 0)]

    # ------------------------------------------------------------ main
    def run(self):
        t0 = time.time()
        s = self.s
        self._step("scan", 0.0, "scanning frames")
        sc = self.scan()
        if not sc["light_groups"]:
            raise ValueError("no light frames found")
        gi = s["light_group"] or 0
        light_key, lights = sc["light_groups"][gi]
        label = frames.describe_key(light_key)
        ov = (s.get("group_settings") or {}).get(label) or (s.get("group_settings") or {}).get(str(gi))
        if ov:
            s.update({k: v for k, v in ov.items() if k in DEFAULTS and k not in ("lights", "calibration", "sources")})
            self.log(f"group settings for {label}: " + ", ".join(f"{k}={v}" for k, v in ov.items()))
        ckw = [str(k).strip().upper() for k in s.get("calibration_keywords") or [] if str(k).strip()]
        self._light_kw = {k: lights[0].keywords.get(k, "") for k in ckw if lights[0].keywords.get(k, "")}
        if self._light_kw:
            self.log("calibration keywords: " + ", ".join(f"{k}={v}" for k, v in self._light_kw.items()))
        for k, fs in sc["light_groups"]:
            tag = "USED" if k == light_key else "skipped"
            self.log(f"lights [{tag}] {len(fs)} x {frames.describe_key(k)}")
        excluded = set(s["exclude"])
        lights = [f for f in lights if f.path not in excluded]
        if len(lights) < 2:
            raise ValueError(f"only {len(lights)} light frame(s) left after the exclusions: need at least 2")
        pattern = lights[0].bayer or s.get("bayer_pattern") or None
        self.report.update(light_setup=frames.describe_key(light_key), lights_total=len(lights), bayer=pattern)

        m = self.masters(sc["groups"], light_key, lights[0].exptime, lights)
        # the Bayer checks need the colour differences between the 2x2 sites: a raw frame has them, the
        # master flat does not any more once it is CFA-normalised (every site ~1 -> wrong pattern kept)
        probe, probe_h = calibrate.load_raw(lights[0].path)
        flipped = str(probe_h.get("ROWORDER", "TOP-DOWN")).upper() == "BOTTOM-UP"
        if not pattern and not lights[0].filter:
            # colour camera whose software did not write BAYERPAT: stacking it as mono gives a grey,
            # checkered image. The 2x2 sites of a CFA have clearly different levels (G vs R/B).
            if calibrate.looks_like_cfa(probe):
                pattern, _ = calibrate.verify_pattern("RGGB", probe)
                self.log(f"WARNING: no BAYERPAT in the headers but the data look like a colour (CFA) sensor: "
                         f"debayering as {pattern}. If red and blue are swapped set 'bayer_pattern'.")
        if pattern:
            if s.get("bayer_pattern"):
                pattern = s["bayer_pattern"]
                self.log(f"Bayer pattern forced to {pattern}")
            else:
                fixed, changed = calibrate.verify_pattern(pattern, probe)
                if changed:
                    self.log(f"Bayer pattern: header says {pattern} but green sites say {fixed}; using {fixed}")
                    if flipped:
                        # the greens alone cannot tell a horizontal shift (APT + QHY168C) from a vertical one (a
                        # BAYERPAT written for the bottom-up stored order): the horizontal one is assumed
                        self.log(f"WARNING: the file is stored bottom-up (ROWORDER): if red and blue come out "
                                 f"swapped set bayer_pattern = {calibrate.vertical_shift(fixed)}")
                pattern = fixed
            self.report["bayer"] = pattern
        # crop overscan / unexposed borders (e.g. ~170 black columns on the QHY168C) from everything
        area = calibrate.active_area(m.get("flat"), None if "flat" in m else calibrate.load_raw(lights[0].path)[0])
        full_shape = (lights[0].height, lights[0].width)
        if area and (area[1] - area[0], area[3] - area[2]) != full_shape:
            y0, y1, x0, x1 = area
            self.log(f"active sensor area: x {x0}-{x1}, y {y0}-{y1} (overscan removed)")
            m = {k: ({n: f[y0:y1, x0:x1] for n, f in v.items()} if isinstance(v, dict) else v[y0:y1, x0:x1])
                 for k, v in m.items()}
        else:
            area = None
        self.report["active_area"] = area
        dark_terms = self._dark_terms(sc["groups"], light_key, lights, m, area)
        dark_signal = m["dark"] - m["bias"] if "dark" in m and "bias" in m else m.get("dark")
        bad = calibrate.defect_map(dark_signal, m.get("flat")) if s["cosmetic"] else None
        if s.get("defect_map") and s["cosmetic"]:
            # saved map (Crea master): hot/dead pixels and bad columns of the sensor, added to the computed one
            try:
                from astropy.io import fits as _fits
                saved = _fits.getdata(s["defect_map"]).astype(bool)
                if area:
                    saved = saved[area[0]:area[1], area[2]:area[3]]
                if bad is None or saved.shape == bad.shape:
                    bad = saved if bad is None else (bad | saved)
                    self.log(f"defect map {Path(s['defect_map']).name}: {int(saved.sum())} pixels")
                else:
                    self.log(f"WARNING: defect map {Path(s['defect_map']).name} has another size, ignored")
            except Exception as e:
                self.log(f"WARNING: defect map not readable ({e})")
        if bad is not None:
            self.log(f"cosmetic: {int(bad.sum())} defective pixels mapped")

        # --- calibrate + analyze
        # one folder per output: filters and mosaic panels keep their own frames (blink, charts, rerun)
        cal_dir = self.work / "calibrated" / s["output"]
        cal_dir.mkdir(parents=True, exist_ok=True)
        def calib_one_raw(i):
            f = lights[i]
            raw, _ = calibrate.load_raw(f.path)
            if area:
                raw = raw[area[0]:area[1], area[2]:area[3]]
            term = dark_terms.get(f.exptime)
            return calibrate.calibrate(raw - term if term is not None else raw, None, None, self._session_flat(m, f))

        def calib_one(i):
            cal = calib_one_raw(i)
            if ped:
                cal = cal + np.float32(ped)
            if bad is not None:
                cal = calibrate.fix_defects(cal, bad, 2 if pattern else 1)
            if s["banding"]:
                cal = calibrate.reduce_banding(cal, pattern, s["banding_axis"], s["banding_amount"])
            np.save(cal_dir / f"{i:04d}.npy", cal.astype(np.float32))
            lum = calibrate.superpixel_lum(cal) if pattern else cal
            lum = np.ascontiguousarray(lum, dtype=np.float32)
            st = analyze.detect(lum, max_stars=1500 if s["distortion"] else 600)
            # measure on the image the stars were found in, then move the coordinates to full resolution
            q = analyze.quality(lum, st, 2.0 if pattern else 1.0)
            if pattern:
                st["x"], st["y"] = st["x"] * 2 + 0.5, st["y"] * 2 + 0.5
            return st, q

        # frames are independent: calibrate and measure several at once (numpy/OpenCV release the GIL);
        # ~12 float32 copies of a frame are alive per worker
        self._lights_for_pedestal = lights
        ped = self._pedestal(calib_one_raw, s.get("pedestal") or 0)
        frame_bytes = lights[0].height * lights[0].width * 4 * 12
        workers = max(1, min(len(lights), (os.cpu_count() or 2) // 2, int(combine.available_memory() * 0.5 // frame_bytes)))
        results = [None] * len(lights)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(calib_one, i): i for i in range(len(lights))}
            for done, fu in enumerate(as_completed(futs), 1):
                i = futs[fu]
                results[i] = fu.result()
                self._step("calibrate", done / len(lights), f"calibrated {Path(lights[i].path).name} ({done}/{len(lights)})")
        stars, quals = [r[0] for r in results], [r[1] for r in results]
        analyze.score_frames(quals, s["fwhm_sigma"], s["ecc_max"])
        if s["weighting"] != "psf" or s["approval"] is not None:
            formula = "standard" if s["weighting"] == "psf" else s["weighting"]
            vals = subframes.apply(quals, formula, s["approval"])
            self.log(f"subframe weighting '{formula}': " + ", ".join(f"{v:.0f}" for v in vals if v is not None)
                     + (f" (approval > {s['approval']:g})" if s["approval"] is not None else ""))
        frame_table = []
        for idx, (f, q) in enumerate(zip(lights, quals)):
            frame_table.append({"file": Path(f.path).name, "path": f.path, "index": idx,
                                "calibrated": str(cal_dir / f"{idx:04d}.npy"), **q.to_dict()})
            self.log(f"  {Path(f.path).name}: stars {q.stars} FWHM {q.fwhm:.2f} ecc {q.eccentricity:.2f} "
                     f"SNR {q.snr:.0f} score {q.score:.0f} {'' if q.accepted else 'REJECTED: ' + q.reason}")
        self.report["frames"] = frame_table
        use = [i for i, q in enumerate(quals) if q.accepted or not s["auto_reject"]]
        if len(use) < 2:
            raise ValueError("fewer than 2 usable light frames")
        ref = max(use, key=lambda i: quals[i].score)
        self.log(f"reference frame: {Path(lights[ref].path).name}")
        self.header = fits.getheader(lights[ref].path)   # pointing/optics keywords for the outputs

        # --- register
        ref_xy = np.stack([stars[ref]["x"], stars[ref]["y"]], 1)
        transforms, distortions = {}, {}
        for n, i in enumerate(use):
            self._step("register", n / len(use), f"registering {Path(lights[i].path).name}")
            if i == ref:
                transforms[i] = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
                continue
            try:
                M, nmatch, rms = register.match(np.stack([stars[i]["x"], stars[i]["y"]], 1), ref_xy)
                transforms[i] = M
                msg = f"  matched {nmatch} stars, rms {rms:.2f}px"
                if s["distortion"]:
                    # field curvature / tilt / different optics between sessions: affine is not enough
                    try:
                        d = register.fit_distortion(np.stack([stars[i]["x"], stars[i]["y"]], 1), ref_xy, M)
                        gain = d.info["distortion_rms_improvement"]
                        if gain > 0.1:
                            distortions[i] = d
                            msg += f", distortion model {d.info['rms_affine']:.2f} -> {d.info['rms']:.2f}px"
                    except Exception as e:
                        msg += f" (distortion fit failed: {e})"
                self.log(msg)
            except Exception as e:
                self.log(f"  registration failed ({e}), frame skipped")
        use = [i for i in use if i in transforms]

        # --- debayer + warp + statistics
        reg_dir = self.work / "registered" / s["output"]
        reg_dir.mkdir(parents=True, exist_ok=True)
        H, W = np.load(cal_dir / f"{ref:04d}.npy", mmap_mode="r").shape[:2]
        norm = {}
        for n, i in enumerate(use):
            self._step("register", 0.5 + 0.5 * n / len(use), f"resampling frame {n + 1}/{len(use)}")
            cal = np.load(cal_dir / f"{i:04d}.npy")
            rgb = calibrate.debayer(cal, pattern, s["debayer"]) if pattern else cal[..., None]
            out = register.warp(rgb, transforms[i], (H, W), distortion=distortions.get(i))
            np.save(reg_dir / f"{i:04d}.npy", out.astype(np.float16) if s["registered_float16"] else out)
            st = []
            for c in range(out.shape[-1]):
                ch = out[::4, ::4, c]
                med = float(np.nanmedian(ch))
                mad = float(np.nanmedian(np.abs(ch - med))) * 1.4826
                hp = rgb[..., c] - cv2.GaussianBlur(rgb[..., c], (0, 0), 2)
                noise = float(np.median(np.abs(hp[::4, ::4]))) * 1.4826 * 1.25
                st.append((med, mad, noise))
            norm[i] = st
        r = norm[ref]
        so = {i: [(r[c][1] / max(norm[i][c][1], 1e-9), r[c][0] - norm[i][c][0] * r[c][1] / max(norm[i][c][1], 1e-9))
                  for c in range(len(r))] for i in use}
        weights = [quals[i].weight for i in use] if s["weights"] else None
        ln_models = {}
        if s["local_norm"]:
            # local normalization: smooth per-frame scale/offset maps onto the reference, so gradients that
            # change between frames (Moon, light pollution, clouds) do not end up in the integration
            # reference = the frame with the weakest gradient (not the sharpest): every frame is brought onto
            # its background, so the integration inherits it
            fstats = [localnorm.frame_stats(np.load(reg_dir / f"{i:04d}.npy", mmap_mode="r"), s["local_norm_scale"])
                      for i in use]
            ln_ref = use[localnorm.select_reference(fstats)]
            nref = int(s.get("local_norm_ref_frames") or 8)
            if s.get("local_norm_reference", "integration") == "integration" and len(use) >= 4 and nref >= 3:
                # WBPP: the reference is an integration of the best frames (weak gradient, high SNR), with
                # less noise than any single frame, so the normalization adds no noise of its own
                order = np.argsort([-sc_ for sc_ in localnorm.reference_scores(fstats)])
                best = [use[k] for k in order[:min(nref, len(use))]]

                class _Norm:
                    height = H

                    def __call__(self, i, y0, y1):
                        a = np.load(reg_dir / f"{i:04d}.npy", mmap_mode="r")[y0:y1].astype(np.float32)
                        k = np.asarray(so[i], np.float32)
                        return a * k[:, 0] + k[:, 1]
                self._step("normalize", 0.0, f"local normalization reference: integrating the {len(best)} best frames")
                ref_img, _, _ = combine.combine_chunked(best, _Norm(), "sigma" if len(best) >= 5 else "median",
                                                        3.0, 3.0, rows=128)
                ref_img = ref_img.astype(np.float32)
                ln_ref = None
                self.log("local normalization reference: integration of " + ", ".join(Path(lights[i].path).name for i in best))
            else:
                self.log(f"local normalization reference: {Path(lights[ln_ref].path).name}")
                ref_img = np.load(reg_dir / f"{ln_ref:04d}.npy").astype(np.float32)
            for n, i in enumerate(use):
                if i == ln_ref:
                    continue
                self._step("normalize", n / len(use), f"local normalization {n + 1}/{len(use)}")
                ln_models[i] = localnorm.fit(np.load(reg_dir / f"{i:04d}.npy").astype(np.float32), ref_img,
                                             s["local_norm_scale"])
            del ref_img

        # --- integrate (1x)
        nch = 3 if pattern else 1
        self._step("integrate", 0.0, f"integrating {len(use)} frames, rejection {s['rejection']}")

        class Loader:
            height = H

            def __call__(self, i, y0, y1):
                a = np.load(reg_dir / f"{i:04d}.npy", mmap_mode="r")[y0:y1].astype(np.float32)
                if ln_models:
                    a = localnorm.apply(a, ln_models[i], y0, y1) if i in ln_models else a
                else:
                    k = np.asarray(so[i], np.float32)
                    a = a * k[:, 0] + k[:, 1]
                if i in trails:
                    a[np.load(trails[i], mmap_mode="r")[y0:y1]] = np.nan
                return a

        def prog(msg):
            self._step("integrate", 0.5, msg)

        trails = {}
        sat = s["satellites"]
        if sat is True or (sat == "auto" and len(use) < 20):
            # with few frames per-pixel rejection misses faint trails: find them as lines in each frame's
            # residual against a median of all frames, and leave them out of the integration
            self._step("integrate", 0.0, "looking for satellite / airplane trails")
            med, _, _ = combine.combine_chunked(use, Loader(), "median", rows=128)
            tr_dir = self.work / "trails" / s["output"]
            tr_dir.mkdir(parents=True, exist_ok=True)
            for i in use:
                frame = Loader()(i, 0, H)
                k = (np.asarray(so[i], np.float32)[:, 0] if not ln_models else np.ones(len(norm[i]), np.float32))
                noise = np.array([norm[i][c][2] for c in range(len(norm[i]))], np.float32) * k
                mask = satellites.trail_mask(frame, med, noise)
                if mask.any():
                    np.save(tr_dir / f"{i:04d}.npy", mask)
                    trails[i] = tr_dir / f"{i:04d}.npy"
                    self.log(f"  {Path(lights[i].path).name}: trail found, {mask.mean():.2%} of the frame masked")
            del med
            self.report["trails"] = {Path(lights[i].path).name: str(p_) for i, p_ in trails.items()}

        lsr = s.get("large_scale_rejection") or None
        if lsr:
            self.log(f"large-scale pixel rejection: high {lsr.get('high') or 0} layers, low {lsr.get('low') or 0}, "
                     f"growth {lsr.get('growth', 2)} px")
        integ, rl, rh = combine.combine_chunked(use, Loader(), s["rejection"], s["reject_low"], s["reject_high"],
                                                weights, rows=128, log=prog, grow_px=s["reject_grow"],
                                                **({"large_scale": lsr} if lsr else {}))
        integ = np.nan_to_num(integ, nan=0.0)
        if s["comet"] and s["comet_first_xy"]:
            integ = self._comet(integ, use, ref, lights, transforms, cal_dir, pattern, so, weights, H, W)
        integ_full = integ          # drizzle outlier rejection works in uncropped reference coordinates
        # keep only the area covered by (almost) every frame: rotated sessions or meridian flips
        # leave borders built from few frames, noisy and with different gradients
        crop = None
        if s.get("coverage_crop", 0.9):
            cov, step = register.coverage([transforms[i] for i in use], (H, W), (H, W))
            y0, y1, x0, x1 = register.best_rect(cov >= s.get("coverage_crop", 0.9))
            crop = (y0 * step, min(H, y1 * step), x0 * step, min(W, x1 * step))
            if crop != (0, H, 0, W):
                self.log(f"coverage crop: x {crop[2]}-{crop[3]}, y {crop[0]}-{crop[1]} "
                         f"(area covered by >= {s.get('coverage_crop', 0.9):.0%} of the frames)")
                integ = integ[crop[0]:crop[1], crop[2]:crop[3]]
            self.report["coverage_crop"] = crop
        out_base = self.work / s["output"]
        xisf.write(integ, f"{out_base}_integration.xisf")
        io.save_fits(integ, f"{out_base}_integration.fits", self.header)
        self._preview(integ, f"{out_base}_integration_preview.jpg")
        self.report["rejected_pct"] = float((rl.sum() + rh.sum()) / (rl.size * len(use)) * 100)
        # rejection maps (how many frames were rejected per pixel, low and high), as PixInsight writes them
        for name, rmap in (("low", rl), ("high", rh)):
            rmap = np.asarray(rmap, np.float32)
            if crop and crop != (0, H, 0, W):
                rmap = rmap[crop[0]:crop[1], crop[2]:crop[3]]
            io.save_fits(rmap if rmap.ndim == 3 else rmap[..., None], f"{out_base}_rejection_{name}.fits")
        self.log(f"integration done, {self.report['rejected_pct']:.2f}% pixels rejected")
        result = integ
        self._solve_output(integ, f"{out_base}_integration.fits", self.header)

        # --- drizzle
        if s["drizzle"] and s["drizzle"] > 1:
            bayer = bool(pattern) and s["bayer_drizzle"]
            bg = [float(np.median(integ_full[::8, ::8, c])) for c in range(nch)]
            devs = gpu.devices()
            # one accumulator per GPU (frames are shared out dynamically), or one on the CPU
            kern = s["drizzle_kernel"]
            parts = {d[0]: drizzle.Drizzle((H, W), s["drizzle"], s["pixfrac"], nch, device=d[0], kernel=kern)
                     for d in devs} or {None: drizzle.Drizzle((H, W), s["drizzle"], s["pixfrac"], nch, kernel=kern)}
            self.log(f"drizzle {s['drizzle']}x on {gpu.describe()}")
            done = [0]

            def drizzle_frame(i, dev=None):
                self._check()
                cal = np.load(cal_dir / f"{i:04d}.npy")
                src = cal if bayer else (calibrate.debayer(cal, pattern, s["debayer"]) if pattern else cal[..., None])
                # with local normalization the frames follow their LN model; the LN reference frame itself (single
                # frame reference) has none and goes in as it is, like in the 1x integration (not scaled to the
                # star reference: that gave it a wrong level and spurious rejections)
                so_i = [(1.0, 0.0)] * nch if ln_models and i not in ln_models else so[i]
                rej = None
                if s["drizzle_reject"]:
                    # noise must be measured on what is drizzled: raw CFA sites are noisier than
                    # debayered pixels (interpolation smooths them)
                    noise = cfa_noise(cal, pattern) if bayer else [norm[i][c][2] for c in range(nch)]
                    noise = [noise[c] * so_i[c][0] for c in range(nch)]
                    rej = drizzle.outlier_mask(src, transforms[i], integ_full, noise, bg, so_i,
                                               pattern if bayer else None, distortion=distortions.get(i),
                                               local_norm=ln_models.get(i))
                w = quals[i].weight if s["weights"] else 1.0
                parts[dev].add(src, transforms[i], w, pattern if bayer else None, so_i, rej,
                               distortion=distortions.get(i), local_norm=ln_models.get(i))
                done[0] += 1
                self.progress("drizzle", done[0] / len(use))
                self.log(f"  frame {done[0]}/{len(use)}" + (f" on GPU{dev}" if dev is not None else "")
                         + (f", {rej.mean() * 100:.3f}% pixels rejected" if rej is not None else ""))

            if devs:
                gpu.run_tasks(use, drizzle_frame)
                dz = drizzle.Drizzle.merge(list(parts.values()))
            else:
                for i in use:
                    drizzle_frame(i)
                dz = parts[None]
            # coverage-aware smoothing: in Bayer drizzle R and B come from 1/4 of the sites, G from 1/2
            sc_ = s["drizzle"]
            smooth = [0.5 * sc_, 0.3 * sc_, 0.5 * sc_] if bayer else 0.25 * sc_
            result, wmap = dz.result(smooth=smooth)
            if crop:
                k = s["drizzle"]
                result = result[crop[0] * k:crop[1] * k, crop[2] * k:crop[3] * k]
            xisf.write(result, f"{out_base}_drizzle{s['drizzle']}x.xisf")
            dz_hdr = self.header.copy()
            for key in ("XPIXSZ", "YPIXSZ"):     # drizzle output pixels are smaller: keeps the plate-solve hint right
                if key in dz_hdr:
                    dz_hdr[key] = float(dz_hdr[key]) / s["drizzle"]
            io.save_fits(result, f"{out_base}_drizzle{s['drizzle']}x.fits", dz_hdr)
            self._solve_output(result, f"{out_base}_drizzle{s['drizzle']}x.fits", dz_hdr)
            self._preview(result, f"{out_base}_drizzle{s['drizzle']}x_preview.jpg")
            self.log(f"drizzle done: {result.shape[1]}x{result.shape[0]}")

        if s["dualband"] and pattern:
            result = self._dualband(result, f"{out_base}{'_drizzle%dx' % s['drizzle'] if s['drizzle'] > 1 else ''}")
        self._step("done", 1.0)
        self.report["seconds"] = round(time.time() - t0, 1)
        self.report["output"] = str(out_base)
        self._save_timings(lights, use)
        with open(f"{out_base}_report.json", "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)
        self._step("done", 1.0, f"finished in {self.report['seconds']} s")
        return result, self.report

    def _comet(self, integ, use, ref, lights, transforms, cal_dir, pattern, so, weights, H, W):
        """Second integration aligned on the comet nucleus, merged with the star-aligned one."""
        s = self.s
        reg_dir = self.work / "registered" / s["output"]

        class Lum:   # lazy: one registered frame in memory at a time
            def __len__(self):
                return len(use)

            def __getitem__(self, n):
                a = np.load(reg_dir / f"{use[n]:04d}.npy", mmap_mode="r")
                return np.nan_to_num(np.asarray(a, np.float32).mean(-1))

        self._step("comet", 0.0, "measuring the comet nucleus in every frame")
        pos = comet.comet_positions(Lum(), None, s["comet_first_xy"], s["comet_last_xy"],
                                    [lights[i].date or None for i in use])
        r = use.index(ref)
        ct = comet.comet_transforms([transforms[i] for i in use], pos, r)
        self.log(f"comet moved {np.hypot(*(pos[-1] - pos[0])):.1f} px during the session")
        com_dir = self.work / "comet" / s["output"]
        com_dir.mkdir(parents=True, exist_ok=True)
        for n, i in enumerate(use):
            self._step("comet", 0.2 + 0.4 * n / len(use), f"resampling on the comet {n + 1}/{len(use)}")
            cal = np.load(cal_dir / f"{i:04d}.npy")
            rgb = calibrate.debayer(cal, pattern, s["debayer"]) if pattern else cal[..., None]
            np.save(com_dir / f"{i:04d}.npy", register.warp(rgb, ct[n], (H, W)))

        class Loader:
            height = H

            def __call__(self, i, y0, y1):
                a = np.load(com_dir / f"{i:04d}.npy", mmap_mode="r")[y0:y1].astype(np.float32)
                k = np.asarray(so[i], np.float32)
                return a * k[:, 0] + k[:, 1]

        # the stars move in this stack: a strong rejection removes their trails
        cint, _, _ = combine.combine_chunked(use, Loader(), "sigma" if len(use) >= 6 else "percentile",
                                             s["reject_low"], 2.0, weights, rows=128)
        cint = np.nan_to_num(cint, nan=0.0)
        out_base = self.work / s["output"]
        io.save_fits(cint, f"{out_base}_comet.fits", self.header)
        both = comet.combine_star_comet(integ, cint, s["comet_radius"], pos[r], trail=pos)
        io.save_fits(both, f"{out_base}_stars_comet.fits", self.header)
        self.report["comet"] = {"positions": pos.tolist(), "comet_stack": f"{out_base}_comet.fits"}
        return both.astype(np.float32)

    def _dualband(self, rgb, base):
        """OSC + dual-band filter: Ha from the red sites, OIII from green + blue, then HOO."""
        s = self.s
        ha, o = dualband._solve(rgb[..., 0], rgb[..., 1], rgb[..., 2], s["dualband_ha_leak_g"], 0.0, 0.0,
                                s["dualband_b_gain"], 1 / 3)
        io.save_fits(ha[..., None], f"{base}_Ha.fits", self.header)
        io.save_fits(o[..., None], f"{base}_OIII.fits", self.header)
        hoo, _ = channels.combine({"Ha": ha, "OIII": o}, "HOO")
        xisf.write(hoo, f"{base}_HOO.xisf")
        io.save_fits(hoo, f"{base}_HOO.fits", self.header, {"PALETTE": "HOO"})
        self._preview(hoo, f"{base}_HOO_preview.jpg")
        self.report["dualband"] = {"Ha": f"{base}_Ha.fits", "OIII": f"{base}_OIII.fits", "HOO": f"{base}_HOO.fits"}
        self.log(f"dual-band: Ha and OIII extracted, HOO saved to {Path(base).name}_HOO.fits")
        return hoo

    @staticmethod
    def _run_summary(name, rep):
        """What the GUI needs to show one filter / panel of a multi-run job (frames table, charts, blink)."""
        keep = ("frames", "output", "light_setup", "lights_total", "rejected_pct", "trails", "darks", "seconds")
        return {"name": name, **{k: rep[k] for k in keep if k in rep}}

    def run_mosaic(self, panels):
        """Mosaic: stack each panel (a list of light folders/files per panel), register the panels on the stars
        of their overlaps, match their backgrounds and feather-blend them on one canvas."""
        from . import mosaic
        base_output, images, outs, headers, runs = self.s["output"], [], [], [], []
        base = dict(self.s)                  # every panel starts from the same settings
        for k, panel in enumerate(panels, 1):
            self.log(f"===== panel {k}/{len(panels)} =====")
            self.s = {**base, "lights": list(panel) if isinstance(panel, (list, tuple)) else [panel],
                      "output": f"{base_output}_panel{k}", "light_group": None}
            self.report = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
            img, rep = self.run()
            images.append(img)
            outs.append(rep["output"])
            headers.append(getattr(self, "header", None))
            runs.append(self._run_summary(f"Pannello {k}", rep))
        self.s["output"] = base_output
        T = None
        from .. import astrometry, pipeline
        tools = pipeline.load_config().get("tools", {})
        hinted = all(h is not None and astrometry.header_hints(h, im.shape[0])[0] is not None
                     for h, im in zip(headers, images))
        if self.s["mosaic_wcs"] and tools.get("astap") and hinted:
            # plate solving places every panel on the sky: works with small overlaps where star matching fails
            # (only with RA/DEC in the headers: a blind solve of every panel would take minutes)
            self._step("mosaic", 0.2, "plate solving the panels")
            wcs = [astrometry.solve(im, tools, h, log=self.log) for im, h in zip(images, headers)]
            if all(w is not None for w in wcs):
                T = wcs
                self.log("mosaic: panels placed by plate solving")
                self.report["panel_centers"] = [astrometry.field_center(w, im.shape[:2]) for w, im in zip(wcs, images)]
            else:
                self.log(f"mosaic: {sum(w is None for w in wcs)} panel(s) not solved, using star matching")
        if T is None:
            self._step("mosaic", 0.3, "registering the panels on their overlaps")
            T = mosaic.register_panels(images)
        canvas, CT = mosaic.plan(T, [im.shape[:2] for im in images])
        self._step("mosaic", 0.6, f"blending {len(images)} panels on a {canvas[1]}x{canvas[0]} canvas")
        out = mosaic.blend(images, CT, canvas)
        out_base = self.work / f"{base_output}_mosaic"
        xisf.write(out, f"{out_base}.xisf")
        io.save_fits(out, f"{out_base}.fits")
        self._preview(out, f"{out_base}_preview.jpg")
        self.report.update(panels=outs, runs=runs, output=str(out_base), combined=f"{out_base}.fits")
        self._step("done", 1.0, f"mosaic of {len(images)} panels: {canvas[1]}x{canvas[0]}")
        return out, self.report

    def _pedestal(self, calib_raw, setting):
        """Output pedestal (fraction of full scale) from the setting in ADU, or "auto" from a few frames."""
        if setting in (0, None, "", False):
            return 0.0
        if setting == "auto":
            lows = []
            for i in range(min(3, len(self._lights_for_pedestal))):
                cal = calib_raw(i)
                lows.append(float(np.quantile(cal[::2, ::2], 1e-4)))
            adu = int(np.ceil(max(0.0, -min(lows)) * 65535))
            adu = min(adu + (5 if adu else 0), 500)
            self.log(f"pedestal auto: {adu} ADU" + ("" if adu else " (no negative pixels)"))
        else:
            adu = int(float(setting))
            self.log(f"pedestal: {adu} ADU")
        self.report["pedestal_adu"] = adu
        return adu / 65535.0

    def _solve_output(self, img, path, header):
        """Plate solve an output and rewrite its FITS with the WCS (SPCC, annotation, mosaics need no solving)."""
        mode = self.s.get("plate_solve", "auto")
        if not mode:
            return None
        from .. import astrometry, pipeline
        tools = pipeline.load_config().get("tools", {})
        if not tools.get("astap"):
            if mode is True:
                self.log("plate solve: ASTAP not configured")
            return None
        if mode == "auto" and (header is None or astrometry.header_hints(header, img.shape[0])[0] is None):
            self.log("plate solve: no RA/DEC in the headers, skipped (set plate_solve true for a blind solve)")
            return None
        self._step("solve", 0.0, f"plate solving {Path(path).name}")
        try:
            wcs = astrometry.solve(img, tools, header, log=self.log)
        except Exception as e:
            self.log(f"plate solve failed: {e}")
            return None
        if wcs is None:
            self.log("plate solve: not solved")
            return None
        wh = astrometry.wcs_to_header(wcs)
        extra = {k: wh[k] for k in wh if k not in ("COMMENT", "HISTORY", "")}
        extra["PEDESTAL"] = self.report.get("pedestal_adu") or None
        io.save_fits(img, path, header, extra)
        ra, dec = astrometry.field_center(wcs, img.shape[:2])
        rot, flipped = astrometry.rotation_deg(wcs)
        sol = {"ra": ra, "dec": dec, "scale": astrometry.pixel_scale_arcsec(wcs), "rotation": rot, "flipped": flipped}
        self.report.setdefault("wcs", {})[Path(path).name] = sol
        self.log(f"plate solve: RA {ra:.4f} DEC {dec:+.4f}, {sol['scale']:.3f}\"/px, rotation {sol['rotation']:.1f} deg "
                 f"-> WCS written in {Path(path).name}")
        return wcs

    @staticmethod
    def _will_solve(mode, lights):
        from .. import astrometry, pipeline
        if not pipeline.load_config().get("tools", {}).get("astap") or not lights:
            return False
        if mode is True:
            return True
        try:
            return astrometry.header_hints(fits.getheader(lights[0].path), lights[0].height)[0] is not None
        except Exception:
            return False

    def _save_timings(self, lights, use):
        """Stage times of this run (seconds per frame and megapixel) for the plan estimates."""
        try:
            mpx = lights[0].width * lights[0].height / 1e6
            rec = {"frames": len(use), "mpx": round(mpx, 2), "gpu": bool(gpu.devices()),
                   "drizzle": self.s.get("drizzle", 1), "timings": self.report.get("timings", {})}
            path = _timings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            hist = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            hist = (hist + [rec])[-30:]
            path.write_text(json.dumps(hist), encoding="utf-8")
        except Exception:
            pass

    def run_groups(self):
        """WBPP: integrate every light group (each with its own settings from group_settings)."""
        sc = self.scan()
        if not sc["light_groups"]:
            raise ValueError("no light frames found")
        base, outs, runs = dict(self.s), [], []
        for gi, (key, fs) in enumerate(sc["light_groups"]):
            if len(fs) < 2:
                self.log(f"group {gi + 1} {frames.describe_key(key)}: {len(fs)} frame, skipped")
                continue
            self.log(f"===== group {gi + 1}/{len(sc['light_groups'])}: {frames.describe_key(key)} =====")
            self.s = {**base, "light_group": gi, "output": f"{base['output']}_g{gi + 1}"}
            self.report = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
            self._stage = None
            _, rep = self.run()
            dz = self.s.get("drizzle", 1)
            outs.append(f"{rep['output']}_drizzle{dz}x.fits" if dz and dz > 1 else f"{rep['output']}_integration.fits")
            runs.append(self._run_summary(frames.describe_key(key), rep))
        self.s = base
        self.report = {"groups": outs, "runs": runs, "output": outs[0] if outs else None,
                       "combined": outs[0] if outs else None}
        return None, self.report

    def plan(self):
        """What a run will do, before running it: groups, frames, masters (build / cache / library / missing),
        stages and an estimated time (from the timings of previous runs on this PC, else rough defaults)."""
        sc = self.scan()
        s = self.s
        out = {"groups": [], "stages": [], "total_s": 0}
        if not sc["light_groups"]:
            out["error"] = "nessun light trovato"
            return out
        gis = [i for i, (_, fs) in enumerate(sc["light_groups"]) if len(fs) >= 2] if s.get("all_groups") else [s["light_group"] or 0]
        rates = _rates(bool(gpu.devices()))
        for gi in gis:
            key, lights = sc["light_groups"][gi]
            lights = [f for f in lights if f.path not in set(s["exclude"])]
            label = frames.describe_key(key)
            ov = (s.get("group_settings") or {}).get(label) or (s.get("group_settings") or {}).get(str(gi)) or {}
            g = {**s, **ov}
            ckw = [str(k).strip().upper() for k in g.get("calibration_keywords") or [] if str(k).strip()]
            self._light_kw = {k: lights[0].keywords.get(k, "") for k in ckw if lights and lights[0].keywords.get(k, "")}
            masters = []
            for kind in ("bias", "dark", "flat", "darkflat"):
                exp = lights[0].exptime if kind == "dark" else None
                if kind == "darkflat":
                    fl = self._pick(sc["groups"], "flat", key)
                    exp = sorted(f.exptime for f in fl)[len(fl) // 2] if fl else None
                    if exp is None:
                        continue
                fs = self._pick(sc["groups"], kind, key, exp)
                if fs:
                    cached = self._master_path(kind, [f.path for f in fs]).exists() if kind in ("bias", "dark") else False
                    masters.append({"kind": kind, "frames": len(fs), "how": "cache" if cached else "crea"})
                elif kind != "darkflat":
                    lib = None
                    try:
                        lib = self._from_library(kind, key, **({"exptime": lights[0].exptime} if kind == "dark" else {}))
                    except Exception:
                        lib = None
                    masters.append({"kind": kind, "frames": 0, "how": "libreria" if lib else "manca"})
            n = len(lights)
            mpx = lights[0].width * lights[0].height / 1e6 if lights else 0
            nm = sum(m["frames"] for m in masters if m["how"] == "crea")
            st = [("masters", nm * mpx * rates["masters"])] if nm else []
            st += [("calibrate", n * mpx * rates["calibrate"]), ("register", n * mpx * rates["register"])]
            if g.get("local_norm"):
                st.append(("normalize", n * mpx * rates["normalize"]))
            st.append(("integrate", n * mpx * rates["integrate"] * (1.5 if g.get("large_scale_rejection") else 1)))
            if g.get("drizzle", 1) and g.get("drizzle", 1) > 1:
                st.append(("drizzle", n * mpx * rates["drizzle"] * g["drizzle"] ** 2 / 4))
            if g.get("plate_solve", "auto") and self._will_solve(g.get("plate_solve", "auto"), lights):
                st.append(("solve", rates["solve"]))
            secs = sum(v for _, v in st)
            out["groups"].append({"index": gi, "label": label, "frames": n, "mpx": round(mpx, 1), "masters": masters,
                                  "overrides": ov, "stages": [{"stage": k, "s": round(v)} for k, v in st],
                                  "seconds": round(secs)})
            out["total_s"] += round(secs)
        out["estimate_from"] = rates["source"]
        return out

    def run_sources(self, sources, mode="multi"):
        """Several cameras / sessions of one target (mode "multi": merged) or colour + narrowband ("hybrid"):
        see stack.sources."""
        from . import sources as _sources
        return _sources.run_sources(self, sources, mode)

    def run_filters(self, palette="auto"):
        """Mono + filters: stack every filter of the main setup, align the channels, combine to color."""
        sc = self.scan()
        if not sc["light_groups"]:
            raise ValueError("no light frames found")
        main = sc["light_groups"][self.s["light_group"] or 0][0]
        idx = [i for i, (k, _) in enumerate(sc["light_groups"]) if k[:7] == main[:7] and k[8]]
        if len(idx) < 2:
            self.log("only one filter found: normal stacking")
            return self.run()
        # one filter may come in several exposures (Ha 300 s one night, 180 s another): each exposure is
        # its own light group with its own darks, so stack them separately and merge them afterwards
        by_filter = {}
        for i in idx:
            by_filter.setdefault(sc["light_groups"][i][0][8], []).append(i)
        if len(by_filter) < 2:
            self.log("only one filter found: normal stacking")
            return self.run()
        self.log(f"filters: {', '.join(f'{n} ({len(g)} exposures)' if len(g) > 1 else n for n, g in by_filter.items())}")
        stacks, exposure, base_output, runs = {}, {}, self.s["output"], []   # stack name -> file, total seconds
        base = dict(self.s)                  # every filter starts from the same settings (group_settings apply per run)
        for name, group_idx in by_filter.items():
            for i in group_idx:
                key, fs = sc["light_groups"][i]
                tag = name if len(group_idx) == 1 else f"{name}_{key[7]:g}s"
                self.log(f"===== filter {tag} =====")
                self.s = {**base, "light_group": i, "output": f"{base_output}_{tag}"}
                self.report = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
                _, rep = self.run()
                runs.append(self._run_summary(tag, rep))
                out = rep["output"]
                dz = self.s["drizzle"]
                stacks[tag] = f"{out}_drizzle{dz}x.fits" if dz and dz > 1 else f"{out}_integration.fits"
                used_frames = sum(1 for fr in rep.get("frames", []) if fr.get("accepted", True)) or len(fs)
                exposure[tag] = (name, used_frames * key[7])
        self.s["output"] = base_output
        self._step("combine", 0.5, "aligning and combining channels")
        aligned = channels.align(stacks)
        chans, paths = {}, {}
        for name in by_filter:
            parts = [(tag, aligned[tag], sec) for tag, (n, sec) in exposure.items() if n == name]
            if len(parts) == 1:
                chans[name], paths[name] = parts[0][1], stacks[parts[0][0]]
                continue
            # bring every exposure to the level of the deepest one, then average weighted by exposure
            parts.sort(key=lambda p: -p[2])
            ref = parts[0][1]
            total = sum(p[2] for p in parts)
            chans[name] = sum(channels._match(a, ref) * (sec / total) for _, a, sec in parts).astype(np.float32)
            paths[name] = str(self.work / f"{base_output}_{name}_merged.fits")
            io.save_fits(chans[name][..., None], paths[name])
            self.log(f"{name}: merged " + ", ".join(f"{t} ({sec / 60:.0f} min)" for t, _, sec in parts))
        rgb, used = channels.combine(chans, palette)
        out_base = self.work / f"{base_output}_{used}"
        xisf.write(rgb, f"{out_base}.xisf")
        io.save_fits(rgb, f"{out_base}.fits", getattr(self, "header", None), {"PALETTE": used})
        self._preview(rgb, f"{out_base}_preview.jpg")
        self.report.update(filters=paths, runs=runs, palette=used, output=str(out_base), combined=f"{out_base}.fits")
        self._step("done", 1.0, f"combined {used} from {', '.join(paths)}")
        return rgb, self.report

    @staticmethod
    def _preview(img, path):
        st, _ = stretch.auto_stretch(img, 0.2, -2.8)
        io.save_preview(st, path)
