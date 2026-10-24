"""Local web interface: wizard for stacking + post-processing with live preview.

Runs on 127.0.0.1 by default. Long operations are background jobs polled by the page.

Network mode (bound to any host other than 127.0.0.1 / localhost / ::1, see create_app):
  * every request (pages, /static, /api) needs the access token: ?token=... once (it sets a cookie), the
    cookie af_token, or the header X-AstroForge-Token; otherwise 401;
  * POST /api/config (paths of the external programs that get executed) is accepted only from a client on
    this machine (127.0.0.1 / ::1), remote clients get 403;
  * endpoints that read or write images (image, frame*, preview, mask, dbe_samples, pixel, fits_header, post) accept only paths
    under the "session folders": folders named by the page in this server session (light / calibration /
    mosaic / master folders sent to scan or stack, the work dir, folders and files chosen in the file
    browser -> POST /api/roots); the export folder must be one of them too. A drive root or the home folder itself is never a
    session folder (their subfolders can be). /api/browse still lists folder names and FITS/XISF names
    everywhere so folders can be chosen.
With 127.0.0.1 / localhost nothing of this applies.
"""
import hashlib
import hmac
import io as _io
import json
import re
import os
import secrets
import socket
import string
import threading
import time
import traceback
import unicodedata
import uuid
from pathlib import Path
from xml.sax.saxutils import quoteattr

import cv2
import numpy as np
from astropy.io import fits
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .. import gpu, io, look, masks, pipeline, plugins, stretch
from ..stack import frames as frames_mod
from ..stats import luminance
from ..stack import xisf
from ..stack.project import DEFAULTS as STACK_DEFAULTS
from ..stack.project import Cancelled, Runner

STATIC = Path(__file__).parent / "static"
router = APIRouter()
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost"}
TOKEN_COOKIE = "af_token"
TOKEN_HEADER = "X-AstroForge-Token"

JOBS = {}
_PREVIEW_CACHE = {}

# Job queue: one FIFO, one worker thread, one job at a time (stacking and export both use all the cores).
QUEUE = []            # ids of the queued jobs, in order
HISTORY = []          # ids of the finished jobs, oldest first
HISTORY_MAX = 30
_FNS = {}
_QLOCK = threading.Condition()
_WORKER = []


class Job:
    def __init__(self, kind, label=""):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind
        self.label = label or kind
        self.stage = "queued"
        self.progress = 0.0
        self.log = []
        self.status = "queued"      # queued | running | done | error | cancelled
        self.result = {}
        self.cancelled = False
        self.created = time.time()
        self.started = None
        self.finished = None

    def add(self, msg):
        self.log.append(f"[{time.time() - (self.started or self.created):6.1f}s] {msg}")

    def summary(self):
        with _QLOCK:
            pos = QUEUE.index(self.id) + 1 if self.id in QUEUE else 0
        err = next((m for m in reversed(self.log) if "ERROR:" in m), "") if self.status == "error" else ""
        return {"id": self.id, "kind": self.kind, "label": self.label, "stage": self.stage,
                "progress": self.progress, "status": self.status, "position": pos, "created": self.created,
                "started": self.started, "finished": self.finished, "error": err}

    def to_dict(self, since=0):
        return {**self.summary(), "log": self.log[since:], "log_len": len(self.log), "result": self.result}


def _finish(job):
    job.finished = time.time()
    with _QLOCK:
        HISTORY.append(job.id)
        while len(HISTORY) > HISTORY_MAX:
            JOBS.pop(HISTORY.pop(0), None)


def _run_job(job, fn):
    job.status, job.stage, job.started = "running", "starting", time.time()
    try:
        fn(job)
        if job.status == "running":
            job.status = "done"
    except Cancelled:
        job.status = "cancelled"
        job.add("cancelled")
    except Exception as e:
        job.status = "error"
        job.add(f"ERROR: {e}")
        job.add(traceback.format_exc())
    finally:
        _finish(job)


def _worker():
    while True:
        with _QLOCK:
            while not QUEUE:
                _QLOCK.wait()
            jid = QUEUE.pop(0)
            fn = _FNS.pop(jid)
        _run_job(JOBS[jid], fn)


def _start(job, fn):
    """Queue a job (FIFO); the worker runs one job at a time."""
    with _QLOCK:
        JOBS[job.id] = job
        _FNS[job.id] = fn
        QUEUE.append(job.id)
        pos = len(QUEUE)
        if not _WORKER or not _WORKER[0].is_alive():
            _WORKER[:] = [threading.Thread(target=_worker, daemon=True, name="astroforge-jobs")]
            _WORKER[0].start()
        _QLOCK.notify()
    return {"job": job.id, "position": pos}


# ------------------------------------------------------------------ network mode: session folders
def _remote(request):
    return bool(getattr(request.app.state, "remote", False))


def _norm(p):
    return os.path.normcase(os.path.abspath(os.path.realpath(str(p))))


def _add_roots(request, paths):
    """Remember folders named by the page (network mode only). A file adds its folder. Drive roots and the
    home folder itself are skipped (too broad); their subfolders are fine."""
    if not _remote(request):
        return
    home = _norm(Path.home())
    for p in paths or []:
        if not p:
            continue
        q = Path(str(p))
        if q.is_file() or q.suffix.lower() in frames_mod.FITS_EXT | {".xisf", ".npy", ".tif", ".tiff", ".jpg", ".png"}:
            q = q.parent
        n = _norm(q)
        if n == home or Path(n).parent == Path(n):
            continue
        request.app.state.roots.add(n)


def _check_path(request, path):
    """In network mode a path must lie under a session folder (403 otherwise)."""
    if not _remote(request):
        return
    n = _norm(path)
    for r in list(request.app.state.roots):        # a copy: another request may add a root meanwhile
        if n == r or n.startswith(r.rstrip("\\/") + os.sep):
            return
    raise HTTPException(403, "percorso fuori dalle cartelle di questa sessione: in modalità rete si possono aprire "
                             "solo file nelle cartelle scelte nella GUI (sorgenti, cartella di lavoro, file aperti)")


@router.post("/api/roots")
def roots_add(request: Request, body: dict):
    """Folders / files chosen in the file browser (network mode: they become session folders)."""
    _add_roots(request, body.get("paths", []))
    return {"remote": _remote(request), "roots": sorted(request.app.state.roots) if _remote(request) else []}


def _read_any(path):
    if str(path).lower().endswith(".xisf"):
        return xisf.read(path)[0], None
    return io.load(path)


# ------------------------------------------------------------------ pages
@router.get("/", response_class=HTMLResponse)
def index():
    """The page, with the version in the addresses of app.js / style.css: after an update the browser cannot keep
    the old files from its cache (the new page with the old script looked like "the update did not arrive")."""
    from .. import disk_version
    v = "".join(c for c in disk_version() if c.isalnum() or c in ".-+")
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    # every file the page links (scripts, styles, icons, manifest: a renamed app kept the old tab icon)
    import re
    html = re.sub(r'((?:href|src)="/static/[^"#?]+)"', lambda m: f'{m.group(1)}?v={v}"', html)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Browsers and Windows ask for /favicon.ico on their own (tab, taskbar, bookmarks)."""
    return FileResponse(STATIC / "icons" / "noctua.ico", media_type="image/x-icon")


# ------------------------------------------------------------------ files
@router.get("/api/browse")
def browse(path: str = "", images: bool = False):
    """images=True also lists finished pictures (JPEG/PNG/TIFF) for "profile from an image"."""
    exts = frames_mod.FITS_EXT | {".xisf"} | ({".jpg", ".jpeg", ".png", ".tif", ".tiff"} if images else set())
    if not path:
        drives = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
        return {"path": "", "parent": None, "dirs": drives, "files": [], "fits": 0}
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(404, "not a folder")
    dirs, files = [], []
    try:
        for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if c.is_dir():
                dirs.append(c.name)
            elif c.suffix.lower() in exts:
                files.append(c.name)
    except PermissionError:
        pass
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files, "fits": len(files)}


@router.get("/api/image")
def image(request: Request, path: str, max_side: int = 1600):
    """JPEG of a FITS/XISF/JPG file, auto-stretched when linear."""
    _check_path(request, path)
    p = Path(path)
    if not p.exists():
        raise HTTPException(404)
    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
        return FileResponse(p)
    return Response(_jpeg(_display(_load_small(p, max_side))), media_type="image/jpeg")


_IDENTIFY = {}


def _main_object(objs, shape):
    """The object the session is about: a Messier object near the centre, else the biggest one that fills a good
    part of the field, else the one nearest the centre."""
    if not objs:
        return None
    h, w = shape[:2]
    near = lambda o: ((o["x"] - w / 2) / w) ** 2 + ((o["y"] - h / 2) / h) ** 2
    dso = [o for o in objs if o.get("kind") != "star"] or objs
    m = [o for o in dso if o.get("messier") and near(o) < 0.18]
    if m:
        return min(m, key=near)
    big = [o for o in dso if near(o) < 0.18 and (o.get("a") or 0) * 2 > 0.15 * max(h, w)]
    return max(big, key=lambda o: o.get("a") or 0) if big else min(dso, key=near)


@router.post("/api/identify")
def identify(request: Request, body: dict):
    """What a frame shows, to name the session as soon as the lights are added (and an image opened in post):
    OBJECT from the header, then a plate solve with ASTAP (seconds with RA/DEC in the header, longer blind) and
    the catalogue objects in the field. Remembered per file."""
    path = str(body.get("path") or "")
    _check_path(request, path)
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "file non trovato")
    key = (str(p), p.stat().st_mtime)
    if key in _IDENTIFY:
        return _IDENTIFY[key]
    from .. import astrometry, catalogs
    img, header = _read_any(p)
    hdr_obj = str((header or {}).get("OBJECT", "") or "").strip()
    out = {"object": hdr_obj, "name": hdr_obj, "solved": False, "objects": []}
    tools = pipeline.load_config().get("tools", {})
    if not tools.get("astap") or not Path(tools["astap"]).exists():
        out["note"] = "plate solving non disponibile: ASTAP non configurato (Programmi esterni)"
        return out
    h0 = img.shape[0]
    _, _, fov = astrometry.header_hints(header, h0)
    mono = img.mean(axis=2) if img.ndim == 3 else img
    f = min(1.0, 1600 / max(mono.shape))            # also averages the 2x2 sites of a raw colour frame
    small = cv2.resize(mono, (round(mono.shape[1] * f), round(mono.shape[0] * f)), interpolation=cv2.INTER_AREA)
    wcs = astrometry.solve(small, tools, header, fov_deg=fov or 0, timeout=90, log=lambda m: None)
    if wcs is None:
        out["note"] = "plate solving non riuscito" + ("" if fov else " (mancano focale e pixel negli header: ricerca alla cieca)")
        _IDENTIFY[key] = out
        return out
    ra, dec = astrometry.field_center(wcs, small.shape)
    ps = astrometry.pixel_scale_arcsec(wcs)          # arcsec per pixel of the reduced copy
    objs = catalogs.objects_in_field(wcs, small.shape, stars=False)
    main = _main_object(objs, small.shape)
    name = (main["label"] if main else "") or hdr_obj
    out.update(solved=True, name=name, ra=ra, dec=dec, scale=ps * f,          # arcsec per pixel of the frame
               fov=[small.shape[1] * ps / 3600, small.shape[0] * ps / 3600],
               rotation=astrometry.rotation_deg(wcs)[0], objects=[o["label"] for o in objs[:8]])
    _IDENTIFY[key] = out
    return out


@router.get("/api/thumb")
def thumb(request: Request, path: str, side: int = 220):
    """Small JPEG of a raw frame for the contact sheet of the Sorgenti page. Kept out of the preview cache (a sheet
    of 100 frames would evict the image being edited); the browser caches it (the URL carries the file time)."""
    _check_path(request, path)
    p = Path(path)
    if not p.exists():
        raise HTTPException(404)
    img, _ = _read_any(p)
    h, w = img.shape[:2]
    f = min(1.0, max(16, min(side, 512)) / max(h, w))
    if f < 1:                   # INTER_AREA over a CFA mosaic averages the 2x2 sites: a grey but faithful frame
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[..., None]
    return Response(_jpeg(_display(img), 80), media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=86400"})


def _load_small(path, max_side):
    key = (str(path), max_side, Path(path).stat().st_mtime)
    hit = _PREVIEW_CACHE.get(key)            # .get: another request may clear the cache between a test and a read
    if hit is not None:
        return hit
    img, _ = _read_any(path)
    h, w = img.shape[:2]
    _FULL_SHAPE[str(path)] = (w, h)
    f = max_side / max(h, w)
    if f < 1:
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[..., None]
    if len(_PREVIEW_CACHE) > 4:
        _PREVIEW_CACHE.clear()
    _PREVIEW_CACHE[key] = img
    return img


def _display(img):
    img = np.nan_to_num(img)
    if float(np.median(img[::4, ::4])) < 0.08:
        img, _ = stretch.auto_stretch(img, 0.2, -2.8)
    return img


def _jpeg(img, q=88):
    a = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    a = a[..., 0] if a.shape[-1] == 1 else a
    buf = _io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=q)
    return buf.getvalue()


# ------------------------------------------------------------------ system / config
@router.get("/api/system")
def system():
    import os
    return {"gpus": [{"index": i, "name": n, "memory_gb": m} for i, n, m in gpu.devices()],
            "compute": gpu.describe(), "cpus": os.cpu_count()}


@router.get("/api/system/load")
def system_load():
    """Live CPU / RAM / GPU load for the top bar (polled by the page)."""
    from .. import sysload
    return sysload.snapshot()


@router.get("/api/config")
def config_get():
    cfg = pipeline.load_config()
    tools = cfg.get("tools", {})
    return {"tools": {k: {"path": tools.get(k, ""), "found": bool(tools.get(k)) and Path(tools[k]).exists()}
                      for k in ("starnet", "deepsnr", "graxpert", "cosmic_clarity", "cosmic_clarity_denoise", "astap", "rc-astro")}}


@router.post("/api/config")
def config_set(request: Request, body: dict):
    if _remote(request) and (request.client.host if request.client else "") not in LOCAL_CLIENTS:
        raise HTTPException(403, "i percorsi dei programmi esterni si possono cambiare solo dal PC su cui gira Noctua "
                                 "(http://127.0.0.1), non da un altro dispositivo in rete")
    cfg = pipeline.load_config()
    cfg["tools"] = {k: v for k, v in body.get("tools", {}).items() if v}
    pipeline.CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return config_get()


@router.get("/api/tools/updates")
def tools_updates(force: bool = False):
    """New StarNet2 / DeepSNR versions (installed one from the program, latest from starnetastro.com, daily)."""
    from .. import toolupdates
    return toolupdates.check(pipeline.load_config().get("tools", {}), force=force)


@router.post("/api/tools/find")
def tools_find(request: Request):
    """Look for the external programs (known folders, PATH, installed programs, N.I.N.A.) and fill in the ones
    missing or wrong in config.json; paths that exist are kept."""
    if _remote(request) and (request.client.host if request.client else "") not in LOCAL_CLIENTS:
        raise HTTPException(403, "la ricerca dei programmi si fa solo dal PC su cui gira Noctua")
    from .. import updater
    lines = []
    updater.write_config(pipeline.CONFIG.parent, log=lines.append)
    return {**config_get(), "log": lines}


# ------------------------------------------------------------------ stacking
@router.post("/api/scan")
def scan(request: Request, body: dict):
    lights, calib = body.get("lights", []), body.get("calibration", [])
    mdirs = body.get("master_dirs", []) or []
    gkw = [str(k) for k in body.get("grouping_keywords") or [] if str(k).strip()]
    kws = gkw + [str(k) for k in body.get("calibration_keywords") or [] if str(k).strip()]
    _add_roots(request, list(lights) + list(calib) + list(mdirs))
    found = []
    for src in lights + calib:
        p = Path(src)
        found += [frames_mod.read_frame(p, kws)] if p.is_file() else frames_mod.scan(p, keywords=kws)
    roots = [str(Path(x)) for x in lights]
    lf = [f for f in found if f.type == "light" and any(f.path.startswith(r) for r in roots)]
    cf = [f for f in found if f.type != "light"]
    groups = frames_mod.group(lf + cf, gkw)
    out = {"light_groups": [], "calibration": {}, "unknown": [f.path for f in found if f.type == "unknown"]}
    for key, fs in sorted(groups["light"].items(), key=lambda kv: -len(kv[1])):
        out["light_groups"].append({"label": frames_mod.describe_key(key), "count": len(fs),
                                    "frames": [f.to_dict() for f in fs]})
    for kind in ("bias", "dark", "flat", "darkflat"):
        out["calibration"][kind] = []
        for k, v in groups[kind].items():
            odd = frames_mod.outlier_exposures(v) if kind == "flat" else []
            out["calibration"][kind].append({"label": frames_mod.describe_key(k), "count": len(v),
                                             "odd_exposure": [Path(f.path).name for f in odd],
                                             "odd_paths": [f.path for f in odd],
                                             "frames": [f.to_dict() for f in v]})
    # ready masters in the calibration folders (FITS or XISF): used as they are, the raw frames of a kind win
    out["masters"] = []
    if calib or mdirs:
        try:
            from ..stack import masterlib
            lib = masterlib.MasterLibrary(STACK_DEFAULTS.get("master_library"), list(mdirs), log=lambda m: None,
                                          strict_extra=list(calib))
            roots = [str(Path(c).resolve()).lower() for c in list(calib) + list(mdirs)]
            for e in lib:
                if e.get("external") and any(str(Path(e["path"]).resolve()).lower().startswith(r) for r in roots):
                    out["masters"].append({"kind": e["kind"], "file": Path(e["path"]).name,
                                           "exptime": e.get("exptime"), "filter": e.get("filter") or "",
                                           "flags": e.get("flags") or []})
        except Exception as ex:          # the summary is informative: never fail the scan for it
            out["masters_error"] = str(ex)
    return out


_VIEW_CACHE = {}


def _view_small(path, side, color):
    """Reduced image of a frame for the quick viewer: colour from the Bayer matrix (2x2 superpixel, header
    BAYERPAT) when asked, else as stored; cached so changing the stretch is instant."""
    p = Path(path)
    key = (str(p), p.stat().st_mtime, side, bool(color))
    hit = _VIEW_CACHE.get(key)
    if hit is not None:
        return hit
    img, header = _read_any(p)
    img = np.nan_to_num(np.asarray(img, np.float32))
    if img.ndim == 2:
        img = img[..., None]
    pat = str((header or {}).get("BAYERPAT", "") or "").strip().upper() if header is not None else ""
    bayer = pat if img.shape[-1] == 1 and len(pat) == 4 and set(pat) <= set("RGB") else ""
    full_h, full_w = img.shape[:2]
    if img.shape[-1] == 1 and min(full_h, full_w) >= 64:
        # as the stacker does: the header pattern checked on the green sites (QHY168C + APT write GRBG for RGGB
        # data: a green nebula), and a colour sensor without BAYERPAT recognised from its 2x2 levels
        from ..stack import calibrate
        if not bayer and not (header or {}).get("FILTER") and calibrate.looks_like_cfa(img[..., 0]):
            bayer = "RGGB"
        if bayer:
            bayer = calibrate.verify_pattern(bayer, img[..., 0])[0]
    if bayer and color:
        h, w = full_h // 2 * 2, full_w // 2 * 2
        sub = max(1, int(np.ceil(max(h, w) / 2 / side)))
        img = _frame_color(img[:h, :w, 0], bayer, 0, h, 0, w, sub)
    f = side / max(img.shape[:2])
    if f < 1:
        img = cv2.resize(img, (max(1, round(img.shape[1] * f)), max(1, round(img.shape[0] * f))), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[..., None]
    info = {"width": int(full_w), "height": int(full_h), "bayer": bayer, "channels": int(img.shape[-1])}
    if header is not None:
        for k, name in (("EXPTIME", "exptime"), ("EXPOSURE", "exptime"), ("GAIN", "gain"), ("ISOSPEED", "iso"),
                        ("OFFSET", "offset"), ("CCD-TEMP", "temp"), ("FILTER", "filter"), ("DATE-OBS", "date"),
                        ("IMAGETYP", "type"), ("OBJECT", "object")):
            if k in header and name not in info and str(header[k]).strip() != "":
                v = header[k]
                info[name] = v if isinstance(v, (int, float)) else str(v).strip()
    if len(_VIEW_CACHE) > 12:
        _VIEW_CACHE.clear()
    _VIEW_CACHE[key] = (img, info)
    return img, info


@router.get("/api/view")
def view(request: Request, path: str, side: int = 1600, stretch_mode: str = "linked", color: bool = True,
         boost: bool = False):
    """Quick viewer of the Sorgenti page (hover popup and full-screen viewer): a frame reduced to `side`, in
    colour from the Bayer matrix, stretched like PixInsight's STF: linked (one curve), unlinked (one per channel,
    removes the colour cast of raw OSC frames), boost (STF Boost) or linear (as stored, min-max). X-Info: size,
    Bayer, exposure, gain, temperature, median and MAD of the shown image."""
    _check_path(request, path)
    if not Path(path).exists():
        raise HTTPException(404, "file not found")
    side = int(np.clip(side, 64, 3000))
    try:
        img, info = _view_small(path, side, color)
    except Exception as e:                       # unreadable file: a message, not a 500
        raise HTTPException(400, f"non riesco a leggere il file: {e}")
    target, clip = (0.3, -1.7) if boost else (0.2, -2.8)
    if stretch_mode == "linear":
        lo, hi = np.percentile(img[::2, ::2], [0.05, 99.95])
        out = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
    elif stretch_mode == "unlinked" and img.shape[-1] == 3:
        out = np.stack([stretch.histogram_transform(img[..., c:c + 1], *stretch.auto_params(img[..., c:c + 1], target, clip))[..., 0]
                        for c in range(3)], -1)
    else:
        out = stretch.histogram_transform(img, *stretch.auto_params(img, target, clip))
    med = [round(float(np.median(img[::2, ::2, c])), 5) for c in range(img.shape[-1])]
    info = dict(info, median=med, shown=[int(out.shape[1]), int(out.shape[0])])
    return Response(_jpeg(np.clip(out, 0, 1), 88), media_type="image/jpeg",
                    headers={"X-Info": json.dumps(info, default=str), "Cache-Control": "no-cache"})


@router.get("/api/frame")
def frame_preview(request: Request, path: str, max_side: int = 1000):
    """Blink: quick color preview of a calibrated CFA frame (2x2 superpixel, auto-stretched)."""
    _check_path(request, path)
    p = Path(path)
    if p.suffix != ".npy" or not p.exists():
        raise HTTPException(404)
    cfa = np.load(p, mmap_mode="r")
    img = _frame_small(cfa, _frame_pattern(p, cfa), max_side)
    return Response(_jpeg(stretch.histogram_transform(img, *_frame_stretch(p)), 82), media_type="image/jpeg")


def _frame_pattern(p, cfa):
    return body_pattern(p) if cfa.ndim == 2 else None


def _frame_color(cfa, pattern, y0, y1, x0, x1, sub=1):
    """Color (or mono) float32 (h, w, C) of the region [y0:y1, x0:x1] of a calibrated frame.

    CFA frames: 2x2 superpixel (half resolution, y0/x0 must be even), `sub` = take one superpixel every `sub`."""
    if cfa.ndim == 2 and pattern:
        sites = [np.asarray(cfa[y0 + dy:y1:2 * sub, x0 + dx:x1:2 * sub], np.float32) for dy in (0, 1) for dx in (0, 1)]
        hh, ww = min(s.shape[0] for s in sites), min(s.shape[1] for s in sites)
        idx = {"R": [], "G": [], "B": []}
        for i, ch in enumerate(pattern):
            idx[ch].append(sites[i][:hh, :ww])
        img = np.stack([np.mean(idx[c], 0) for c in "RGB"], -1)
    elif cfa.ndim == 2:
        img = np.asarray(cfa[y0:y1:sub, x0:x1:sub], np.float32)[..., None]
    else:
        img = np.asarray(cfa[y0:y1:sub, x0:x1:sub], np.float32)
    return np.nan_to_num(img)


def _frame_small(cfa, pattern, max_side):
    h, w = cfa.shape[0] // 2 * 2, cfa.shape[1] // 2 * 2
    if cfa.ndim == 2 and pattern:
        return _frame_color(cfa, pattern, 0, h, 0, w, max(1, int(np.ceil(max(h, w) / 2 / max_side))))
    return _frame_color(cfa, pattern, 0, h, 0, w, max(1, int(np.ceil(max(h, w) / max_side))))


_FRAME_STRETCH = {}


def _frame_stretch(p):
    """Auto-stretch (c0, m) of a calibrated frame, measured on its small preview: the zoomed crops use the same
    stretch so they look like the blink preview."""
    key = (str(p), p.stat().st_mtime)
    if key not in _FRAME_STRETCH:
        cfa = np.load(p, mmap_mode="r")
        if len(_FRAME_STRETCH) > 64:
            _FRAME_STRETCH.clear()
        _FRAME_STRETCH[key] = stretch.auto_params(_frame_small(cfa, _frame_pattern(p, cfa), 1000), 0.2, -2.8)
    return _FRAME_STRETCH[key]


def _npy(path):
    p = Path(path)
    if p.suffix != ".npy" or not p.exists():
        raise HTTPException(404)
    return p


@router.get("/api/frame_info")
def frame_info(request: Request, path: str):
    _check_path(request, path)
    p = _npy(path)
    cfa = np.load(p, mmap_mode="r")
    return {"width": int(cfa.shape[1]), "height": int(cfa.shape[0]), "bayer": _frame_pattern(p, cfa)}


@router.get("/api/frame_crop")
def frame_crop(request: Request, path: str, x: float, y: float, zoom: float = 1.0, w: int = 800, h: int = 600):
    """Blink zoom: crop of a calibrated frame around (x, y) (full-resolution pixels) at `zoom` screen pixels per
    sensor pixel (1 = 1:1), at most w x h pixels, stretched like the blink preview. The crop is kept inside the
    frame; headers X-Cx / X-Cy give the center actually used, X-Frame-Width / X-Frame-Height the frame size."""
    _check_path(request, path)
    p = _npy(path)
    zoom = float(np.clip(zoom, 0.25, 16))
    w, h = int(np.clip(w, 16, 2400)), int(np.clip(h, 16, 2400))
    cfa = np.load(p, mmap_mode="r")
    pattern = _frame_pattern(p, cfa)
    H, W = cfa.shape[:2]
    cw, ch = min(W, int(np.ceil(w / zoom))), min(H, int(np.ceil(h / zoom)))
    if pattern:
        cw, ch = max(2, cw // 2 * 2), max(2, ch // 2 * 2)
    x0 = int(np.clip(round(x - cw / 2), 0, W - cw))
    y0 = int(np.clip(round(y - ch / 2), 0, H - ch))
    if pattern:
        x0, y0 = x0 // 2 * 2, y0 // 2 * 2
    img = _frame_color(cfa, pattern, y0, y0 + ch, x0, x0 + cw)
    ow, oh = max(1, round(cw * zoom)), max(1, round(ch * zoom))
    big = ow >= img.shape[1] * 2
    img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_NEAREST if big else
                     cv2.INTER_LINEAR if ow > img.shape[1] else cv2.INTER_AREA)
    if img.ndim == 2:
        img = img[..., None]
    img = stretch.histogram_transform(img, *_frame_stretch(p))
    return Response(_jpeg(img, 85), media_type="image/jpeg",
                    headers={"X-Cx": f"{x0 + cw / 2:.1f}", "X-Cy": f"{y0 + ch / 2:.1f}", "X-Zoom": f"{zoom:g}",
                             "X-Frame-Width": str(W), "X-Frame-Height": str(H)})


def body_pattern(npy_path):
    """Bayer pattern used for a work folder, from its report (None for mono)."""
    for rep in Path(npy_path).parent.parent.glob("*_report.json"):
        try:
            return json.loads(rep.read_text(encoding="utf-8")).get("bayer")
        except (OSError, ValueError):
            pass
    return None


@router.get("/api/stack/defaults")
def stack_defaults():
    return STACK_DEFAULTS


@router.post("/api/plan")
def stack_plan(request: Request, settings: dict):
    """What the stacking will do and an estimated time, without running it (WBPP-like diagram)."""
    _add_roots(request, list(settings.get("lights") or []) + list(settings.get("calibration") or [])
               + list(settings.get("master_dirs") or []))
    s = dict(settings)
    s.setdefault("work_dir", STACK_DEFAULTS.get("work_dir"))
    try:
        return Runner(s, log=lambda m: None).plan()
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/api/masters/defaults")
def masters_defaults():
    """Common and per-type master settings (the defaults are the recommended ones)."""
    from ..stack.makemasters import MASTER_DEFAULTS, TYPE_DEFAULTS
    return {"common": MASTER_DEFAULTS, "types": TYPE_DEFAULTS}


def _stack_label(settings):
    panels = [p for p in (settings.get("mosaic") or []) if p]
    first = (panels[0][0] if panels else (settings.get("lights") or [""])[0]) or ""
    name = Path(first).name or first
    if panels:
        return f"Mosaico {len(panels)} pannelli · {name}"
    srcs = [x for x in (settings.get("sources") or []) if isinstance(x, dict)]
    if settings.get("mode") in ("multi", "hybrid") and srcs:
        what = "Più camere" if settings["mode"] == "multi" else "Ibrida RGB + banda stretta"
        from ..stack.sources import source_name
        return f"{what} · " + " + ".join(source_name(x, i) for i, x in enumerate(srcs))
    return ("Stacking filtri · " if settings.get("filters") else "Stacking · ") + name


@router.post("/api/reset")
def reset():
    """Forget the images kept in memory for the live preview (downscaled copies, StarNet results)."""
    n = len(_PREVIEW_CACHE) + len(_STARNET_CACHE)
    _PREVIEW_CACHE.clear()
    _STARNET_CACHE.clear()
    _FULL_SHAPE.clear()
    return {"ok": True, "freed": n}


@router.post("/api/masters")
def masters_job(request: Request, body: dict):
    """Queue "Crea i master": body = {folders: [...], out_dir, options: {...}} (stack.makemasters)."""
    from ..stack.makemasters import MASTER_DEFAULTS, TYPE_DEFAULTS, make_masters
    folders = [f for f in body.get("folders") or [] if f]
    if not folders:
        raise HTTPException(400, "scegli almeno una cartella con i file di calibrazione")
    out_dir = body.get("out_dir") or str(Path(folders[0]) / "AstroForge masters")
    _add_roots(request, folders + [out_dir])
    opts = {k: v for k, v in (body.get("options") or {}).items() if k in MASTER_DEFAULTS}
    for kind in TYPE_DEFAULTS:            # per type (the PixInsight ImageIntegration settings of each master)
        t = (body.get("options") or {}).get(kind)
        if isinstance(t, dict):
            opts[kind] = {k: v for k, v in t.items() if k in TYPE_DEFAULTS[kind]}
    job = Job("masters", f"Crea master · {Path(folders[0]).name}")

    def fn(job):
        def progress(frac, msg=""):
            job.stage, job.progress = msg or "masters", frac
        job.result = make_masters(folders, out_dir, opts, log=job.add, progress=progress, cancel=lambda: job.cancelled)
    return _start(job, fn) | {"out_dir": out_dir}


@router.post("/api/stack")
def stack(request: Request, settings: dict):
    """Queue a stacking job. settings["mosaic"] = [[light folders of panel 1], [panel 2], ...] makes a mosaic."""
    srcs = [x for x in (settings.get("sources") or []) if isinstance(x, dict)]
    _add_roots(request, list(settings.get("lights") or []) + list(settings.get("calibration") or [])
               + [f for p in (settings.get("mosaic") or []) for f in (p or [])]
               + [f for x in srcs for f in list(x.get("lights") or []) + list(x.get("calibration") or [])
                  + list(x.get("master_dirs") or []) + ([x["file"]] if x.get("file") else [])]
               + list(settings.get("master_dirs") or []) + [settings.get("work_dir") or STACK_DEFAULTS.get("work_dir")])
    job = Job("stack", _stack_label(settings))

    def fn(job):
        def progress(stage, frac, msg=""):
            job.stage, job.progress = stage, frac

        runner = Runner(settings, progress=progress, log=job.add, cancel=lambda: job.cancelled)
        if settings.get("mode") in ("multi", "hybrid"):
            _, report = runner.run_sources(srcs, settings["mode"])
            job.result = {"report": report, "combined": report["combined"], "final_input": report["combined"],
                          "sources": [r.get("file") for r in report["sources"]]}
            if report.get("nb"):     # hybrid: the post-production starts with the narrowband step and its sliders
                # on finished (stretched) files the emission is blended like a screen layer: the added
                # continuum-subtracted signal of NBRGBCombination is meant for linear data
                mode = "screen" if report.get("stretched") else "nbrgb"
                job.result["hybrid_step"] = {"plugin": "color.nb_rgb", "params": dict(report["nb"], mode=mode)}
                job.result["stretched"] = bool(report.get("stretched"))
            # every source a file already integrated: only the alignment was asked, the post opens on the sliders
            job.result["align_only"] = all(x.get("file") for x in srcs if x.get("file") or x.get("lights"))
            return
        panels = [list(p) for p in (settings.get("mosaic") or []) if p]
        if panels:
            if len(panels) < 2:
                raise ValueError("a mosaic needs at least 2 panels")
            _, report = runner.run_mosaic(panels)
            dz = settings.get("drizzle", 1) or 1
            suffix = f"_drizzle{dz}x.fits" if dz > 1 else "_integration.fits"
            job.result = {"report": report, "mosaic": report["combined"],
                          "panels": [f"{o}{suffix}" for o in report["panels"]], "final_input": report["combined"]}
            return
        if settings.get("all_groups") and not settings.get("filters"):
            _, report = runner.run_groups()
            job.result = {"report": report, "groups": report["groups"], "combined": report["combined"],
                          "final_input": report["combined"]}
            return
        if settings.get("filters"):
            _, report = runner.run_filters(settings.get("palette", "auto"))
        else:
            _, report = runner.run()
        out = report["output"]
        if report.get("combined"):
            job.result = {"report": report, "filters": report["filters"], "combined": report["combined"],
                          "final_input": report["combined"]}
            return
        job.result = {"report": report, "integration": f"{out}_integration.fits"}
        if settings.get("drizzle", 1) > 1:
            job.result["drizzle"] = f"{out}_drizzle{settings['drizzle']}x.fits"
        job.result["final_input"] = job.result.get("drizzle", job.result["integration"])
        if report.get("dualband"):
            job.result["final_input"] = report["dualband"]["HOO"]
        elif report.get("comet"):
            job.result["final_input"] = report["comet"]["comet_stack"].replace("_comet.fits", "_stars_comet.fits")
    return _start(job, fn)


@router.get("/api/jobs")
def job_list():
    """Queue panel: running + queued jobs in execution order, then the finished ones (newest first)."""
    with _QLOCK:
        queued = list(QUEUE)
        hist = list(reversed(HISTORY))
        running = [j for j in JOBS.values() if j.status == "running"]
    return {"active": [j.summary() for j in running] + [JOBS[i].summary() for i in queued if i in JOBS],
            "history": [JOBS[i].summary() for i in hist if i in JOBS]}


@router.get("/api/jobs/{jid}")
def job_status(jid: str, since: int = 0):
    if jid not in JOBS:
        raise HTTPException(404)
    return JOBS[jid].to_dict(since)


@router.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    """Cancel a job: a queued one is dropped at once, a running one stops at its next check."""
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404)
    with _QLOCK:
        queued = jid in QUEUE
        if queued:
            QUEUE.remove(jid)
            _FNS.pop(jid, None)
    job.cancelled = True
    if queued:
        job.status = "cancelled"
        job.add("cancelled before starting")
        _finish(job)
    return {"ok": True, "status": job.status}


# ------------------------------------------------------------------ post-processing
@router.get("/api/plugins")
def plugin_list():
    """help: the short Italian text of the parameter (plugins.help_it), the English one when missing (help_en)."""
    from ..plugins import help_it
    out = []
    for pid, pl in sorted(plugins.REGISTRY.items(), key=lambda kv: (kv[1].category, kv[0])):
        out.append({"id": pid, "name": pl.name, "category": pl.category, "stage": pl.stage,
                    "description": pl.description,
                    "params": {k: {"default": v.default, "min": v.min, "max": v.max,
                                   "help": help_it.text(pid, k) or v.help, "help_en": v.help,
                                   "choices": v.choices, "int": _int_param(v)} for k, v in pl.params.items()}})
    return out


def _int_param(p):
    """Whole-number parameter? JSON writes 1.0 as 1, so the page cannot tell 0.0-5.0 from 0-5 by itself and
    gave decimal sliders steps of 1 (the hybrid Ha slider went 0, 1, 2, 3)."""
    def whole(x):
        return isinstance(x, int) and not isinstance(x, bool)
    return whole(p.default) if p.default is not None else whole(p.min) and whole(p.max)


@router.get("/api/presets")
def preset_list():
    return [pipeline.load_preset(f) | {"id": f.stem} for f in sorted(pipeline.PRESETS.glob("*.json"))]


@router.post("/api/presets/{name}")
def preset_save(name: str, preset: dict):
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "custom"
    preset["name"] = safe
    (pipeline.PRESETS / f"{safe}.json").write_text(json.dumps(preset, indent=2), encoding="utf-8")
    return {"ok": True, "id": safe}


@router.post("/api/presets/{name}/rename")
def preset_rename(name: str, body: dict):
    """Name shown for a preset (the id stays, see pipeline.rename_preset)."""
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    try:
        p = pipeline.rename_preset(safe, body.get("label", ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "id": safe, "label": p["label"]}


@router.post("/api/presets_from_image")
def preset_from_image(request: Request, body: dict):
    """New profile: steps of `base`, final look.match with the look measured on a finished image you like."""
    path = body.get("path", "")
    _check_path(request, path)
    if not Path(path).is_file():
        raise HTTPException(404, "immagine non trovata")
    base = pipeline.load_preset(body.get("base") or "nebula_pro")
    img = io.load_display(path)
    stats = look.measure(img)
    steps = [s for s in base["steps"] if s["plugin"] != "look.match"]
    steps.append({"plugin": "look.match", "params": look.target_from(stats)})
    label = " ".join(str(body.get("label") or f"come {Path(path).stem}").split())[:60]
    pid = "".join(c for c in label.lower().replace(" ", "_") if c.isalnum() or c in "-_") or "da_immagine"
    while (pipeline.PRESETS / f"{pid}.json").exists():
        pid += "_2"
    preset = {"name": pid, "label": label, "description": f"look di {Path(path).name} su base {base.get('name')}",
              "reference": {"image": Path(path).name, "look": stats}, "steps": steps}
    (pipeline.PRESETS / f"{pid}.json").write_text(json.dumps(preset, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "id": pid, "label": label, "look": stats}


@router.get("/api/suggest_preset")
def suggest(request: Request, path: str):
    """Automatic profile for an integrated image (pipeline.suggest_preset)."""
    _check_path(request, path)
    header, n = {}, 3
    try:
        if path.lower().endswith((".fits", ".fit", ".fts")):
            from astropy.io import fits as _fits
            h = _fits.getheader(path)
            header = {k: h.get(k) for k in ("PALETTE", "OBJECT") if k in h}
            n = int(h.get("NAXIS3", 1)) if int(h.get("NAXIS", 2)) == 3 else 1
    except OSError:
        pass
    pid, why = pipeline.suggest_preset(path, header, n)
    return {"id": pid, "reason": why}


@router.get("/api/presets/{name}")
def preset_get(name: str):
    """One preset, steps as saved (params, "mask", "enabled", "preview")."""
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    f = pipeline.PRESETS / f"{safe}.json"
    if not safe or not f.exists():
        raise HTTPException(404, "preset not found")
    return pipeline.load_preset(f) | {"id": safe}


@router.post("/api/histogram")
def histogram(request: Request, body: dict):
    """Histogram of the image as it enters a step (the curves editor draws it under the curve): body = input,
    steps (those before the step), target "stars" for the star layer. Bins of R, G, B and luminance, each scaled to
    its own maximum (log, so the faint end shows). The image is displayed stretched when still linear."""
    img, ctx = _preview_input(body | {"max_side": min(int(body.get("max_side", 600)), 900)}, request)
    try:
        img = _preview_steps(img, body.get("steps", []), ctx)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    if body.get("target") == "stars" and ctx.layers.get("stars") is not None:
        img = ctx.layers["stars"]
    elif ctx.linear:
        img, _ = stretch.auto_stretch(img, 0.2, -2.8)
    n = int(np.clip(body.get("bins", 128), 16, 512))
    chans = [img[..., c] for c in range(img.shape[-1])]
    out = {"bins": n, "linear": bool(ctx.linear)}
    for name, ch in zip("rgb" if len(chans) == 3 else "l", chans):
        out[name] = ch
    out["l"] = luminance(img) if len(chans) == 3 else chans[0]
    for k in ("r", "g", "b", "l"):
        if k in out:
            h = np.histogram(np.clip(out[k], 0, 1), bins=n, range=(0, 1))[0].astype(np.float64)
            h = np.log1p(h)
            out[k] = [round(float(v), 4) for v in h / max(h.max(), 1e-9)]
    return out


@router.post("/api/preview")
def preview(request: Request, body: dict):
    """Run post steps on a downscaled copy and return a JPEG (live preview).

    Pixel-sized parameters are scaled to the preview size (ctx.scale) so it matches the full result.
    StarNet runs on the small image and is cached, so 'stars.recombine' works; the other external
    programs are too slow for a live preview and are skipped.
    """
    steps = body.get("steps", [])
    if body.get("upto") is not None:
        steps = steps[:body["upto"]]
    view = body.get("view") or ""        # "stars": the star layer only, "starless": the nebula without the stars
    if view == "starless":
        steps = [s for s in steps if s.get("plugin") != "stars.recombine"]
    img, ctx = _preview_input(body, request)
    t = time.time()
    try:
        out = _preview_steps(img, steps, ctx)
    except ValueError as e:          # bad parameters (e.g. too few DBE samples): a message, not a 500
        raise HTTPException(400, str(e))
    except KeyError as e:            # a plugin this process does not know: GUI started before an update
        raise HTTPException(400, f"{e.args[0].split('.')[0]}: riavvia la GUI (il codice è stato aggiornato "
                                 "dopo l'avvio)")
    if view == "stars":
        if ctx.layers.get("stars") is None:
            raise HTTPException(400, "nessun livello stelle: aggiungi prima un passo che toglie le stelle (StarNet)")
        out = ctx.layers["stars"]
        if ctx.layers.get("stars_mode") == "add":      # removed while linear: shown stretched, like the stars step does
            out = np.clip(np.arcsinh(out * 60.0) / np.arcsinh(60.0), 0, 1).astype(np.float32)
    elif ctx.linear:
        out, _ = stretch.auto_stretch(out, 0.2, -2.8)
    return Response(_jpeg(out), media_type="image/jpeg", headers={"X-Elapsed": f"{time.time() - t:.2f}",
                                                                  "X-Scale": f"{ctx.scale:.8f}"})


@router.post("/api/preview_zoom")
def preview_zoom(request: Request, body: dict):
    """Detail of the zoomed area at (up to) full resolution. The WHOLE image is processed at the resolution that
    shows the area 1:1 on screen and only the area is returned: steps that use statistics of the whole image
    (auto stretch, look.match, gradient models) see the same image as the export, so the detail matches it.
    body: input, steps, crop [x0, y0, x1, y1] (0..1 of the image), width (screen px of the area),
    external (also run the external programs, slower; otherwise skipped as in the live preview)."""
    _check_path(request, body["input"])
    path = Path(body["input"])
    if not path.exists():
        raise HTTPException(404, "input not found")
    x0, y0, x1, y1 = (float(np.clip(v, 0, 1)) for v in body.get("crop", [0, 0, 1, 1]))
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        raise HTTPException(400, "empty zoom area")
    full_w, full_h = _full_shape(path)                   # (the full image was loaded just to know the height:
                                                          # ~800 MB for a drizzle 2x, and it evicted the preview)
    scale = min(1.0, float(body.get("width", 1200)) / ((x1 - x0) * full_w))
    # pixel budget: a deep zoom on a drizzle 2x (10080 x 6692) meant the whole 67 Mpx image through every step,
    # minutes (or the GPU memory) with the spinner turning; above the budget the scale is lowered (X-Scale says it)
    budget = float(body.get("max_mpx", 24)) * 1e6
    if scale * scale * full_w * full_h > budget:
        scale = (budget / (full_w * full_h)) ** 0.5
    max_side = max(64, round(max(full_w, full_h) * scale))
    img = _load_small(path, max_side if scale < 1 else max(full_w, full_h)).copy()
    ctx = plugins.Context(log=lambda m: None, tools=pipeline.load_config().get("tools", {}),
                          scale=img.shape[1] / full_w, linear=not pipeline.looks_stretched(img))
    t = time.time()
    try:
        if body.get("external"):
            steps = [s for s in body.get("steps", []) if s["plugin"] not in _PREVIEW_SKIP]
            out = pipeline.run_steps(img, steps, ctx)
        else:
            out = _preview_steps(img, body.get("steps", []), ctx)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except KeyError as e:
        raise HTTPException(400, f"{e.args[0].split('.')[0]}: riavvia la GUI (il codice è stato aggiornato dopo l'avvio)")
    if ctx.linear:
        out, _ = stretch.auto_stretch(out, 0.2, -2.8)
    h, w = out.shape[:2]
    crop = out[int(y0 * h):max(int(y0 * h) + 1, int(round(y1 * h))), int(x0 * w):max(int(x0 * w) + 1, int(round(x1 * w)))]
    return Response(_jpeg(crop, 92), media_type="image/jpeg",
                    headers={"X-Elapsed": f"{time.time() - t:.2f}", "X-Scale": f"{ctx.scale:.8f}",
                             "X-Size": f"{crop.shape[1]}x{crop.shape[0]}"})


@router.get("/api/version")
def version():
    """Running version and the one on disk: different after an update while the GUI kept running (new plugins
    and profiles are then missing from this process: the GUI shows a restart notice)."""
    from .. import disk_version, version_string
    run, disk = version_string(), disk_version()
    return {"version": run, "disk": disk, "restart_needed": run.split(" ")[:2] != disk.split(" ")[:2],
            "pid": os.getpid()}


# ------------------------------------------------------------------ stop / restart from the page
ROOT = Path(__file__).resolve().parents[2]


def _local_only(request, what):
    if _remote(request) and (request.client.host if request.client else "") not in LOCAL_CLIENTS:
        raise HTTPException(403, f"{what} si fa solo dal PC su cui gira Noctua (http://127.0.0.1)")


def _restart_cmd(launch):
    """Same server again, after the update check (as start-gui.cmd does)."""
    import sys
    cmd = [sys.executable, "-m", "astroforge", "gui", "--host", launch.get("host", "127.0.0.1"),
           "--port", str(launch.get("port", 8765)), "--update"]
    return cmd + (["--token", launch["token"]] if launch.get("token") else [])


def _exit_soon(spawn=None, delay=0.6):
    """Leave after the HTTP answer went out; `spawn` (a command) is started first, in its own console window."""
    import subprocess
    import sys

    def go():
        if spawn:
            kw = ({"creationflags": subprocess.CREATE_NEW_CONSOLE} if sys.platform == "win32"
                  else {"start_new_session": True})
            subprocess.Popen(spawn, cwd=str(ROOT), **kw)
        os._exit(0)
    threading.Timer(delay, go).start()


@router.post("/api/server/restart")
def server_restart(request: Request):
    """Start a fresh server (new code, new plugins, empty caches) and stop this one; the page waits for it.
    Running jobs are lost: the page asks first."""
    _local_only(request, "il riavvio del server")
    _exit_soon(_restart_cmd(getattr(request.app.state, "launch", None) or {}))
    return {"ok": True, "pid": os.getpid()}


@router.post("/api/server/stop")
def server_stop(request: Request):
    _local_only(request, "lo spegnimento del server")
    _exit_soon()
    return {"ok": True, "pid": os.getpid()}


def _preview_input(body, request):
    _check_path(request, body["input"])
    path = Path(body["input"])
    if not path.exists():
        raise HTTPException(404, "input not found")
    img = _load_small(path, int(body.get("max_side", 1200))).copy()
    # an image already stretched (finished file) is not stretched again for the preview
    ctx = plugins.Context(log=lambda m: None, tools=pipeline.load_config().get("tools", {}),
                          scale=img.shape[1] / _full_shape(path)[0], linear=not pipeline.looks_stretched(img))
    return img, ctx


# plate solving / downloads take tens of seconds to minutes: not run on every preview change
_PREVIEW_SKIP = {"color.spcc", "color.pcc", "analysis.plate_solve", "annotate.objects"}
_PREVIEW_SUBST = {"background.allsky": "background.dbe"}


def _preview_steps(img, steps, ctx):
    """Run steps on the preview image: external programs and plate-solving steps skipped (the all-sky gradient
    is previewed with DBE), StarNet cached."""
    done = []
    for step in steps:
        if step["plugin"].startswith("ext.") or step["plugin"] in _PREVIEW_SKIP:
            continue
        if step["plugin"] in _PREVIEW_SUBST:
            step = {**step, "plugin": _PREVIEW_SUBST[step["plugin"]], "params": {}}
        if step["plugin"] == "stars.starnet" and step.get("enabled", True) is not False:
            img = pipeline.run_steps(img, done, ctx) if done else img
            done = []
            out = _cached_starnet(img, step, ctx)
            if step.get("mask"):           # same blend as pipeline.apply_step
                m = masks.make_mask(img, step["mask"], ctx)[..., None]
                out = (img * (1 - m) + out * m).astype(np.float32)
            img = out
        else:
            done.append(step)
    return pipeline.run_steps(img, done, ctx)


@router.post("/api/dbe_samples")
def dbe_samples(request: Request, body: dict):
    """DBE samples of step `index` (a background.dbe step) on the preview image: runs the steps before it, then
    the DBE itself, and returns ctx.layers["background_samples"] in preview pixels ("samples") and full-resolution
    pixels ("full", what goes into the step's `samples` param). auto=true ignores the manual samples."""
    steps = body.get("steps", [])
    i = int(body.get("index", -1))
    if not 0 <= i < len(steps) or steps[i].get("plugin") != "background.dbe":
        raise HTTPException(400, "index must point to a background.dbe step")
    img, ctx = _preview_input(body, request)
    img = _preview_steps(img, steps[:i], ctx)
    params = dict(steps[i].get("params") or {})
    if body.get("auto"):
        params["samples"] = []
    pl = plugins.get("background.dbe")
    p = pl.scaled(pl.resolve(params), ctx.scale)
    try:
        pl.run(img, ctx, **p)
    except ValueError as e:
        raise HTTPException(400, str(e))
    pts = [[float(x), float(y)] for x, y in ctx.layers.get("background_samples", [])]
    s = ctx.scale
    return {"samples": pts, "full": [[round(x / s, 1), round(y / s, 1)] for x, y in pts],
            "auto": not params.get("samples"), "width": int(img.shape[1]), "height": int(img.shape[0]),
            "scale": s, "box": float(p["box"])}


@router.post("/api/mask")
def mask_png(request: Request, body: dict):
    """Mask of step `index` on the preview image, as a grayscale PNG (white = processed, black = protected).

    Runs the steps before it (like the live preview), then masks.make_mask on that image with ctx.scale: the
    same mask pipeline.apply_step uses. body["mask"] (optional) overrides the step's own "mask" spec."""
    steps = body.get("steps", [])
    i = int(body.get("index", -1))
    if not 0 <= i < len(steps):
        raise HTTPException(400, "index must point to a step")
    spec = body.get("mask") or steps[i].get("mask")
    if not spec:
        raise HTTPException(400, "this step has no mask")
    img, ctx = _preview_input(body, request)
    try:
        img = _step_input(img, steps, i, ctx)
        m = masks.make_mask(img, spec, ctx)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    buf = _io.BytesIO()
    Image.fromarray((np.clip(np.nan_to_num(m), 0, 1) * 255 + 0.5).astype(np.uint8), "L").save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"X-Width": str(m.shape[1]), "X-Height": str(m.shape[0]), "X-Mean": f"{float(m.mean()):.4f}",
                             "X-Scale": f"{ctx.scale:.8f}"})


def _step_input(img, steps, i, ctx):
    """The image step i works on in the preview: the result of the steps before it, or the star layer when
    the step has "target": "stars" (as pipeline.apply_step does)."""
    img = _preview_steps(img, steps[:i], ctx)
    if steps[i].get("target", "image") == "stars":
        stars = ctx.layers.get("stars")
        if stars is None:
            raise ValueError("nessun livello stelle: serve prima un passo di rimozione stelle (StarNet, Dark Star, StarX)")
        img = np.asarray(stars, np.float32)
        img = img[..., None] if img.ndim == 2 else img
    return img


@router.post("/api/pixel")
def pixel_value(request: Request, body: dict):
    """Value of the image entering step `index` (what that step receives, e.g. GHS's symmetry point) at the
    preview pixel (x, y) (preview pixels, continuous: pixel i covers [i, i + 1)). Median of a (2*radius+1)^2
    box (radius in preview pixels, default 2) so a click on the background is not one noisy pixel.
    Returns "value" (luminance; the value for mono), "channels" (per channel), "mean" (of the channels),
    "full" = [x, y] in full-resolution pixels, "width"/"height" of the preview and ctx.scale."""
    steps = body.get("steps", [])
    i = int(body.get("index", len(steps)))
    if not 0 <= i <= len(steps):
        raise HTTPException(400, "index must point to a step (or the end of the list)")
    try:
        x, y = float(body["x"]), float(body["y"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "x and y (preview pixels) are required")
    img, ctx = _preview_input(body, request)
    try:
        img = _step_input(img, steps, i, ctx) if i < len(steps) else _preview_steps(img, steps, ctx)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    h, w = img.shape[:2]
    if not (0 <= x <= w and 0 <= y <= h):
        raise HTTPException(400, f"point outside the preview ({w}x{h})")
    r = int(np.clip(int(body.get("radius", 2)), 0, 20))
    cx, cy = min(int(x), w - 1), min(int(y), h - 1)
    box = np.nan_to_num(img[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].reshape(-1, img.shape[-1]))
    ch = [float(v) for v in np.median(box, 0)]
    lum = float(luminance(np.asarray(ch, np.float32)[None, None])[0, 0]) if len(ch) in (1, 3) else float(np.mean(ch))
    return {"value": lum, "channels": ch, "mean": float(np.mean(ch)), "x": x, "y": y,
            "full": [round(x / ctx.scale, 1), round(y / ctx.scale, 1)], "width": int(w), "height": int(h),
            "scale": ctx.scale, "linear": bool(ctx.linear)}


@router.get("/api/fits_header")
def fits_header(request: Request, path: str):
    """Header of a FITS file (or the FITS keywords of an XISF): {"cards": [[key, value, comment], ...]}."""
    _check_path(request, path)
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "file not found")
    suffix = p.suffix.lower()
    if suffix != ".xisf" and suffix not in frames_mod.FITS_EXT:
        raise HTTPException(400, "not a FITS / XISF file")
    cards = []
    try:
        if suffix == ".xisf":
            cards = [[k, v, ""] for k, v in xisf.read(p)[1].items()]
        else:
            with fits.open(p, memmap=False) as hdul:
                for k, hdu in enumerate(hdul):
                    if k:
                        cards.append([f"--- HDU {k}", "", hdu.name])
                    for c in hdu.header.cards:
                        v = c.value
                        cards.append([c.keyword, v if isinstance(v, (int, float, str, bool)) else str(v), c.comment])
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"header non leggibile: {e}")
    return {"path": str(p), "cards": cards}


def _clean_metadata(meta):
    """{KEYWORD: value} from the export page -> FITS keywords (8 characters, upper case) with ASCII text values
    (accents dropped: FITS headers are ASCII). Empty values are skipped. COMMENT / HISTORY may be multi-line."""
    out = {}
    for k, v in (meta or {}).items():
        key = "".join(c for c in str(k).upper() if c.isalnum() or c in "-_")[:8]
        if not key or v is None or (isinstance(v, str) and not v.strip()):
            continue
        if isinstance(v, str):
            v = unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode()
            if key in ("COMMENT", "HISTORY"):
                v = "\n".join("".join(c for c in line if c.isprintable()) for line in v.strip().splitlines())
            else:
                v = " ".join("".join(c if c.isprintable() else " " for c in v).split())[:68]
        out[key] = v
    return out


_STARNET_CACHE = {}
_STARNET_LOCK = threading.Lock()


def _cached_starnet(img, step, ctx):
    key = (hashlib.sha1(img.tobytes()).hexdigest(), ctx.linear, json.dumps(step.get("params", {}), sort_keys=True))
    with _STARNET_LOCK:      # preview, histogram and mask requests arrive together: StarNet runs once, not three times
        hit = _STARNET_CACHE.get(key)
        if hit is None:
            pl = plugins.get("stars.starnet")
            out = pl.run(img, ctx, **pl.scaled(pl.resolve(step.get("params", {})), ctx.scale))
            if len(_STARNET_CACHE) > 8:
                _STARNET_CACHE.clear()
            hit = _STARNET_CACHE[key] = (out, ctx.layers.get("stars"), ctx.layers.get("stars_mode"))
    out, stars, mode = hit
    if stars is not None:
        ctx.layers.update(stars=stars, stars_mode=mode)
    return out


_FULL_SHAPE = {}   # (width, height) of the files opened, filled by _load_small


def _full_shape(path):
    """Full-resolution (width, height) of an image: from the cache, else from the file header without reading
    the pixels (FITS) or by loading it."""
    s = _FULL_SHAPE.get(str(path))
    if s:
        return s
    p = Path(path)
    if p.suffix.lower() in (".fits", ".fit", ".fts"):
        try:
            from astropy.io import fits
            with fits.open(p, memmap=True) as hd:
                sh = hd[0].shape
                if sh and len(sh) >= 2:
                    s = (int(sh[-1]), int(sh[-2]))
        except Exception:
            s = None
    if not s:
        im = _read_any(p)[0]
        s = (int(im.shape[1]), int(im.shape[0]))
    _FULL_SHAPE[str(path)] = s
    return s


@router.post("/api/post")
def post(request: Request, body: dict):
    _check_path(request, body.get("input", ""))
    _check_path(request, body.get("out_dir") or str(Path(body.get("input", "")).parent))
    job = Job("post", f"Export {body.get('name', 'final')} · {Path(body.get('input', '')).name}")

    def fn(job):
        name = body.get("name", "final")
        out_dir = body.get("out_dir") or str(Path(body["input"]).parent)
        job.stage = "post"
        img, header = _read_any(body["input"])
        ctx = plugins.Context(log=job.add, header=header, out_dir=out_dir,
                              tools=pipeline.load_config().get("tools", {}), linear=not pipeline.looks_stretched(img))

        def on_step(i, n, pl):
            if job.cancelled:
                raise Cancelled()
            job.progress = (i - 1) / n
            job.add(f"#{i} {pl.name}")
        img = pipeline.run_steps(img, body["steps"], ctx, on_step)
        base = Path(out_dir) / f"{Path(body['input']).stem}_{name}"
        fmts = body.get("formats", ["fits", "tif", "jpg"])
        meta = _clean_metadata(body.get("metadata"))
        if "fits" in fmts:
            from .. import version_string
            io.save_fits(img, f"{base}.fits", header, extra={"SWCREATE": f"Noctua {version_string()}", **meta})
        if "xisf" in fmts:
            xisf.write(img, f"{base}.xisf", {k: quoteattr(str(v))[1:-1] for k, v in meta.items()
                                             if k not in ("COMMENT", "HISTORY")})
        if "tif" in fmts:
            io.save_tiff(img, f"{base}.tif")
        if "jpg" in fmts:
            io.save_jpeg(img, f"{base}.jpg", quality=95)
        io.save_preview(img, f"{base}_preview.jpg")
        job.result = {"files": [f"{base}.{f}" for f in fmts], "preview": f"{base}_preview.jpg"}
        job.progress = 1.0
        job.add(f"saved {base}")
    return _start(job, fn)


def is_local(host):
    return str(host or "").strip().lower() in LOCAL_HOSTS


def create_app(host="127.0.0.1", token=None):
    """The FastAPI app. On a local host (127.0.0.1 / localhost / ::1) there are no restrictions. Any other host
    turns on network mode (module docstring): access token on every request, POST /api/config only from this
    machine, file endpoints limited to the session folders. token=None there -> ASTROFORGE_TOKEN or random."""
    a = FastAPI(title="Noctua")
    a.include_router(router)
    a.mount("/static", StaticFiles(directory=STATIC), name="static")

    @a.middleware("http")
    async def no_stale_static(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"       # always revalidated (ETag): never an old app.js
        return resp
    remote = not is_local(host)
    a.state.remote = remote
    a.state.roots = set()
    a.state.token = (token or os.environ.get("ASTROFORGE_TOKEN") or secrets.token_urlsafe(18)) if remote else None
    if remote:
        @a.middleware("http")
        async def require_token(request: Request, call_next):
            want = request.app.state.token
            got_q = request.query_params.get("token")
            for got in (request.headers.get(TOKEN_HEADER), request.cookies.get(TOKEN_COOKIE), got_q):
                if got and hmac.compare_digest(str(got).encode(), want.encode()):
                    resp = await call_next(request)
                    if got_q and request.cookies.get(TOKEN_COOKIE) != want:
                        resp.set_cookie(TOKEN_COOKIE, want, httponly=True, samesite="strict")
                    return resp
            msg = ("Noctua: serve il token di accesso. Apri l'indirizzo con ?token=... stampato all'avvio "
                   "del server (oppure header X-AstroForge-Token).")
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": msg}, status_code=401)
            return Response(msg, status_code=401, media_type="text/plain; charset=utf-8")
    return a


def _lan_ip():
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))        # no packet is sent: only picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


def _port_free(host, port):
    s = socket.socket(socket.AF_INET6 if ":" in host.strip("[]") else socket.AF_INET)
    try:
        s.bind((host.strip("[]") if host not in ("", "localhost") else "127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _free_port(host, port, wait=30, log=print):
    """A previous AstroForge still holding the port (window closed but process alive, a restart in progress,
    an old version without the stop button) is stopped; anything else holding it is an error."""
    if _port_free(host, port):
        return
    try:
        import psutil
        owners = {c.pid for c in psutil.net_connections("tcp")
                  if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid}
    except Exception:
        owners = set()
    for pid in owners - {os.getpid()}:
        try:
            p = psutil.Process(pid)
            cmd = " ".join(p.cmdline()).lower()
        except Exception:
            continue
        # the AstroForge server itself, not any program run from a folder or venv that is called astroforge
        if not re.search(r"-m\s+astroforge(\s|$)|astroforge[\\/]__main__\.py|astroforge\.gui\.server", cmd):
            raise SystemExit(f"la porta {port} è usata da un altro programma ({p.name()}): usa --port")
        log(f"chiudo il server Noctua precedente (processo {pid}) che occupa la porta {port}")
        try:
            p.terminate()
            p.wait(5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    t = time.time()
    while not _port_free(host, port):
        if time.time() - t > wait:
            raise SystemExit(f"la porta {port} è ancora occupata: chiudi l'altro Noctua o usa --port")
        time.sleep(0.5)


def open_browser_when_up(url, probe, timeout=30.0):
    """Open the page in the default browser once the server answers (start-gui.cmd: the user double-clicked
    the icon and expects the page, not a black window). A thread, so the server starts meanwhile."""
    def go():
        import urllib.error
        import urllib.request
        import webbrowser
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                urllib.request.urlopen(probe, timeout=2).close()
                break
            except urllib.error.HTTPError:   # an answer (even 401 in network mode): the server is up
                break
            except OSError:
                time.sleep(0.3)
        try:
            webbrowser.open(url)
        except Exception:                    # no browser: the address is printed anyway
            pass
    threading.Thread(target=go, daemon=True).start()


def serve(host="127.0.0.1", port=8765, token=None, open_browser=False):
    import uvicorn
    _free_port(host, port)
    a = create_app(host, token)
    a.state.launch = {"host": host, "port": port, "token": a.state.token if a.state.remote else None}
    if a.state.remote:
        shown = _lan_ip() if host in ("0.0.0.0", "::", "") else host
        print(f"Noctua GUI (rete, con token): http://{shown}:{port}/?token={a.state.token}", flush=True)
        print("  il token resta valido finché il server è acceso; i programmi esterni si configurano solo da "
              f"http://127.0.0.1:{port}/?token=... su questo PC", flush=True)
        url = f"http://127.0.0.1:{port}/?token={a.state.token}"
    else:
        print(f"Noctua GUI: http://{host}:{port}", flush=True)
        url = f"http://{host}:{port}/"
    if open_browser:
        open_browser_when_up(url, f"http://127.0.0.1:{port}/api/version")
    uvicorn.run(a, host=host, port=port, log_level="warning")


app = create_app()          # local app (tests, `uvicorn astroforge.gui.server:app`)
