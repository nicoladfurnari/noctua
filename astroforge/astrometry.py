"""Plate solving with ASTAP and Gaia DR3 star catalogue from VizieR TAP (urllib only).

WCS pixel coordinates follow the numpy layout of the image: x = column, y = row (0-based with origin=0).
"""
import csv
import hashlib
import io as _io
import subprocess
import tempfile
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

from . import net

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def _sexa(value, hours=False):
    """'HH MM SS' / '+DD MM SS' / number -> degrees (or hours when hours=True)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(":", " ").replace("h", " ").replace("m", " ").replace("d", " ").rstrip("s")
    parts = s.split()
    if not parts:
        return None
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    sign = -1.0 if parts[0].startswith("-") else 1.0
    v = abs(nums[0]) + (nums[1] / 60 if len(nums) > 1 else 0) + (nums[2] / 3600 if len(nums) > 2 else 0)
    return sign * v


def header_hints(header, height):
    """(ra_hours, dec_deg, fov_deg) from FITS header keywords, None when missing."""
    if header is None:
        return None, None, None
    ra = dec = fov = None
    if header.get("RA") is not None and header.get("DEC") is not None:
        try:
            ra, dec = float(header["RA"]) / 15.0, float(header["DEC"])
        except (TypeError, ValueError):
            ra = dec = None
    if ra is None and header.get("OBJCTRA") and header.get("OBJCTDEC"):
        ra, dec = _sexa(header["OBJCTRA"]), _sexa(header["OBJCTDEC"])
    try:
        fl, px = float(header.get("FOCALLEN") or 0), float(header.get("XPIXSZ") or 0)
        if fl > 0 and px > 0:
            fov = height * px / fl * 206.265 / 3600.0
    except (TypeError, ValueError):
        pass
    return ra, dec, fov


def _write_mono(img, path):
    a = np.asarray(img, np.float32)
    lum = a if a.ndim == 2 else (a[..., 0] if a.shape[-1] == 1 else a @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    lo, hi = np.percentile(lum[::4, ::4], [0.1, 100])
    data = np.clip((lum - lo) / max(hi - lo, 1e-12), 0, 1) * 65535
    fits.PrimaryHDU(data.astype(np.uint16)).writeto(path, overwrite=True)


def parse_wcs_text(text):
    """FITS header (text with newlines or raw 80-char cards) -> WCS."""
    if isinstance(text, bytes):
        text = text.decode("ascii", "replace")
    if "\n" in text:
        cards = [ln.rstrip("\r") for ln in text.splitlines()]
        text = "".join(c[:80].ljust(80) for c in cards if c.strip())
    hdr = fits.Header.fromstring(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(hdr, relax=True)


def _read_ini(path):
    out = {}
    if Path(path).exists():
        for ln in Path(path).read_text(errors="replace").splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def solve(img, tools, header=None, fov_deg=0, ra_hours=None, dec_deg=None, radius_deg=30, downsample=0,
          timeout=300, log=print, retry_auto_fov=True):
    """Plate solve with astap_cli. Returns astropy WCS (with SIP) or None when unsolved.
    retry_auto_fov: when a field size is given and fails, try again with ASTAP's own (fov 0)."""
    exe = (tools or {}).get("astap")
    if not exe or not Path(exe).exists():
        log(f"plate solve: ASTAP not configured or missing ({exe})")
        return None
    h = np.asarray(img).shape[0]
    hra, hdec, hfov = header_hints(header, h)
    ra_hours = ra_hours if ra_hours is not None else hra
    dec_deg = dec_deg if dec_deg is not None else hdec
    fov = fov_deg or hfov or 0
    attempts = [fov] + ([0] if fov and retry_auto_fov else [])
    with tempfile.TemporaryDirectory(prefix="astroforge_astap_") as tmp:
        f = Path(tmp) / "solve.fits"
        _write_mono(img, f)
        for fv in attempts:
            cmd = [str(exe), "-f", str(f), "-fov", f"{fv:.4f}", "-z", str(int(downsample)), "-sip"]
            if ra_hours is not None and dec_deg is not None:
                cmd += ["-ra", f"{ra_hours:.5f}", "-spd", f"{dec_deg + 90:.5f}", "-r", f"{radius_deg:g}"]
            else:
                cmd += ["-r", "180"]
            log("plate solve: " + " ".join(cmd[1:]))
            try:
                subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout, cwd=tmp)
            except subprocess.TimeoutExpired:
                log(f"plate solve: ASTAP timed out after {timeout}s")
                continue
            except OSError as e:
                log(f"plate solve: cannot run ASTAP: {e}")
                return None
            ini = _read_ini(f.with_suffix(".ini"))
            wcs_file = f.with_suffix(".wcs")
            if ini.get("PLTSOLVD", "F").upper().startswith("T") and wcs_file.exists():
                return parse_wcs_text(wcs_file.read_bytes())
            log(f"plate solve: not solved (fov {fv:g}) {ini.get('ERROR', '') or ini.get('WARNING', '')}".rstrip())
            for p in (f.with_suffix(".ini"), wcs_file):
                p.unlink(missing_ok=True)
    return None


def pixel_scale_arcsec(wcs):
    return float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0)


def field_center(wcs, shape=None):
    """(ra_deg, dec_deg) of the image center (or CRVAL without shape)."""
    if shape is None:
        return tuple(float(v) for v in wcs.wcs.crval)
    h, w = shape[:2]
    ra, dec = wcs.all_pix2world([[(w - 1) / 2, (h - 1) / 2]], 0)[0]
    return float(ra), float(dec)


def rotation_deg(wcs):
    """Position angle of image 'up' (decreasing row index is north-ish) east of north, plus flip flag."""
    cd = wcs.pixel_scale_matrix
    flipped = np.linalg.det(cd) > 0
    return float(np.degrees(np.arctan2(cd[0, 1], cd[1, 1])) % 360), bool(flipped)


def wcs_to_header(wcs):
    return wcs.to_header(relax=True)


def field_radius_deg(wcs, shape):
    h, w = shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    ra0, dec0 = field_center(wcs, shape)
    rd = wcs.all_pix2world(corners, 0)
    return float(max(_sep(ra0, dec0, r, d) for r, d in rd))


def _sep(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def query_gaia(ra, dec, radius, mag_limit=14, timeout=60, url=VIZIER_TAP, max_rows=200000):
    """Cone search on Gaia DR3 (VizieR I/355/gaiadr3). Returns dict of arrays ra, dec, g, bp_rp."""
    q = (f'SELECT TOP {int(max_rows)} "RA_ICRS", "DE_ICRS", "Gmag", "BP-RP" FROM "I/355/gaiadr3" '
         f"WHERE 1=CONTAINS(POINT('ICRS', \"RA_ICRS\", \"DE_ICRS\"), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.5f})) "
         f'AND "Gmag" < {mag_limit:g} AND "BP-RP" IS NOT NULL')
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q}).encode()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "AstroForge"})
    with net.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    rows = list(csv.reader(_io.StringIO(text)))
    if not rows or "RA_ICRS" not in rows[0]:
        raise ValueError(f"unexpected TAP answer: {text[:200]}")
    vals = np.array([[float(v) if v else np.nan for v in r] for r in rows[1:] if len(r) == 4], float).reshape(-1, 4)
    vals = vals[np.isfinite(vals).all(axis=1)]
    return {"ra": vals[:, 0], "dec": vals[:, 1], "g": vals[:, 2], "bp_rp": vals[:, 3]}


def gaia_stars(wcs, shape, mag_limit=14, cache_dir=None, timeout=60, log=print):
    """Gaia DR3 stars inside the image: dict ra, dec, x, y, g, bp_rp (x, y 0-based pixels). None on failure."""
    ra, dec = field_center(wcs, shape)
    radius = field_radius_deg(wcs, shape) * 1.02
    cat = None
    path = None
    if cache_dir:
        key = hashlib.md5(f"{ra:.3f}_{dec:.3f}_{radius:.3f}_{mag_limit:g}".encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"gaia_{key}.npz"
        if path.exists():
            with np.load(path) as z:
                cat = {k: z[k] for k in z.files}
    if cat is None:
        try:
            cat = query_gaia(ra, dec, radius, mag_limit, timeout)
        except Exception as e:  # network, HTTP, parsing
            log(f"gaia: query failed ({type(e).__name__}: {e})")
            return None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **cat)
    if len(cat["ra"]) == 0:
        return {k: np.zeros(0) for k in ("ra", "dec", "x", "y", "g", "bp_rp")}
    x, y = wcs.all_world2pix(cat["ra"], cat["dec"], 0)
    h, w = shape[:2]
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    out = {k: np.asarray(v)[ok] for k, v in cat.items()}
    out["x"], out["y"] = x[ok], y[ok]
    log(f"gaia: {ok.sum()} stars G<{mag_limit:g} in the field (radius {radius:.2f} deg)")
    return out


# ------------------------------------------------------------------ APASS (photometric colour calibration)
# APASS DR10 (what PixInsight ships as a local database) is not published on VizieR (checked 2026-10:
# the only APASS table there is II/336/apass9), so the online query uses APASS DR9.
APASS_TABLE = "II/336/apass9"
APASS_NAME = "APASS DR9 (VizieR II/336/apass9)"
_APASS_COLS = [("RAJ2000", "ra"), ("DEJ2000", "dec"), ("Vmag", "V"), ("e_Vmag", "e_V"), ("Bmag", "B"),
               ("e_Bmag", "e_B"), ("g'mag", "g"), ("e_g'mag", "e_g"), ("r'mag", "r"), ("e_r'mag", "e_r"),
               ("i'mag", "i"), ("e_i'mag", "e_i")]
APASS_KEYS = [k for _, k in _APASS_COLS]


def query_apass(ra, dec, radius, mag_limit=15, timeout=60, url=VIZIER_TAP, max_rows=200000):
    """Cone search on APASS DR9 (VizieR II/336/apass9). Dict of arrays ra, dec, V, B, g, r, i and their
    e_* errors (NaN when the catalogue has no value). Stars with V < mag_limit, or without V but r' < mag_limit."""
    cols = ", ".join(f'"{c}"' for c, _ in _APASS_COLS)
    q = (f'SELECT TOP {int(max_rows)} {cols} FROM "{APASS_TABLE}" '
         f"WHERE 1=CONTAINS(POINT('ICRS', \"RAJ2000\", \"DEJ2000\"), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.5f})) "
         f'AND ("Vmag" < {mag_limit:g} OR ("Vmag" IS NULL AND "r\'mag" < {mag_limit:g}))')
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q}).encode()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "AstroForge"})
    with net.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    rows = list(csv.reader(_io.StringIO(text)))
    if not rows or "RAJ2000" not in rows[0]:
        raise ValueError(f"unexpected TAP answer: {text[:200]}")
    idx = [rows[0].index(c) for c, _ in _APASS_COLS]

    def num(v):
        try:
            return float(v) if v.strip() else np.nan
        except ValueError:
            return np.nan

    vals = np.array([[num(r[j]) for j in idx] for r in rows[1:] if len(r) >= len(rows[0])], float)
    vals = vals.reshape(-1, len(idx))
    vals = vals[np.isfinite(vals[:, :2]).all(axis=1)]
    return {k: vals[:, n] for n, k in enumerate(APASS_KEYS)}


def apass_stars(wcs, shape, mag_limit=15, cache_dir=None, timeout=60, log=print):
    """APASS DR9 stars inside the image: dict ra, dec, x, y, V, B, g, r, i (+ e_V ... e_i), NaN where the
    catalogue has no magnitude (x, y 0-based pixels). None on failure (offline, HTTP, parsing)."""
    ra, dec = field_center(wcs, shape)
    radius = field_radius_deg(wcs, shape) * 1.02
    cat = None
    path = None
    if cache_dir:
        key = hashlib.md5(f"apass9_{ra:.3f}_{dec:.3f}_{radius:.3f}_{mag_limit:g}".encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"apass_{key}.npz"
        if path.exists():
            with np.load(path) as z:
                cat = {k: z[k] for k in z.files}
    if cat is None:
        try:
            cat = query_apass(ra, dec, radius, mag_limit, timeout)
        except Exception as e:  # network, HTTP, parsing
            log(f"apass: query failed ({type(e).__name__}: {e})")
            return None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **cat)
    if len(cat["ra"]) == 0:
        return {k: np.zeros(0) for k in APASS_KEYS + ["x", "y"]}
    x, y = wcs.all_world2pix(cat["ra"], cat["dec"], 0)
    h, w = shape[:2]
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    out = {k: np.asarray(v)[ok] for k, v in cat.items()}
    out["x"], out["y"] = x[ok], y[ok]
    log(f"apass: {ok.sum()} {APASS_NAME} stars V<{mag_limit:g} in the field (radius {radius:.2f} deg)")
    return out
