"""Loading and saving images.

Internal convention: float32 array, shape (H, W, C) with C = 1 or 3, values in [0, 1].
"""
import warnings
from pathlib import Path

import cv2
import numpy as np
import tifffile
from astropy.io import fits
from PIL import Image

from .colorspace import resolve_icc


def load(path):
    path = Path(path)
    if path.suffix.lower() == ".xisf":          # PixInsight masters / integrations
        from .stack import xisf
        data, keywords = xisf.read(path)
        header = fits.Header()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, v in keywords.items():
                if k and len(k) <= 8 and k not in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
                                                   "EXTEND", "END", "COMMENT", "HISTORY"):
                    try:
                        header[k] = v.strip("'").strip() if isinstance(v, str) else v
                    except ValueError:
                        pass
        return data, header
    if path.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):     # finished images (16-bit TIFF...)
        data = tifffile.imread(path) if path.suffix.lower() in (".tif", ".tiff") else np.asarray(Image.open(path))
        if data.ndim == 3 and data.shape[0] in (1, 3, 4) and data.shape[-1] not in (1, 3, 4):
            data = np.moveaxis(data, 0, -1)                                  # planar TIFF
        if data.ndim == 3 and data.shape[-1] == 4:
            data = data[..., :3]                                             # alpha
        data = (data.astype(np.float32) / np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer)
                else data.astype(np.float32))
        return (data if data.ndim == 3 else data[..., None]), fits.Header()
    with fits.open(path, memmap=False) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        data = hdu.data
        header = hdu.header
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32, copy=False)
    if data.ndim == 3:
        data = np.ascontiguousarray(np.moveaxis(data, 0, -1))  # (C,H,W) -> (H,W,C)
    else:
        data = data[..., None]
    if str(header.get("ROWORDER", "")).upper() == "BOTTOM-UP":
        data = data[::-1].copy()
    return data, header


def save_fits(img, path, header=None, extra=None):
    """extra: {KEYWORD: value or (value, comment)} written as given (author, copyright, notes...)."""
    data = np.moveaxis(img, -1, 0) if img.shape[-1] > 1 else img[..., 0]
    hdr = fits.Header()
    if header is not None:
        # acquisition + pointing keywords: plate solving (SPCC, mosaics) uses RA/DEC/FOCALLEN/XPIXSZ as hints
        for key in ("OBJECT", "EXPTIME", "DATE-OBS", "INSTRUME", "TELESCOP", "FOCALLEN", "GAIN", "OFFSET",
                    "RA", "DEC", "OBJCTRA", "OBJCTDEC", "XPIXSZ", "YPIXSZ", "XBINNING", "YBINNING", "FILTER",
                    "SITELAT", "SITELONG"):
            if key in header:
                hdr[key] = header[key]
    for key, val in (extra or {}).items():
        if val in (None, ""):
            continue
        key = str(key).upper().strip()
        if key in ("COMMENT", "HISTORY"):
            for line in str(val).splitlines():
                hdr.add_comment(line) if key == "COMMENT" else hdr.add_history(line)
        else:
            with warnings.catch_warnings():    # keywords longer than 8 chars (COPYRIGHT) go out as HIERARCH
                warnings.simplefilter("ignore", fits.verify.VerifyWarning)
                hdr[key] = val if not isinstance(val, (list, tuple)) else tuple(val)
    fits.PrimaryHDU(np.ascontiguousarray(data, dtype=np.float32), hdr).writeto(path, overwrite=True)


def _to_uint(img, bits):
    top = 65535 if bits == 16 else 255
    out = np.clip(img, 0, 1) * top + 0.5
    return out.astype(np.uint16 if bits == 16 else np.uint8)


def _icc_for(img, icc):
    """Embedded profile bytes for an RGB image (profiles are RGB, so mono images get none)."""
    return resolve_icc(icc) if img.shape[-1] == 3 else None


def save_tiff(img, path, icc="srgb"):
    """16-bit TIFF. icc: colour space name ('srgb', 'display_p3', 'adobe_rgb'), ICC bytes or None.
    The profile only labels the data (TIFF tag 34675): convert first with colorspace.convert()."""
    prof = _icc_for(img, icc)
    extra = [(34675, "B", len(prof), prof, True)] if prof else []
    tifffile.imwrite(path, _to_uint(img, 16), photometric="rgb" if img.shape[-1] == 3 else "minisblack",
                     extratags=extra)


def save_jpeg(img, path, quality=92, icc="srgb"):
    arr = _to_uint(img, 8)
    prof = _icc_for(img, icc)
    kw = {"icc_profile": prof} if prof else {}
    Image.fromarray(arr if arr.shape[-1] == 3 else arr[..., 0]).save(path, quality=quality, **kw)


def save_png(img, path, icc="srgb"):
    """8-bit PNG with an embedded ICC profile (iCCP chunk)."""
    arr = _to_uint(img, 8)
    prof = _icc_for(img, icc)
    kw = {"icc_profile": prof} if prof else {}
    Image.fromarray(arr if arr.shape[-1] == 3 else arr[..., 0]).save(path, **kw)


def save_preview(img, path, max_side=1600):
    h, w = img.shape[:2]
    f = max_side / max(h, w)
    small = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else img
    if small.ndim == 2:
        small = small[..., None]
    save_jpeg(small, path, quality=88)


def load_display(path, max_side=2400):
    """A finished image to look at or measure (JPEG/PNG/TIFF as displayed, FITS/XISF as stored),
    reduced to max_side: float32 (H, W, 3) in 0..1."""
    path = Path(path)
    if path.suffix.lower() in (".fits", ".fit", ".fts", ".xisf"):
        a = load(path)[0]
    else:
        Image.MAX_IMAGE_PIXELS = None          # 80 MB web finals are fine here, they are reduced at once
        im = Image.open(path)
        im.draft("RGB", (max_side, max_side))
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        a = np.asarray(im, np.float32) / 255
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, -1)
    return np.clip(np.nan_to_num(a), 0, 1).astype(np.float32)
