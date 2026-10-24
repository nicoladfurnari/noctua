"""RGB colour spaces for export: sRGB, Display P3, Adobe RGB (1998).

AstroForge images after the stretch are display-referred and treated as sRGB-encoded. convert() moves them
to another space (linearise with the source TRC, 3x3 matrix through CIE XYZ, encode with the destination
TRC); icc_bytes() gives a matching ICC v2 matrix/TRC display profile written by a small built-in writer
(no external profile files), so viewers that honour embedded profiles show the intended colours.
"""
import struct
from functools import lru_cache

import numpy as np

D65 = (0.3127, 0.3290)
D50_XYZ = (0.9642, 1.0, 0.8249)

SPACES = {
    "srgb": {"desc": "sRGB IEC61966-2.1 (AstroForge)", "primaries": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
             "trc": "srgb"},
    "display_p3": {"desc": "Display P3 (AstroForge)", "primaries": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),
                   "trc": "srgb"},
    "adobe_rgb": {"desc": "Adobe RGB (1998) compatible (AstroForge)",
                  "primaries": ((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)), "trc": 563 / 256},
}
ALIASES = {"sRGB": "srgb", "p3": "display_p3", "displayp3": "display_p3", "display-p3": "display_p3",
           "adobe": "adobe_rgb", "adobergb": "adobe_rgb", "adobe-rgb": "adobe_rgb", "argb": "adobe_rgb"}


def canonical(name):
    n = str(name).strip()
    n = ALIASES.get(n, ALIASES.get(n.lower(), n.lower()))
    if n not in SPACES:
        raise ValueError(f"unknown colour space '{name}' (use {', '.join(SPACES)})")
    return n


def _xyz(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz_matrix(name):
    """3x3 matrix linear RGB -> XYZ (D65 white, Y of white = 1)."""
    p = SPACES[canonical(name)]["primaries"]
    m = np.stack([_xyz(c) for c in p], axis=1)
    s = np.linalg.solve(m, _xyz(D65))
    return m * s


def _bradford_to_d50():
    b = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
    src, dst = b @ _xyz(D65), b @ np.array(D50_XYZ)
    return np.linalg.inv(b) @ np.diag(dst / src) @ b


def decode(v, trc):
    """Encoded [0,1] -> linear light."""
    v = np.asarray(v, np.float64)
    if trc == "srgb":
        return np.where(v <= 0.04045, v / 12.92, ((np.maximum(v, 0) + 0.055) / 1.055) ** 2.4)
    return np.maximum(v, 0) ** trc


def encode(v, trc):
    """Linear light [0,1] -> encoded."""
    v = np.clip(np.asarray(v, np.float64), 0, 1)
    if trc == "srgb":
        return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)
    return v ** (1 / trc)


def convert(img, src="srgb", dst="display_p3"):
    """Convert display-encoded RGB (H, W, 3) between colour spaces. Out-of-gamut colours are clipped.
    Single-channel images are returned unchanged."""
    s, d = canonical(src), canonical(dst)
    img = np.asarray(img)
    if s == d or img.ndim < 3 or img.shape[-1] != 3:
        return img
    m = np.linalg.inv(rgb_to_xyz_matrix(d)) @ rgb_to_xyz_matrix(s)
    lin = decode(img, SPACES[s]["trc"]) @ m.T
    return encode(lin, SPACES[d]["trc"]).astype(np.float32)


# ------------------------------------------------------------------ ICC v2 writer
def _s15(v):
    return struct.pack(">i", int(round(v * 65536)))


def _xyz_tag(xyz):
    return b"XYZ " + b"\0" * 4 + b"".join(_s15(c) for c in xyz)


def _curv_tag(trc):
    if trc == "srgb":
        t = decode(np.linspace(0, 1, 1024), "srgb")
        vals = np.round(np.clip(t, 0, 1) * 65535).astype(">u2")
        return b"curv" + b"\0" * 4 + struct.pack(">I", len(vals)) + vals.tobytes()
    return b"curv" + b"\0" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(trc * 256)))


def _desc_tag(text):
    a = text.encode("ascii", "replace") + b"\0"
    return (b"desc" + b"\0" * 4 + struct.pack(">I", len(a)) + a + struct.pack(">II", 0, 0)
            + struct.pack(">HB", 0, 0) + b"\0" * 67)


def _text_tag(text):
    return b"text" + b"\0" * 4 + text.encode("ascii", "replace") + b"\0"


def _build_icc(name):
    sp = SPACES[name]
    m50 = _bradford_to_d50() @ rgb_to_xyz_matrix(name)   # colorants adapted to the D50 PCS
    tags = [(b"desc", _desc_tag(sp["desc"])),
            (b"cprt", _text_tag("No copyright, use freely")),
            (b"wtpt", _xyz_tag(D50_XYZ)),
            (b"rXYZ", _xyz_tag(m50[:, 0])), (b"gXYZ", _xyz_tag(m50[:, 1])), (b"bXYZ", _xyz_tag(m50[:, 2]))]
    curv = _curv_tag(sp["trc"])
    tags += [(b"rTRC", curv), (b"gTRC", curv), (b"bTRC", curv)]
    table_len = 4 + 12 * len(tags)
    offset = 128 + table_len
    table, data, placed = b"", b"", {}
    for sig, body in tags:
        if id(body) in placed:                     # the three TRC tags share one curve
            off, size = placed[id(body)]
        else:
            off, size = offset + len(data), len(body)
            placed[id(body)] = (off, size)
            data += body + b"\0" * (-len(body) % 4)
        table += sig + struct.pack(">II", off, size)
    body = struct.pack(">I", len(tags)) + table + data
    size = 128 + len(body)
    header = (struct.pack(">I", size) + b"lcms" + bytes([2, 0x10, 0, 0]) + b"mntr" + b"RGB " + b"XYZ "
              + struct.pack(">6H", 2026, 1, 1, 0, 0, 0) + b"acsp" + b"\0" * 4 + struct.pack(">I", 0)
              + b"\0" * 8 + b"\0" * 8 + struct.pack(">I", 0) + b"".join(_s15(c) for c in D50_XYZ)
              + b"\0" * 4 + b"\0" * 16 + b"\0" * 28)
    assert len(header) == 128
    return header + body


@lru_cache(maxsize=None)
def _icc_cached(name):
    return _build_icc(name)


def icc_bytes(name):
    """ICC profile (bytes) for 'srgb', 'display_p3' or 'adobe_rgb'."""
    return _icc_cached(canonical(name))


def resolve_icc(icc):
    """io helpers: None/False/'' -> None, bytes -> as is, name -> generated profile."""
    if icc is None or icc is False or icc == "":
        return None
    if isinstance(icc, (bytes, bytearray)):
        return bytes(icc)
    return icc_bytes(icc)
