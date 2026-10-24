"""Minimal XISF 1.0 reader/writer (PixInsight format), enough for masters and integrations.

Supports monolithic files, attached blocks, planar/normal storage, zlib compression with
optional byte shuffling. Values are returned as float32 (H, W, C).
"""
import re
import struct
import zlib
import xml.etree.ElementTree as ET

import numpy as np

_DTYPES = {"UInt8": np.uint8, "UInt16": np.uint16, "UInt32": np.uint32,
           "Float32": np.float32, "Float64": np.float64}


def _unshuffle(raw, itemsize):
    a = np.frombuffer(raw, np.uint8)
    n = len(a) // itemsize
    return a[:n * itemsize].reshape(itemsize, n).T.tobytes() + a[n * itemsize:].tobytes()


def read(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"XISF0100":
        raise ValueError(f"{path}: not an XISF file")
    hlen = struct.unpack("<I", data[8:12])[0]
    xml = data[16:16 + hlen].decode("utf-8")
    xml = re.sub(r'\sxmlns(:\w+)?="[^"]*"', "", xml)      # drop namespaces
    xml = re.sub(r'\s\w+:[\w-]+="[^"]*"', "", xml)          # and prefixed attributes (xsi:...)
    xml = xml.rstrip("\0")
    root = ET.fromstring(xml)
    img = root.find(".//Image")
    if img is None:
        raise ValueError(f"{path}: no Image element")
    w, h, c = (int(v) for v in img.get("geometry").split(":"))
    dt = np.dtype(_DTYPES[img.get("sampleFormat")]).newbyteorder("<")
    kind, pos, size = img.get("location").split(":")[:3]
    if kind != "attachment":
        raise ValueError(f"{path}: only attached image blocks are supported")
    raw = data[int(pos):int(pos) + int(size)]
    comp = img.get("compression")
    if comp:
        codec = comp.split(":")[0]
        if codec.startswith("zlib"):
            raw = zlib.decompress(raw)
            if codec.endswith("+sh"):
                raw = _unshuffle(raw, dt.itemsize)
        else:
            raise ValueError(f"{path}: compression {codec} not supported")
    arr = np.frombuffer(raw, dt)
    if img.get("pixelStorage", "Planar") == "Planar":
        arr = arr.reshape(c, h, w).transpose(1, 2, 0)
    else:
        arr = arr.reshape(h, w, c)
    arr = arr.astype(np.float32)
    if np.issubdtype(dt, np.integer):
        arr /= np.iinfo(dt).max
    keywords = {k.get("name"): k.get("value") for k in root.iter("FITSKeyword")}
    return np.ascontiguousarray(arr), keywords


def write(img, path, keywords=None):
    """Write float32 (H, W, C) as an uncompressed planar XISF."""
    h, w, c = img.shape
    planar = np.ascontiguousarray(img.transpose(2, 0, 1), dtype="<f4").tobytes()
    from xml.sax.saxutils import quoteattr
    kw = "".join(f'<FITSKeyword name={quoteattr(str(k))} value={quoteattr(str(v))} comment=""/>'
                 for k, v in (keywords or {}).items())
    cs = "RGB" if c == 3 else "Gray"

    def header(pos):
        return (f'<?xml version="1.0" encoding="UTF-8"?><xisf version="1.0" xmlns="http://www.pixinsight.com/xisf">'
                f'<Image geometry="{w}:{h}:{c}" sampleFormat="Float32" bounds="0:1" colorSpace="{cs}" '
                f'location="attachment:{pos}:{len(planar)}">{kw}</Image></xisf>').encode()

    pos = 16 + len(header(0)) + 64
    pos = (pos + 4095) // 4096 * 4096
    hdr = header(pos)
    with open(path, "wb") as f:
        f.write(b"XISF0100" + struct.pack("<I", len(hdr)) + b"\0" * 4 + hdr)
        f.write(b"\0" * (pos - 16 - len(hdr)))
        f.write(planar)
