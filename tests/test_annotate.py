"""Object annotation (offline catalogues) and colour management. Run: .venv\\Scripts\\python.exe -m pytest tests"""
import io as _io

import numpy as np
import pytest
import tifffile
from astropy.wcs import WCS
from PIL import Image, ImageCms

from astroforge import catalogs, colorspace, io, plugins
from astroforge.plugins import annotate  # noqa: F401  (registers annotate.objects)

RA0, DEC0, SCALE, W, H = 270.9, -24.38, 1.5, 3000, 2000


def m8_wcs(w=W, h=H, scale=SCALE):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [RA0, DEC0]
    wcs.wcs.crpix = [(w - 1) / 2 + 1, (h - 1) / 2 + 1]   # FITS 1-based -> image centre at 0-based (w-1)/2
    wcs.wcs.cdelt = [-scale / 3600, scale / 3600]
    return wcs


def tan_pixel(ra, dec, w=W, h=H, scale=SCALE):
    """Gnomonic projection by hand (independent of astropy) for the WCS above."""
    a, d, a0, d0 = map(np.radians, (ra, dec, RA0, DEC0))
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    s = np.radians(scale / 3600)
    return (w - 1) / 2 - xi / s, (h - 1) / 2 + eta / s


def test_catalogue_loads_offline():
    cat = catalogs.load()
    assert len(cat["id"]) > 13000
    assert len(set(cat["messier"][cat["messier"] > 0])) >= 109
    assert len(catalogs.bright_stars()["name"]) > 150


@pytest.mark.parametrize("name, ra, dec, alt", [
    ("M42", 83.8187, -5.3897, "NGC 1976"),
    ("M8", 270.9220, -24.3802, "NGC 6523"),
    ("M31", 10.6848, 41.2691, "NGC 224"),
])
def test_messier_coordinates(name, ra, dec, alt):
    o = catalogs.find(name)
    assert o is not None and alt in o["names"]
    assert abs(o["ra"] - ra) < 0.01 and abs(o["dec"] - dec) < 0.01
    assert catalogs.find(alt.replace(" ", ""))["id"] == o["id"]
    assert catalogs.find("Lagoon Nebula")["names"][0] == "M8"


def test_objects_in_field_m8():
    objs = catalogs.objects_in_field(m8_wcs(), (H, W, 3))
    by = {n: o for o in objs for n in o["names"]}
    assert "M8" in by and "NGC 6530" in by
    for name in ("M8", "NGC 6530"):
        o = by[name]
        ex, ey = tan_pixel(o["ra"], o["dec"])
        assert abs(o["x"] - ex) < 0.5 and abs(o["y"] - ey) < 0.5
        assert 0 <= o["x"] < W and 0 <= o["y"] < H
    m8 = by["M8"]
    # 45' x 30' at 1.5"/px -> semi-axes 900 x 600 px (PA unknown -> major axis north-south)
    assert abs(m8["a"] - 900) < 2 and abs(m8["b"] - 600) < 2
    assert abs(abs(m8["angle"]) - 90) < 1
    assert "Lagoon Nebula" in m8["label"]
    # NGC 6530 lies east of M8 (east = smaller x with this WCS) and slightly north (larger y)
    assert by["NGC 6530"]["x"] < m8["x"] and by["NGC 6530"]["y"] > m8["y"]


def test_objects_in_field_scaled_and_filtered():
    full = {o["id"]: o for o in catalogs.objects_in_field(m8_wcs(), (H, W))}
    half = {o["id"]: o for o in catalogs.objects_in_field(m8_wcs(), (H // 2, W // 2), wcs_shape=(H, W))}
    o, p = full["NGC 6523"], half["NGC 6523"]
    assert abs(p["x"] - o["x"] / 2) < 0.5 and abs(p["a"] - o["a"] / 2) < 0.5
    big = catalogs.objects_in_field(m8_wcs(), (H, W), min_size_px=500)
    assert all(o["messier"] or 2 * o["a"] >= 500 for o in big if o["kind"] != "star")


def test_ellipse_position_angle():
    jac = catalogs.local_jacobian(m8_wcs(), RA0, DEC0)
    a, b, ang = catalogs.sky_ellipse_to_pixels(jac, np.array([10.0]), np.array([5.0]), np.array([90.0]))
    # PA 90 = major axis east-west -> horizontal in the image
    assert abs(a[0] - 200) < 1 and abs(b[0] - 100) < 1 and abs(np.sin(np.radians(ang[0]))) < 0.01


def test_plugin_writes_annotated_jpeg(tmp_path):
    rng = np.random.default_rng(1)
    img = (0.1 + 0.02 * rng.random((H // 2, W // 2, 3))).astype(np.float32)
    ctx = plugins.Context(log=lambda m: None, out_dir=tmp_path)
    ctx.layers["wcs"] = m8_wcs()
    ctx.layers["wcs_shape"] = (H, W)
    pl = plugins.get("annotate.objects")
    out = pl.run(img, ctx, **pl.resolve({"grid": True, "filename": "m8_annotated"}))
    assert out is img                                          # annotation is an output only
    path = tmp_path / "m8_annotated.jpg"
    assert path.exists()
    ann = np.asarray(Image.open(path), np.float32) / 255
    assert ann.shape == img.shape and np.abs(ann - img).max() > 0.3   # something was drawn
    names = [n for o in ctx.layers["annotations"] for n in o["names"]]
    assert "M8" in names and "NGC 6530" in names
    applied = pl.run(img, ctx, **pl.resolve({"apply": True, "filename": "x"}))
    assert applied.shape == img.shape and np.abs(applied - img).max() > 0.3


def test_plugin_without_wcs_or_astap_is_noop(tmp_path):
    img = np.full((50, 60, 3), 0.2, np.float32)
    ctx = plugins.Context(log=lambda m: None, out_dir=tmp_path, tools={})
    pl = plugins.get("annotate.objects")
    assert pl.run(img, ctx, **pl.resolve({})) is img
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["srgb", "display_p3", "adobe_rgb"])
def test_icc_profiles_open_and_match_convert(name):
    prof = ImageCms.ImageCmsProfile(_io.BytesIO(colorspace.icc_bytes(name)))
    assert ImageCms.getProfileDescription(prof).strip()
    rng = np.random.default_rng(2)
    arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    src = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    lcms = np.asarray(ImageCms.profileToProfile(Image.fromarray(arr), src, prof, renderingIntent=1), np.float32)
    ours = colorspace.convert(arr.astype(np.float32) / 255, "srgb", name) * 255
    assert np.abs(lcms - ours).max() <= 1.5


def test_srgb_p3_round_trip():
    rng = np.random.default_rng(3)
    img = rng.random((40, 40, 3)).astype(np.float32)
    p3 = colorspace.convert(img, "srgb", "display_p3")
    assert np.abs(p3 - img).max() > 0.01
    back = colorspace.convert(p3, "display_p3", "srgb")
    assert np.abs(back - img).max() < 1 / 255
    mono = img[..., :1]
    assert colorspace.convert(mono, "srgb", "display_p3") is mono


def test_profiles_embedded_in_files(tmp_path):
    img = np.random.default_rng(4).random((20, 30, 3)).astype(np.float32)
    io.save_tiff(img, tmp_path / "a.tif")
    io.save_jpeg(img, tmp_path / "a.jpg")
    io.save_png(img, tmp_path / "a.png", icc="display_p3")
    io.save_tiff(img, tmp_path / "p3.tif", icc="display_p3")
    with tifffile.TiffFile(tmp_path / "a.tif") as t:
        assert bytes(t.pages[0].tags[34675].value) == colorspace.icc_bytes("srgb")
    with tifffile.TiffFile(tmp_path / "p3.tif") as t:
        assert bytes(t.pages[0].tags[34675].value) == colorspace.icc_bytes("display_p3")
    assert Image.open(tmp_path / "a.jpg").info.get("icc_profile") == colorspace.icc_bytes("srgb")
    assert Image.open(tmp_path / "a.png").info.get("icc_profile") == colorspace.icc_bytes("display_p3")
    io.save_jpeg(img, tmp_path / "none.jpg", icc=None)
    assert not Image.open(tmp_path / "none.jpg").info.get("icc_profile")
    io.save_tiff(img[..., :1], tmp_path / "mono.tif")          # mono: no RGB profile
    with tifffile.TiffFile(tmp_path / "mono.tif") as t:
        assert 34675 not in t.pages[0].tags


def test_pipeline_colorspace_export(tmp_path):
    from astroforge import pipeline
    rng = np.random.default_rng(5)
    lin = (0.01 + 0.002 * rng.random((64, 64, 3))).astype(np.float32)
    lin[20:40, 20:40] += 0.3
    src = tmp_path / "t.fits"
    io.save_fits(lin, src)
    preset = {"name": "p3", "steps": [{"plugin": "stretch.statistical", "params": {"target": 0.2}}]}
    out = pipeline.run(src, preset, out_dir=tmp_path, previews=False, tools={}, colorspace="display_p3")
    jpg = Image.open(tmp_path / "t_p3.jpg")
    assert jpg.info.get("icc_profile") == colorspace.icc_bytes("display_p3")
    with tifffile.TiffFile(tmp_path / "t_p3.tif") as t:
        assert bytes(t.pages[0].tags[34675].value) == colorspace.icc_bytes("display_p3")
        tif = t.asarray().astype(np.float32) / 65535
    assert np.abs(tif - colorspace.convert(out, "srgb", "display_p3")).max() < 1e-3
    fits_img, _ = io.load(tmp_path / "t_p3.fits")
    assert np.abs(fits_img - out).max() < 1e-6                     # FITS untouched
