"""The atlas (astroforge.atlas + /api/atlas): distances, galactic coordinates, the register of the targets
photographed, the scene for the 3D map, and the hook that fills it when a job ends."""
import time

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from astroforge import atlas, catalogs
from astroforge.gui import server

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def empty_atlas(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROFORGE_ATLAS", str(tmp_path / "atlas"))


def _fits(path, obj=None, shape=(60, 90)):
    h = fits.Header()
    if obj:
        h["OBJECT"] = obj
    img = (0.05 + np.random.default_rng(3).normal(0, 0.003, (3,) + shape)).astype(np.float32)
    fits.PrimaryHDU(np.clip(img, 0, 1), h).writeto(path, overwrite=True)
    return str(path)


def test_distances_resolve_and_messier_is_complete():
    d = atlas.distances()
    assert len(d) > 300
    for m in range(1, 111):
        if m in (40, 73, 102):               # asterisms / the debated M102: may be missing from the catalogue
            continue
        o = catalogs.find(f"M{m}")
        assert o is not None and o["id"] in d, f"M{m}"
    assert 1200 < d[catalogs.find("M42")["id"]] < 1500
    assert 2.3e6 < d[catalogs.find("M31")["id"]] < 2.7e6


def test_galactic_coordinates():
    m31 = catalogs.find("M31")
    l, b = atlas.galactic_lb(m31["ra"], m31["dec"])
    assert abs(l - 121.17) < 0.05 and abs(b + 21.57) < 0.05
    l, b = atlas.galactic_lb(266.417, -29.008)            # Sgr A*: the galactic centre is l = 0, b = 0
    assert min(l, 360 - l) < 0.1 and abs(b) < 0.1
    x, y, z = atlas.galactic_xyz(266.417, -29.008, 26700)
    assert x > 26600 and abs(y) < 60 and abs(z) < 60        # +x points to the galactic centre


def test_record_scene_and_remove(tmp_path):
    f1, f2 = tmp_path / "a.fits", tmp_path / "a_final.jpg"
    t = atlas.record("M 42", file=f1, kind="stack", exposure_s=3600, frames=12, date="2026-11-01", thumb=b"jpg")
    assert t["name"] == "M42" and len(t["sessions"]) == 1
    atlas.record("NGC 1976", file=f2, kind="final", thumb=b"jpg2")          # the same object under another name
    atlas.record("M42", file=f1, kind="stack", exposure_s=7200, frames=24)   # the same file again replaces itself
    assert atlas.record("uno sconosciuto", file=tmp_path / "x.fits") is None  # no catalogue object, no coordinates
    s = atlas.scene()
    got = [o for o in s["objects"] if o["captured"]]
    assert len(got) == 1 and got[0]["name"] == "M42" and got[0]["known"]
    assert got[0]["exposure_s"] == 7200 and got[0]["frames"] == 24 and len(got[0]["sessions"]) == 2
    assert got[0]["thumb"] == atlas.thumb_name(f2)                           # the final picture is the one shown
    assert s["stats"]["targets"] == 1 and s["stats"]["catalogue"] > 300
    assert abs(np.linalg.norm(got[0]["xyz"]) - got[0]["ly"]) < 1
    key = got[0]["key"]
    assert atlas.remove(key, str(f2)) and len(atlas.load()["targets"][key]["sessions"]) == 1
    assert not (atlas.folder() / atlas.thumb_name(f2)).exists()
    assert atlas.remove(key) and not atlas.load()["targets"] and not atlas.remove(key)


def test_a_field_without_a_catalogue_object_is_placed_by_its_coordinates(tmp_path):
    t = atlas.record("cometa", 83.8, -5.4, file=tmp_path / "c.fits")
    assert t and t["key"].startswith("pos:")
    o = next(o for o in atlas.scene()["objects"] if o["captured"])
    assert not o["known"] and o["name"] == "cometa"


def test_api_add_by_header_name_and_thumbnail(tmp_path):
    path = _fits(tmp_path / "stack.fits", "NGC 7000")
    r = client.post("/api/atlas/add", json={"path": path})
    assert r.status_code == 200 and r.json()["target"]["name"] == "NGC 7000"
    s = client.get("/api/atlas").json()
    o = next(o for o in s["objects"] if o["captured"])
    assert 2000 < o["ly"] < 3000 and o["thumb"]
    img = client.get(f"/api/atlas/thumb/{o['thumb']}")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    assert client.get("/api/atlas/thumb/..%2fatlas.json").status_code == 404
    assert client.post("/api/atlas/remove", json={"key": o["key"]}).status_code == 200
    assert client.post("/api/atlas/remove", json={"key": o["key"]}).status_code == 404


def test_api_add_asks_for_a_name_when_the_image_says_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(server.pipeline, "load_config", lambda: {"tools": {}})      # no ASTAP: no plate solve
    path = _fits(tmp_path / "anon.fits")
    r = client.post("/api/atlas/add", json={"path": path})
    assert r.status_code == 400 and "non riconosciuto" in r.json()["detail"]
    r = client.post("/api/atlas/add", json={"path": path, "name": "M 31"})
    assert r.status_code == 200 and r.json()["target"]["name"] == "M31"


def test_a_finished_export_is_registered(tmp_path):
    path = _fits(tmp_path / "stack_integration.fits", "M 8")
    r = client.post("/api/post", json={"input": path, "steps": [], "name": "final", "formats": ["jpg"]})
    assert r.status_code == 200
    jid = r.json()["job"]
    for _ in range(200):
        if client.get(f"/api/jobs/{jid}").json()["status"] != "running" and client.get(f"/api/jobs/{jid}").json()["status"] != "queued":
            break
        time.sleep(0.1)
    for _ in range(100):                          # the atlas is written in the background, after the job
        got = [o for o in atlas.scene()["objects"] if o["captured"]]
        if got:
            break
        time.sleep(0.1)
    assert got and got[0]["name"] == "M8" and got[0]["sessions"][0]["kind"] == "final"


def test_a_picture_made_elsewhere_is_placed_by_the_name_of_its_file_or_folder(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(server.pipeline, "load_config", lambda: {"tools": {}})      # no ASTAP
    assert server._object_from_names(tmp_path / "M31 - Andromeda.jpg") == "M 31"
    assert server._object_from_names(tmp_path / "NGC 7000 - Nord America" / "final.tif") == "NGC 7000"
    assert server._object_from_names(tmp_path / "ngc6888_final.jpg") == "NGC 6888"
    assert server._object_from_names(tmp_path / "sh2-155 cave.png") == "Sh2-155"
    assert server._object_from_names(tmp_path / "IMG_0031.jpg") == ""
    assert server._object_from_names(tmp_path / "DSC_M5000.jpg") == ""
    folder = tmp_path / "IC 1396 - Proboscide"
    folder.mkdir()
    pic = folder / "finale.jpg"
    Image.fromarray(np.full((40, 60, 3), 90, np.uint8)).save(pic)
    r = client.post("/api/atlas/add", json={"path": str(pic)})
    assert r.status_code == 200 and r.json()["target"]["name"] == "IC 1396"
    assert r.json()["target"]["sessions"][0]["kind"] == "final"


def test_observing_site_defaults_to_rieti_and_can_be_changed():
    assert client.get("/api/atlas/site").json()["name"] == "Rieti, Italia"
    assert client.get("/api/atlas").json()["site"]["lat"] == pytest.approx(42.404)
    r = client.post("/api/atlas/site", json={"name": "Campo Imperatore", "lat": 42.44, "lon": 13.56})
    assert r.status_code == 200 and client.get("/api/atlas/site").json()["name"] == "Campo Imperatore"
    assert client.post("/api/atlas/site", json={"name": "x", "lat": 95, "lon": 0}).status_code == 400
    assert client.post("/api/atlas/site", json={"lat": "nord"}).status_code == 400
    o = next(o for o in client.get("/api/atlas").json()["objects"] if o["name"] == "M51")
    assert o["mag"] and o["size_arcmin"] and o["size2_arcmin"]          # what the integration advice needs


def test_sky_preview_is_downloaded_once_and_falls_back(monkeypatch):
    calls = []

    def cds(o, fov, timeout):
        calls.append("cds")
        raise OSError("CDS giù")

    def skyview(o, fov, timeout):
        calls.append("skyview")
        assert 0.25 <= fov <= 8
        return bytes([0xFF, 0xD8]) + b"jpeg"

    monkeypatch.setitem(atlas._CDS_DOWN, "until", 0.0)
    monkeypatch.setattr(atlas, "_sky_hips2fits", cds)
    monkeypatch.setattr(atlas, "_sky_skyview", skyview)
    r = client.get("/api/atlas/sky", params={"id": "M 51"})
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8" and calls == ["cds", "skyview"]
    assert client.get("/api/atlas/sky", params={"id": "NGC 5194"}).status_code == 200 and len(calls) == 2   # cached
    assert client.get("/api/atlas/sky", params={"id": "non esiste"}).status_code == 404
    assert client.get("/api/atlas/sky", params={"id": "M 101"}).status_code == 200
    assert calls == ["cds", "skyview", "skyview"]          # CDS failed: not asked again for a while (it can hang)
    monkeypatch.setattr(atlas, "_sky_skyview", cds)
    assert client.get("/api/atlas/sky", params={"id": "M 52"}).status_code == 502


def test_prefetch_downloads_the_missing_previews_once(monkeypatch):
    got = []
    monkeypatch.setitem(atlas._PREFETCH, "started", False)
    monkeypatch.setenv("ASTROFORGE_ATLAS_PREFETCH", "1")
    monkeypatch.setattr(atlas, "sky_preview", lambda oid: got.append(oid))
    assert atlas.prefetch_sky(["M 1", "M 2"], pause=0) and not atlas.prefetch_sky(["M 3"], pause=0)
    for _ in range(50):
        if len(got) == 2:
            break
        time.sleep(0.05)
    assert got == ["M 1", "M 2"]
