"""Look measurement / matching, automatic profile choice, profile names."""
import json

import numpy as np
import pytest

from astroforge import look, pipeline


def _scene(sky=(0.12, 0.10, 0.08), neb=0.35, sat=0.3, seed=0):
    """Stretched-looking image: tinted sky, a red-ish nebula blob, a few stars."""
    rng = np.random.default_rng(seed)
    h, w = 400, 600
    y, x = np.mgrid[:h, :w]
    blob = np.exp(-(((x - 300) / 120.0) ** 2 + ((y - 200) / 90.0) ** 2))
    img = np.empty((h, w, 3), np.float32)
    col = np.array([1 + sat, 1 - sat / 2, 1 - sat / 2])
    for c in range(3):
        img[..., c] = sky[c] + (neb - np.mean(sky)) * blob * col[c]
    img += rng.normal(0, 0.004, img.shape)
    for _ in range(150):
        yy, xx = rng.integers(3, h - 3), rng.integers(3, w - 3)
        img[yy - 1:yy + 2, xx - 1:xx + 2] = 0.95
    return np.clip(img, 0, 1).astype(np.float32)


def test_measure_reads_sky_and_tint():
    m = look.measure(_scene())
    assert abs(m["sky"] - 0.10) < 0.01
    assert m["sky_rgb"][0] > m["sky_rgb"][2] + 0.03
    assert m["nebula"] > m["sky"] + 0.1 and m["hues"]["red"] > 0.5


def test_measure_ignores_painted_frame():
    img = _scene()
    framed = np.zeros((440, 640, 3), np.float32)
    framed[20:420, 20:620] = img
    a, b = look.measure(img), look.measure(framed)
    assert abs(a["sky"] - b["sky"]) < 0.01 and abs(a["nebula"] - b["nebula"]) < 0.02


def test_match_moves_towards_target_and_neutralises_sky():
    img = _scene(sky=(0.16, 0.13, 0.10), neb=0.3, sat=0.15)
    out = look.match(img, sky=0.07, nebula=0.40, saturation=0.40, white=0.95)
    m0, m = look.measure(img), look.measure(out)
    assert abs(m["sky"] - 0.07) < 0.015
    assert np.ptp(m["sky_rgb"]) < 0.01 < np.ptp(m0["sky_rgb"])
    assert abs(m["nebula"] - 0.40) < abs(m0["nebula"] - 0.40)
    assert abs(m["saturation"] - 0.40) < abs(m0["saturation"] - 0.40)
    assert out.dtype == np.float32 and out.shape == img.shape


def test_match_strength_zero_keeps_image():
    img = _scene()
    out = look.match(img, sky=0.02, nebula=0.6, saturation=0.8, white=1.0, strength=0.0, neutral_sky=False)
    assert np.abs(out - img).max() < 1e-4


def test_match_is_limited_on_strange_targets():
    img = _scene()
    out = look.match(img, sky=0.3, nebula=0.9, saturation=1.0)
    m = look.measure(out)
    assert m["sky"] < 0.2 and np.isfinite(out).all()


def test_distance_zero_for_same_look():
    m = look.measure(_scene())
    assert look.distance(m, m)["total"] == 0


def test_look_match_plugin_in_a_preset(tmp_path):
    from astroforge import plugins
    pl = plugins.get("look.match")
    out = pl.run(_scene(), plugins.Context(log=lambda m: None), **pl.resolve({"sky": 0.08}))
    assert abs(look.measure(out)["sky"] - 0.08) < 0.015


@pytest.mark.parametrize("path,header,n,expect", [
    ("work/stack_SHO.fits", {}, 3, "sho_starless"),
    ("x/whatever.fits", {"PALETTE": "HOO"}, 3, "hoo_starless"),
    ("x/stack_integration.fits", {}, 1, "mono"),
    ("x/M31_stack.fits", {}, 3, "galaxy"),
    ("x/stack.fits", {"OBJECT": "NGC 891"}, 3, "galaxy"),
    ("x/M8_stack_integration.fits", {}, 3, pipeline.AUTO_PRESETS["RGB"]),
])
def test_suggest_preset(path, header, n, expect):
    assert pipeline.suggest_preset(path, header, n)[0] == expect
    assert (pipeline.PRESETS / f"{expect}.json").exists()


def test_rename_preset_keeps_id_and_name_works(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PRESETS", tmp_path)
    monkeypatch.setattr(pipeline, "LABELS", tmp_path / "home" / "labels.json")
    (tmp_path / "a.json").write_text(json.dumps({"name": "a", "steps": []}))
    (tmp_path / "b.json").write_text(json.dumps({"name": "b", "steps": []}))
    pipeline.rename_preset("a", "Il mio  RGB")
    assert pipeline.load_preset("il mio rgb")["name"] == "a"
    assert pipeline.load_preset("a")["label"] == "Il mio RGB"
    assert "label" not in json.loads((tmp_path / "a.json").read_text())      # shipped file untouched
    with pytest.raises(ValueError):
        pipeline.rename_preset("b", "il mio rgb")
    with pytest.raises(ValueError):
        pipeline.rename_preset("b", "a")


def test_gui_rename_and_profile_from_image(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from PIL import Image
    from astroforge.gui.server import create_app
    monkeypatch.setattr(pipeline, "PRESETS", tmp_path)
    monkeypatch.setattr(pipeline, "LABELS", tmp_path / "labels.json")
    (tmp_path / "nebula_pro.json").write_text(json.dumps({"name": "nebula_pro", "steps": [{"plugin": "crop.auto"}]}))
    ref = tmp_path / "final.jpg"
    Image.fromarray((_scene() * 255).astype(np.uint8)).save(ref)
    c = TestClient(create_app())
    r = c.post("/api/presets/nebula_pro/rename", json={"label": "Nebulose a colori"}).json()
    assert r["label"] == "Nebulose a colori"
    r = c.post("/api/presets_from_image", json={"path": str(ref), "base": "nebula_pro", "label": "come finale"}).json()
    p = json.loads((tmp_path / f"{r['id']}.json").read_text(encoding="utf-8"))
    assert p["label"] == "come finale" and p["steps"][-1]["plugin"] == "look.match"
    assert abs(p["steps"][-1]["params"]["sky"] - 0.10) < 0.02
    names = {x["id"]: x.get("label") for x in c.get("/api/presets").json()}
    assert names["nebula_pro"] == "Nebulose a colori"
    assert c.get("/api/suggest_preset", params={"path": str(tmp_path / "x_SHO.fits")}).json()["id"] == "sho_starless"


def _star_layer():
    y, x = np.mgrid[:80, :80]
    st = np.zeros((80, 80, 3), np.float32)
    for cx, cy, s, a in ((25, 25, 2.5, 1.0), (55, 55, 4.0, 0.6)):
        st += (a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)))[..., None]
    return st


def test_star_morph_shrinks_edges_and_keeps_cores():
    from astroforge import plugins
    pl = plugins.get("stars.morph")
    st = _star_layer()
    for keep, peak_min in ((1.0, 0.99), (0.0, 0.0)):
        ctx = plugins.Context(log=lambda m: None)
        ctx.layers["stars"] = st.copy()
        img = np.zeros_like(st)
        assert pl.run(img, ctx, **pl.resolve({"keep_core": keep})) is img       # works on the layer only
        o = ctx.layers["stars"]
        assert (o[..., 0] > 0.1).sum() < 0.85 * (st[..., 0] > 0.1).sum()       # smaller stars
        if keep:
            assert o[25, 25, 0] >= peak_min * st[25, 25, 0] and o[55, 55, 0] >= peak_min * st[55, 55, 0]
        else:
            assert o[25, 25, 0] < 0.9                                          # plain erosion dims the cores


def test_starless_profiles_are_valid():
    for name in ("rgb_starless", "sho_starless", "hoo_starless"):
        p = pipeline.load_preset(name)
        assert pipeline.validate(p) == []
        ids = [s["plugin"] for s in p["steps"]]
        assert ids.index("stars.starnet") < ids.index("stars.morph") < ids.index("stars.recombine") < ids.index("look.match")


def test_match_lowers_bright_faint_background():
    img = _scene()
    rng = np.random.default_rng(3)
    from scipy import ndimage as ndi
    dust = ndi.gaussian_filter(rng.random(img.shape[:2]), 12)
    dust = (dust - dust.min()) / np.ptp(dust)
    img = np.clip(img + 0.25 * dust[..., None] * (img < 0.3), 0, 1).astype(np.float32)
    m0 = look.measure(img)
    out = look.match(img, sky=m0["sky"], faint=0.02, nebula=m0["nebula"], saturation=m0["saturation"])
    m = look.measure(out)
    assert m["faint"] < m0["faint"] - 0.01
    assert abs(m["nebula"] - m0["nebula"]) < 0.04 and abs(m["sky"] - m0["sky"]) < 0.015


def test_halo_clean_removes_disc_and_keeps_nebula():
    """A dust donut (OIII and fainter Ha) in the sky goes, the Ha nebula and its OIII stay."""
    from astroforge import plugins
    rng = np.random.default_rng(5)
    h, w = 400, 500
    y, x = np.mgrid[:h, :w]
    img = np.full((h, w, 3), 0.01, np.float32)
    neb = 0.03 * np.exp(-(((x - 350) / 60.0) ** 2 + ((y - 200) / 60.0) ** 2))      # Ha nebula (G) ...
    img[..., 1] += neb
    img[..., 2] += 0.6 * neb                                                     # ... with real OIII (B)
    r = np.hypot(x - 120, y - 150)
    disc_shape = np.clip((30 - r) / 5, 0, 1)
    img[..., 2] += 0.004 * disc_shape                                            # dust donut: OIII ...
    img[..., 1] += 0.0015 * disc_shape                                           # ... and fainter in Ha
    img += rng.normal(0, 0.0006, img.shape).astype(np.float32)
    pl = plugins.get("stars.halo_clean")
    img[..., 2] += 0.003 * np.clip((25 - np.hypot(x - 160, y - 175)) / 5, 0, 1)      # a second disc overlapping
    out = pl.run(img, plugins.Context(log=lambda m: None), **pl.resolve({"max_radius": 90, "min_radius": 6}))
    two = np.hypot(x - 175, y - 190) < 12
    assert float(np.median(out[..., 2][two])) - 0.01 < 0.0015
    disc = r < 22
    before = float(np.median(img[..., 2][disc])) - 0.01
    after = float(np.median(out[..., 2][disc])) - 0.01
    assert before > 0.0035 and abs(after) < 0.25 * before                  # disc gone
    core = np.hypot(x - 350, y - 200) < 40
    assert abs(float(np.median(out[..., 2][core])) - float(np.median(img[..., 2][core]))) < 0.0005   # OIII kept
    assert abs(float(np.median(out[..., 1][disc])) - 0.01) < 0.0005                                # Ha disc gone too
    assert abs(float(np.median(out[..., 1][core])) - float(np.median(img[..., 1][core]))) < 0.0005   # Ha nebula kept


def test_gui_zoom_render_and_version(tmp_path):
    from fastapi.testclient import TestClient
    from PIL import Image
    import io as _io
    from astroforge import io
    from astroforge.gui.server import create_app
    rng = np.random.default_rng(2)
    img = (0.01 + 0.002 * rng.random((300, 400, 3))).astype(np.float32)
    img[100:103, 200:203] = 0.5
    src = tmp_path / "lin.fits"
    io.save_fits(img, src)
    c = TestClient(create_app())
    steps = [{"plugin": "stretch.auto"}]
    r = c.post("/api/preview_zoom", json={"input": str(src), "steps": steps, "crop": [0.25, 0.25, 0.75, 0.75], "width": 400})
    assert r.status_code == 200
    w, h = Image.open(_io.BytesIO(r.content)).size
    assert (w, h) == (200, 150) and float(r.headers["X-Scale"]) == 1.0          # full resolution, only the area
    r = c.post("/api/preview_zoom", json={"input": str(src), "steps": steps, "crop": [0, 0, 1, 1], "width": 100})
    assert Image.open(_io.BytesIO(r.content)).size[0] == 100                       # whole image reduced
    bad = c.post("/api/preview", json={"input": str(src), "steps": [{"plugin": "nonesiste.plugin"}]})
    assert bad.status_code == 400 and "riavvia" in bad.json()["detail"]
    v = c.get("/api/version").json()
    assert v["version"].startswith("0.") and v["restart_needed"] is False


def test_page_assets_carry_the_version_and_are_not_cached():
    from fastapi.testclient import TestClient
    from astroforge.gui.server import create_app
    c = TestClient(create_app())
    r = c.get("/")
    assert r.headers["cache-control"] == "no-cache"
    assert '/static/app.js?v=' in r.text and '/static/style.css?v=' in r.text
    assert c.get("/static/app.js").headers["cache-control"] == "no-cache"


def test_reset_forgets_preview_caches():
    from fastapi.testclient import TestClient
    from astroforge.gui import server
    c = TestClient(server.create_app())
    server._PREVIEW_CACHE["x"] = 1
    server._STARNET_CACHE["y"] = 2
    assert c.post("/api/reset").json()["ok"] and not server._PREVIEW_CACHE and not server._STARNET_CACHE
