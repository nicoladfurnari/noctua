"""GUI server endpoints: DBE samples on the preview, blink zoom crops, job queue (FIFO, cancel, history), step masks,
presets, network mode (token, local-only config, session folders)."""
import io as _io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient
from PIL import Image

from astroforge import io
from astroforge.gui import server
from astroforge.stack.project import Cancelled

client = TestClient(server.app)
H, W = 160, 200


def _stars(h, w, rng, n=40, dx=0.0, dy=0.0, pts=None):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w), 0.03, np.float32)
    for x, y, a in pts if pts is not None else [(rng.uniform(10, w - 10), rng.uniform(10, h - 10), rng.uniform(0.1, 0.6))
                                                  for _ in range(n)]:
        img += a * np.exp(-((xx - x - dx) ** 2 + (yy - y - dy) ** 2) / (2 * 1.4 ** 2))
    return img


@pytest.fixture(scope="module")
def lights(tmp_path_factory):
    root = tmp_path_factory.mktemp("gui_sky")
    (root / "lights").mkdir()
    rng = np.random.default_rng(5)
    pts = [(rng.uniform(10, W - 10), rng.uniform(10, H - 10), rng.uniform(0.1, 0.6)) for _ in range(40)]
    for k in range(4):
        img = _stars(H, W, rng, dx=k * 1.3, dy=-k * 0.8, pts=pts) + rng.normal(0, 0.003, (H, W))
        h = fits.Header({"IMAGETYP": "Light Frame", "EXPTIME": 60.0, "INSTRUME": "QHY163M", "GAIN": 100,
                         "OFFSET": 30, "XBINNING": 1, "SET-TEMP": -10.0})
        fits.PrimaryHDU((np.clip(img, 0, 1) * 65535).astype(np.uint16), h).writeto(root / "lights" / f"L{k}.fits")
    return root


@pytest.fixture(scope="module")
def linear_image(tmp_path_factory):
    """Linear color image with a gradient, 600x400 (preview at max_side 300 -> scale 0.5)."""
    p = tmp_path_factory.mktemp("gui_post") / "integ.fits"
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:400, 0:600].astype(np.float32)
    base = 0.02 + 0.03 * xx / 600 + _stars(400, 600, rng, 60) * 0.3
    img = np.repeat(base[..., None], 3, -1) + rng.normal(0, 0.002, (400, 600, 3)).astype(np.float32)
    io.save_fits(np.clip(img, 0, 1).astype(np.float32), str(p))
    return p


def _wait(jid, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.1)
    raise AssertionError(f"job {jid} still {j['status']}")


# ------------------------------------------------------------------ pages
def test_pages_load():
    assert client.get("/").status_code == 200
    for f in ("app.js", "style.css", "vendor/uPlot.iife.min.js", "vendor/uPlot.min.css"):
        assert client.get(f"/static/{f}").status_code == 200
    html = client.get("/").text        # the charts library is loaded before app.js, with the cache-busting ?v=
    assert html.index("vendor/uPlot.iife.min.js?v=") < html.index("/static/app.js?v=")
    assert client.get("/api/jobs").json().keys() == {"active", "history"}


# ------------------------------------------------------------------ DBE samples
def test_dbe_samples_auto_and_manual(linear_image):
    steps = [{"plugin": "background.dbe", "params": {"samples_per_side": 8}}]
    r = client.post("/api/dbe_samples", json={"input": str(linear_image), "steps": steps, "index": 0, "max_side": 300})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["auto"] and abs(d["scale"] - 0.5) < 1e-6 and (d["width"], d["height"]) == (300, 200)
    assert len(d["samples"]) >= 3
    for (x, y), (fx, fy) in zip(d["samples"], d["full"]):
        assert 0 <= x < 300 and 0 <= y < 200
        assert abs(fx - x / 0.5) < 0.2 and abs(fy - y / 0.5) < 0.2
    # manual samples (full resolution) come back in preview coordinates
    manual = [[100, 100], [500, 100], [300, 300], [550, 350]]
    steps[0]["params"]["samples"] = manual
    d2 = client.post("/api/dbe_samples", json={"input": str(linear_image), "steps": steps, "index": 0,
                                               "max_side": 300}).json()
    assert not d2["auto"] and d2["full"] == [[float(x), float(y)] for x, y in manual]
    assert d2["samples"][0] == [50.0, 50.0]
    # auto=true ignores them (reset to automatic)
    d3 = client.post("/api/dbe_samples", json={"input": str(linear_image), "steps": steps, "index": 0,
                                               "max_side": 300, "auto": True}).json()
    assert d3["auto"] and len(d3["samples"]) == len(d["samples"])
    # the preview with the manual samples works
    r = client.post("/api/preview", json={"input": str(linear_image), "steps": steps, "max_side": 300})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_dbe_samples_bad_index(linear_image):
    r = client.post("/api/dbe_samples", json={"input": str(linear_image),
                                              "steps": [{"plugin": "stretch.auto", "params": {}}], "index": 0})
    assert r.status_code == 400


# ------------------------------------------------------------------ blink zoom
def _jpeg_size(content):
    return Image.open(_io.BytesIO(content)).size


@pytest.mark.parametrize("bayer", [None, "RGGB"])
def test_frame_crop(tmp_path, bayer):
    cal = tmp_path / "calibrated"
    cal.mkdir()
    rng = np.random.default_rng(2)
    frame = (_stars(300, 400, rng) + rng.normal(0, 0.003, (300, 400))).astype(np.float32)
    np.save(cal / "0000.npy", frame)
    (tmp_path / "stack_report.json").write_text(json.dumps({"bayer": bayer}), encoding="utf-8")
    path = str(cal / "0000.npy")
    info = client.get("/api/frame_info", params={"path": path}).json()
    assert (info["width"], info["height"], info["bayer"]) == (400, 300, bayer)
    r = client.get("/api/frame_crop", params={"path": path, "x": 200, "y": 150, "zoom": 1, "w": 120, "h": 80})
    assert r.status_code == 200 and _jpeg_size(r.content) == (120, 80)
    assert float(r.headers["X-Cx"]) == 200 and r.headers["X-Frame-Width"] == "400"
    r = client.get("/api/frame_crop", params={"path": path, "x": 200, "y": 150, "zoom": 2, "w": 120, "h": 80})
    assert _jpeg_size(r.content) == (120, 80)
    # near the corner the crop is kept inside the frame: the center moves
    r = client.get("/api/frame_crop", params={"path": path, "x": 0, "y": 0, "zoom": 1, "w": 120, "h": 80})
    assert (float(r.headers["X-Cx"]), float(r.headers["X-Cy"])) == (60, 40)
    # zoomed out further than the frame: the whole frame
    r = client.get("/api/frame_crop", params={"path": path, "x": 200, "y": 150, "zoom": 0.5, "w": 1000, "h": 1000})
    assert _jpeg_size(r.content) == (200, 150)
    assert client.get("/api/frame", params={"path": path}).status_code == 200
    assert client.get("/api/frame_crop", params={"path": str(tmp_path / "x.fits"), "x": 0, "y": 0}).status_code == 404


# ------------------------------------------------------------------ job queue
def _fake_job(label, seconds, log):
    def fn(job):
        log.append(("start", label, time.time()))
        t = time.time()
        while time.time() - t < seconds:
            if job.cancelled:
                raise Cancelled()
            time.sleep(0.02)
        job.result = {"label": label}
        log.append(("end", label, time.time()))
    return server._start(server.Job("test", label), fn)["job"]


def test_queue_fifo_cancel_history():
    log = []
    a = _fake_job("a", 0.4, log)
    b = _fake_job("b", 0.1, log)
    c = _fake_job("c", 0.1, log)
    lst = client.get("/api/jobs").json()
    ids = [j["id"] for j in lst["active"]]
    assert ids.index(a) < ids.index(b) < ids.index(c)
    assert client.get(f"/api/jobs/{c}").json()["position"] >= 1
    assert client.post(f"/api/jobs/{b}/cancel").json()["status"] == "cancelled"   # queued: dropped at once
    assert _wait(a)["status"] == "done" and _wait(c)["status"] == "done"
    assert [e[:2] for e in log] == [("start", "a"), ("end", "a"), ("start", "c"), ("end", "c")]
    jb = client.get(f"/api/jobs/{b}").json()
    assert jb["status"] == "cancelled" and jb["started"] is None
    # a running job stops at its next check
    d = _fake_job("d", 30, log)
    t = time.time()
    while client.get(f"/api/jobs/{d}").json()["status"] != "running" and time.time() - t < 10:
        time.sleep(0.02)
    client.post(f"/api/jobs/{d}/cancel")
    assert _wait(d, 10)["status"] == "cancelled"
    hist = [j["id"] for j in client.get("/api/jobs").json()["history"]]
    assert hist[:4] == [d, c, a, b]
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_two_stack_jobs_run_sequentially(lights, tmp_path):
    def settings(name):
        return {"lights": [str(lights / "lights")], "calibration": [], "work_dir": str(tmp_path / name),
                "auto_reject": False, "cosmetic": False, "gpu": False, "satellites": False, "distortion": False}
    j1 = client.post("/api/stack", json=settings("w1")).json()["job"]
    j2 = client.post("/api/stack", json=settings("w2")).json()
    assert j2["position"] >= 1
    r1, r2 = _wait(j1, 300), _wait(j2["job"], 300)
    assert r1["status"] == "done", r1["log"][-3:]
    assert r2["status"] == "done", r2["log"][-3:]
    assert r2["started"] >= r1["finished"]
    assert r1["label"].startswith("Stacking")
    assert len(r1["result"]["report"]["frames"]) == 4
    assert (tmp_path / "w2").exists() and r2["result"]["final_input"].endswith("_integration.fits")


def test_mosaic_from_gui(tmp_path):
    from test_stack_modes import H as MH, W as MW, _cal, _sky, _write
    rng = np.random.default_rng(4)
    root = tmp_path / "mosaic"
    panels = []
    for p, x0 in enumerate((0, 280)):
        for k in range(4):
            img = 0.04 + 0.01 * p + _sky(x0, 50, k * 1.2, -k * 0.6) + rng.normal(0, 0.002, (MH, MW))
            _write(root / f"panel{p}" / f"L_{k}.fits", img + 0.01, IMAGETYP="Light Frame", EXPTIME=60.0)
        panels.append([str(root / f"panel{p}")])
    _cal(root)
    jid = client.post("/api/stack", json={"lights": [p[0] for p in panels], "mosaic": panels,
                                          "calibration": [str(root / "bias")], "work_dir": str(tmp_path / "work"),
                                          "auto_reject": False, "cosmetic": False, "gpu": False}).json()["job"]
    j = _wait(jid, 300)
    assert j["status"] == "done", j["log"][-3:]
    res = j["result"]
    assert j["label"].startswith("Mosaico 2")
    assert len(res["panels"]) == 2 and all(Path(p).exists() for p in res["panels"])
    assert res["final_input"] == res["mosaic"] and Path(res["mosaic"]).exists()
    assert client.get("/api/image", params={"path": res["mosaic"], "max_side": 400}).status_code == 200


# ------------------------------------------------------------------ masks, presets, step order
def _png(content):
    return Image.open(_io.BytesIO(content))


def _jpeg_array(content):
    return np.asarray(Image.open(_io.BytesIO(content)).convert("RGB"), np.float32)


MASK = {"type": "range", "lower": 0.3, "upper": 1.0, "fuzziness": 0.2, "smoothness": 4}


def test_mask_png(linear_image):
    steps = [{"plugin": "stretch.auto", "params": {}},
             {"plugin": "contrast.saturation", "params": {"amount": 1.5}, "mask": MASK}]
    body = {"input": str(linear_image), "steps": steps, "index": 1, "max_side": 300}
    r = client.post("/api/mask", json=body)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    im = _png(r.content)
    assert im.mode == "L" and im.size == (300, 200)
    m = np.asarray(im, np.float32)
    assert 0 < m.mean() < 255 and m.max() > m.min()
    # invert -> complementary mask; an override spec in the body wins over the step's own
    inv = np.asarray(_png(client.post("/api/mask", json=body | {"mask": MASK | {"invert": True}}).content), np.float32)
    assert np.abs(inv + m - 255).mean() < 1.5
    # matches masks.make_mask on the preview image after the steps before it (ctx.scale applied)
    from astroforge import masks, plugins
    img = server._load_small(linear_image, 300).copy()
    ctx = plugins.Context(scale=img.shape[1] / 600)
    ref = masks.make_mask(server._preview_steps(img, steps[:1], ctx), MASK, ctx)
    assert np.abs(ref * 255 - m).mean() < 1.0
    # no mask / bad index / bad spec: 400
    assert client.post("/api/mask", json=body | {"index": 0}).status_code == 400
    assert client.post("/api/mask", json=body | {"index": 5}).status_code == 400
    assert client.post("/api/mask", json=body | {"mask": {"type": "nope"}}).status_code == 400


def test_preview_uses_masks(linear_image):
    base = [{"plugin": "stretch.auto", "params": {}}]
    sat = {"plugin": "contrast.saturation", "params": {"amount": 2.0}}

    def prev(steps):
        r = client.post("/api/preview", json={"input": str(linear_image), "steps": steps, "max_side": 300})
        assert r.status_code == 200, r.text
        return _jpeg_array(r.content)
    a = prev(base)
    full = prev(base + [sat])
    zero = prev(base + [sat | {"mask": {"type": "lum", "strength": 0.0}}])    # mask 0 everywhere: no effect
    part = prev(base + [sat | {"mask": MASK}])
    assert np.abs(zero - a).mean() < 0.5
    assert np.abs(full - a).mean() > np.abs(part - a).mean() > 0.01


def test_step_order_changes_preview(linear_image):
    s1 = {"plugin": "stretch.auto", "params": {}}
    s2 = {"plugin": "contrast.curves", "params": {"points": [[0, 0], [0.25, 0.5], [1, 1]]}}
    outs = []
    for steps in ([s1, s2], [s2, s1]):
        r = client.post("/api/preview", json={"input": str(linear_image), "steps": steps, "max_side": 300})
        assert r.status_code == 200, r.text
        outs.append(_jpeg_array(r.content))
    assert np.abs(outs[0] - outs[1]).mean() > 2


def test_presets_save_load_with_masks():
    from astroforge import pipeline
    name = "zz_gui_test_masks"
    f = pipeline.PRESETS / f"{name}.json"
    steps = [{"plugin": "stretch.auto", "params": {"target": 0.2}, "enabled": True},
             {"plugin": "contrast.saturation", "params": {"amount": 1.3}, "mask": MASK | {"invert": True}},
             {"plugin": "color.scnr", "params": {}, "enabled": False,
              "mask": {"type": "color", "color": "green", "width": 50, "blur": 2, "strength": 0.8}}]
    try:
        r = client.post(f"/api/presets/{name}", json={"name": name, "description": "test", "steps": steps})
        assert r.status_code == 200 and r.json()["id"] == name
        got = client.get(f"/api/presets/{name}").json()
        assert got["steps"] == steps and got["id"] == name
        listed = next(p for p in client.get("/api/presets").json() if p["id"] == name)
        assert listed["steps"][1]["mask"] == MASK | {"invert": True}
        assert pipeline.validate(pipeline.load_preset(name)) == []
        assert client.get("/api/presets/does_not_exist").status_code == 404
    finally:
        f.unlink(missing_ok=True)


# ------------------------------------------------------------------ network mode (token, local-only config, folders)
def test_localhost_mode_unchanged(tmp_path, monkeypatch):
    from astroforge import pipeline
    monkeypatch.setattr(pipeline, "CONFIG", tmp_path / "config.json")
    assert server.app.state.remote is False and server.app.state.token is None
    for host in ("127.0.0.1", "localhost", "::1"):
        a = server.create_app(host, token="ignored")
        c = TestClient(a)
        assert not a.state.remote
        assert c.get("/").status_code == 200 and c.get("/api/jobs").status_code == 200
    # config from a "non-local" test client is fine on localhost
    r = TestClient(server.create_app("127.0.0.1")).post("/api/config", json={"tools": {"astap": "C:/x/astap.exe"}})
    assert r.status_code == 200 and r.json()["tools"]["astap"]["path"] == "C:/x/astap.exe"


def test_network_mode_requires_token(monkeypatch):
    a = server.create_app("0.0.0.0", token="s3cret-token")
    assert a.state.remote and a.state.token == "s3cret-token"
    c = TestClient(a)
    for url in ("/", "/static/app.js", "/api/jobs", "/api/plugins"):
        assert c.get(url).status_code == 401, url
    assert c.get("/api/jobs", headers={"X-AstroForge-Token": "wrong"}).status_code == 401
    assert c.get("/api/jobs?token=wrong").status_code == 401
    assert c.get("/api/jobs", headers={"X-AstroForge-Token": "s3cret-token"}).status_code == 200
    # first visit with ?token= sets the cookie, then plain requests work
    r = c.get("/?token=s3cret-token")
    assert r.status_code == 200 and "af_token" in r.cookies
    assert c.get("/api/jobs").status_code == 200 and c.get("/static/style.css").status_code == 200
    # token from the environment, or a random one
    monkeypatch.setenv("ASTROFORGE_TOKEN", "from-env")
    assert server.create_app("192.168.1.10").state.token == "from-env"
    monkeypatch.delenv("ASTROFORGE_TOKEN")
    t1, t2 = server.create_app("0.0.0.0").state.token, server.create_app("0.0.0.0").state.token
    assert t1 != t2 and len(t1) >= 20


def test_network_mode_config_only_from_localhost(tmp_path, monkeypatch):
    from astroforge import pipeline
    monkeypatch.setattr(pipeline, "CONFIG", tmp_path / "config.json")
    a = server.create_app("0.0.0.0", token="tok")
    hdr = {"X-AstroForge-Token": "tok"}
    remote = TestClient(a, headers=hdr, client=("192.168.1.50", 50000))
    r = remote.post("/api/config", json={"tools": {"starnet": "C:/evil.exe"}})
    assert r.status_code == 403 and "127.0.0.1" in r.json()["detail"]
    assert not (tmp_path / "config.json").exists()
    assert remote.get("/api/config").status_code == 200          # reading is fine
    local = TestClient(a, headers=hdr, client=("127.0.0.1", 50001))
    assert local.post("/api/config", json={"tools": {"starnet": "C:/ok/starnet2.exe"}}).status_code == 200
    assert json.loads((tmp_path / "config.json").read_text())["tools"]["starnet"] == "C:/ok/starnet2.exe"


def test_network_mode_session_folders(linear_image, tmp_path):
    a = server.create_app("0.0.0.0", token="tok")
    c = TestClient(a, headers={"X-AstroForge-Token": "tok"})
    body = {"input": str(linear_image), "steps": [], "max_side": 300}
    assert c.get("/api/image", params={"path": str(linear_image), "max_side": 200}).status_code == 403
    assert c.post("/api/preview", json=body).status_code == 403
    # a drive root or the home folder is never a session folder
    roots = c.post("/api/roots", json={"paths": [str(Path(linear_image).anchor), str(Path.home())]}).json()
    assert roots["remote"] and roots["roots"] == []
    # the file chosen in the browser adds its folder
    assert c.post("/api/roots", json={"paths": [str(linear_image)]}).json()["roots"]
    assert c.get("/api/image", params={"path": str(linear_image), "max_side": 200}).status_code == 200
    assert c.post("/api/preview", json=body).status_code == 200
    # export only into a session folder
    other = tmp_path / "elsewhere"
    assert c.post("/api/post", json=body | {"out_dir": str(other)}).status_code == 403
    # the server-wide local app is not affected
    assert client.get("/api/image", params={"path": str(linear_image), "max_side": 200}).status_code == 200


# ------------------------------------------------------------------ pixel value (GHS symmetry point), painted masks
def test_pixel_value(linear_image):
    body = {"input": str(linear_image), "steps": [], "index": 0, "max_side": 300, "x": 150.5, "y": 100.5}
    r = client.post("/api/pixel", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    img = server._load_small(linear_image, 300)
    ref = np.median(img[98:103, 148:153].reshape(-1, 3), 0)
    assert np.allclose(d["channels"], ref, atol=1e-6) and abs(d["mean"] - ref.mean()) < 1e-6
    assert abs(d["value"] - float(ref @ [0.2126, 0.7152, 0.0722])) < 2e-3
    assert d["full"] == [301.0, 201.0] and abs(d["scale"] - 0.5) < 1e-6 and (d["width"], d["height"]) == (300, 200)
    assert d["linear"] is True
    # the image entering a later step: after the stretch the values are higher
    steps = [{"plugin": "stretch.auto", "params": {}}, {"plugin": "stretch.ghs", "params": {}}]
    d2 = client.post("/api/pixel", json=body | {"steps": steps, "index": 1}).json()
    assert d2["value"] > d["value"] + 0.05 and d2["linear"] is False
    # radius 0 = the single pixel
    d3 = client.post("/api/pixel", json=body | {"radius": 0}).json()
    assert np.allclose(d3["channels"], img[100, 150], atol=1e-6)
    # outside, missing coordinates, bad index, star layer missing: 400
    assert client.post("/api/pixel", json=body | {"x": 400}).status_code == 400
    assert client.post("/api/pixel", json={k: v for k, v in body.items() if k != "y"}).status_code == 400
    assert client.post("/api/pixel", json=body | {"index": 3}).status_code == 400
    stars_step = [{"plugin": "contrast.saturation", "params": {}, "target": "stars"}]
    assert client.post("/api/pixel", json=body | {"steps": stars_step}).status_code == 400


def test_mask_png_paint_and_channel(linear_image):
    steps = [{"plugin": "stretch.auto", "params": {}},
             {"plugin": "contrast.saturation", "params": {"amount": 1.5},
              "mask": {"type": "paint", "base": {"type": "channel", "channel": "G", "mid": 0.5},
                       "strokes": [[[100, 100, 40, 1, 0], [200, 100, 40, 1, 0]], [[500, 300, 30, 0, 0]]]}}]
    r = client.post("/api/mask", json={"input": str(linear_image), "steps": steps, "index": 1, "max_side": 300})
    assert r.status_code == 200, r.text
    assert abs(float(r.headers["X-Scale"]) - 0.5) < 1e-6
    m = np.asarray(_png(r.content), np.float32)
    assert m[50, 50] == 255 and m[50, 75] == 255 and m[150, 250] == 0       # painted / erased (preview px)
    # the preview reports its scale too (the brush converts preview <-> full-resolution pixels with it)
    r = client.post("/api/preview", json={"input": str(linear_image), "steps": steps, "max_side": 300})
    assert r.status_code == 200 and abs(float(r.headers["X-Scale"]) - 0.5) < 1e-6


# ------------------------------------------------------------------ FITS header, export metadata
def test_fits_header(linear_image, tmp_path):
    r = client.get("/api/fits_header", params={"path": str(linear_image)})
    assert r.status_code == 200, r.text
    keys = [c[0] for c in r.json()["cards"]]
    assert "SIMPLE" in keys and "NAXIS" in keys
    assert client.get("/api/fits_header", params={"path": str(tmp_path / "none.fits")}).status_code == 404
    (tmp_path / "x.txt").write_text("hi")
    assert client.get("/api/fits_header", params={"path": str(tmp_path / "x.txt")}).status_code == 400
    # network mode: only session folders
    a = server.create_app("0.0.0.0", token="tok")
    c = TestClient(a, headers={"X-AstroForge-Token": "tok"})
    assert c.get("/api/fits_header", params={"path": str(linear_image)}).status_code == 403
    c.post("/api/roots", json={"paths": [str(linear_image)]})
    assert c.get("/api/fits_header", params={"path": str(linear_image)}).status_code == 200


def test_export_writes_metadata(linear_image, tmp_path):
    meta = {"OBSERVER": "Nicola Furnari", "AUTHOR": "Nicola Furnari", "COPYRIGHT": "(c) 2026 Nicòla",
            "TELESCOP": "Esprit 100", "FILTER": "L-eXtreme", "COMMENT": "riga uno\nriga due", "EMPTY": "",
            "notes": "x" * 100}
    body = {"input": str(linear_image), "steps": [{"plugin": "stretch.auto", "params": {}}], "name": "meta",
            "out_dir": str(tmp_path), "formats": ["fits", "xisf"], "metadata": meta}
    j = _wait(client.post("/api/post", json=body).json()["job"])
    assert j["status"] == "done", j["log"][-3:]
    h = fits.getheader(tmp_path / "integ_meta.fits")
    assert h["OBSERVER"] == "Nicola Furnari" and h["AUTHOR"] == "Nicola Furnari"
    assert h["COPYRIGHT"[:8]] == "(c) 2026 Nicola" and h["TELESCOP"] == "Esprit 100" and h["FILTER"] == "L-eXtreme"
    assert "riga uno" in str(h["COMMENT"]) and "riga due" in str(h["COMMENT"])
    assert "EMPTY" not in h and h["NOTES"] == "x" * 68
    from astroforge.stack import xisf
    kw = xisf.read(tmp_path / "integ_meta.xisf")[1]
    assert kw["OBSERVER"] == "Nicola Furnari" and kw["TELESCOP"] == "Esprit 100"


def test_preview_skips_plate_solving_steps():
    from astroforge.gui import server
    for pid in ("color.spcc", "color.pcc", "analysis.plate_solve", "annotate.objects"):
        assert pid in server._PREVIEW_SKIP
    assert server._PREVIEW_SUBST["background.allsky"] == "background.dbe"


def test_preview_of_a_finished_file_is_not_stretched_again(tmp_path):
    # a FITS already stretched (finished elsewhere) keeps its look in the preview: no automatic stretch
    rng = np.random.default_rng(3)
    img = np.clip(0.25 + 0.05 * rng.standard_normal((120, 160, 3)), 0, 1).astype(np.float32)
    p = tmp_path / "finished.fits"
    fits.PrimaryHDU(np.moveaxis(img, -1, 0)).writeto(p)
    r = client.post("/api/preview", json={"input": str(p), "steps": [], "max_side": 160})
    assert r.status_code == 200, r.text
    out = _jpeg_array(r.content)
    assert abs(float(np.median(out)) - 0.25 * 255) < 3


def test_hybrid_with_integrated_files_opens_the_post_on_the_sliders(tmp_path):
    # every source an integrated file (linear masters here): the job only aligns, flags align_only, and gives the
    # narrowband step with every role (the GUI then opens the post on it instead of "Risultati" + a profile)
    from test_sources import field, save
    rgb = np.repeat(field()[..., None], 3, -1)
    srcs = [{"name": "RGB", "role": "rgb", "file": save(tmp_path / "rgb.fits", rgb)}]
    for role, ang in (("ha", 3), ("oiii", -2), ("sii", 1)):
        srcs.append({"name": role, "role": role, "file": save(tmp_path / f"{role}.fits", field(scale=1.1, angle=ang))})
    r = client.post("/api/stack", json={"mode": "hybrid", "sources": srcs, "work_dir": str(tmp_path / "w"), "output": "x"})
    j = _wait(r.json()["job"])
    assert j["status"] == "done", j.get("log")
    res = j["result"]
    assert res["align_only"] is True and res["stretched"] is False
    step = res["hybrid_step"]
    assert step["plugin"] == "color.nb_rgb" and set(step["params"]) >= {"ha", "oiii", "sii"} and step["params"]["mode"] == "nbrgb"


def test_server_restart_and_stop(monkeypatch):
    """The buttons at the foot of the rail: restart starts the same server again (after the update check),
    stop just leaves; never from another device in network mode."""
    calls = []
    monkeypatch.setattr(server, "_exit_soon", lambda spawn=None, delay=0.6: calls.append(spawn))
    v = client.get("/api/version").json()
    assert v["pid"] > 0
    assert client.post("/api/server/stop", json={}).json()["ok"] and calls[-1] is None
    a = server.create_app("0.0.0.0", token="tok")
    a.state.launch = {"host": "0.0.0.0", "port": 8800, "token": "tok"}
    hdr = {"X-AstroForge-Token": "tok"}
    remote = TestClient(a, headers=hdr, client=("192.168.1.50", 50000))
    assert remote.post("/api/server/restart", json={}).status_code == 403
    assert remote.post("/api/server/stop", json={}).status_code == 403
    assert len(calls) == 1
    local = TestClient(a, headers=hdr, client=("127.0.0.1", 50001))
    assert local.post("/api/server/restart", json={}).status_code == 200
    cmd = calls[-1]
    assert cmd[1:4] == ["-m", "astroforge", "gui"] and "--update" in cmd
    assert cmd[cmd.index("--port") + 1] == "8800" and cmd[cmd.index("--token") + 1] == "tok"


def test_free_port_refuses_other_programs():
    """A port held by something that is not the AstroForge server (here a plain Python process started from the
    same venv, whose path contains "astroforge") is an error, not a process to stop."""
    import subprocess
    import sys
    code = ("import socket,sys,time; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(1); "
            "print(s.getsockname()[1], flush=True); time.sleep(30)")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        port = int(p.stdout.readline())
        assert not server._port_free("127.0.0.1", port)
        with pytest.raises(SystemExit, match="altro programma"):
            server._free_port("127.0.0.1", port, wait=1, log=lambda *_: None)
        assert p.poll() is None                      # left alone
    finally:
        p.kill()
        p.wait()


def test_decimal_params_are_not_sent_as_whole_numbers():
    """JSON writes 1.0 as 1: without the "int" flag the page gave 0.0-5.0 sliders steps of 1 (hybrid Ha 0, 1, 2)."""
    specs = {p["id"]: p["params"] for p in client.get("/api/plugins").json()}
    nb = specs["color.nb_rgb"]
    assert nb["ha_amount"]["int"] is False and nb["oiii_amount"]["int"] is False and nb["ha_blue"]["int"] is False
    assert nb["continuum"]["int"] is False                     # default None, float range
    assert any(p["int"] for ps in specs.values() for p in ps.values())   # whole-number params still say so
    assert not any(p["int"] for ps in specs.values() for p in ps.values() if isinstance(p["default"], bool))


def test_scan_shows_every_file_and_thumbnails(lights, tmp_path):
    """The Sorgenti page lists the frames (lights and calibration), not only how many there are, and draws them."""
    flats = tmp_path / "flats"
    flats.mkdir()
    for k, exp in enumerate([2.0, 2.0, 2.0, 0.01]):
        h = fits.Header({"IMAGETYP": "Flat Field", "EXPTIME": exp, "INSTRUME": "QHY163M", "GAIN": 100, "OFFSET": 30,
                         "XBINNING": 1, "SET-TEMP": -10.0, "FILTER": "L"})
        fits.PrimaryHDU(np.full((H, W), 30000, np.uint16), h).writeto(flats / f"F{k}.fits")
    r = client.post("/api/scan", json={"lights": [str(lights / "lights")], "calibration": [str(flats)]}).json()
    g = r["light_groups"][0]
    assert g["count"] == 4 and len(g["frames"]) == 4 and all(f["path"].endswith(".fits") for f in g["frames"])
    flat = r["calibration"]["flat"][0]
    assert len(flat["frames"]) == 4
    assert [Path(p).name for p in flat["odd_paths"]] == ["F3.fits"]          # the odd exposure, ignored by itself
    t = client.get("/api/thumb", params={"path": g["frames"][0]["path"], "side": 120})
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"
    im = Image.open(_io.BytesIO(t.content))
    assert max(im.size) == 120


def test_excluded_calibration_frames_never_reach_a_master(lights, tmp_path):
    from astroforge.stack.project import Runner
    darks = tmp_path / "darks"
    darks.mkdir()
    for k in range(3):
        h = fits.Header({"IMAGETYP": "Dark Frame", "EXPTIME": 60.0, "INSTRUME": "QHY163M", "GAIN": 100, "OFFSET": 30,
                         "XBINNING": 1, "SET-TEMP": -10.0})
        fits.PrimaryHDU(np.full((H, W), 500, np.uint16), h).writeto(darks / f"D{k}.fits")
    bad = str(darks / "D1.fits")
    run = Runner({"lights": [str(lights / "lights")], "calibration": [str(darks)], "exclude": [bad],
                  "work_dir": str(tmp_path / "w")}, log=lambda *a: None)
    darks_seen = [f.path for fs in run.scan()["groups"]["dark"].values() for f in fs]
    assert len(darks_seen) == 2 and bad not in darks_seen


def test_identify_without_astap_uses_the_header(lights, monkeypatch, tmp_path):
    """No ASTAP: the session still gets the OBJECT of the header and says why it was not solved."""
    import shutil
    from astroforge import pipeline
    monkeypatch.setattr(pipeline, "load_config", lambda: {"tools": {}})
    f = tmp_path / "L_M42.fits"
    shutil.copy(sorted((lights / "lights").glob("*.fits"))[0], f)
    with fits.open(f, mode="update") as hdul:
        hdul[0].header["OBJECT"] = "M42"
    server._IDENTIFY.clear()
    r = client.post("/api/identify", json={"path": str(f)}).json()
    assert r["name"] == "M42" and r["solved"] is False and "ASTAP" in r["note"]
    assert client.post("/api/identify", json={"path": str(f) + "x"}).status_code == 404


def test_main_object_prefers_a_central_messier():
    objs = [{"label": "NGC 6526", "kind": "neb", "x": 500, "y": 300, "a": 300, "messier": 0},
            {"label": "M20 Trifid Nebula", "kind": "neb", "x": 900, "y": 100, "a": 40, "messier": 20},
            {"label": "M8 Lagoon Nebula", "kind": "neb", "x": 520, "y": 340, "a": 120, "messier": 8}]
    assert server._main_object(objs, (667, 1000))["label"] == "M8 Lagoon Nebula"
    assert server._main_object(objs[:1], (667, 1000))["label"] == "NGC 6526"
    assert server._main_object([], (667, 1000)) is None


def test_preview_star_layer_and_starless_views(linear_image):
    # "Mostra: solo stelle" returns the star layer, "solo nebulosa" skips stars.recombine; no layer -> 400
    steps = [{"plugin": "stretch.auto", "params": {}}, {"plugin": "stars.remove_basic", "params": {}},
             {"plugin": "stars.recombine", "params": {}}]
    get = lambda view: client.post("/api/preview", json={"input": str(linear_image), "steps": steps, "max_side": 300, "view": view})
    full, stars, neb = (get(v) for v in ("", "stars", "starless"))
    assert full.status_code == stars.status_code == neb.status_code == 200
    a, s, n = (_jpeg_array(r.content) for r in (full, stars, neb))
    assert np.median(s) < np.median(n) and np.median(s) < 0.5 * np.median(a) + 1     # mostly black around the stars
    assert np.abs(a - n).mean() > 0.01                                                 # the stars are not added back
    r = client.post("/api/preview", json={"input": str(linear_image), "steps": steps[:1], "max_side": 300, "view": "stars"})
    assert r.status_code == 400


def test_histogram_endpoint(linear_image):
    h = client.post("/api/histogram", json={"input": str(linear_image), "steps": [], "max_side": 200}).json()
    assert h["bins"] == len(h["l"]) == 128 and max(h["l"]) == 1.0 and h["linear"] is True


def test_icons_manifest_and_favicon():
    # the tab / taskbar / installed-app icon is the owl (PNG + ICO for Windows, manifest for the app window)
    html = client.get("/").text
    assert 'rel="manifest" href="/static/manifest.json?v=' in html and "favicon-32.png?v=" in html
    assert "/static/app.js?v=" in html and "/static/theme.css?v=" in html and 'icons.svg#' in html  # sprites untouched
    m = client.get("/static/manifest.json").json()
    assert m["name"] == "Noctua" and any(i.get("purpose") == "maskable" for i in m["icons"])
    for i in m["icons"]:
        assert client.get(i["src"]).status_code == 200
    r = client.get("/favicon.ico")
    assert r.status_code == 200 and r.content[:4] == b"\x00\x00\x01\x00"


def test_zoom_render_keeps_to_the_pixel_budget(linear_image):
    # a deep zoom of a big image is processed at the largest scale within the budget (max_mpx), not at 1:1
    body = {"input": str(linear_image), "steps": [{"plugin": "stretch.auto", "params": {}}], "crop": [0.4, 0.4, 0.5, 0.5],
            "width": 4000}
    full = client.post("/api/preview_zoom", json=body)
    small = client.post("/api/preview_zoom", json=body | {"max_mpx": 0.01})
    assert full.status_code == small.status_code == 200
    assert float(full.headers["X-Scale"]) == 1.0 and float(small.headers["X-Scale"]) < 0.5


def test_quick_viewer_raw_bayer_frame(tmp_path):
    # /api/view: a raw OSC frame (BAYERPAT) comes out in colour at half resolution, stretched; linked vs unlinked
    rng = np.random.default_rng(4)
    cfa = (0.02 + 0.002 * rng.standard_normal((200, 300))).astype(np.float32)
    cfa[0::2, 0::2] *= 2.0                                   # R sites brighter: a colour cast to remove
    hdr = fits.Header({"BAYERPAT": "RGGB", "EXPTIME": 300, "GAIN": 120, "CCD-TEMP": -10})
    p = tmp_path / "L_0001.fits"
    fits.PrimaryHDU((cfa * 65535).astype(np.uint16), hdr).writeto(p)
    get = lambda **kw: client.get("/api/view", params={"path": str(p), "side": 400, **kw})
    r = get()
    assert r.status_code == 200, r.text
    info = json.loads(r.headers["X-Info"])
    assert info["bayer"] == "RGGB" and info["width"] == 300 and info["exptime"] == 300 and info["shown"] == [150, 100]
    linked, unlinked = _jpeg_array(r.content), _jpeg_array(get(stretch_mode="unlinked").content)
    cast = lambda a: float(np.median(a[..., 0]) - np.median(a[..., 2]))
    assert cast(linked) > 10 and abs(cast(unlinked)) < 0.5 * cast(linked)      # unlinked removes the cast
    mono = get(color="false")
    assert json.loads(mono.headers["X-Info"])["shown"] == [300, 200] and get(stretch_mode="linear").status_code == 200
    assert client.get("/api/view", params={"path": str(tmp_path / "missing.fits")}).status_code == 404


def test_quick_viewer_fixes_a_wrong_header_pattern(tmp_path):
    # QHY168C + APT write GRBG for RGGB data: the viewer checks the green sites like the stacker does
    rng = np.random.default_rng(5)
    cfa = (0.02 + 0.001 * rng.standard_normal((200, 300))).astype(np.float32)
    cfa[0::2, 0::2] *= 1.6                                   # R
    cfa[1::2, 1::2] *= 0.7                                   # B; greens at (0,1) and (1,0): RGGB data
    p = tmp_path / "L_0002.fits"
    fits.PrimaryHDU((cfa * 65535).astype(np.uint16), fits.Header({"BAYERPAT": "GRBG"})).writeto(p)
    info = json.loads(client.get("/api/view", params={"path": str(p), "side": 400}).headers["X-Info"])
    assert info["bayer"] == "RGGB"
