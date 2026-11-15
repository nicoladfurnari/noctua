"""Defects found by the code review of 2026-11-15: shipped presets by any letter case, pip marker written before the
install, same-origin writes, "this PC" with a specific host, star layer steps with a mask, linear clip of the star
layer, TGV on a layer without noise, auto crop on a zero background with a gradient, 16-bit mono finals, samples
outside the image, look strength 0, red hues, ISO in the setup key, master names, work-cache masters."""
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from astroforge import background, crop, io, look, pipeline, plugins, tgv
from astroforge.stack import frames, makemasters, masterlib

RNG = np.random.default_rng(15)
H, W = 96, 128


def ctx(**kw):
    return plugins.Context(log=lambda m: None, **kw)


def star_layer():
    yy, xx = np.mgrid[:H, :W]
    st = np.zeros((H, W, 3), np.float32)
    for x, y in ((10, 20), (90, 30), (60, 70)):
        st += (0.3 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4))[..., None].astype(np.float32)
    return st


# ------------------------------------------------------------------ post-production
def test_mask_is_applied_to_steps_that_replace_the_star_layer():
    c = ctx(linear=False)
    st = star_layer()
    c.layers["stars"], c.layers["stars_mode"], c.layers["none"] = st, "screen", np.zeros((H, W), np.float32)
    step = {"plugin": "stars.size", "params": {"amount": 1.0}, "target": "stars", "mask": {"type": "layer", "name": "none"}}
    pipeline.apply_step(np.full((H, W, 3), 0.1, np.float32), step, c)
    assert np.allclose(c.layers["stars"], st, atol=1e-6)          # a mask of zeros: nothing changes (it changed by 0.5)
    c.layers["all"] = np.ones((H, W), np.float32)
    step["mask"] = {"type": "layer", "name": "all"}
    pipeline.apply_step(np.full((H, W, 3), 0.1, np.float32), step, c)
    assert not np.allclose(c.layers["stars"], st, atol=1e-3)


def test_a_linear_clip_of_the_star_layer_keeps_it_linear():
    c = ctx(linear=False)
    c.layers["stars"], c.layers["stars_mode"] = star_layer() * 0.05, "add"
    img = np.full((H, W, 3), 0.1, np.float32)
    pipeline.apply_step(img, {"plugin": "stretch.histogram", "params": {"shadows": 0.0001}, "target": "stars"}, c)
    assert c.layers["stars_mode"] == "add" and c.linear is False     # not stretched: still stretched at recombine
    pipeline.apply_step(img, {"plugin": "stretch.auto", "params": {}, "target": "stars"}, c)
    assert c.layers["stars_mode"] == "screen" and c.linear is False


def test_tgv_edge_weight_without_noise_is_finite():
    w = tgv.edge_weight(np.zeros((32, 32), np.float32), 0.0)
    assert np.isfinite(w).all() and (w == 1).all()


def test_auto_crop_on_a_zero_background_with_a_gradient_keeps_the_frame():
    img = RNG.normal(0, 0.003, (800, 1200, 3)).astype(np.float32)
    img += (np.linspace(-0.005, 0.005, 1200, dtype=np.float32))[None, :, None]       # median 0, real gradient
    assert crop.detect_borders(img, pad=0) == (0, 0, 0, 0)                           # was (48, 48, 0, 0)
    assert max(crop.detect_borders(np.clip(img, 0, 1), pad=0)) <= 2                  # as io.load returns it (was 48)
    img[:5] = img[-5:] = 0.2
    img[:, :5] = img[:, -5:] = 0.2
    assert crop.detect_borders(img, pad=0) == (5, 5, 5, 5)                           # a real border is still cut


def test_mono_16_bit_tiff_and_palette_png_are_read_as_pictures(tmp_path):
    from PIL import Image
    ramp = np.tile(np.linspace(0, 0.3, 256, dtype=np.float32), (64, 1))[..., None]
    io.save_tiff(ramp, tmp_path / "mono.tif")
    a = io.load_display(tmp_path / "mono.tif")
    assert a.shape == (64, 256, 3) and abs(float(a.mean()) - 0.15) < 0.01           # was 0.99: all white
    pal = Image.fromarray((np.tile(np.arange(256, dtype=np.uint8), (64, 1)))).convert("RGB").convert("P")
    pal.save(tmp_path / "pal.png")
    assert io.load(tmp_path / "pal.png")[0].shape == (64, 256, 3)
    Image.fromarray(np.full((8, 8), 200, np.uint8)).convert("LA").save(tmp_path / "la.png")
    assert io.load(tmp_path / "la.png")[0].shape == (8, 8, 1)


def test_dbe_sample_outside_the_image_does_not_poison_it():
    img = (0.05 + RNG.normal(0, 0.001, (300, 400, 3))).astype(np.float32)
    out = background.extract_rbf(img, samples=[[50, 50], [350, 50], [200, 250], [60, 240], [900, 700]])
    out = out[0] if isinstance(out, tuple) else out
    assert np.isfinite(out).all()
    with pytest.raises(ValueError):
        background.extract_rbf(img, samples=[[50, 50], [900, 700], [950, 10]])


def test_look_match_strength_zero_changes_nothing_and_red_hues_are_counted():
    yy, xx = np.mgrid[:120, :160]
    neb = np.exp(-((xx - 80) ** 2 + (yy - 60) ** 2) / 900)[..., None]
    img = np.clip(np.array([0.10, 0.07, 0.05], np.float32) + neb * np.array([0.6, 0.15, 0.25], np.float32), 0, 1).astype(np.float32)
    assert np.allclose(look.match(img, strength=0.0), img)                           # the sky was neutralised anyway
    red = np.clip(0.06 + neb * np.array([0.7, 0.1, 0.28], np.float32), 0, 1).astype(np.float32)   # hue ~ 342 deg
    assert look.measure(red)["hues"]["red"] > 0.6


# ------------------------------------------------------------------ stacker
def test_iso_separates_dslr_groups_and_master_names_are_file_names(tmp_path):
    base = dict(camera="EOS", width=100, height=100, binning=1, gain=None, offset=None, temp=None)
    a, b = SimpleNamespace(iso=800.0, **base), SimpleNamespace(iso=1600.0, **base)
    assert frames.setup_key(a) != frames.setup_key(b)
    assert frames.setup_key(SimpleNamespace(iso=800.0, **{**base, "gain": 120.0}))[4] == 120.0
    f0 = SimpleNamespace(filter="UV/IR Cut", temp=-10.0, gain=100.0, binning=1)
    name = makemasters._name("flat", f0, 20, "2026-11-01", 2.0)
    assert "/" not in name and "\\" not in name and " " not in name and name.startswith("master_flat_UVIRCut")
    p = tmp_path / "x.fits"
    p.write_bytes(b"1")
    assert frames.file_sig(p).startswith("1:") and frames.file_sig(tmp_path / "missing") == ""


def test_work_cache_masters_are_not_library_masters(tmp_path):
    assert masterlib.WORK_CACHE.match("master_flat_0123456789.xisf")
    assert not masterlib.WORK_CACHE.match("master_flat_Ha_2s_-10C_g100_n20.xisf")
    from astroforge.stack import calibrate
    work = tmp_path / "session" / "astroforge" / "masters"
    work.mkdir(parents=True)
    calibrate.save_master(np.full((20, 30), 0.5, np.float32), work / "master_flat_0123456789.xisf")
    lib = masterlib.MasterLibrary(tmp_path / "lib", (), strict_extra=[tmp_path / "session"], log=lambda m: None)
    assert not [e for e in lib.entries.values() if e.get("external")]


# ------------------------------------------------------------------ server, updater
def test_shipped_presets_are_recognised_in_any_letter_case():
    if shutil.which("git") is None or not (pipeline.ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    assert pipeline.shipped_preset("galaxy") and pipeline.shipped_preset("Galaxy") and pipeline.shipped_preset("GALAXY")
    assert not pipeline.shipped_preset("mio_profilo_che_non_esiste")


def test_this_pc_with_a_specific_host_and_ipv6_loopback():
    from astroforge.gui import server
    req = lambda host, bind: SimpleNamespace(client=SimpleNamespace(host=host),
                                             app=SimpleNamespace(state=SimpleNamespace(launch={"host": bind})))
    assert server._from_this_pc(req("127.0.0.1", "0.0.0.0"))
    assert server._from_this_pc(req("192.168.1.50", "192.168.1.50"))       # was refused: "only from this PC"
    assert not server._from_this_pc(req("192.168.1.77", "192.168.1.50"))
    assert not server._from_this_pc(req("192.168.1.77", "0.0.0.0"))
    assert server._local_host("::") == "[::1]" and server._local_host("0.0.0.0") == "127.0.0.1"
    assert {"color.lrgb", "color.nb_rgb"} <= set(server._PATH_PARAMS)


def test_posts_from_another_site_are_refused():
    from fastapi.testclient import TestClient
    from astroforge.gui import server
    c = TestClient(server.app)
    r = c.post("/api/roots", json={"paths": []}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert c.post("/api/roots", json={"paths": []}, headers={"Origin": "http://testserver"}).status_code == 200
    assert c.post("/api/roots", json={"paths": []}).status_code == 200
    assert c.get("/api/jobs", headers={"Origin": "http://evil.example"}).status_code == 200


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_pip_marker_is_written_before_the_install(tmp_path, monkeypatch):
    from astroforge import updater

    def git(cwd, *args):
        subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, stdin=subprocess.DEVNULL)
    src, inst = tmp_path / "src", tmp_path / "inst"
    src.mkdir()
    git(src, "init", "-q", "-b", "main")
    git(src, "config", "user.email", "t@example.invalid")
    git(src, "config", "user.name", "t")
    (src / "requirements.txt").write_text("numpy\n")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "first")
    subprocess.run(["git", "clone", "-q", str(src), str(inst)], check=True, capture_output=True)
    (src / "requirements.txt").write_text("numpy\nscipy\n")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "new package")
    seen = []

    def interrupted(root, log):                 # the window closed during the download
        seen.append((root / updater.DEPS_PENDING).exists())
        raise OSError("interrupted")
    monkeypatch.setattr(updater, "install_deps", interrupted)
    updater.update(inst, log=lambda m: None)
    assert seen == [True] and (inst / updater.DEPS_PENDING).exists()
    monkeypatch.setattr(updater, "install_deps", lambda root, log: True)
    assert updater.update(inst, log=lambda m: None) == "current" and not (inst / updater.DEPS_PENDING).exists()


# ------------------------------------------------------------------ stacker numerics
def test_debayer_keeps_highlights_when_a_pixel_is_negative():
    from astroforge.stack import calibrate
    cfa = np.full((32, 32), 0.97, np.float32)
    cfa[0, 0] = -0.1                                    # a hot pixel of the dark: the pedestal goes to 0.1
    for method in ("bilinear", "ea", "rcd"):
        out = calibrate.debayer(cfa, "RGGB", method)
        assert abs(float(out[8:24, 8:24].min()) - 0.97) < 1e-3, method      # was cut at 0.90


def test_percentile_rejection_does_not_reject_pure_noise():
    from astroforge.stack import combine
    x = (0.003 + RNG.normal(0, 0.001, (3, 1, 100000))).astype(np.float32)
    out, lo, hi = combine.combine(x, "percentile", 4.0, 3.0, None)
    assert (lo.sum() + hi.sum()) / x.size < 0.01                            # was 8.5 % of the samples
    assert float(np.std(out)) < 1.05 * float(np.std(x.mean(0)))             # was 21 % noisier than the mean


def test_run_tasks_fails_when_the_gpu_worker_dies(monkeypatch):
    from astroforge import gpu

    class Dead:
        class cuda:
            @staticmethod
            def Device(i):
                raise RuntimeError("device lost")
    monkeypatch.setattr(gpu, "devices", lambda: [(0, "fake")])
    monkeypatch.setattr(gpu, "cupy", lambda: Dead)
    done = []
    with pytest.raises(RuntimeError):
        gpu.run_tasks([1, 2, 3], lambda t, d: done.append(t), lambda t: done.append(t), cpu_workers=0)
    assert done == []
