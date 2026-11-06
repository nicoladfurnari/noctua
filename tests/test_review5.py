"""Defects found by the code review of 2026-11-06: star layer steps, geometric steps and the layers, auto stretch and
auto crop on a zero background, NaN borders of colour frames, frames listed twice, dark matching by temperature,
cancel in the row-band combine, server restart / open / preset save, updater retry of a failed pip install."""
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from astroforge import color, crop, pipeline, plugins, stretch
from astroforge.stack import combine, frames, register

RNG = np.random.default_rng(11)
H, W = 96, 128


def ctx(**kw):
    return plugins.Context(log=lambda m: None, **kw)


def star_layer():
    yy, xx = np.mgrid[:H, :W]
    st = np.zeros((H, W, 3), np.float32)
    for x, y in ((10, 20), (90, 30), (60, 70)):
        st += (0.02 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4))[..., None].astype(np.float32)
    return st


# ------------------------------------------------------------------ star layer
def test_layer_aware_plugins_keep_their_result_with_target_stars():
    for plugin, params in (("stars.stretch", {"amount": 4.0}), ("stars.size", {"amount": 0.5}),
                           ("stars.morph", {})):
        c = ctx(linear=False)
        st = star_layer()
        c.layers["stars"] = st
        img = np.full((H, W, 3), 0.1, np.float32)
        out = pipeline.apply_step(img, {"plugin": plugin, "params": params, "target": "stars"}, c)
        assert out is img
        assert not np.allclose(c.layers["stars"], st), plugin      # was written back unchanged


def test_starless_profiles_leave_the_image_alone_without_a_star_layer():
    """StarNet not installed: no star layer; the star steps of the automatic profiles must not stretch the image."""
    for name in ("rgb_starless", "hoo_starless", "sho_starless", "video_trifida"):
        steps = [s for s in pipeline.load_preset(name)["steps"] if s["plugin"] in ("stars.stretch", "stars.morph", "stars.size")]
        assert steps and all(s.get("target") == "stars" for s in steps), name
        img = np.full((H, W, 3), 0.08, np.float32)
        c = ctx(linear=False)
        for s in steps:
            img = pipeline.apply_step(img, s, c)
        assert np.allclose(img, 0.08), name


def test_stretch_of_the_star_layer_marks_it_for_screen_blend():
    c = ctx(linear=True)
    c.layers["stars"], c.layers["stars_mode"] = star_layer(), "add"
    pipeline.apply_step(np.full((H, W, 3), 0.01, np.float32), {"plugin": "stretch.auto", "params": {}, "target": "stars"}, c)
    assert c.layers["stars_mode"] == "screen" and c.linear is True


def test_auto_stretch_with_a_zero_median_keeps_faint_pixels_faint():
    st = star_layer()
    st[0, 0] = 1e-5                                    # a faint pixel far below the stars
    c0, m = stretch.auto_params(st)
    assert c0 == 0.0 and m > 1e-4                      # was 1e-6: everything above zero went to white
    out = stretch.histogram_transform(st, c0, m)
    assert out[0, 0].max() < 0.2
    assert stretch.auto_params(np.zeros((8, 8, 3), np.float32)) == (0.0, 0.5)


# ------------------------------------------------------------------ geometry and layers
def test_crop_and_flip_move_the_star_layer_with_the_image():
    img = np.full((H, W, 3), 0.1, np.float32)
    c = ctx(linear=False)
    c.layers["stars"] = star_layer()
    out = pipeline.apply_step(img, {"plugin": "crop.manual", "params": {"x0": 8, "y0": 16, "x1": 108, "y1": 80}}, c)
    assert out.shape == c.layers["stars"].shape == (64, 100, 3)
    assert c.layers["stars"][20 - 16, 10 - 8].max() > 0.015       # the star at (10, 20) followed the crop
    before = c.layers["stars"].copy()
    out = pipeline.apply_step(out, {"plugin": "geometry.rotate", "params": {"angle": 0, "flip": "horizontal"}}, c)
    assert np.allclose(c.layers["stars"], before[:, ::-1])
    out = pipeline.apply_step(out, {"plugin": "geometry.rotate", "params": {"angle": 90}}, c)
    assert out.shape[:2] == c.layers["stars"].shape[:2] == (100, 64)
    out = pipeline.apply_step(out, {"plugin": "stars.recombine", "params": {}}, c)   # no size error
    assert out.shape == (100, 64, 3)


def test_crop_shifts_the_plate_solution_and_a_flip_drops_it():
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix, w.wcs.cdelt, w.wcs.crval, w.wcs.ctype = [64, 48], [-0.001, 0.001], [10, 20], ["RA---TAN", "DEC--TAN"]
    img = np.full((H, W, 3), 0.1, np.float32)
    c = ctx(linear=False)
    c.layers["wcs"], c.layers["wcs_shape"] = w, (H, W)
    sky = w.pixel_to_world_values(50, 40)
    out = pipeline.apply_step(img, {"plugin": "crop.manual", "params": {"x0": 8, "y0": 16, "x1": 108, "y1": 80}}, c)
    assert np.allclose(c.layers["wcs"].pixel_to_world_values(50 - 8, 40 - 16), sky)
    assert tuple(c.layers["wcs_shape"]) == (64, 100) and list(w.wcs.crpix) == [64, 48]   # the original is untouched
    pipeline.apply_step(out, {"plugin": "geometry.rotate", "params": {"angle": 0, "flip": "vertical"}}, c)
    assert "wcs" not in c.layers and "wcs_shape" not in c.layers


def test_auto_crop_with_a_zero_background_cuts_only_the_ragged_border():
    img = RNG.normal(0, 0.001, (300, 400, 3)).astype(np.float32)      # background-subtracted: median 0
    img[:5] = img[-5:] = 0.2
    img[:, :5] = img[:, -5:] = 0.2
    assert crop.detect_borders(img, pad=0) == (5, 5, 5, 5)             # was the maximum (6% of the side) everywhere


def test_star_white_balance_with_an_empty_channel_is_finite():
    yy, xx = np.mgrid[:200, :200]
    img = np.full((200, 200, 3), 0.02, np.float32)
    for x, y in RNG.integers(15, 185, (60, 2)):
        img[..., :2] += (0.5 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 3))[..., None].astype(np.float32)
    img[..., 2] = 0.0
    out, k, n = color.white_balance_stars(np.clip(img, 0, 1))
    assert np.isfinite(out).all() and np.isfinite(k).all()


# ------------------------------------------------------------------ stacker
def test_warp_marks_uncovered_borders_as_nan_in_every_channel():
    img = np.ones((40, 60, 3), np.float32)
    out = register.warp(img, np.array([[1, 0, 5.3], [0, 1, 2.1]]), (40, 60))
    n = [int(np.isnan(out[..., ch]).sum()) for ch in range(3)]
    assert n[0] > 0 and n[0] == n[1] == n[2]           # was [n, 0, 0]: zeros taken as data in G and B


def test_frames_listed_twice_are_kept_once(tmp_path):
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    fs = [SimpleNamespace(path=str(a)), SimpleNamespace(path=str(b)), SimpleNamespace(path=str(tmp_path / "." / "a.fits"))]
    assert [f.path for f in frames.unique(fs)] == [str(a), str(b)]


def test_darks_are_picked_by_temperature_before_frame_count():
    from astroforge.stack.project import Runner
    r = Runner.__new__(Runner)
    r.log, r._kw_filter = (lambda m: None), (lambda fs: fs)
    setup = ("cam", 100, 100, 1, 100, 10)
    groups = {"dark": {setup + (-10.0, 300.0): ["cold"] * 20, setup + (0.0, 300.0): ["warm"] * 30}}
    assert r._pick(groups, "dark", setup + (-10.0, 300.0, ""), 300.0)[0] == "cold"
    assert r._pick(groups, "dark", setup + (-8.0, 300.0, ""), 300.0)[0] == "cold"     # nearest, not largest
    assert r._pick(groups, "dark", setup + (None, 300.0, ""), 300.0)[0] == "warm"     # unknown: the larger group


def test_combine_chunked_stops_on_cancel():
    data = [np.full((64, 32), v, np.float32) for v in (0.1, 0.2, 0.3)]

    class Loader:
        height = 64

        def __call__(self, a, y0, y1):
            return a[y0:y1]
    assert combine.combine_chunked(data, Loader(), "median", rows=16)[0].shape == (64, 32)
    with pytest.raises(combine.Cancelled):
        combine.combine_chunked(data, Loader(), "median", rows=16, cancel=lambda: True)
    from astroforge.stack import project
    assert project.Cancelled is combine.Cancelled


# ------------------------------------------------------------------ server, updater
def test_restart_command_and_local_address():
    from astroforge.gui import server
    cmd = server._restart_cmd({"host": "0.0.0.0", "port": 8765, "token": "-abc"})
    assert "--token=-abc" in cmd and "--token" not in cmd          # "--token -abc" was read as an option
    assert server._local_host("0.0.0.0") == server._local_host("") == "127.0.0.1"
    assert server._local_host("192.168.1.5") == "192.168.1.5" and server._local_host("::1") == "[::1]"


def test_saving_over_a_shipped_profile_is_refused():
    from fastapi.testclient import TestClient
    from astroforge.gui import server
    if shutil.which("git") is None or not (pipeline.ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    f = pipeline.PRESETS / "quick.json"
    before = f.read_bytes()
    r = TestClient(server.app).post("/api/presets/quick", json={"steps": []})
    assert r.status_code == 409 and f.read_bytes() == before


def test_missing_input_is_a_400():
    from fastapi.testclient import TestClient
    from astroforge.gui import server
    c = TestClient(server.app)
    assert c.post("/api/preview", json={"steps": []}).status_code == 400
    assert c.post("/api/preview_zoom", json={"steps": []}).status_code in (400, 404, 405)
    assert c.post("/api/post", json={"steps": []}).status_code == 400


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_failed_pip_install_is_retried_at_the_next_start(tmp_path, monkeypatch):
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
    results = [False, True]
    calls = []
    monkeypatch.setattr(updater, "install_deps", lambda root, log: (calls.append(root), results.pop(0))[1])
    assert updater.update(inst, log=lambda m: None) == "updated"
    assert (inst / updater.DEPS_PENDING).exists() and len(calls) == 1
    assert updater.update(inst, log=lambda m: None) == "current"       # nothing new, but pip runs again
    assert len(calls) == 2 and not (inst / updater.DEPS_PENDING).exists()
    assert updater.update(inst, log=lambda m: None) == "current" and len(calls) == 2
