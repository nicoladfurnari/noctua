"""Several sources of one target (stack.sources): cameras merged (multi) and colour + narrowband (hybrid)."""
import cv2
import numpy as np
import pytest
from astropy.io import fits

from astroforge.stack import sources
from astroforge.stack.project import Runner

H, W = 260, 340
RNG = np.random.default_rng(11)
YY, XX = np.mgrid[:H, :W]
STARS = [(x, y, f) for x, y, f in zip(RNG.uniform(15, W - 15, 150), RNG.uniform(15, H - 15, 150),
                                      RNG.uniform(0.05, 0.5, 150))]
NEB = np.exp(-((XX - 170) ** 2 + (YY - 130) ** 2) / 3000.0)


def field(scale=1.0, angle=0.0, noise=0.001, gain=1.0, offset=0.0):
    st = np.zeros((H, W), np.float32)
    for x, y, f in STARS:
        st += f * np.exp(-((XX - x) ** 2 + (YY - y) ** 2) / 3.0)
    img = 0.02 + st + 0.02 * NEB
    if scale != 1.0 or angle:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
        img = cv2.warpAffine(img.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0.02)
    img = offset + gain * img + RNG.normal(0, noise, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def save(path, a):
    fits.PrimaryHDU(a if a.ndim == 2 else np.moveaxis(a, -1, 0)).writeto(path, overwrite=True)
    return str(path)


def runner(tmp_path):
    return Runner({"work_dir": str(tmp_path / "work"), "output": "t"}, log=lambda m: None)


def test_multi_merges_two_cameras_weighted_by_noise(tmp_path):
    a = np.repeat(field(noise=0.002)[..., None], 3, -1)
    b = np.repeat(field(scale=0.85, angle=4, noise=0.0007, gain=1.3, offset=0.01)[..., None], 3, -1)   # other camera
    srcs = [{"name": "533MC", "file": save(tmp_path / "a.fits", a)}, {"name": "2600MC", "file": save(tmp_path / "b.fits", b)}]
    out, rep = runner(tmp_path).run_sources(srcs, "multi")
    w = {x["name"]: x["share"] for x in rep["weights"]}
    assert w["2600MC"] > 0.75                              # 3x less noise -> ~9x the weight
    y0, y1, x0, x1 = rep["crop"]
    ref = a[y0:y1, x0:x1]
    sky = (slice(5, 40), slice(5, 60))
    assert sources.background_noise(out[..., 0]) < 0.8 * sources.background_noise(ref[..., 0])
    assert abs(float(np.median(out[sky])) - float(np.median(ref[sky]))) < 0.003     # reference levels kept
    assert (tmp_path / "work" / "t_multi.fits").exists()


def test_hybrid_saves_registered_narrowband(tmp_path):
    rgb = np.repeat(field()[..., None], 3, -1)
    ha = field(scale=1.2, angle=-6, gain=0.1)
    srcs = [{"name": "RGB", "role": "rgb", "file": save(tmp_path / "rgb.fits", rgb)},
            {"name": "Ha", "role": "ha", "file": save(tmp_path / "ha.fits", ha)}]
    out, rep = runner(tmp_path).run_sources(srcs, "hybrid")
    assert set(rep["nb"]) == {"ha"}
    reg = fits.getdata(rep["nb"]["ha"]).astype(np.float32)
    reg = reg[0] if reg.ndim == 3 else reg
    assert reg.shape == out.shape[:2]
    # the registered Ha lines up with the colour image: its brightest star is where the RGB one is
    ry, rx = np.unravel_index(np.argmax(cv2.GaussianBlur(out[..., 0], (0, 0), 2)), reg.shape)
    hy, hx = np.unravel_index(np.argmax(cv2.GaussianBlur(reg, (0, 0), 2)), reg.shape)
    assert abs(ry - hy) <= 2 and abs(rx - hx) <= 2


def test_hybrid_colour_source_not_first(tmp_path):
    """Reported by Nicola: after deleting and re-adding cards the RGB was not the first source -> error."""
    rgb = np.repeat(field()[..., None], 3, -1)
    ha = field(scale=1.2, angle=-6, gain=0.1)
    srcs = [{"role": "ha", "file": save(tmp_path / "ha.fits", ha)},
            {"role": "rgb", "file": save(tmp_path / "rgb.fits", rgb)}]
    out, rep = runner(tmp_path).run_sources(srcs, "hybrid")
    assert out.shape[-1] == 3 and set(rep["nb"]) == {"ha"}
    assert [x["name"] for x in rep["sources"]] == ["RGB rgb", "Ha ha"]      # names from role + file


def test_mode_checks(tmp_path):
    rgb = save(tmp_path / "rgb.fits", np.repeat(field()[..., None], 3, -1))
    r = runner(tmp_path)
    with pytest.raises(ValueError, match="at least 2"):
        r.run_sources([{"file": rgb}], "multi")
    with pytest.raises(ValueError, match="una sola sorgente a colori"):
        r.run_sources([{"file": rgb}, {"name": "x", "file": rgb}], "hybrid")
    mono = save(tmp_path / "m.fits", field())
    with pytest.raises(ValueError, match="channels"):
        r.run_sources([{"name": "a", "file": rgb}, {"name": "b", "file": mono}], "multi")


def test_hybrid_finished_files_are_flagged_stretched(tmp_path):
    # finished (stretched) files: the GUI opens the post-production without a linear profile, screen blend
    st = lambda a: np.clip(a ** 0.25, 0, 1).astype(np.float32)
    rgb = np.repeat(st(field())[..., None], 3, -1)
    srcs = [{"name": "RGB", "role": "rgb", "file": save(tmp_path / "rgb.fits", rgb)},
            {"name": "Ha", "role": "ha", "file": save(tmp_path / "ha.fits", st(field(scale=1.1, angle=3)))}]
    _, rep = runner(tmp_path).run_sources(srcs, "hybrid")
    assert rep["stretched"] is True and set(rep["nb"]) == {"ha"}
    lin = [{"name": "RGB", "role": "rgb", "file": save(tmp_path / "rgb2.fits", np.repeat(field()[..., None], 3, -1))},
           {"name": "Ha", "role": "ha", "file": save(tmp_path / "ha2.fits", field(scale=1.1, angle=3))}]
    _, rep = runner(tmp_path).run_sources(lin, "hybrid")
    assert rep["stretched"] is False


def test_hybrid_registers_finished_stretched_files(tmp_path):
    # stretched files: the nebula is as bright as the stars; the retry on the high-pass stars must register them
    h, w = 600, 800
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[:h, :w]
    st = np.zeros((h, w), np.float32)
    for x, y, f in zip(rng.uniform(10, w - 10, 400), rng.uniform(10, h - 10, 400), rng.uniform(0.05, 0.6, 400)):
        st += f * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 3.0)
    neb = np.exp(-((xx - 420) ** 2 / 30000 + (yy - 300) ** 2 / 15000)).astype(np.float32)
    ring = np.exp(-((np.hypot(xx - 420, yy - 300) - 150) ** 2) / 800).astype(np.float32)
    stretch = lambda a: np.clip(np.clip(a, 0, 1) ** 0.3, 0, 1).astype(np.float32)
    rgb = stretch(np.stack([0.01 + st + 0.01 * neb, 0.01 + st + 0.006 * neb, 0.01 + st + 0.008 * neb], -1)
                  + rng.normal(0, 0.0015, (h, w, 3)))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 7, 0.8)
    ha = stretch(cv2.warpAffine((0.01 + 0.3 * st + 0.05 * neb + 0.08 * ring).astype(np.float32), M, (w, h),
                                borderValue=0.01) + rng.normal(0, 0.001, (h, w)))
    srcs = [{"name": "RGB", "role": "rgb", "file": save(tmp_path / "rgb.fits", rgb)},
            {"name": "Ha", "role": "ha", "file": save(tmp_path / "ha.fits", ha)}]
    msgs = []
    r = Runner({"work_dir": str(tmp_path / "work"), "output": "t"}, log=msgs.append)
    _, rep = r.run_sources(srcs, "hybrid")
    assert rep["stretched"] is True
    line = next(m for m in msgs if "Ha: registered" in m)
    assert "scale x1.25" in line or "scale x1.24" in line or "scale x1.26" in line, line


def test_finished_tiff_loads_as_source(tmp_path):
    import tifffile
    from astroforge import io
    a = (np.random.default_rng(1).random((40, 60, 3)) * 65535).astype(np.uint16)
    tifffile.imwrite(tmp_path / "final.tif", a)
    img, h = io.load(tmp_path / "final.tif")
    assert img.shape == (40, 60, 3) and img.dtype == np.float32 and abs(img[3, 4, 1] - a[3, 4, 1] / 65535) < 1e-6
    tifffile.imwrite(tmp_path / "planar.tif", np.moveaxis(a, -1, 0))
    assert io.load(tmp_path / "planar.tif")[0].shape == (40, 60, 3)
