"""background.allsky: gradient correction against an all-sky survey reference (download mocked, one live test)."""
import io
import json
import urllib.parse

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from astroforge import allsky, plugins
from astroforge.plugins import gradient_ref  # noqa: F401  (registers background.allsky)

H = W = 768
RA0, DEC0 = 270.904, -24.386      # M8


def _wcs(scale_arcsec=2.0, rot_deg=30.0, w=W, h=H):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [RA0, DEC0]
    wcs.wcs.crpix = [(w + 1) / 2, (h + 1) / 2]
    s, t = scale_arcsec / 3600, np.radians(rot_deg)
    wcs.wcs.cd = [[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]]
    wcs.wcs.set()
    return wcs


def _nebula(x, y):
    """Extended faint nebula profile (peak ~1) in image pixel coordinates."""
    blob = np.exp(-((x - 300) ** 2 + (y - 420) ** 2) / (2 * 110 ** 2))
    u, v = (x - 470) * np.cos(0.6) + (y - 330) * np.sin(0.6), -(x - 470) * np.sin(0.6) + (y - 330) * np.cos(0.6)
    fil = 0.7 * np.exp(-u ** 2 / (2 * 200 ** 2) - v ** 2 / (2 * 28 ** 2))
    glow = 0.35 * np.exp(-((x - 380) ** 2 + (y - 380) ** 2) / (2 * 260 ** 2))
    return blob + fil + glow


AMP = np.array([0.04, 0.025, 0.02], np.float32)
rng = np.random.default_rng(3)
STARS = np.c_[rng.uniform(0, W, 500), rng.uniform(0, H, 500), rng.lognormal(-3.5, 1.0, 500)]


def _field():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    neb = _nebula(xx, yy)[..., None] * AMP
    grad = (0.035 * xx / W + 0.03 * (2 * yy / H - 1) ** 2 + 0.01 * xx * yy / (W * H))[..., None] \
        * np.array([1.0, 0.8, 0.6], np.float32)
    clean = 0.05 + neb
    stars = np.zeros((H, W), np.float32)
    for x, y, f in STARS:
        x0, y0 = int(x), int(y)
        sl = (slice(max(0, y0 - 6), y0 + 7), slice(max(0, x0 - 6), x0 + 7))
        stars[sl] += f * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * 1.5 ** 2))
    clean = clean + stars[..., None] * np.array([1.0, 0.95, 0.9], np.float32)
    noise = rng.normal(0, 0.002, (H, W, 3)).astype(np.float32)
    img = np.clip(clean + grad + noise, 0, 1).astype(np.float32)
    return img, (clean + noise).astype(np.float32), neb.astype(np.float32), grad.astype(np.float32)


def _fake_download(full_wcs, calls=None):
    """Mock of allsky._download: renders the survey on the WCS asked in the URL (no gradient, own units)."""
    def dl(url, timeout):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        hips, wd = q["hips"][0], json.loads(q["wcs"][0])
        if calls is not None:
            calls.append(hips)
        hdr = fits.Header()
        for k, v in wd.items():
            hdr[k] = v
        sw = WCS(hdr)
        h2, w2 = wd["NAXIS2"], wd["NAXIS1"]
        jj, ii = np.mgrid[0:h2, 0:w2]
        ra, dec = sw.all_pix2world(ii.ravel(), jj.ravel(), 0)
        x, y = full_wcs.all_world2pix(ra, dec, 0)
        neb = _nebula(x, y).reshape(h2, w2)
        level, gain = (3000.0, 20000.0) if "red" in hips else (2000.0, 15000.0)
        data = level + gain * neb
        # survey stars (brighter, other photometry) on the coarse grid
        k = W / w2
        for sx, sy, f in STARS[::2]:
            px, py = (sx + 0.5) / k - 0.5, (sy + 0.5) / k - 0.5
            data += 4e5 * f * np.exp(-((ii - px) ** 2 + (jj - py) ** 2) / (2 * 0.6 ** 2))
        data += np.random.default_rng(1).normal(0, 40, data.shape)
        buf = io.BytesIO()
        fits.PrimaryHDU(data.astype(np.float32), hdr).writeto(buf)
        return buf.getvalue()
    return dl


def _metrics(out, clean, neb, grad):
    """(residual gradient fraction, kept nebula fraction) of a corrected image.

    Per channel the difference to the gradient-free image is regressed on [1, nebula, true gradient]:
    the nebula coefficient is what was removed of the nebula (-1 = all), the rest is residual gradient."""
    from scipy import ndimage
    diff = (out - clean)[::4, ::4].astype(np.float64)
    nb, gr = neb[::4, ::4], grad[::4, ::4]
    kept, frac = [], []
    for c in range(3):
        A = np.c_[np.ones(nb[..., c].size), nb[..., c].ravel(), gr[..., c].ravel()]
        coef, *_ = np.linalg.lstsq(A, diff[..., c].ravel(), rcond=None)
        kept.append(1 + coef[1])
        resid = (diff[..., c] - coef[0] - coef[1] * nb[..., c])
        sm = ndimage.uniform_filter(resid, 8)[8:-8, 8:-8]
        frac.append(np.ptp(sm) / np.ptp(gr[8:-8, 8:-8, c]))
    return float(max(frac)), float(min(kept))


@pytest.fixture
def mocked(monkeypatch, tmp_path):
    wcs = _wcs()
    calls = []
    monkeypatch.setattr(allsky, "CACHE", tmp_path / "allsky")
    monkeypatch.setattr(allsky, "_download", _fake_download(wcs, calls))
    return wcs, calls


def test_reduced_wcs_maps_the_same_sky():
    wcs = _wcs()
    wd, (h2, w2) = allsky.reduced_wcs(wcs, (H, W), 8)
    assert (h2, w2) == (96, 96)
    hdr = fits.Header()
    for k, v in wd.items():
        hdr[k] = v
    sw = WCS(hdr)
    for x, y in [(0, 0), (767, 0), (383.5, 383.5), (100, 700)]:
        ra, dec = wcs.all_pix2world([[x, y]], 0)[0]
        px, py = sw.all_world2pix([[ra, dec]], 0)[0]
        assert abs(px - ((x + 0.5) / 8 - 0.5)) < 1e-3 and abs(py - ((y + 0.5) / 8 - 0.5)) < 1e-3


def test_allsky_keeps_nebula_dbe_does_not(mocked):
    wcs, calls = mocked
    img, clean, neb, grad = _field()
    f0, k0 = _metrics(img, clean, neb, grad)
    assert f0 > 0.9
    ctx = plugins.Context(log=lambda m: None)
    ctx.layers["wcs"] = wcs
    pl = plugins.get("background.allsky")
    out = pl.run(img, ctx, **pl.resolve({}))
    frac, kept = _metrics(out, clean, neb, grad)
    assert frac < 0.10, frac
    assert kept > 0.80, kept
    assert ctx.layers["background_model"].shape == img.shape
    assert ctx.layers["allsky_reference"].shape == (96, 96, 3)
    assert sorted(set(calls)) == ["CDS/P/DSS2/blue", "CDS/P/DSS2/red"]
    # second run: from the cache, no download
    n = len(calls)
    pl.run(img, ctx, **pl.resolve({}))
    assert len(calls) == n

    dbe = plugins.get("background.dbe")
    out_d = dbe.run(img, plugins.Context(log=lambda m: None), **dbe.resolve({}))
    frac_d, kept_d = _metrics(out_d, clean, neb, grad)
    assert kept > kept_d + 0.1, (kept, kept_d)


def test_allsky_division_and_protect(mocked):
    wcs, _ = mocked
    img, clean, neb, grad = _field()
    ctx = plugins.Context(log=lambda m: None)
    ctx.layers["wcs"] = wcs
    pl = plugins.get("background.allsky")
    out = pl.run(img, ctx, **pl.resolve({"protect": 0.5, "scale": 6.0}))
    frac, kept = _metrics(out, clean, neb, grad)
    assert frac < 0.15 and kept > 0.75, (frac, kept)
    out = pl.run(img, ctx, **pl.resolve({"correction": "division"}))
    assert np.isfinite(out).all() and out.shape == img.shape


def test_fallback_without_wcs_runs_dbe():
    img, *_ = _field()
    ctx = plugins.Context(log=lambda m: None, tools={"astap": "C:/nonexistent/astap_cli.exe"})
    pl = plugins.get("background.allsky")
    out = pl.run(img, ctx, **pl.resolve({}))
    dbe = plugins.get("background.dbe")
    ref = dbe.run(img, plugins.Context(log=lambda m: None), **dbe.resolve({}))
    assert np.allclose(out, ref)
    assert "background_samples" in ctx.layers and "allsky_reference" not in ctx.layers
    ctx = plugins.Context(log=lambda m: None, tools={"astap": "C:/nonexistent/astap_cli.exe"})
    assert pl.run(img, ctx, **pl.resolve({"fallback": "none"})) is img


def test_fallback_when_download_fails(monkeypatch, tmp_path):
    def boom(url, timeout):
        raise allsky.urllib.error.URLError("offline")
    monkeypatch.setattr(allsky, "CACHE", tmp_path / "allsky")
    monkeypatch.setattr(allsky, "_download", boom)
    img, *_ = _field()
    logs = []
    ctx = plugins.Context(log=logs.append)
    ctx.layers["wcs"] = _wcs()
    pl = plugins.get("background.allsky")
    out = pl.run(img, ctx, **pl.resolve({}))
    assert out is not img and "background_samples" in ctx.layers
    assert any("falling back" in m for m in logs)
    assert allsky.reference_background(_wcs(), img.shape, log=lambda m: None) is None


def test_live_hips2fits_m8(tmp_path):
    """Real download from CDS hips2fits: M8, 0.5 deg, 200x200 (skipped when offline)."""
    wcs = _wcs(scale_arcsec=0.5 * 3600 / 200, rot_deg=0.0, w=200, h=200)
    wd, shape = allsky.reduced_wcs(wcs, (200, 200), 1)
    logs = []
    data = allsky.fetch_hips("CDS/P/DSS2/red", wd, timeout=60, cache_dir=tmp_path, log=logs.append)
    if data is None:
        pytest.skip("hips2fits unreachable: " + " ".join(logs))
    assert data.shape == (200, 200) and np.isfinite(data).all()
    # M8 (Lagoon) is bright in the middle of the field
    c = data[70:130, 70:130].mean()
    edge = np.median(np.r_[data[:20].ravel(), data[-20:].ravel()])
    assert c > edge
