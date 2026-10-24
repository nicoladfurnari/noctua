"""Local normalization, dual-band extraction, dark optimization and comet stacking on synthetic data."""
from datetime import datetime, timedelta

import cv2
import numpy as np

from astroforge.stack import comet, darkopt, dualband, localnorm


def _gauss(shape, x, y, sigma, amp=1.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def _stars(shape, n, rng, sigma=1.3, amp=(0.05, 0.6), avoid=None, margin=8):
    img = np.zeros(shape, np.float32)
    pos = []
    while len(pos) < n:
        x, y = rng.uniform(margin, shape[1] - margin), rng.uniform(margin, shape[0] - margin)
        if avoid is not None and avoid(x, y):
            continue
        a = rng.uniform(*amp)
        img += _gauss(shape, x, y, sigma, a)
        pos.append((x, y, a))
    return img, pos


# ---------------------------------------------------------------- local normalization

def test_localnorm_removes_frame_gradient(tmp_path):
    rng = np.random.default_rng(1)
    H, W = 384, 512
    stars, _ = _stars((H, W), 150, rng)
    neb = _gauss((H, W), 300, 180, 50, 0.08)
    scene = stars + neb
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = 0.06 * (xx / W) + 0.03 * ((yy - H / 2) / H) ** 2 * 4          # moonlight-like gradient
    ref_clean = 0.10 + scene
    ref = ref_clean + rng.normal(0, 0.004, (H, W))
    frame = (0.12 + grad + 1.15 * scene + rng.normal(0, 0.004, (H, W))).astype(np.float32)
    frame[:, :20] = np.nan                                                  # registration border
    model = localnorm.fit(frame, ref, scale_px=128)
    out = localnorm.apply(frame, model, 0, H)
    resid = out - ref_clean
    # residual background: robust tile medians (stars excluded) vs the gradient amplitude
    t = 32
    meds = [np.nanmedian(resid[y:y + t, x:x + t]) for y in range(0, H, t) for x in range(32, W, t)]
    span = np.nanmax(meds) - np.nanmin(meds)
    gspan = grad.max() - grad.min()
    assert span < 0.1 * gspan, (span, gspan)
    assert abs(model["scale"] - 1 / 1.15) < 0.03
    # star fluxes match the reference too
    assert abs(np.nansum(out[100:140, 100:140] - 0) - np.nansum(ref_clean[100:140, 100:140])) \
        < 0.05 * np.nansum(ref_clean[100:140, 100:140])
    # row bands and serialisation
    band = localnorm.apply(frame[50:90], model, 50, 90)
    assert np.allclose(band, out[50:90], equal_nan=True, atol=1e-6)
    p = tmp_path / "ln.npz"
    localnorm.save(model, p)
    m2 = localnorm.load(p)
    assert m2["a"].size < 2000 and m2["tile"] == model["tile"]
    assert np.allclose(localnorm.apply(frame, m2, 0, H), out, equal_nan=True)
    # 3-channel frames
    m3 = localnorm.fit(np.dstack([frame] * 3), np.dstack([ref] * 3), scale_px=128)
    assert m3["a"].shape[-1] == 3
    # reference selection: the flat one wins
    st = [localnorm.frame_stats(frame), localnorm.frame_stats(ref)]
    assert localnorm.select_reference(st) == 1


# ---------------------------------------------------------------- dual band

def test_dualband_superpixel_and_full():
    rng = np.random.default_rng(2)
    h, w = 64, 80
    ha = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2) + _gauss((h, w), 30, 30, 6, 0.5)
    o3 = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2) * 0.5
    lg, lr, gb = 0.08, 0.02, 0.85
    cfa = np.zeros((2 * h, 2 * w), np.float32)
    cfa[0::2, 0::2] = ha + lr * o3            # RGGB
    cfa[0::2, 1::2] = o3 + lg * ha
    cfa[1::2, 0::2] = o3 + lg * ha
    cfa[1::2, 1::2] = gb * o3
    out = dualband.extract(cfa, "RGGB", ha_leak_g=lg, oiii_leak_r=lr, b_gain=gb)
    assert out["Ha"].shape == (h, w) and out["OIII"].shape == (h, w)
    assert np.allclose(out["Ha"], ha, atol=1e-5)
    assert np.allclose(out["OIII"], o3, atol=1e-5)
    # another pattern: same sites relabelled
    out2 = dualband.extract(cfa[::-1, ::-1], "BGGR", ha_leak_g=lg, oiii_leak_r=lr, b_gain=gb)
    assert np.allclose(out2["Ha"], ha[::-1, ::-1], atol=1e-5)
    assert np.allclose(out2["OIII"], o3[::-1, ::-1], atol=1e-5)
    assert set(dualband.extract(cfa, "RGGB", mode="ha")) == {"Ha"}

    # full resolution from smooth images
    H, W = 128, 160
    HA = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 6) * 4
    O3 = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 6) * 3
    c = np.zeros((H, W), np.float32)
    c[0::2, 0::2] = (HA + lr * O3)[0::2, 0::2]
    c[0::2, 1::2] = (O3 + lg * HA)[0::2, 1::2]
    c[1::2, 0::2] = (O3 + lg * HA)[1::2, 0::2]
    c[1::2, 1::2] = (gb * O3)[1::2, 1::2]
    full = dualband.extract_full(c, "RGGB", ha_leak_g=lg, oiii_leak_r=lr, b_gain=gb)
    assert full["Ha"].shape == (H, W)
    s = np.s_[4:-4, 4:-4]
    assert np.abs(full["Ha"][s] - HA[s]).max() < 0.02 * np.ptp(HA)
    assert np.abs(full["OIII"][s] - O3[s]).max() < 0.02 * np.ptp(O3)


# ---------------------------------------------------------------- dark optimization

def _dark_set(rng, glow):
    H, W = 512, 512
    bias = (0.05 + rng.normal(0, 0.001, (H, W))).astype(np.float32)
    P = (0.004 * rng.gamma(2.0, 1.0, (H, W))).astype(np.float32)        # dark current, per pixel
    hot = rng.random((H, W)) < 0.001
    P[hot] += rng.uniform(0.05, 0.3, hot.sum())
    G = _gauss((H, W), 0, 0, 90, 0.08) if glow else np.zeros((H, W), np.float32)
    dark = bias + P + G + rng.normal(0, 0.0004, (H, W)).astype(np.float32)   # master: low noise
    k_true, kg_true = 0.7, 0.9
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sky = 0.03 + 0.01 * xx / W
    st, _ = _stars((H, W), 120, rng, amp=(0.05, 0.5))
    light = bias + k_true * P + kg_true * G + sky + st + rng.normal(0, 0.004, (H, W))
    return light.astype(np.float32), dark.astype(np.float32), bias, k_true, kg_true, P, G, sky + st


def test_darkopt_scale():
    rng = np.random.default_rng(4)
    light, dark, bias, kt, _, P, _, _ = _dark_set(rng, glow=False)
    k = darkopt.optimal_scale(light, dark, bias)
    assert abs(k - kt) < 0.05 * kt, k
    assert not darkopt.glow_mask(dark, bias).any()
    k2, kg2, m = darkopt.optimal_scale(light, dark, bias, glow=True)
    assert abs(k2 - kt) < 0.05 * kt and not m.any()
    # ROI still works
    assert abs(darkopt.optimal_scale(light, dark, bias, roi=(100, 400, 100, 400)) - kt) < 0.05 * kt


def test_darkopt_amp_glow():
    rng = np.random.default_rng(5)
    light, dark, bias, kt, kgt, P, G, signal = _dark_set(rng, glow=True)
    m = darkopt.glow_mask(dark, bias)
    assert m[:40, :40].all() and not m[300:, 300:].any()
    k, kg, mask = darkopt.optimal_scale(light, dark, bias, glow=True)
    assert abs(k - kt) < 0.05 * kt, k
    assert abs(kg - kgt) < 0.05 * kgt, kg
    cal = darkopt.subtract(light, dark, bias, k, kg, mask)
    # the glow corner is flat after calibration (residual much smaller than the uncorrected one)
    plain = darkopt.subtract(light, dark, bias, k)
    r_opt = np.median((cal - signal)[:60, :60]) - np.median((cal - signal)[200:260, 200:260])
    r_plain = np.median((plain - signal)[:60, :60]) - np.median((plain - signal)[200:260, 200:260])
    assert abs(r_opt) < 0.25 * abs(r_plain)


# ---------------------------------------------------------------- comet

def test_comet_stack():
    rng = np.random.default_rng(6)
    H, W, N = 200, 240, 12
    t0 = datetime(2026, 10, 21, 22, 0, 0)
    times = [t0 + timedelta(seconds=300 * i + (1800 if i >= 6 else 0)) for i in range(N)]   # a gap
    secs = np.array([(t - t0).total_seconds() for t in times])
    v = np.array([0.012, 0.006])                                                           # px / s
    truth = np.array([60.0, 70.0]) + secs[:, None] * v
    path_near = lambda x, y: np.min(np.hypot(truth[:, 0] - x, truth[:, 1] - y)) < 12
    stars, spos = _stars((H, W), 60, rng, sigma=1.2, amp=(0.1, 0.8), avoid=path_near)
    frames = []
    for i in range(N):
        f = 0.05 + stars + _gauss((H, W), *truth[i], 3.0, 0.3) + rng.normal(0, 0.005, (H, W))
        frames.append(f.astype(np.float32))
    first = truth[0] + [0.8, -0.7]
    last = truth[-1] + [-0.9, 0.6]
    iso = [t.isoformat() for t in times]
    pos = comet.comet_positions(frames, None, first, last, times=iso)
    assert np.hypot(*(pos - truth).T).max() < 0.5, np.hypot(*(pos - truth).T)
    # tracking without the last position
    pos2 = comet.comet_positions(frames, None, first, times=times)
    assert np.hypot(*(pos2 - truth).T).max() < 0.5

    ident = [np.array([[1, 0, 0], [0, 1, 0]], np.float64) for _ in range(N)]
    ref = 0
    ct = comet.comet_transforms(ident, pos, ref)
    assert np.allclose(comet.comet_transforms({3: ident[3]}, {3: pos[3], 0: pos[0]}, 0)[3], ct[3])
    warped = [cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LANCZOS4, borderValue=np.nan) for f, M in zip(frames, ct)]
    comet_stack = np.nanmedian(np.stack(warped), 0)
    star_stack = np.median(np.stack(frames), 0)
    # comet sharp in the comet stack: peak and centroid at the reference position
    cx, cy = truth[ref]
    peak_c = comet_stack[int(round(cy)) - 1:int(round(cy)) + 2, int(round(cx)) - 1:int(round(cx)) + 2].max() - 0.05
    assert peak_c > 0.27
    p = comet.nucleus(comet_stack, truth[ref], radius=6)
    assert np.hypot(p[0] - cx, p[1] - cy) < 0.5
    # combine: stars from the star stack, comet from the comet stack
    final = comet.combine_star_comet(star_stack, comet_stack, 15, comet_xy=tuple(truth[ref]), trail=pos)
    assert final[int(round(cy)), int(round(cx))] - 0.05 > 0.27
    x, y, a = max(spos, key=lambda s: s[2])
    assert abs(final[int(round(y)), int(round(x))] - star_stack[int(round(y)), int(round(x))]) < 0.02
    # the smeared comet along the path (away from the reference nucleus) is gone
    mx, my = truth[N - 1]
    assert abs(final[int(round(my)), int(round(mx))] - 0.05) < 0.03
