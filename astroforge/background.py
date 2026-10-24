"""Gradient removal: robust polynomial fit on background tiles (ABE-like) and RBF surface (DBE-like)."""
import numpy as np

from .stats import LUMA, luminance


def _design(x, y, degree):
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append((x ** i) * (y ** j))
    return np.stack(cols, axis=-1)


def extract(img, degree=2, tiles=40, reject_sigma=1.5, iterations=4):
    """Subtract a smooth background model. Returns (corrected, model, kept_tile_fraction)."""
    h, w, nc = img.shape
    ts = max(16, max(h, w) // tiles)
    ny, nx = h // ts, w // ts
    lum = luminance(img)

    # per-tile medians (subsampled inside each tile)
    yy, xx, tl, tc = [], [], [], []
    for ty in range(ny):
        for tx in range(nx):
            sl = (slice(ty * ts, (ty + 1) * ts, 2), slice(tx * ts, (tx + 1) * ts, 2))
            yy.append((ty + 0.5) * ts)
            xx.append((tx + 0.5) * ts)
            tl.append(np.median(lum[sl]))
            tc.append([np.median(img[sl + (c,)]) for c in range(nc)])
    yy, xx = np.array(yy), np.array(xx)
    tl, tc = np.array(tl), np.array(tc)

    # start from the darker tiles: bright ones are nebulae, star clusters, galaxies
    med = np.median(tl)
    mad = np.median(np.abs(tl - med)) * 1.4826
    keep = tl < med + reject_sigma * mad

    xn, yn = xx / w * 2 - 1, yy / h * 2 - 1
    A = _design(xn, yn, degree)
    for _ in range(iterations):
        coef, *_ = np.linalg.lstsq(A[keep], tl[keep], rcond=None)
        res = tl - A @ coef
        s = np.median(np.abs(res[keep])) * 1.4826
        keep = res < reject_sigma * s

    coefs = [np.linalg.lstsq(A[keep], tc[keep, c], rcond=None)[0] for c in range(nc)]

    # evaluate the model on a coarse grid, then upscale (polynomial is smooth)
    gy = np.linspace(-1, 1, 256, dtype=np.float32)
    gx = np.linspace(-1, 1, 256, dtype=np.float32)
    GX, GY = np.meshgrid(gx, gy)
    G = _design(GX, GY, degree)
    import cv2
    model = np.empty_like(img)
    out = np.empty_like(img)
    for c in range(nc):
        small = (G @ coefs[c]).astype(np.float32)
        model[..., c] = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        out[..., c] = img[..., c] - model[..., c] + float(np.median(model[..., c][::16, ::16]))
    np.clip(out, 0, 1, out=out)
    return out, model, float(keep.mean())


def _sample_values(img, pts, size):
    """Per-channel median of each sample box, stars clipped. Returns (values (n, C), box MAD, clipped fraction)."""
    h, w, nc = img.shape
    lum = luminance(img)
    r = max(1, int(size) // 2)
    step = max(1, (2 * r + 1) // 32)
    vals, disp, frac = [], [], []
    for x, y in pts:
        x0, x1 = int(max(0, x - r)), int(min(w, x + r + 1))
        y0, y1 = int(max(0, y - r)), int(min(h, y + r + 1))
        sl = (slice(y0, max(y1, y0 + 1), step), slice(x0, max(x1, x0 + 1), step))
        lb = lum[sl].ravel()
        med = np.median(lb)
        mad = np.median(np.abs(lb - med)) * 1.4826
        ok = lb < med + 2.5 * mad if mad > 0 else np.ones(lb.shape, bool)
        box = img[sl].reshape(-1, nc)
        vals.append(np.median(box[ok], axis=0))
        disp.append(mad)
        frac.append(1 - ok.mean())
    return np.array(vals, np.float64), np.array(disp), np.array(frac)


def samples_auto(img, n=20, size=25, reject_sigma=2.5, iterations=4):
    """DBE-like automatic samples on a grid (n along the long side), boxes of `size` px.

    Boxes on stars clusters, nebulae and galaxies are rejected: too many clipped pixels, internal structure
    (MAD well above the typical box) or brighter than a robust quadratic trend. Returns array (k, 2) of [x, y].
    """
    h, w = img.shape[:2]
    sp = max(h, w) / n
    nx, ny = max(2, int(round(w / sp))), max(2, int(round(h / sp)))
    xs, ys = (np.arange(nx) + 0.5) * w / nx, (np.arange(ny) + 0.5) * h / ny
    pts = np.array([[x, y] for y in ys for x in xs])
    vals, disp, frac = _sample_values(img, pts, size)
    lv = vals @ LUMA if vals.shape[1] == 3 else vals[:, 0]
    dm = np.median(disp)
    keep = (frac < 0.3) & (disp < dm + 4 * np.median(np.abs(disp - dm)) * 1.4826 + 1e-12)
    npx = (min(size, 32) ** 2)
    A = _design(pts[:, 0] / w * 2 - 1, pts[:, 1] / h * 2 - 1, 2)
    for _ in range(iterations):
        if keep.sum() < 8:
            break
        coef, *_ = np.linalg.lstsq(A[keep], lv[keep], rcond=None)
        res = lv - A @ coef
        s = max(np.median(np.abs(res[keep])) * 1.4826, 2 * dm / np.sqrt(npx), 1e-9)
        keep &= (res < reject_sigma * s) & (res > -5 * s)
    return pts[keep]


def extract_rbf(img, samples=None, smoothing=0.02, size=25, n=20, correction="subtraction", grid=192,
                reject_sigma=2.5):
    """Background model from a thin-plate RBF through sample-box medians (DynamicBackgroundExtraction-like).

    `samples`: list of [x, y] in pixels (used as given) or None/empty = `samples_auto`. The surface is evaluated
    on a coarse grid (`grid` points along the long side) and upsampled. Returns (corrected, model, samples).
    """
    import cv2
    from scipy.interpolate import RBFInterpolator
    h, w, nc = img.shape
    user = samples is not None and len(samples) > 0
    pts = np.asarray(samples, np.float64).reshape(-1, 2) if user else samples_auto(img, n, size, reject_sigma)
    if len(pts) < 3:
        raise ValueError("background: need at least 3 samples")
    vals, _, _ = _sample_values(img, pts, size)
    L = float(max(h, w))
    gw, gh = max(2, int(round(grid * w / L))), max(2, int(round(grid * h / L)))
    gx = ((np.arange(gw) + 0.5) * w / gw - 0.5) / L
    gy = ((np.arange(gh) + 0.5) * h / gh - 0.5) / L
    G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    rbf = RBFInterpolator(pts / L, vals, kernel="thin_plate_spline", smoothing=smoothing, degree=1)
    small = rbf(G).reshape(gh, gw, nc).astype(np.float32)
    model = np.empty_like(img)
    out = np.empty_like(img)
    for c in range(nc):
        model[..., c] = cv2.resize(np.ascontiguousarray(small[..., c]), (w, h), interpolation=cv2.INTER_CUBIC)
        ref = float(np.median(model[::16, ::16, c]))
        if correction.startswith("div"):
            out[..., c] = img[..., c] / np.maximum(model[..., c], 1e-6) * ref
        else:
            out[..., c] = img[..., c] - model[..., c] + ref
    np.clip(out, 0, 1, out=out)
    return out, model, pts
