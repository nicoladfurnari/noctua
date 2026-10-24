"""Star-based registration: triangle matching (astroalign) refined with an affine fit, optional distortion
correction (polynomial or thin-plate spline on the residuals) and clamped Lanczos resampling."""
import astroalign
import cv2
import numpy as np
from scipy.spatial import cKDTree


def match(src_xy, ref_xy, max_points=80):
    """Affine 2x3 matrix mapping src -> ref coordinates, plus number of matched stars and RMS error.
    max_points: control points for the triangle matching (the affine refinement always uses every star)."""
    T, (s, r) = astroalign.find_transform(src_xy[:max_points], ref_xy[:max_points], max_control_points=max_points)
    # refine with every star that lands near a reference star
    M = T.params[:2]
    proj = src_xy @ M[:, :2].T + M[:, 2]
    d, j = cKDTree(ref_xy).query(proj)
    good = d < 2.0
    if good.sum() >= 6:
        A = np.hstack([src_xy[good], np.ones((good.sum(), 1))])
        sol, *_ = np.linalg.lstsq(A, ref_xy[j[good]], rcond=None)
        M = sol.T
    proj = src_xy @ M[:, :2].T + M[:, 2]
    d, _ = cKDTree(ref_xy).query(proj)
    inl = d < 2.0
    rms = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("nan")
    return M.astype(np.float64), int(inl.sum()), rms


def _affine(xy, M):
    return xy @ M[:, :2].T + M[:, 2]


def _inv(M):
    return cv2.invertAffineTransform(np.asarray(M, np.float64))


class Distortion:
    """Residual field E on reference coordinates: a reference pixel p corresponds to the affine-mapped source
    position p + E(p), i.e. source = M^-1 (p + E(p)). kind 'poly' (order n) or 'tps'."""

    def __init__(self, kind, center, scale, coef=None, order=3, rbf=None, info=None):
        self.kind, self.center, self.scale, self.coef, self.order, self.rbf = kind, center, scale, coef, order, rbf
        self.info = info or {}

    def _terms(self, xy):
        u = (np.asarray(xy, np.float64) - self.center) / self.scale
        return np.stack([u[:, 0] ** i * u[:, 1] ** j for i in range(self.order + 1)
                         for j in range(self.order + 1 - i)], 1)

    def residual(self, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        if self.kind == "tps":
            return self.rbf((xy - self.center) / self.scale)
        return self._terms(xy) @ self.coef

    def __call__(self, xy):
        """Reference pixel -> position in the affine-mapped source frame."""
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        return xy + self.residual(xy)

    def source(self, xy, M):
        """Reference pixel -> source pixel."""
        return _affine(self(xy), _inv(M))

    def maps(self, M, shape, grid=16):
        """Dense cv2.remap maps (float32) for the reference grid `shape`. The smooth residual is evaluated on a
        coarse grid and upsampled; the affine part is exact."""
        h, w = shape
        ny, nx = -(-h // grid), -(-w // grid)
        gy, gx = np.mgrid[0:ny, 0:nx]
        nodes = np.stack([(gx.ravel() + 0.5) * grid - 0.5, (gy.ravel() + 0.5) * grid - 0.5], 1)
        Mi = _inv(M)
        e = (self.residual(nodes) @ Mi[:, :2].T).astype(np.float32).reshape(ny, nx, 2)
        e = cv2.resize(e, (nx * grid, ny * grid), interpolation=cv2.INTER_LINEAR)[:h, :w]
        x = np.arange(w, dtype=np.float32)[None, :]
        y = np.arange(h, dtype=np.float32)[:, None]
        mx = (Mi[0, 0] * x + Mi[0, 1] * y + Mi[0, 2]).astype(np.float32) + e[..., 0]
        my = (Mi[1, 0] * x + Mi[1, 1] * y + Mi[1, 2]).astype(np.float32) + e[..., 1]
        return mx, my


def _pairs(proj, ref_xy, tol):
    """Mutual nearest neighbours closer than tol: indices into proj and ref."""
    d, j = cKDTree(ref_xy).query(proj, distance_upper_bound=tol)
    ok = np.isfinite(d)
    i = np.nonzero(ok)[0]
    j = j[ok]
    _, back = cKDTree(proj).query(ref_xy[j])
    keep = back == i
    return i[keep], j[keep]


def fit_distortion(src_xy, ref_xy, M, order=3, tol=4.0, kappa=3.0, max_stars=2000, smoothing=None):
    """Distortion model on the star matches left by the affine M (src -> ref), with iterative re-matching and
    kappa-sigma outlier rejection. order: polynomial degree (int) or 'tps' (thin-plate spline, smoothed).
    The returned Distortion has .info with rms_affine (best affine on the same stars), rms,
    distortion_rms_improvement (px) and n: use it only when the improvement is worth it (> ~0.1 px)."""
    src_xy = np.asarray(src_xy, np.float64)[:max_stars]
    ref_xy = np.asarray(ref_xy, np.float64)[:max_stars]
    proj = _affine(src_xy, M)
    center = ref_xy.mean(0)
    scale = float(max(np.ptp(ref_xy, 0).max() / 2, 1.0))
    poly_order = 3 if order == "tps" else int(order)
    model = Distortion("poly", center, scale, np.zeros(((poly_order + 1) * (poly_order + 2) // 2, 2)), poly_order)
    i, j = _pairs(proj, ref_xy, tol)
    keep = np.ones(len(i), bool)
    for it in range(6):
        if len(i) < max(12, 2 * model._terms(ref_xy[:1]).shape[1]):
            raise ValueError(f"too few star matches for a distortion model ({len(i)})")
        P, R = proj[i[keep]], ref_xy[j[keep]]
        model.coef, *_ = np.linalg.lstsq(model._terms(R), P - R, rcond=None)
        res = np.hypot(*(proj[i] - model(ref_xy[j])).T)
        sig = max(1.4826 * float(np.median(res[keep])), 0.01)
        new = res < kappa * sig + 0.05
        if it < 3:   # re-match with the improved model (stars far off the affine at the corners)
            pred = proj - model.residual(proj)
            i, j = _pairs(pred, ref_xy, max(1.0, min(tol, 6 * sig + 0.5)))
            keep = np.ones(len(i), bool)
            continue
        if np.array_equal(new, keep):
            break
        keep = new
    P, R = proj[i[keep]], ref_xy[j[keep]]
    model.coef, *_ = np.linalg.lstsq(model._terms(R), P - R, rcond=None)
    if order == "tps":
        from scipy.interpolate import RBFInterpolator
        sm = len(P) * 1e-5 if smoothing is None else smoothing
        rbf = RBFInterpolator((R - center) / scale, P - R, kernel="thin_plate_spline", smoothing=sm, degree=1)
        model = Distortion("tps", center, scale, rbf=rbf)
    # baseline: the best affine on the same stars, so the improvement is the non-linear part only
    S = src_xy[i[keep]]
    A = np.hstack([S, np.ones((len(S), 1))])
    Ma = np.linalg.lstsq(A, R, rcond=None)[0].T
    rms_aff = float(np.sqrt(np.mean(np.sum((_affine(S, Ma) - R) ** 2, 1))))
    rms = float(np.sqrt(np.mean(np.sum((P - model(R)) ** 2, 1))))
    model.info = {"kind": model.kind, "order": order, "n": int(len(P)), "rms_affine": rms_aff, "rms": rms,
                  "distortion_rms_improvement": rms_aff - rms}
    return model


def _remap_channels(img, fn):
    if img.ndim == 3 and img.shape[2] > 4:
        return np.stack([fn(np.ascontiguousarray(img[..., c])) for c in range(img.shape[2])], -1)
    return fn(img)


def warp(img, M, shape, interpolation=cv2.INTER_LANCZOS4, clamp=True, distortion=None, grid=16):
    """Resample img into the reference grid; pixels outside the frame become NaN (also where the interpolation
    kernel touches the outside). clamp: Lanczos/cubic results are limited to the min/max of the 3x3 source
    neighbourhood of the nearest pixel, which removes the dark/bright rings around bright stars.
    distortion: a Distortion from fit_distortion (dense cv2.remap instead of warpAffine)."""
    h, w = shape
    if distortion is None:
        def rs(a, flags):
            return cv2.warpAffine(a, M, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    else:
        mx, my = distortion.maps(M, shape, grid)

        def rs(a, flags):
            return cv2.remap(a, mx, my, flags, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    out = _remap_channels(img, lambda a: rs(a, interpolation))
    if clamp and interpolation in (cv2.INTER_LANCZOS4, cv2.INTER_CUBIC):
        k = np.ones((3, 3), np.uint8)
        lo = _remap_channels(img, lambda a: rs(cv2.erode(a, k), cv2.INTER_NEAREST))
        hi = _remap_channels(img, lambda a: rs(cv2.dilate(a, k), cv2.INTER_NEAREST))
        np.maximum(out, lo, out=out)
        np.minimum(out, hi, out=out)
    if out.ndim == 2 and img.ndim == 3:
        out = out[..., None]
    return out


def coverage(transforms, src_shape, ref_shape, step=8):
    """Fraction of frames covering each reference pixel (computed on a grid `step` times smaller)."""
    h, w = ref_shape
    hs, ws = (h + step - 1) // step, (w + step - 1) // step
    cov = np.zeros((hs, ws), np.float32)
    ones = np.ones(src_shape, np.float32)
    S = np.diag([1 / step, 1 / step, 1])
    for M in transforms:
        Ms = (S @ np.vstack([M, [0, 0, 1]]))[:2]
        cov += cv2.warpAffine(ones[::step, ::step], Ms @ np.diag([step, step, 1]), (ws, hs),
                              flags=cv2.INTER_NEAREST, borderValue=0)
    return cov / max(len(transforms), 1), step


def best_rect(mask):
    """Largest-ish axis aligned rectangle inside a boolean mask, by peeling the worst border."""
    y0, y1, x0, x1 = 0, mask.shape[0], 0, mask.shape[1]
    while y1 - y0 > 2 and x1 - x0 > 2:
        sub = mask[y0:y1, x0:x1]
        edges = {"t": sub[0].mean(), "b": sub[-1].mean(), "l": sub[:, 0].mean(), "r": sub[:, -1].mean()}
        worst = min(edges, key=edges.get)
        if edges[worst] >= 0.999:
            break
        if worst == "t":
            y0 += 1
        elif worst == "b":
            y1 -= 1
        elif worst == "l":
            x0 += 1
        else:
            x1 -= 1
    return y0, y1, x0, x1


def scale_matrix(M, factor):
    """Matrix for coordinates multiplied by `factor` (e.g. superpixel -> full resolution)."""
    M = M.copy()
    M[:, 2] *= factor
    return M
