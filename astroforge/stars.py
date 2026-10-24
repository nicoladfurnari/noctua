"""Star mask and star reduction (morphological selection under the mask)."""
import cv2
import numpy as np

from .stats import luminance


def star_mask(img, k=4.0, grow=2, blur=1.5):
    """Mask of small, bright structures (stars) from a stretched image. Values in [0,1]."""
    lum = luminance(img)
    h, w = lum.shape
    # large-scale background via downsampled median, so nebulae are not in the mask
    small = cv2.resize(lum, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    detail = lum - bg
    d = detail[::4, ::4]
    sigma = float(np.median(np.abs(d - np.median(d)))) * 1.4826
    if sigma <= 0:                       # a star layer has the sky at exactly 0: MAD 0 gave a NaN mask
        nz = np.abs(d[d != 0])
        sigma = float(np.median(nz)) * 1.4826 if nz.size else 1e-6
    sigma = max(sigma, 1e-6)
    mask = np.clip((detail - k * sigma) / (2 * k * sigma), 0, 1)
    if grow:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        mask = cv2.dilate(mask, ker)
    if blur:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    return mask.astype(np.float32)


def reduce_stars(img, mask, amount=0.75, iterations=2, size=5):
    """Shrink stars: blend of erosion and median (selection ~0.25), applied under the mask."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    sel = img.copy()
    for _ in range(iterations):
        for c in range(img.shape[-1]):
            ch = np.ascontiguousarray(sel[..., c])
            ero = cv2.erode(ch, ker)
            med = cv2.medianBlur(ch, 5)
            sel[..., c] = 0.5 * ero + 0.5 * med
    m = (mask * amount)[..., None]
    return (img * (1 - m) + sel * m).astype(np.float32)
