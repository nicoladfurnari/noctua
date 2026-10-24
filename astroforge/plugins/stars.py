"""Star reduction, built-in star removal and recombination."""
import cv2
import numpy as np

from .. import masks, stars
from .base import Param, Plugin, register


@register
class StarReduce(Plugin):
    id = "stars.reduce"
    name = "Star reduction"
    category = "stars"
    stage = "nonlinear"
    description = "Shrinks stars with a morphological selection under a star mask."
    params = {"amount": Param(0.75, 0.0, 1.0, "strength"),
              "iterations": Param(2, 1, 5, "repeat count, more = smaller stars"),
              "size": Param(5, 3, 9, "structuring element size (odd)", px="odd")}

    def run(self, img, ctx, amount, iterations, size):
        m = masks.star_mask(img, ctx.linear, scale=ctx.scale)
        ctx.log(f"star mask coverage {m.mean():.1%}")
        return stars.reduce_stars(img, m, amount=amount, iterations=iterations, size=size)


@register
class StarMorph(Plugin):
    id = "stars.morph"
    name = "Morphological transformation (stars)"
    category = "stars"
    stage = "any"
    description = ("PixInsight MorphologicalTransformation repeated on the star layer left by a star removal "
                   "(StarNet/Dark Star): erosion, or morphological selection between erosion and dilation, with a "
                   "round structuring element, blended by amount at every iteration. Many light iterations shrink "
                   "the stars without the dark rings of one strong pass. Without a star layer it works on the image "
                   "under a star mask.")
    params = {"operator": Param("selection", choices=["erosion", "selection", "dilation"],
                                help="selection = mix of erosion and dilation (PixInsight Morphological Selection)"),
              "selection": Param(0.25, 0.0, 1.0, "with 'selection': 0 = erosion .. 1 = dilation"),
              "size": Param(3, 3, 9, "structuring element size (odd)", px="odd"),
              "iterations": Param(4, 1, 12, "repeat count"),
              "amount": Param(0.6, 0.0, 1.0, "blend of each iteration"),
              "keep_core": Param(1.0, 0.0, 1.0, "keep the star cores as bright as before (only the edges and halos "
                                                "shrink): 0 = plain transformation, 1 = core fully kept")}

    @staticmethod
    def _apply(x, operator, selection, size, iterations, amount, keep_core=0.0):
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        x = np.ascontiguousarray(np.clip(x, 0, 1).astype(np.float32))
        orig = x
        for _ in range(iterations):
            ch = []
            for c in range(x.shape[-1]):
                a = np.ascontiguousarray(x[..., c])
                er = cv2.erode(a, ker)
                if operator == "erosion":
                    y = er
                elif operator == "dilation":
                    y = cv2.dilate(a, ker)
                else:
                    y = er * (1 - selection) + cv2.dilate(a, ker) * selection
                ch.append(a * (1 - amount) + y * amount)
            x = np.stack(ch, -1)
        if keep_core:
            # core = pixels near the local peak of each star: they keep the original brightness, the wings
            # (below ~60% of the peak) keep the eroded value, smooth in between
            L = orig.max(-1)
            peak = cv2.dilate(np.ascontiguousarray(L), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1,) * 2))
            w = np.clip((L / np.maximum(peak, 1e-6) - 0.6) / 0.3, 0, 1) * (peak > 0.05)
            w = (keep_core * w * w * (3 - 2 * w))[..., None]
            x = np.maximum(x, orig * w + x * (1 - w))
        return x.astype(np.float32)

    def run(self, img, ctx, operator, selection, size, iterations, amount, keep_core):
        st = ctx.layers.get("stars")
        ctx.log(f"morphological {operator} {size}px x{iterations} amount {amount} on the "
                + ("star layer" if st is not None else "image (star mask)"))
        if st is not None:
            ctx.layers["stars"] = self._apply(st, operator, selection, size, iterations, amount, keep_core)
            return img
        m = masks.star_mask(img, ctx.linear, scale=ctx.scale)[..., None]
        out = self._apply(img, operator, selection, size, iterations, amount, keep_core)
        return (img * (1 - m) + out * m).astype(np.float32)


@register
class StarRemoveBasic(Plugin):
    id = "stars.remove_basic"
    name = "Star removal (built-in)"
    category = "stars"
    stage = "nonlinear"
    description = ("Removes stars without AI: stars are the significant positive structures of the small "
                   "wavelet scales; they are subtracted under a star mask and the cores are filled by a "
                   "morphological opening. Stars are kept for 'stars.recombine'. StarNet is still better.")
    params = {"size": Param(11, 5, 31, "largest star diameter removed (odd)", px="odd"),
              "scales": Param(4, 2, 5, "wavelet scales treated as stars (more = bigger stars)", px="octave"),
              "threshold": Param(3.0, 1.0, 8.0, "detection threshold in noise units (lower = more stars)")}

    def run(self, img, ctx, size, scales, threshold):
        from .. import denoise
        from ..stats import luminance
        lum = luminance(img)
        w = denoise.starlet(np.ascontiguousarray(lum), scales)
        sig = []
        for j in range(scales):
            d = w[j][::4, ::4]
            sig.append(float(np.median(np.abs(d - np.median(d)))) * 1.4826)
        # star signal: significant positive small-scale coefficients
        S = sum(np.maximum(w[j] - threshold * sig[j], 0) for j in range(scales))
        m = np.clip(S / (threshold * sig[0] + 1e-9), 0, 1)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        m = cv2.GaussianBlur(m, (0, 0), 1.5)[..., None]
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        starless = np.empty_like(img)
        for c in range(img.shape[-1]):
            ch = np.ascontiguousarray(img[..., c])
            wc = denoise.starlet(ch, scales)
            peaks = sum(np.maximum(wc[j], 0) for j in range(scales))
            sub = ch - peaks * m[..., 0]
            opened = cv2.GaussianBlur(cv2.morphologyEx(ch, cv2.MORPH_OPEN, ker), (0, 0), size / 4)
            # never brighter than the opened background, never darker than the wavelet-cleaned value
            starless[..., c] = np.maximum(np.minimum(sub, opened), sub - 0.02) * m[..., 0] + ch * (1 - m[..., 0])
        starless = np.clip(starless, 0, 1)
        ctx.log(f"star mask coverage {float((m > 0.5).mean()):.1%}")
        ctx.layers["stars"] = np.clip(1 - (1 - img) / np.maximum(1 - starless, 1e-6), 0, 1)
        ctx.layers["stars_mode"] = "screen"
        return starless.astype(np.float32)


@register
class StarRecombine(Plugin):
    id = "stars.recombine"
    name = "Add stars back"
    category = "stars"
    stage = "nonlinear"
    description = "Screen-blends the stars saved by a star removal step back onto the image."
    params = {"amount": Param(1.0, 0.0, 1.5, "star brightness"),
              "reduce": Param(0.0, 0.0, 1.0, "shrink the stars before adding them"),
              "star_stretch": Param(60.0, 1.0, 1000.0,
                                    "arcsinh stretch for stars removed while linear (keeps their color)")}

    def run(self, img, ctx, amount, reduce, star_stretch):
        st = ctx.layers.get("stars")
        if st is None:
            ctx.log("no star layer saved, skipped")
            return img
        if reduce:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            er = np.stack([cv2.erode(np.ascontiguousarray(st[..., c]), ker) for c in range(st.shape[-1])], -1)
            st = st * (1 - reduce) + er * reduce
        if ctx.layers.get("stars_mode") == "add":
            if ctx.linear:
                return np.clip(img + st * amount, 0, 1).astype(np.float32)
            # stars were removed on linear data: stretch them on their own, color-preserving
            lum = st.mean(axis=-1, keepdims=True)
            st = st * (np.arcsinh(star_stretch * lum) / np.arcsinh(star_stretch)) / np.maximum(lum, 1e-7)
            st = np.clip(st / np.maximum(st.max(axis=-1, keepdims=True), 1), 0, 1).astype(np.float32)
        out = 1 - (1 - img) * (1 - np.clip(st * amount, 0, 1))
        return out.astype(np.float32)


def _disc_kernel(r, ring=False):
    """Normalised disc (or annulus 1.3r..1.8r) kernel for the matched filter of StarHaloClean."""
    import cv2 as _cv2
    R = int(np.ceil(1.8 * r)) if ring else int(np.ceil(r))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.hypot(yy, xx)
    k = ((d >= 1.3 * r) & (d <= 1.8 * r)) if ring else (d <= r)
    k = k.astype(np.float32)
    return k / max(k.sum(), 1)


@register
class StarHaloClean(Plugin):
    id = "stars.halo_clean"
    name = "Remove round discs (dust donuts / reflections)"
    category = "stars"
    stage = "any"
    description = ("Removes round, flat, soft-edged discs from the background of the starless image: dust donuts "
                   "left by flats that do not match the lights (same place in every filter) and filter "
                   "reflections. Like CLEAN in radio astronomy: a disc-shaped matched filter (disc minus the ring "
                   "around it, several radii) finds the strongest disc, its model is subtracted, and so on until no "
                   "disc stands above the noise; overlapping discs are taken one by one. Never inside bright "
                   "nebula (main channel, Ha); the noise texture stays. Real fix: matching flats.")
    params = {"min_radius": Param(10, 2, 200, "smallest disc radius in px", px=True),
              "max_radius": Param(160, 10, 600, "largest disc radius in px", px=True),
              "threshold": Param(5.0, 2.0, 30.0, "detection: disc contrast in noise sigmas of the disc average"),
              "nebula_sigma": Param(6.0, 1.0, 30.0, "main-channel level (noise sigmas) above which it is nebula"),
              "nebula_share": Param(0.15, 0.0, 1.0, "... and at least this share of the main channel's range "
                                                     "(a dust disc is faint, the nebula is not)"),
              "max_discs": Param(80, 1, 500, "per channel"),
              "amount": Param(1.0, 0.0, 1.0, "share of the disc excess removed")}

    def run(self, img, ctx, min_radius, max_radius, threshold, nebula_sigma, nebula_share, max_discs, amount):
        import cv2 as _cv2
        from scipy import ndimage as ndi
        h, w = img.shape[:2]
        nch = img.shape[-1]
        f = max(1, int(round(max_radius / 24)))                  # work reduced: discs are large and smooth
        small = _cv2.resize(img, (max(8, w // f), max(8, h // f)), interpolation=_cv2.INTER_AREA)
        small = small[..., None] if small.ndim == 2 else small
        flat = small.reshape(-1, nch)
        bg = np.median(flat, 0)
        noise = np.median(np.abs(flat - bg), 0) * 1.4826 + 1e-9      # per pixel of the reduced image
        sm_ref = ndi.gaussian_filter(small[..., 0 if nch == 1 else 0], 1.5)
        sig = [(np.percentile(ndi.gaussian_filter(small[..., c], 1.5), 99, None) - bg[c]) / noise[c] for c in range(nch)]
        ref = int(np.argmax(sig))                                 # the channel with the most nebula
        sm_ref = ndi.gaussian_filter(small[..., ref], 1.5)
        top = float(np.percentile(sm_ref, 99.5)) - bg[ref]
        level = max(nebula_sigma * noise[ref] / 3, nebula_share * top)
        nebula = ndi.binary_dilation((sm_ref - bg[ref]) > level, iterations=3)
        rmin, rmax = max(1.5, min_radius / f), max_radius / f
        radii = []
        r = rmin
        while r <= rmax:
            radii.append(r); r *= 1.25
        q = max(1, int(rmax))            # sky: median over ~9x the largest disc, so dense discs do not lift it
        out = np.array(img, np.float32, copy=True)
        found = []
        yy, xx = np.mgrid[:small.shape[0], :small.shape[1]]
        for c in range(nch):
            ch = small[..., c].astype(np.float32)
            tiny = _cv2.resize(ch, (max(4, ch.shape[1] // q), max(4, ch.shape[0] // q)), interpolation=_cv2.INTER_AREA)
            sky = _cv2.resize(ndi.median_filter(tiny, size=7, mode="reflect"), (ch.shape[1], ch.shape[0]),
                              interpolation=_cv2.INTER_LINEAR)
            exc = np.where(nebula, 0, ch - sky).astype(np.float32)
            ceiling = np.clip(ndi.gaussian_filter(exc, 2.0), 0, None)   # the model never goes above the data
            model = np.zeros_like(exc)
            for _ in range(int(max_discs)):
                res = exc - model
                best = None
                for r in radii:
                    inner = _cv2.filter2D(res, -1, _disc_kernel(r), borderType=_cv2.BORDER_REFLECT)
                    outer = _cv2.filter2D(res, -1, _disc_kernel(r, ring=True), borderType=_cv2.BORDER_REFLECT)
                    snr = (inner - outer) / (noise[c] / np.sqrt(np.pi * r * r))
                    snr[nebula] = 0
                    k = int(np.argmax(snr))
                    if best is None or snr.flat[k] > best[0]:
                        best = (float(snr.flat[k]), r, k, float(inner.flat[k] - outer.flat[k]))
                s_, r, k, amp = best
                if s_ < threshold or amp <= 0:
                    break
                cy, cx = divmod(k, exc.shape[1])
                d = np.hypot(yy - cy, xx - cx)
                amp = float(np.median(res[(d < 0.8 * r) & ~nebula])) if ((d < 0.8 * r) & ~nebula).any() else 0.0
                if amp <= 0:
                    break
                disc = np.clip((r * 1.1 - d) / (r * 0.25), 0, 1).astype(np.float32)
                model = np.minimum(model + amp * disc * ~nebula, ceiling)
                found.append((c, int(cx * f), int(cy * f), int(round(r * f))))
            if model.any():
                full = _cv2.resize(ndi.gaussian_filter(model, 1.0), (w, h), interpolation=_cv2.INTER_CUBIC)
                out[..., c] -= amount * np.clip(full, 0, None)
        per = {c: sum(1 for d in found if d[0] == c) for c in range(nch)}
        ctx.log("halo clean: discs removed per channel " + ", ".join(f"{c}: {n}" for c, n in per.items()))
        ctx.layers["discs"] = found
        return np.clip(out, 0, None).astype(np.float32)
