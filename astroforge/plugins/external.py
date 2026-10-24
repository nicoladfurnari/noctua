"""Wrappers around external programs: GraXpert, StarNet, Cosmic Clarity, RC-Astro CLI.

Tool paths come from ctx.tools (config.json "tools" section). A missing tool is skipped
with a log line, so presets keep working on machines without it.

Improvements over calling the tools by hand:
- linear images are handled: StarNet gets a reversible stretch and the result is
  brought back to linear, so star removal can happen before the stretch;
- the star layer is kept for 'stars.recombine' (screen blend, unscreen to extract it);
- outputs are always checked, never assumed.
"""
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np

from .. import io, stretch
from .base import Param, Plugin, register


def _tool(ctx, name):
    path = ctx.tools.get(name)
    if path and Path(path).exists():
        return Path(path)
    found = shutil.which(name)
    return Path(found) if found else None


def _run(cmd, cwd=None, log=print, timeout=1800):
    log("  $ " + " ".join(str(c) for c in cmd))
    try:
        res = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL, errors="replace")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{Path(cmd[0]).name} did not finish within {timeout} s")
    if res.returncode != 0:
        tail = (res.stdout + res.stderr).strip().splitlines()[-5:]
        raise RuntimeError(f"{Path(cmd[0]).name} exited with {res.returncode}: " + " | ".join(tail))
    return res


def _read_any(path):
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(path)
        a = a.astype(np.float32) / (65535 if a.dtype == np.uint16 else 255 if a.dtype == np.uint8 else 1)
        return a if a.ndim == 3 else a[..., None]
    return io.load(path)[0]


def _pad_min(img, size=512):
    """StarNet2/DeepSNR refuse images smaller than their 512 px window: mirror-pad, crop back after."""
    h, w = img.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if not (ph or pw):
        return img, lambda a: a
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="symmetric"), lambda a: a[:h, :w]


def _rescale_like(out, ref, n=200_000, seed=0):
    """Undo a tool's own normalisation (GraXpert denoise writes (x - min) / (max - min)): per channel linear
    fit out -> ref on a random pixel sample. Denoising keeps the local mean, so the fit is the inverse map."""
    rng = np.random.default_rng(seed)
    o, r = out.reshape(-1, out.shape[-1]), ref.reshape(-1, ref.shape[-1])
    idx = rng.integers(0, len(o), min(n, len(o)))
    res = np.empty_like(out, dtype=np.float32)
    for c in range(out.shape[-1]):
        x, y = o[idx, c].astype(np.float64), r[idx, c].astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y) & (y < 0.98)
        a, b = np.polyfit(x[ok], y[ok], 1) if ok.sum() > 100 else (1.0, 0.0)
        res[..., c] = out[..., c] * a + b
    return res


def _unscreen(img, starless):
    return np.clip(1 - (1 - img) / np.maximum(1 - starless, 1e-6), 0, 1).astype(np.float32)


@register
class GraXpertBackground(Plugin):
    id = "ext.graxpert_background"
    name = "GraXpert AI background extraction"
    category = "background"
    stage = "linear"
    description = "Gradient removal with GraXpert's AI model (free, open source)."
    params = {"smoothing": Param(0.1, 0.0, 1.0, "background model smoothing"),
              "correction": Param("Subtraction", choices=["Subtraction", "Division"]),
              "gpu": Param(True)}

    def run(self, img, ctx, smoothing, correction, gpu):
        exe = _tool(ctx, "graxpert")
        if exe is None:
            ctx.log("GraXpert not configured, skipped")
            return img
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.fits"
            io.save_fits(img, src)
            _run([exe, src, "-cli", "-cmd", "background-extraction", "-output", Path(td) / "out",
                  "-correction", correction, "-smoothing", smoothing, "-gpu", str(gpu).lower()], log=ctx.log)
            outs = list(Path(td).glob("out*.fits")) + list(Path(td).glob("in_GraXpert*.fits"))
            if not outs:
                raise RuntimeError("GraXpert produced no output file")
            return np.clip(io.load(outs[0])[0], 0, 1)


@register
class GraXpertDenoise(Plugin):
    id = "ext.graxpert_denoise"
    name = "GraXpert AI denoise"
    category = "noise"
    description = "AI noise reduction from GraXpert (free). Works best on linear data."
    params = {"strength": Param(0.5, 0.0, 1.0), "batch_size": Param(4, 1, 32), "gpu": Param(True)}

    def run(self, img, ctx, strength, batch_size, gpu):
        exe = _tool(ctx, "graxpert")
        if exe is None:
            ctx.log("GraXpert not configured, skipped")
            return img
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.fits"
            io.save_fits(img, src)
            _run([exe, src, "-cli", "-cmd", "denoising", "-output", Path(td) / "out",
                  "-strength", strength, "-batch_size", batch_size, "-gpu", str(gpu).lower()], log=ctx.log)
            outs = list(Path(td).glob("out*.fits")) + list(Path(td).glob("in_GraXpert*.fits"))
            if not outs:
                raise RuntimeError("GraXpert produced no output file")
            # GraXpert rescales to its own min..max: back to the input scale, or the sky ends at 0
            return np.clip(_rescale_like(io.load(outs[0])[0], img), 0, 1).astype(np.float32)


@register
class StarNet(Plugin):
    id = "stars.starnet"
    name = "StarNet star removal"
    category = "stars"
    description = ("AI star removal with StarNet (free). Works on linear images too: they are stretched "
                   "reversibly, processed, and brought back. Stars are kept for 'stars.recombine'.")
    params = {"stride": Param(256, 64, 512, "tile stride; smaller = slower, fewer tile seams"),
              "upsample": Param(False, help="2x intermediate upsampling (StarNet2 2.6+), for small stars")}

    def run(self, img, ctx, stride, upsample):
        exe = _tool(ctx, "starnet")
        if exe is None:
            ctx.log("StarNet not configured, skipped")
            return img
        if exe.stem.lower() == "starnet2":
            return self._run_v26(exe, img, ctx, stride, upsample)
        work = img
        if ctx.linear:
            c0, m = stretch.auto_params(img, 0.25, -2.8)
            work = stretch.histogram_transform(img, c0, m)
        if work.shape[-1] == 1:
            work = np.repeat(work, 3, axis=-1)
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "in.tif", Path(td) / "out.tif"
            io.save_tiff(work, src)
            # StarNet CLI builds differ: positional (v2.0) or -i/-o flags (newer); try both
            try:
                _run([exe, src, dst, stride], cwd=exe.parent, log=ctx.log)
            except RuntimeError:
                pass
            if not dst.exists():
                _run([exe, "-i", src, "-o", dst, "-s", stride], cwd=exe.parent, log=ctx.log)
            if not dst.exists():
                raise RuntimeError("StarNet produced no output file")
            starless = _read_any(dst)[..., :img.shape[-1]]
        if ctx.linear:
            # invert the midtones transfer (MTF is its own inverse with 1-m), then undo the clip
            lin = stretch.mtf(1 - m, starless) * (1 - c0) + c0
            ctx.layers["stars"] = np.clip(img - lin, 0, 1).astype(np.float32)
            ctx.layers["stars_mode"] = "add"
            return np.clip(lin, 0, 1).astype(np.float32)
        ctx.layers["stars"] = _unscreen(img, starless)
        ctx.layers["stars_mode"] = "screen"
        return starless

    def _run_v26(self, exe, img, ctx, stride, upsample):
        # StarNet2 2.6+: float FITS in/out (no 16-bit round trip) and its own reversible linear MTF
        # its own star layer too (-n, "unscreen": original = starless + U (1 - starless)), computed against the
        # source it actually processed (highlights clipped, shadows restored): ours is only the fallback
        with tempfile.TemporaryDirectory() as td:
            src, dst, stl = Path(td) / "in.fits", Path(td) / "out.fits", Path(td) / "stars.fits"
            padded, unpad = _pad_min(np.clip(img, 0, 1))
            io.save_fits(padded, src)
            cmd = [exe, "-i", src, "-o", dst, "-n", stl, "-s", stride - stride % 2, "-q"]
            if ctx.linear:
                cmd.append("--linear")
            if upsample:
                cmd.append("-u")
            _run(cmd, cwd=exe.parent, log=ctx.log)
            if not dst.exists():
                raise RuntimeError("StarNet produced no output file")
            starless = np.clip(unpad(io.load(dst)[0])[..., :img.shape[-1]], 0, 1).astype(np.float32)
            u = np.clip(unpad(io.load(stl)[0])[..., :img.shape[-1]], 0, 1).astype(np.float32) if stl.exists() else None
        if u is not None and u.shape != starless.shape:
            u = None
        if ctx.linear:
            # linear: additive layer (stars stretched on their own later), U (1 - starless) = source - starless
            ctx.layers["stars"] = (u * (1 - starless) if u is not None else np.clip(img - starless, 0, 1)).astype(np.float32)
            ctx.layers["stars_mode"] = "add"
        else:
            ctx.layers["stars"] = u if u is not None else _unscreen(img, starless)
            ctx.layers["stars_mode"] = "screen"
        ctx.log("StarNet: star layer " + ("from StarNet (unscreen)" if u is not None else "computed (no unscreen output)"))
        return starless


@register
class DeepSNR(Plugin):
    id = "ext.deepsnr"
    name = "DeepSNR AI denoise"
    category = "noise"
    description = ("AI noise reduction from the StarNet author (free). Linear or stretched, colour or mono. "
                   "Best on well-calibrated data without walking noise.")
    params = {"amount": Param(1.0, 0.0, 1.0, "blend with the original (1 = full denoise)"),
              "model": Param(2, 1, 2, "1 = older RGB-only model, 2 = current"),
              "stride": Param(256, 64, 512)}

    def run(self, img, ctx, amount, model, stride):
        exe = _tool(ctx, "deepsnr")
        if exe is None:
            ctx.log("DeepSNR not configured, skipped")
            return img
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "in.fits", Path(td) / "out.fits"
            padded, unpad = _pad_min(np.clip(img, 0, 1))
            io.save_fits(padded, src)
            cmd = [exe, "-i", src, "-o", dst, "-m", model, "-s", stride - stride % 2, "-q"]
            if ctx.linear:
                cmd.append("--linear")
            _run(cmd, cwd=exe.parent, log=ctx.log)
            if not dst.exists():
                raise RuntimeError("DeepSNR produced no output file")
            out = np.clip(unpad(io.load(dst)[0])[..., :img.shape[-1]], 0, 1)
        return (img + amount * (out - img)).astype(np.float32)


@register
class CosmicClarity(Plugin):
    id = "ext.cosmic_clarity"
    name = "Cosmic Clarity (Seti Astro)"
    category = "sharpen"
    stage = "nonlinear"
    description = ("AI sharpening/denoise from Seti Astro (free). Uses the program's input/output folders. "
                   "Experimental: argument names may change between releases.")
    params = {"mode": Param("sharpen", choices=["sharpen", "denoise"]),
              "stellar_amount": Param(0.5, 0.0, 1.0), "nonstellar_amount": Param(0.5, 0.0, 1.0),
              "nonstellar_strength": Param(3.0, 1.0, 8.0, "non-stellar PSF size in px (ignored with auto_psf)"),
              "auto_psf": Param(True, help="measure the PSF per chunk and pick the nearest models"),
              "denoise_strength": Param(0.5, 0.0, 1.0), "gpu": Param(True)}

    def run(self, img, ctx, mode, stellar_amount, nonstellar_amount, nonstellar_strength, auto_psf,
            denoise_strength, gpu):
        exe = _tool(ctx, "cosmic_clarity" if mode == "sharpen" else "cosmic_clarity_denoise")
        if exe is None:
            ctx.log(f"Cosmic Clarity ({mode}) not configured, skipped")
            return img
        # the program only reads <exe>/input and writes <exe>/output: use a unique name so parallel runs
        # (GUI + CLI) never pick up each other's files, and remove only our own files
        inp, outp = exe.parent / "input", exe.parent / "output"
        inp.mkdir(exist_ok=True)
        outp.mkdir(exist_ok=True)
        if mode == "sharpen":
            # every option must be given: a missing one makes the program wait for input forever
            args = ["--sharpening_mode", "Both", "--stellar_amount", stellar_amount,
                    "--nonstellar_amount", nonstellar_amount, "--nonstellar_strength", nonstellar_strength]
            if auto_psf:
                args.append("--auto_detect_psf")
        else:
            args = ["--denoise_strength", denoise_strength, "--denoise_mode", "full"]
        if not gpu:
            args.append("--disable_gpu")
        # Cosmic Clarity 'sharpen' crashes on some image sizes (its last chunk ends up 0 px wide when stitching):
        # mirror-pad to another size and crop back, trying a few sizes
        h, w = img.shape[:2]
        err = None
        for extra in (0, 256, 97):
            ph, pw = (-h) % 256 + extra if extra else 0, (-w) % 256 + extra if extra else 0
            work = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="symmetric") if (ph or pw) else img
            tag = f"astroforge_{uuid.uuid4().hex[:8]}"
            io.save_fits(np.clip(work, 0, 1), inp / f"{tag}.fits")   # 32-bit float in and out
            try:
                _run([exe, *args], cwd=exe.parent, log=ctx.log, timeout=900)
                outs = sorted(outp.glob(f"{tag}*"))
                if not outs:
                    raise RuntimeError("Cosmic Clarity produced no output file")
                return np.clip(_read_any(outs[0])[:h, :w, :img.shape[-1]], 0, 1)
            except RuntimeError as e:
                err = e
                if "broadcast" not in str(e):
                    raise
                ctx.log(f"Cosmic Clarity failed on {work.shape[1]}x{work.shape[0]}, retrying with padding")
            finally:
                for f in list(inp.glob(f"{tag}*")) + list(outp.glob(f"{tag}*")):
                    f.unlink(missing_ok=True)
        raise err


def _cc_tool(ctx, name):
    """Cosmic Clarity Suite programs: their own config key, else next to the configured sharpen exe."""
    exe = _tool(ctx, f"cosmic_clarity_{name}")
    if exe is None:
        base = _tool(ctx, "cosmic_clarity")
        cand = base.parent / f"setiastrocosmicclarity_{name}.exe" if base else None
        exe = cand if cand and cand.exists() else None
    return exe


@register
class CosmicDarkStar(Plugin):
    id = "stars.darkstar"
    name = "Dark Star star removal (Cosmic Clarity)"
    category = "stars"
    stage = "nonlinear"
    description = ("Free AI star removal from Seti Astro, alternative to StarNet. Stars are kept for "
                   "'stars.recombine'.")
    params = {"mode": Param("unscreen", choices=["unscreen", "additive"]), "gpu": Param(True)}

    def run(self, img, ctx, mode, gpu):
        exe = _cc_tool(ctx, "darkstar")
        if exe is None:
            ctx.log("Cosmic Clarity Dark Star not found, skipped")
            return img
        inp, outp = exe.parent / "input", exe.parent / "output"
        inp.mkdir(exist_ok=True)
        outp.mkdir(exist_ok=True)
        tag = f"astroforge_{uuid.uuid4().hex[:8]}"
        work = img if img.shape[-1] == 3 else np.repeat(img, 3, -1)
        io.save_fits(np.clip(work, 0, 1), inp / f"{tag}.fits")
        args = ["--star_removal_mode", mode, "--chunk_size", 512] + ([] if gpu else ["--disable_gpu"])
        try:
            _run([exe, *args], cwd=exe.parent, log=ctx.log, timeout=1800)
            outs = [f for f in sorted(outp.glob(f"{tag}*")) if "star" not in f.stem.lower() or "starless" in f.stem.lower()]
            if not outs:
                raise RuntimeError("Dark Star produced no output file")
            starless = np.clip(_read_any(outs[0])[..., :img.shape[-1]], 0, 1).astype(np.float32)
        finally:
            for f in list(inp.glob(f"{tag}*")) + list(outp.glob(f"{tag}*")):
                f.unlink(missing_ok=True)
        ctx.layers["stars"] = _unscreen(img, starless) if mode == "unscreen" else np.clip(img - starless, 0, 1)
        ctx.layers["stars_mode"] = "screen" if mode == "unscreen" else "add"
        return starless


@register
class CosmicSuperRes(Plugin):
    id = "ext.cc_superres"
    name = "Super-resolution (Cosmic Clarity)"
    category = "geometry"
    description = "AI upscaling 2x-4x from Seti Astro, e.g. before cropping a small galaxy."
    params = {"scale": Param(2, 2, 4, "upscale factor")}

    def run(self, img, ctx, scale):
        exe = _cc_tool(ctx, "superres")
        if exe is None:
            ctx.log("Cosmic Clarity super-resolution not found, skipped")
            return img
        with tempfile.TemporaryDirectory() as td:
            src, outdir = Path(td) / "in.fits", Path(td) / "out"
            outdir.mkdir()
            io.save_fits(np.clip(img, 0, 1), src)
            _run([exe, "--input", src, "--output_dir", outdir, "--scale", int(scale), "--model_dir", exe.parent],
                 cwd=exe.parent, log=ctx.log, timeout=3600)
            outs = sorted(outdir.iterdir())
            if not outs:
                raise RuntimeError("super-resolution produced no output file")
            out = _read_any(outs[0])[..., :img.shape[-1]]
        return np.clip(out, 0, 1).astype(np.float32)


@register
class CosmicSatellite(Plugin):
    id = "ext.cc_satellite"
    name = "Satellite trail removal (Cosmic Clarity)"
    category = "cosmetic"
    description = ("AI satellite/plane trail removal on a single image (the stacking already rejects trails; "
                   "this is for trails left in the final image).")
    params = {"mode": Param("full", choices=["full", "luminance"]),
              "sensitivity": Param(0.1, 0.01, 0.5, "trail clipping sensitivity"), "gpu": Param(True)}

    def run(self, img, ctx, mode, sensitivity, gpu):
        exe = _cc_tool(ctx, "satellite")
        if exe is None:
            ctx.log("Cosmic Clarity satellite removal not found, skipped")
            return img
        with tempfile.TemporaryDirectory() as td:
            indir, outdir = Path(td) / "in", Path(td) / "out"
            indir.mkdir(), outdir.mkdir()
            io.save_fits(np.clip(img, 0, 1), indir / "image.fits")
            # --no-clip-trail: the default sets the trail pixels to 0, leaving a black line in a final image
            args = ["--input", indir, "--output", outdir, "--mode", mode, "--batch", "--sensitivity", sensitivity,
                    "--no-clip-trail"]
            _run([exe, *args, *(["--use-gpu"] if gpu else [])], cwd=exe.parent, log=ctx.log, timeout=1800)
            outs = sorted(outdir.iterdir())
            if not outs:
                ctx.log("no satellite trail found")
                return img
            return np.clip(_read_any(outs[0])[..., :img.shape[-1]], 0, 1).astype(np.float32)


@register
class RCAstro(Plugin):
    id = "ext.rc_astro"
    name = "RC-Astro CLI (BlurX / NoiseX / StarX)"
    category = "sharpen"
    description = ("Paid RC-Astro tools through their stand-alone 'rc-astro' CLI (same licence as the PixInsight "
                   "versions). Syntax from rc-astro.com: 'rc-astro <bxt|nxt|sxt> [OPTIONS] input --output folder'. "
                   "'options' are passed as given, e.g. ['--sharpen-stars', '0.3'].")
    params = {"product": Param("bxt", choices=["bxt", "nxt", "sxt"]),
              "options": Param([], help="extra command line options for the product")}

    def run(self, img, ctx, product, options):
        exe = _tool(ctx, "rc-astro")
        if exe is None:
            ctx.log("rc-astro CLI not configured, skipped")
            return img
        if product == "bxt" and not ctx.linear:
            ctx.log("WARNING: BlurXTerminator is meant for linear data (before the stretch)")
        with tempfile.TemporaryDirectory() as td:
            src, outdir = Path(td) / "in.tif", Path(td) / "out"
            outdir.mkdir()
            import tifffile
            data = np.ascontiguousarray(np.clip(img, 0, 1), dtype=np.float32)
            tifffile.imwrite(src, data[..., 0] if data.shape[-1] == 1 else data,
                             photometric="minisblack" if data.shape[-1] == 1 else "rgb")
            _run([exe, product, *options, src, "--output", outdir, "--overwrite"], log=ctx.log, timeout=3600)
            outs = sorted(p for p in outdir.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".fits", ".fit", ".xisf"))
            if not outs:
                raise RuntimeError("rc-astro produced no output file")
            # StarXTerminator may also write a stars image: the starless one is the main output
            main = next((p for p in outs if "star" not in p.stem.lower() or "starless" in p.stem.lower()), outs[0])
            out = _read_any(main)[..., :img.shape[-1]]
            if out.shape[-1] < img.shape[-1]:
                out = np.repeat(out, img.shape[-1], axis=-1)
        if product == "sxt":
            ctx.layers["stars"] = (_unscreen(img, out) if not ctx.linear
                                   else np.clip(img - out, 0, 1).astype(np.float32))
            ctx.layers["stars_mode"] = "add" if ctx.linear else "screen"
        return np.clip(out, 0, 1).astype(np.float32)
