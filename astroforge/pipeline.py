"""Runs a preset (ordered list of plugin steps) on an integrated linear image."""
import json
import time
from pathlib import Path

import numpy as np

from . import colorspace as _colorspace
from . import io, masks, plugins
from .stats import describe

ROOT = Path(__file__).resolve().parent.parent
PRESETS = ROOT / "presets"
CONFIG = ROOT / "config.json"


class Log:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, msg):
        print(f"[{time.time() - self.t0:6.1f}s] {msg}", flush=True)


# names given to presets in the GUI / `astroforge rename`: kept per user, outside the repository, so that
# renaming never touches the shipped preset files (git pull stays clean)
LABELS = Path.home() / ".astroforge" / "preset_labels.json"


def _labels():
    try:
        return json.loads(LABELS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_preset(name_or_path):
    """By file, id or the name given in the GUI ("label", see rename_preset)."""
    p = Path(name_or_path)
    if not p.exists():
        p = PRESETS / f"{name_or_path}.json"
    if not p.exists():
        want = str(name_or_path).strip().lower()
        hit = next((pid for pid, lab in _labels().items() if lab.lower() == want), None)
        if hit is None:
            for f in PRESETS.glob("*.json"):
                with open(f, encoding="utf-8") as fh:
                    if str(json.load(fh).get("label", "")).strip().lower() == want:
                        hit = f.stem
                        break
        if hit is not None:
            p = PRESETS / f"{hit}.json"
    with open(p, encoding="utf-8") as f:
        preset = json.load(f)
    label = _labels().get(p.stem)
    if label:
        preset["label"] = label
    return preset


def rename_preset(pid, label):
    """Give a preset a name of your choice. The id (file name) stays, so `--post nebula` and saved GUI
    sessions keep working; the new name is accepted everywhere a preset is asked for."""
    label = " ".join(str(label).split())[:60]
    if not label or not (PRESETS / f"{pid}.json").exists():
        raise ValueError("preset or name missing")
    for f in PRESETS.glob("*.json"):
        if f.stem != pid and (f.stem.lower() == label.lower() or
                              str(load_preset(f).get("label", "")).lower() == label.lower()):
            raise ValueError(f"name already used by preset {f.stem}")
    labels = _labels()
    labels[pid] = label
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
    return load_preset(pid)


# what the data is -> profile: the automatic choice of the GUI and of `stack --post auto`
AUTO_PRESETS = {"SHO": "sho_starless", "HOO": "hoo_starless", "MONO": "mono", "galaxy": "galaxy", "RGB": "rgb_starless"}


def suggest_preset(path=None, header=None, n_channels=3):
    """Profile for an integrated image: palette written by the stacker (PALETTE keyword) or in the file name
    (_SHO, _HOO), mono data, galaxies from the OBJECT keyword / file name (OpenNGC type), else broadband
    nebula. Returns (preset id, reason)."""
    import re
    from . import catalogs
    header = header or {}
    name = Path(path).stem if path else ""
    pal = str(header.get("PALETTE", "")).upper()
    if not pal:
        m = re.search(r"(?:^|[_\-\s])(SHO|HOO|HSO)(?:$|[_\-\s])", name, re.I)
        pal = m.group(1).upper().replace("HSO", "SHO") if m else ""
    if pal in ("SHO", "HOO"):
        return AUTO_PRESETS[pal], f"banda stretta {pal}"
    if n_channels == 1 or pal == "MONO":
        return AUTO_PRESETS["MONO"], "mono"
    cands = [str(header.get("OBJECT", ""))] + re.findall(r"(?:M|NGC|IC|Sh2-?)\s?\d+", name, re.I)
    for c in cands:
        o = catalogs.find(c) if c.strip() else None
        if o and catalogs.TYPE_CLASS.get(o["type"]) == "galaxy":
            return AUTO_PRESETS["galaxy"], f"galassia {o['id']}"
    return AUTO_PRESETS["RGB"], "nebulosa a colori (banda larga)"


def looks_stretched(img):
    """True for an image already stretched (a finished FITS/TIFF): linear integrations have the sky near zero and
    90% of the pixels under ~0.07 (Hawk stacks); finished ones have p90 >= 0.3 even with a black sky."""
    med, p90 = np.percentile(img[::8, ::8], [50, 90])
    return bool(med > 0.08 or p90 > 0.2)


def load_config():
    if CONFIG.exists():
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    return {}


def validate(preset):
    """Check plugin ids, parameters and linear/nonlinear order before running anything."""
    linear = True
    problems = []
    for i, step in enumerate(preset["steps"]):
        pl = plugins.get(step["plugin"])
        pl.resolve(step.get("params", {}))
        # a disabled step, or one applied to the star layer, does not change the main image
        if step.get("enabled", True) is False or step.get("target", "image") == "stars":
            continue
        if pl.stage == "linear" and not linear:
            problems.append(f"step {i} {pl.id} expects linear data but comes after a stretch")
        if pl.category == "stretch":
            linear = False
    return problems


def apply_step(img, step, ctx):
    """One step; with "mask" ({type: range|color|luminance|stars|lum|layer, ..., invert}) the result is blended
    through the mask like a PixInsight process applied to a masked image. With "target": "stars" the step works
    on the star layer left by star removal (StarNet, Dark Star, StarX), like processing the separate star image
    in PixInsight before putting it back with stars.recombine."""
    if step.get("target", "image") == "stars":
        stars = ctx.layers.get("stars")
        if stars is None:
            ctx.log(f"{step['plugin']}: no star layer yet (run a star removal step first), skipped")
            return img
        layer_step = {k: v for k, v in step.items() if k != "target"}
        linear = ctx.linear                       # a stretch of the star layer alone leaves the image linear
        ctx.layers["stars"] = apply_step(stars, layer_step, ctx)
        ctx.linear = linear
        return img
    pl = plugins.get(step["plugin"])
    spec = step.get("mask")
    # the mask is measured on the input, before the step changes ctx.linear (a masked stretch read the mask of
    # the linear image as if stretched: a mask of zeros, the stretch cancelled)
    m = masks.make_mask(img, spec, ctx)[..., None] if spec else None
    out = pl.run(img, ctx, **pl.scaled(pl.resolve(step.get("params", {})), ctx.scale))
    if spec:
        if out.shape != img.shape:
            raise ValueError(f"{pl.id}: a mask cannot be used with a step that changes the image size")
        out = (img * (1 - m) + out * m).astype(np.float32)
    return out


def run_steps(img, steps, ctx, on_step=None):
    """Apply plugin steps to an image in memory. on_step(i, n, plugin) is called before each step."""
    for i, step in enumerate(steps, 1):
        pl = plugins.get(step["plugin"])
        if step.get("enabled", True) is False:
            continue
        if on_step:
            on_step(i, len(steps), pl)
        img = apply_step(img, step, ctx)
    return img


def run(src, preset="nebula", out_dir="out", previews=True, tools=None, colorspace="srgb", metadata=None):
    """colorspace: 'srgb', 'display_p3' or 'adobe_rgb' for the .tif/.jpg (converted + profile embedded);
    the .fits keeps the unconverted data. metadata: FITS keywords for the output (OBSERVER, COPYRIGHT, ...)."""
    log = Log()
    img, header = io.load(src)
    log(f"loaded {img.shape[1]}x{img.shape[0]}x{img.shape[2]}  {describe(img)}")
    if isinstance(preset, str) and preset == "auto":
        preset, why = suggest_preset(src, header, img.shape[-1])
        log(f"automatic profile: {preset} ({why})")
    preset = load_preset(preset) if isinstance(preset, (str, Path)) else preset
    for p in validate(preset):
        raise ValueError(p)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    if looks_stretched(img):
        raise ValueError("image looks already stretched; presets expect linear integrated data")

    ctx = plugins.Context(log=log, header=header, out_dir=out_dir,
                          tools=tools if tools is not None else load_config().get("tools", {}))
    ctx.layers["stem"] = stem      # output name for plugins that write files (annotate.objects)
    for i, step in enumerate(preset["steps"], 1):
        if step.get("enabled", True) is False:
            continue
        pl = plugins.get(step["plugin"])
        log(f"#{i} {pl.name}" + (f" (mask: {step['mask'].get('type')})" if step.get("mask") else ""))
        img = apply_step(img, step, ctx)
        if previews and step.get("preview"):
            io.save_preview(img, out_dir / f"{stem}_{i:02d}_{pl.id}.jpg")

    log(f"final  {describe(img)}")
    base = out_dir / f"{stem}_{preset.get('name', 'astroforge')}"
    if previews:
        io.save_preview(img, f"{base}_preview.jpg")
    from . import version_string
    io.save_fits(img, f"{base}.fits", header, {"SWCREATE": f"Noctua {version_string()}", **(metadata or {})})
    cs = _colorspace.canonical(colorspace or "srgb")
    disp = _colorspace.convert(img, "srgb", cs)
    io.save_tiff(disp, f"{base}.tif", icc=cs)
    io.save_jpeg(disp, f"{base}.jpg", icc=cs)
    log(f"saved {base}.fits/.tif/.jpg" + ("" if cs == "srgb" else f" (tif/jpg in {cs})"))
    return img
