"""Command line.

python -m astroforge run <image.fit> [--preset nebula] [-o out_dir]
python -m astroforge stack <lights...> -c <calibration...> [-d 2] [--filters SHO] [--mosaic] [--dualband] ...
python -m astroforge plugins            list plugins and their parameters
python -m astroforge presets            list presets
"""
import argparse
import sys

if len(sys.argv) > 1 and sys.argv[1] == "gui" and "--update" in sys.argv:
    # "Riavvia server": update first, before pipeline / plugins / the version are imported or cached, so the
    # new process runs the new code everywhere (not new server.py on old plugins, asking for another restart)
    from . import updater
    updater.main([])
    sys.argv.remove("--update")

from . import pipeline, plugins


def main():
    ap = argparse.ArgumentParser(prog="astroforge")
    from . import version_string
    ap.add_argument("--version", action="version", version=f"Noctua {version_string()}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="process an integrated linear image with a preset")
    run.add_argument("image")
    run.add_argument("-p", "--preset", default="auto", help="profile id or name; auto = chosen from the data")
    run.add_argument("-o", "--out", default="out")
    run.add_argument("--no-previews", action="store_true")
    run.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                     help="FITS keyword for the output, e.g. --meta OBSERVER=Nicola --meta COPYRIGHT='All rights reserved'")
    run.add_argument("--colorspace", default="srgb", help="srgb, display_p3, adobe_rgb")
    sub.add_parser("plugins", help="list plugins")
    sub.add_parser("presets", help="list presets")
    lk = sub.add_parser("look", help="measure the look of finished images; --save makes a profile from it")
    lk.add_argument("images", nargs="+")
    lk.add_argument("--save", metavar="NAME", help="new profile with this name: steps of --base + this look")
    lk.add_argument("--base", default="nebula_pro", help="profile whose steps the new one uses")
    lk.add_argument("--compare", action="store_true", help="distance of every image from the first one")
    mk = sub.add_parser("masters", help="build master bias/dark/flat from folders of raw calibration frames")
    mk.add_argument("folders", nargs="+", help="folders with raw bias/dark/flat/dark flat (sub folders included)")
    mk.add_argument("-o", "--out", required=True, help="output folder of the masters")
    mk.add_argument("--rejection", default="auto", help="auto, winsorized, linear_fit, esd, sigma, percentile")
    mk.add_argument("--low", type=float, default=4.0, help="low rejection sigma")
    mk.add_argument("--high", type=float, default=3.0, help="high rejection sigma")
    mk.add_argument("--superbias", action="store_true", help="noise-free bias (columns, rows, large scale)")
    mk.add_argument("--no-defects", action="store_true", help="no defect map")
    mk.add_argument("--no-flat-per-night", action="store_true", help="one master flat per filter, all nights together")
    mk.add_argument("--min-frames", type=int, default=3)
    mk.add_argument("--format", choices=["xisf", "fits"], default="xisf")
    mk.add_argument("--no-library", action="store_true", help="do not add them to the master library")
    rn = sub.add_parser("rename", help="give a profile a name of your choice (the id stays)")
    rn.add_argument("preset")
    rn.add_argument("label")
    st = sub.add_parser("stack", help="calibrate, register, integrate (and drizzle) raw frames")
    st.add_argument("lights", nargs="+", help="folders or files with light frames")
    st.add_argument("-c", "--calibration", nargs="*", default=[], help="folders with bias/dark/flat")
    st.add_argument("-w", "--work", default="work")
    st.add_argument("-d", "--drizzle", type=int, default=1, help="drizzle scale: 1 (off), 2, 3")
    st.add_argument("--pixfrac", type=float, default=0.9)
    st.add_argument("--rejection", default="auto")
    st.add_argument("--post", default=None, help="post-processing preset to run on the result")
    st.add_argument("--filters", nargs="?", const="auto", default=None,
                    help="mono camera: stack every filter and combine (auto, RGB, LRGB, HOO, SHO, HaRGB)")
    st.add_argument("--mosaic", action="store_true", help="each 'lights' folder is one mosaic panel")
    st.add_argument("--debayer", default="rcd", help="rcd, bilinear, ea")
    st.add_argument("--local-norm", action="store_true", help="local normalization (changing gradients)")
    st.add_argument("--dualband", action="store_true", help="OSC + dual-band filter: Ha/OIII and HOO")
    st.add_argument("--no-dark-optimize", action="store_true")
    st.add_argument("--no-distortion", action="store_true")
    st.add_argument("--satellites", action="store_true", help="detect and reject satellite/plane trails")
    st.add_argument("--weighting", default="psf",
                    help="psf, standard, nebula, galaxy, cluster or a SubframeSelector expression")
    st.add_argument("--approval", type=float, default=None, help="reject frames whose weighting is below (e.g. 70)")
    st.add_argument("--exposure-tolerance", type=float, default=0,
                    help="seconds: integrate lights of different exposures together (WBPP)")
    st.add_argument("--dark-optimize", choices=["auto", "always", "never"], default="auto")
    st.add_argument("--comet", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                    help="comet nucleus in the first and last frame (reference-frame pixels)")
    gui = sub.add_parser("gui", help="start the web interface")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--host", default="127.0.0.1",
                     help="127.0.0.1 = solo questo PC; 0.0.0.0 = rete (serve il token di accesso)")
    gui.add_argument("--token", default=None,
                     help="token di accesso in modalità rete (default: env ASTROFORGE_TOKEN, altrimenti casuale)")
    gui.add_argument("--update", action="store_true",
                     help="prima controlla e scarica gli aggiornamenti (come start-gui.cmd; usato dal riavvio)")
    args = ap.parse_args()

    if args.cmd == "stack":
        from .stack.project import Runner
        settings = {"lights": args.lights, "calibration": args.calibration, "work_dir": args.work,
                    "drizzle": args.drizzle, "pixfrac": args.pixfrac, "rejection": args.rejection,
                    "debayer": args.debayer, "local_norm": args.local_norm, "dualband": args.dualband,
                    "dark_optimize": False if args.no_dark_optimize else {"auto": "auto", "always": True, "never": False}[args.dark_optimize],
                    "distortion": not args.no_distortion, "satellites": True if args.satellites else "auto",
                    "weighting": args.weighting, "approval": args.approval, "exposure_tolerance": args.exposure_tolerance}
        if args.comet:
            settings.update(comet=True, comet_first_xy=args.comet[:2], comet_last_xy=args.comet[2:])
        from . import gpu
        print(f"compute: {gpu.describe()}", flush=True)
        runner = Runner(settings, log=lambda m: print(m, flush=True))
        if args.mosaic:
            _, report = runner.run_mosaic([[p] for p in args.lights])
        else:
            _, report = runner.run_filters(args.filters) if args.filters else runner.run()
        if args.post:
            out = report["output"]
            src = report.get("combined") or (f"{out}_drizzle{args.drizzle}x.fits" if args.drizzle > 1
                                             else f"{out}_integration.fits")
            pipeline.run(src, args.post, args.work)
    elif args.cmd == "gui":
        if args.update:
            from . import updater
            updater.main([])
        from .gui.server import serve
        serve(args.host, args.port, args.token)
    elif args.cmd == "run":
        meta = dict(m.split("=", 1) for m in args.meta if "=" in m)
        pipeline.run(args.image, args.preset, args.out, previews=not args.no_previews,
                     colorspace=args.colorspace, metadata=meta)
    elif args.cmd == "plugins":
        for pid, pl in sorted(plugins.REGISTRY.items(), key=lambda kv: (kv[1].category, kv[0])):
            print(f"{pid:30s} [{pl.category}/{pl.stage}] {pl.name}")
            for k, v in pl.params.items():
                rng = f" [{v.min}..{v.max}]" if v.min is not None else ""
                print(f"    {k} = {v.default!r}{rng}  {v.help}")
    elif args.cmd == "presets":
        for f in sorted(pipeline.PRESETS.glob("*.json")):
            p = pipeline.load_preset(f)
            name = f"{f.stem} ({p['label']})" if p.get("label") else f.stem
            print(f"{name:28s} {p.get('description', '')}")
    elif args.cmd == "masters":
        from .stack.makemasters import make_masters
        make_masters(args.folders, args.out, {
            "rejection": args.rejection, "low": args.low, "high": args.high, "superbias": args.superbias,
            "defect_map": not args.no_defects, "flat_per_night": not args.no_flat_per_night,
            "min_frames": args.min_frames, "format": args.format, "library": not args.no_library},
            log=lambda m: print(m, flush=True))
    elif args.cmd == "rename":
        p = pipeline.rename_preset(args.preset, args.label)
        print(f"{args.preset}: ora si chiama \"{p['label']}\"")
    elif args.cmd == "look":
        import json
        from . import io, look
        looks = [look.measure(io.load_display(f)) for f in args.images]
        for f, m in zip(args.images, looks):
            extra = f"  distanza dal primo {look.distance(m, looks[0])['total']}" if args.compare else ""
            print(f"{f}: cielo {m['sky']} {m['sky_rgb']}  nebulosa {m['nebula']}  saturazione {m['saturation']}"
                  f"  bianco {m['white']}{extra}")
        if args.save:
            base = pipeline.load_preset(args.base)
            avg = {k: sum(m[k] for m in looks) / len(looks) for k in look.DEFAULT}
            steps = [s for s in base["steps"] if s["plugin"] != "look.match"]
            steps.append({"plugin": "look.match", "params": look.target_from(avg)})
            pid = "".join(c for c in args.save.lower().replace(" ", "_") if c.isalnum() or c in "-_")
            preset = {"name": pid, "label": args.save, "description": f"look di {len(looks)} immagini su base {args.base}",
                      "steps": steps}
            (pipeline.PRESETS / f"{pid}.json").write_text(json.dumps(preset, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"profilo {pid} salvato")


if __name__ == "__main__":
    main()
