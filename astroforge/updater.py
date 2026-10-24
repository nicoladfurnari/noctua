"""Install and update AstroForge from its git checkout.

python -m astroforge.updater            update (run by start-gui.cmd before the GUI starts)
python -m astroforge.updater --deps     install the Python packages (CuPy only with an NVIDIA GPU)
python -m astroforge.updater --config   write / complete config.json with the external programs found on this PC
python -m astroforge.updater --find     list the external programs found (registry, PATH, folders, N.I.N.A.)

The update never blocks the start: no network, no GitHub login, local changes or another branch -> it says why
and AstroForge starts with the version it has. Only fast-forward pulls of main; the packages are reinstalled only
when requirements.txt changed. ASTROFORGE_NO_UPDATE=1 (or a file named no-auto-update next to start-gui.cmd)
turns the automatic update off.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCH = "main"


def _git(*args, timeout=60, root=ROOT):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")   # never wait for a password
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=timeout,
                          env=env, stdin=subprocess.DEVNULL, errors="replace")


def has_nvidia():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, timeout=20, stdin=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# CUDA libraries CuPy needs (what cupy-cuda12x[ctk] would download: cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE,
# NVRTC, runtime): an installed CUDA Toolkit 12 that has them all saves ~1.6 GB of pip packages
_CUDA_DLLS = ("cublas64_12", "cufft64_11", "curand64_10", "cusolver64_11", "cusparse64_12", "nvrtc64_12", "cudart64_12")


def system_cuda():
    """Folder of an installed CUDA Toolkit 12.x with every library CuPy needs, else None."""
    cands = [os.environ.get("CUDA_PATH", "")]
    cands += [v for k, v in os.environ.items() if k.upper().startswith("CUDA_PATH_V12")]
    base = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if base.exists():
        cands += [str(d) for d in sorted(base.glob("v12*"), reverse=True)]
    for c in cands:
        if not c:
            continue
        b = Path(c) / "bin"
        if not b.exists():
            continue
        names = [f.name.lower() for f in b.glob("*.dll")] + [f.name.lower() for f in (b / "x64").glob("*.dll")]
        if all(any(n.startswith(d) for n in names) for d in _CUDA_DLLS):
            return str(Path(c))
    return None


def requirements(root=ROOT, gpu=None, cuda=None):
    """Package lines of requirements.txt; the CuPy line only when there is an NVIDIA GPU, and without its [ctk]
    CUDA packages when a CUDA Toolkit 12 is already installed (cuda: folder, False = none, None = look)."""
    gpu = has_nvidia() if gpu is None else gpu
    out = []
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        pkg = line.split("#")[0].strip()
        if not pkg or (pkg.lower().startswith("cupy") and not gpu):
            continue
        if pkg.lower().startswith("cupy") and "[ctk]" in pkg:
            found = system_cuda() if cuda is None else cuda
            if found:
                pkg = pkg.replace("[ctk]", "")
        out.append(pkg)
    return out


def install_deps(root=ROOT, log=print, gpu=None):
    pkgs = requirements(root, gpu)
    if any(x.lower().startswith("cupy") for x in pkgs):
        cuda = system_cuda()
        log(f"CUDA già installato in {cuda}: lo uso (niente 1,6 GB di pacchetti CUDA)" if cuda
            else "CUDA Toolkit 12 non trovato: scarico i pacchetti CUDA per CuPy (~1,6 GB, una volta sola)")
    log("installo i pacchetti Python: " + ", ".join(p.split("[")[0] for p in pkgs))
    sys.stdout.flush()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(pkgs) + "\n")
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--timeout", "120",
                            "--retries", "10", "-r", f.name], stdin=subprocess.DEVNULL)
    finally:
        os.unlink(f.name)
    if r.returncode != 0:
        log("ATTENZIONE: installazione dei pacchetti non riuscita (vedi sopra)")
    return r.returncode == 0


# where the free external programs usually are; config.json gets the ones that exist
TOOL_GUESSES = {
    "starnet": ["C:/Programs/StarNet2/starnet2.exe", "C:/Program Files/StarNet2/starnet2.exe"],
    "deepsnr": ["C:/Programs/DeepSNR/deepsnr.exe", "C:/Program Files/DeepSNR/deepsnr.exe"],
    "graxpert": ["C:/Programs/GraXpert/GraXpert.exe", "C:/Program Files/GraXpert/GraXpert.exe",
                 str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/GraXpert/GraXpert.exe")],
    "cosmic_clarity": ["C:/Programs/CosmicClarity/setiastrocosmicclarity.exe"],
    "cosmic_clarity_denoise": ["C:/Programs/CosmicClarity/setiastrocosmicclarity_denoise.exe"],
    "astap": ["C:/Programs/astap/astap_cli.exe", "C:/Program Files/astap/astap_cli.exe", "C:/Program Files/astap/astap.exe"],
    "rc-astro": ["C:/Program Files/RC-Astro/rc-astro.exe"],
}


# executable names (first = preferred) and words of their folder / installed program name
TOOL_EXES = {
    "starnet": (["starnet2.exe", "starnet++.exe"], ["starnet"]),
    "deepsnr": (["deepsnr.exe"], ["deepsnr"]),
    "graxpert": (["graxpert.exe"], ["graxpert"]),
    "cosmic_clarity": (["setiastrocosmicclarity.exe"], ["cosmic", "setiastro"]),
    "cosmic_clarity_denoise": (["setiastrocosmicclarity_denoise.exe"], ["cosmic", "setiastro"]),
    "astap": (["astap_cli.exe", "astap.exe"], ["astap"]),
    "rc-astro": (["rc-astro.exe"], ["rc-astro", "rc astro", "rcastro"]),
}


def _roots():
    env = os.environ
    roots = ["C:/Programs", env.get("ProgramFiles", "C:/Program Files"), env.get("ProgramFiles(x86)", ""),
             str(Path(env.get("LOCALAPPDATA", "")) / "Programs"), env.get("LOCALAPPDATA", ""),
             str(Path.home() / "Desktop"), str(Path.home() / "Documents"), "D:/Programs", "D:/Program Files"]
    return [Path(r) for r in dict.fromkeys(roots) if r and Path(r).exists()]


def _find_in(folder, names, depth=3):
    """First of `names` (in order of preference) under folder, depth-limited."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    level = [folder]
    hits = {}
    for _ in range(depth + 1):
        nxt = []
        for d in level:
            try:
                for e in os.scandir(d):
                    if e.is_file() and e.name.lower() in names and e.name.lower() not in hits:
                        hits[e.name.lower()] = e.path
                    elif e.is_dir(follow_symlinks=False) and not e.name.startswith((".", "$")):
                        nxt.append(e.path)
            except OSError:
                continue
        for n in names:                 # the preferred name wins as soon as it is found
            if n in hits:
                return str(Path(hits[n]))
        level = nxt
    return None


def _registry_dirs():
    """InstallLocation (or the folder of DisplayIcon / UninstallString) of the installed programs, with their names."""
    try:
        import winreg
    except ImportError:
        return []
    out = []
    keys = [(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")]
    for hive, path in keys:
        try:
            k = winreg.OpenKey(hive, path)
        except OSError:
            continue
        for i in range(winreg.QueryInfoKey(k)[0]):
            try:
                sk = winreg.OpenKey(k, winreg.EnumKey(k, i))
            except OSError:
                continue

            def val(name):
                try:
                    return str(winreg.QueryValueEx(sk, name)[0])
                except OSError:
                    return ""
            name = val("DisplayName")
            loc = val("InstallLocation")
            if not loc:
                for v in (val("DisplayIcon"), val("UninstallString")):
                    v = v.split(",")[0].strip().strip('"')
                    if v and Path(v).parent.exists():
                        loc = str(Path(v).parent)
                        break
            if name and loc:
                out.append((name, loc.strip('"')))
    return out


def _nina_astap():
    """ASTAP path set in the N.I.N.A. profiles (%LOCALAPPDATA%/NINA/Profiles/*.profile)."""
    prof = Path(os.environ.get("LOCALAPPDATA", "")) / "NINA" / "Profiles"
    for f in sorted(prof.glob("*.profile"), key=lambda x: -x.stat().st_mtime) if prof.exists() else []:
        try:
            m = re.search(r"<[^>]*ASTAPLocation>([^<]+)<", f.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if m and Path(m.group(1).strip()).exists():
            exe = Path(m.group(1).strip())
            cli = exe.with_name("astap_cli.exe")
            return str(cli if cli.exists() else exe)
    return None


def find_tools(log=None):
    """{tool: (path, where)} of the external programs on this PC: known paths, PATH, the installed programs
    (registry), N.I.N.A. (ASTAP), then a depth-limited look in the usual folders."""
    found = {}
    reg = _registry_dirs()
    for k, (exes, words) in TOOL_EXES.items():
        hit = next(((c, "percorso abituale") for c in TOOL_GUESSES.get(k, []) if Path(c).exists()), None)
        if not hit:
            w = next((shutil.which(e) for e in exes if shutil.which(e)), None)
            hit = (str(Path(w)), "PATH") if w else None
        if not hit:
            for name, loc in reg:
                if any(x in name.lower() for x in words):
                    f = _find_in(loc, exes, 2)
                    if f:
                        hit = (f, f"programmi installati ({name})")
                        break
        if not hit and k == "astap":
            n = _nina_astap()
            hit = (n, "profilo di N.I.N.A.") if n else None
        if not hit:
            for root in _roots():
                try:
                    subs = [e.path for e in os.scandir(root) if e.is_dir() and any(x in e.name.lower() for x in words)]
                except OSError:
                    continue
                f = next((x for x in (_find_in(d, exes, 3) for d in subs) if x), None)
                if f:
                    hit = (f, f"cartella {root}")
                    break
        if hit:
            found[k] = hit
    return found


def write_config(root=ROOT, log=print):
    """config.json with the programs found. An existing file keeps every path that exists; only the missing or
    wrong ones are filled in with what is found now."""
    cfg = root / "config.json"
    data = {}
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    tools = dict(data.get("tools") or {})
    found = find_tools()
    changed = False
    for k, cands in TOOL_GUESSES.items():
        cur = tools.get(k)
        if cur and Path(cur).exists():
            log(f"  {k:24s} già impostato: {cur}")
            continue
        if k in found:
            tools[k] = found[k][0]
            changed = True
            log(f"  {k:24s} trovato: {found[k][0]} ({found[k][1]})")
        else:
            if not cur:
                tools[k] = cands[0]
                changed = True
            log(f"  {k:24s} non trovato (si imposta dalla GUI, Programmi esterni)")
    if changed or not cfg.exists():
        data["tools"] = tools
        cfg.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update(root=ROOT, log=print, deps=True):
    """Fast-forward to origin/main when possible. Returns 'updated', 'current' or the reason it was skipped."""
    if os.environ.get("ASTROFORGE_NO_UPDATE") == "1" or (root / "no-auto-update").exists():
        log("aggiornamento automatico disattivato")
        return "disabled"
    if not (root / ".git").exists() or shutil.which("git") is None:
        log("aggiornamento saltato: non è un'installazione da git (o git manca)")
        return "no-git"
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD", root=root).stdout.strip()
        if branch != BRANCH:
            log(f"aggiornamento saltato: sei sul ramo '{branch}', non '{BRANCH}'")
            return "branch"
        if _git("status", "--porcelain", "--untracked-files=no", root=root).stdout.strip():
            log("aggiornamento saltato: ci sono modifiche locali non salvate (git status)")
            return "dirty"
        f = _git("fetch", "--quiet", "origin", BRANCH, timeout=45, root=root)
        if f.returncode != 0:
            log("aggiornamento saltato: GitHub non raggiungibile o login mancante")
            return "offline"
        behind = int(_git("rev-list", "--count", f"HEAD..origin/{BRANCH}", root=root).stdout.strip() or 0)
        if not behind:
            log("Noctua è aggiornato")
            return "current"
        old = _git("rev-parse", "HEAD", root=root).stdout.strip()
        old_req = _git("show", "HEAD:requirements.txt", root=root).stdout
        m = _git("merge", "--ff-only", "--quiet", f"origin/{BRANCH}", root=root)
        if m.returncode != 0:
            log("aggiornamento saltato: la tua copia ha commit propri (serve un merge a mano)")
            return "diverged"
        news = _git("log", "--oneline", "--no-decorate", f"{old}..HEAD", root=root).stdout.strip().splitlines()
        log(f"Noctua aggiornato ({len(news)} novità):")
        for line in news[:10]:
            log("  " + line)
        if deps and (root / "requirements.txt").read_text(encoding="utf-8") != old_req:
            install_deps(root, log)
        return "updated"
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        log(f"aggiornamento saltato: {e}")
        return "error"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="astroforge.updater")
    ap.add_argument("--deps", action="store_true", help="install the Python packages")
    ap.add_argument("--config", action="store_true", help="write config.json with the programs found")
    ap.add_argument("--find", action="store_true", help="list the external programs found on this PC")
    a = ap.parse_args(argv)
    if a.find:
        for k, (path, where) in find_tools().items():
            print(f"{k:24s} {path}  ({where})")
        print("CUDA Toolkit 12:", system_cuda() or "non trovato")
        return 0
    if a.deps or a.config:
        ok = install_deps() if a.deps else True
        if a.config:
            write_config()
        return 0 if ok else 1
    update()
    shortcut_icon()
    return 0                      # never stop the GUI from starting


def shortcut_icon(root=ROOT):
    """Give the desktop shortcut made by the installer (Noctua.lnk) the owl icon: shortcuts made before the icon
    existed show the generic .cmd one. Windows only; silent on any error."""
    ico = Path(root) / "astroforge" / "gui" / "static" / "icons" / "noctua.ico"
    if os.name != "nt" or not ico.exists():
        return False
    # the Desktop as Windows knows it (OneDrive moves it): the installer made the shortcut there; the paths go
    # through the environment, an apostrophe in a user name broke a quoted command
    ps = ("$l=Join-Path ([Environment]::GetFolderPath('Desktop')) 'Noctua.lnk'; if (-not (Test-Path $l)) { exit };"
          "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($l);"
          "if ($s.IconLocation -notlike '*noctua.ico*') { $s.IconLocation=\"$env:AF_ICO,0\"; $s.Save(); 'set' }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20,
                           stdin=subprocess.DEVNULL, env=dict(os.environ, AF_ICO=str(ico)))
        return "set" in r.stdout
    except (OSError, subprocess.SubprocessError):
        return False


if __name__ == "__main__":
    sys.exit(main())
