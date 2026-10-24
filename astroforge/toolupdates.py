"""New versions of StarNet2 and DeepSNR (command-line tools from starnetastro.com).

The installed version comes from the program itself (--machine-info, JSON). Their own --check-updates is used when
the build supports it (the Windows builds 2.6.2 / 1.3.2 answer "not available"); otherwise the current releases
are read from the starnetastro.com home page ("Current command-line releases are StarNet2 x and DeepSNR y"),
at most once a day (cache in ~/.astroforge/tool_updates.json). Nothing about the user is sent: a plain GET.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

SITE = "https://starnetastro.com/"
DOWNLOAD = "https://starnetastro.com/download/"
TOOLS = {"starnet": "StarNet2", "deepsnr": "DeepSNR"}
_RELEASES = re.compile(r"command-line releases are\s+StarNet2\s+([\d.]+)\s+and\s+DeepSNR\s+([\d.]+)", re.I)


def _cache_path():
    return Path(os.environ.get("ASTROFORGE_TOOL_UPDATES") or Path.home() / ".astroforge" / "tool_updates.json")


def _v(s):
    return tuple(int(x) for x in re.findall(r"\d+", str(s or ""))[:4])


def installed_version(exe):
    """Version reported by the program (--machine-info JSON), None when it cannot tell."""
    if not exe or not Path(exe).exists():
        return None
    try:
        r = subprocess.run([str(exe), "--machine-info"], capture_output=True, text=True, timeout=30,
                           cwd=str(Path(exe).parent), stdin=subprocess.DEVNULL, errors="replace")
        return json.loads(r.stdout).get("version")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def own_check(exe):
    """Output of the program's --check-updates when its build supports it, else None."""
    try:
        r = subprocess.run([str(exe), "--check-updates"], capture_output=True, text=True, timeout=30,
                           cwd=str(Path(exe).parent), stdin=subprocess.DEVNULL, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    out = (r.stdout + r.stderr).strip()
    return None if (not out or "not available" in out.lower()) else out


def latest_releases(fetch=None, timeout=10):
    """{"starnet": "2.6.2", "deepsnr": "1.3.2"} from the starnetastro.com home page, {} when unreachable."""
    if fetch is None:
        def fetch(url):
            import urllib.request
            try:
                import truststore          # the Windows certificate store, like the other HTTPS calls
                truststore.inject_into_ssl()
            except ImportError:
                pass
            req = urllib.request.Request(url, headers={"User-Agent": "AstroForge"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
    try:
        text = re.sub(r"<[^>]+>", " ", fetch(SITE))
    except Exception:
        return {}
    m = _RELEASES.search(re.sub(r"\s+", " ", text))
    return {"starnet": m.group(1), "deepsnr": m.group(2)} if m else {}


def check(tools, fetch=None, max_age=86400, force=False):
    """{tool: {name, installed, latest, update, message, url}} for StarNet2 and DeepSNR in `tools` (config)."""
    cache = _cache_path()
    cached = {}
    try:
        cached = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if not force and time.time() - cached.get("time", 0) < max_age and cached.get("latest"):
        latest = cached["latest"]
    else:
        latest = latest_releases(fetch) or cached.get("latest") or {}
        if latest:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"time": time.time(), "latest": latest}), encoding="utf-8")
            except OSError:
                pass
    out = {}
    for k, name in TOOLS.items():
        exe = (tools or {}).get(k)
        if not exe or not Path(exe).exists():
            continue
        have = installed_version(exe)
        msg = own_check(exe)
        new = latest.get(k)
        update = bool(have and new and _v(new) > _v(have))
        out[k] = {"name": name, "installed": have, "latest": new, "update": update, "url": DOWNLOAD,
                  "message": msg or (f"{name} {new} disponibile (hai la {have})" if update else
                                     f"{name} {have} è l'ultima versione" if have and new else
                                     f"{name} {have or '?'}: ultima versione non verificabile (sito non raggiungibile)")}
    return out
