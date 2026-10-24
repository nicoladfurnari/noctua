"""AstroForge: astrophotography processing from raw frames to the final image."""
from functools import lru_cache

__version__ = "0.6.0-beta.1"


@lru_cache(maxsize=1)
def version_string():
    """Version of the code that is running: read once, at the first call."""
    return disk_version()


def disk_version():
    """'0.6.0-beta.1 (39d8f29, 2026-10-25)' from the git checkout when there is one, else the package version.
    Read now: after a git pull it differs from version_string() of a program started before."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %cs"], capture_output=True,
                             text=True, timeout=5, stdin=subprocess.DEVNULL)
        if out.returncode == 0 and out.stdout.strip():
            commit, date = out.stdout.split()
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL).stdout.strip()
            return f"{__version__} ({commit}{'+modifiche' if dirty else ''}, {date})"
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return __version__
