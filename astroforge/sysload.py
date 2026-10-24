"""Live machine load for the GUI bar: CPU, RAM and every NVIDIA GPU (utilisation, memory, temperature).

CPU/RAM come from psutil; GPUs from NVML (nvidia-ml-py), else from `nvidia-smi`. Anything missing is simply
left out, so the bar works on machines without psutil or without an NVIDIA card.
"""
import shutil
import subprocess

_NVML = {"ok": None}


def _nvml():
    if _NVML["ok"] is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML.update(ok=True, mod=pynvml)
        except Exception:
            _NVML["ok"] = False
    return _NVML.get("mod") if _NVML["ok"] else None


def _gpus_nvml(nv):
    out = []
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        u = nv.nvmlDeviceGetUtilizationRates(h)
        m = nv.nvmlDeviceGetMemoryInfo(h)
        name = nv.nvmlDeviceGetName(h)
        try:
            temp = nv.nvmlDeviceGetTemperature(h, 0)
        except Exception:
            temp = None
        out.append({"index": i, "name": name.decode() if isinstance(name, bytes) else str(name),
                    "util": float(u.gpu), "mem_used_gb": m.used / 2 ** 30, "mem_total_gb": m.total / 2 ** 30,
                    "temp": temp})
    return out


def _gpus_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        res = subprocess.run([exe, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in res.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 6:
            out.append({"index": int(p[0]), "name": p[1], "util": float(p[2]), "mem_used_gb": float(p[3]) / 1024,
                        "mem_total_gb": float(p[4]) / 1024, "temp": float(p[5])})
    return out


def snapshot():
    """{'cpu': %, 'ram': %, 'ram_used_gb', 'ram_total_gb', 'gpus': [...]} (keys missing when unknown)."""
    snap = {"gpus": []}
    try:
        import psutil
        snap["cpu"] = psutil.cpu_percent(interval=None)     # since the previous call: no waiting
        vm = psutil.virtual_memory()
        snap.update(ram=vm.percent, ram_used_gb=(vm.total - vm.available) / 2 ** 30, ram_total_gb=vm.total / 2 ** 30)
    except ImportError:
        pass
    nv = _nvml()
    try:
        snap["gpus"] = _gpus_nvml(nv) if nv else _gpus_smi()
    except Exception:
        snap["gpus"] = _gpus_smi()
    return snap
