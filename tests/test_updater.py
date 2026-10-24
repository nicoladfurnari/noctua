"""Automatic update at start (astroforge.updater) with throw-away git repositories."""
import json
import shutil
import subprocess

import pytest

from astroforge import updater

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, stdin=subprocess.DEVNULL)


@pytest.fixture
def repos(tmp_path):
    """origin (stands for GitHub) and an installed clone of it."""
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    _git(src, "config", "user.email", "t@example.invalid")
    _git(src, "config", "user.name", "t")
    (src / "requirements.txt").write_text("numpy\ncupy-cuda12x[ctk]   # optional GPU\npytest   # tests\n")
    (src / "a.txt").write_text("1\n")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "first")
    inst = tmp_path / "inst"
    subprocess.run(["git", "clone", "-q", str(src), str(inst)], check=True, capture_output=True)
    return src, inst


def _commit(src, name, text, msg):
    (src / name).write_text(text)
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", msg)


def test_update_pulls_new_version_and_reinstalls_deps_only_when_needed(repos, monkeypatch):
    src, inst = repos
    calls = []
    monkeypatch.setattr(updater, "install_deps", lambda root, log: calls.append(root))
    assert updater.update(inst, log=lambda m: None) == "current"
    _commit(src, "a.txt", "2\n", "new feature")
    assert updater.update(inst, log=lambda m: None) == "updated"
    assert (inst / "a.txt").read_text() == "2\n" and calls == []
    _commit(src, "requirements.txt", "numpy\nscipy\n", "new package")
    assert updater.update(inst, log=lambda m: None) == "updated" and calls == [inst]


def test_update_never_touches_local_changes_or_other_branches(repos):
    src, inst = repos
    _commit(src, "a.txt", "2\n", "new")
    (inst / "a.txt").write_text("my change\n")
    assert updater.update(inst, log=lambda m: None) == "dirty"
    assert (inst / "a.txt").read_text() == "my change\n"
    _git(inst, "checkout", "-q", "--", "a.txt")
    _git(inst, "checkout", "-q", "-b", "prova")
    assert updater.update(inst, log=lambda m: None) == "branch"


def test_update_offline_and_disabled(repos, monkeypatch):
    src, inst = repos
    _git(inst, "remote", "set-url", "origin", str(src.parent / "missing"))
    assert updater.update(inst, log=lambda m: None) == "offline"
    monkeypatch.setenv("ASTROFORGE_NO_UPDATE", "1")
    assert updater.update(inst, log=lambda m: None) == "disabled"


def test_requirements_cupy_only_with_gpu(repos):
    _, inst = repos
    assert updater.requirements(inst, gpu=False) == ["numpy", "pytest"]
    assert updater.requirements(inst, gpu=True) == ["numpy", "cupy-cuda12x[ctk]", "pytest"]


def test_write_config_keeps_existing(tmp_path):
    updater.write_config(tmp_path, log=lambda m: None)
    cfg = (tmp_path / "config.json").read_text()
    assert '"starnet"' in cfg
    # every path valid: the file stays exactly as it is
    exe = tmp_path / "tool.exe"
    exe.write_text("x")
    text = json.dumps({"tools": {k: str(exe) for k in updater.TOOL_GUESSES}})
    (tmp_path / "config.json").write_text(text)
    updater.write_config(tmp_path, log=lambda m: None)
    assert (tmp_path / "config.json").read_text() == text


def test_find_tools_in_folders_and_nina(tmp_path, monkeypatch):
    prog = tmp_path / "Programs"
    (prog / "ASTAP 2024").mkdir(parents=True)
    (prog / "ASTAP 2024" / "astap.exe").write_text("x")
    (prog / "ASTAP 2024" / "astap_cli.exe").write_text("x")
    (prog / "my-tools" / "GraXpert-3.0").mkdir(parents=True)
    (prog / "GraXpert3" / "bin").mkdir(parents=True)
    (prog / "GraXpert3" / "bin" / "GraXpert.exe").write_text("x")
    monkeypatch.setattr(updater, "TOOL_GUESSES", {k: [] for k in updater.TOOL_GUESSES})
    monkeypatch.setattr(updater, "_registry_dirs", lambda: [])
    monkeypatch.setattr(updater, "_roots", lambda: [prog])
    monkeypatch.setattr(updater.shutil, "which", lambda name: None)
    found = updater.find_tools()
    assert found["astap"][0].endswith("astap_cli.exe")               # the command-line version preferred
    assert found["graxpert"][0].endswith("GraXpert.exe")
    assert "starnet" not in found
    # N.I.N.A. remembers where ASTAP is
    monkeypatch.setattr(updater, "_roots", lambda: [])
    nina = tmp_path / "local" / "NINA" / "Profiles"
    nina.mkdir(parents=True)
    exe = tmp_path / "elsewhere" / "astap.exe"
    exe.parent.mkdir()
    exe.write_text("x")
    (nina / "a.profile").write_text(f"<Profile><PlateSolveSettings><ASTAPLocation>{exe}</ASTAPLocation></PlateSolveSettings></Profile>")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert updater.find_tools()["astap"] == (str(exe), "profilo di N.I.N.A.")


def test_write_config_keeps_valid_paths_and_fills_missing(tmp_path, monkeypatch):
    good = tmp_path / "starnet2.exe"
    good.write_text("x")
    new = tmp_path / "astap_cli.exe"
    new.write_text("x")
    (tmp_path / "config.json").write_text(json.dumps({"tools": {"starnet": str(good), "astap": "C:/nope/astap.exe"},
                                                      "other": 1}))
    monkeypatch.setattr(updater, "find_tools", lambda log=None: {"astap": (str(new), "test"), "starnet": ("X", "test")})
    updater.write_config(tmp_path, log=lambda m: None)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["tools"]["starnet"] == str(good) and cfg["tools"]["astap"] == str(new) and cfg["other"] == 1


def test_system_cuda_reused_instead_of_pip_packages(tmp_path, monkeypatch):
    cuda = tmp_path / "CUDA" / "v12.6"
    (cuda / "bin").mkdir(parents=True)
    for d in updater._CUDA_DLLS:
        (cuda / "bin" / f"{d}.dll").write_text("x")
    monkeypatch.setenv("CUDA_PATH", str(cuda))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "none"))
    assert updater.system_cuda() == str(cuda)
    (tmp_path / "requirements.txt").write_text("numpy\ncupy-cuda12x[ctk]   # gpu\n")
    assert updater.requirements(tmp_path, gpu=True) == ["numpy", "cupy-cuda12x"]
    assert updater.requirements(tmp_path, gpu=True, cuda=False) == ["numpy", "cupy-cuda12x[ctk]"]
    (cuda / "bin" / "cusolver64_11.dll").unlink()                        # incomplete toolkit: not used
    assert updater.system_cuda() is None


def test_tool_updates_from_the_site_and_installed_version(tmp_path, monkeypatch):
    from astroforge import toolupdates
    monkeypatch.setenv("ASTROFORGE_TOOL_UPDATES", str(tmp_path / "upd.json"))
    page = "<p>Current command-line releases are <b>StarNet2 2.7.0</b> and DeepSNR 1.3.2, available for ...</p>"
    assert toolupdates.latest_releases(lambda url: page) == {"starnet": "2.7.0", "deepsnr": "1.3.2"}
    assert toolupdates.latest_releases(lambda url: (_ for _ in ()).throw(OSError("offline"))) == {}
    exe = tmp_path / "starnet2.exe"
    exe.write_text("x")
    monkeypatch.setattr(toolupdates, "installed_version", lambda e: "2.6.2")
    monkeypatch.setattr(toolupdates, "own_check", lambda e: None)
    r = toolupdates.check({"starnet": str(exe), "deepsnr": str(exe)}, fetch=lambda url: page)
    assert r["starnet"]["update"] and "2.7.0" in r["starnet"]["message"]
    assert r["deepsnr"]["latest"] == "1.3.2" and r["deepsnr"]["update"] is (toolupdates._v("1.3.2") > toolupdates._v("2.6.2"))
    # cached: a second check within a day does not fetch the page again
    r2 = toolupdates.check({"starnet": str(exe)}, fetch=lambda url: (_ for _ in ()).throw(AssertionError("fetched")))
    assert r2["starnet"]["latest"] == "2.7.0"


def test_shortcut_icon_needs_the_shortcut(tmp_path, monkeypatch):
    monkeypatch.setattr(updater.Path, "home", lambda: tmp_path)         # no Desktop\Noctua.lnk here
    assert updater.shortcut_icon() is False
