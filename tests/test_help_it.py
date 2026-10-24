"""Every parameter of every plugin has its short Italian help (shown by the GUI's "?")."""
from astroforge import plugins
from astroforge.plugins import help_it


def test_every_parameter_has_italian_help():
    missing = [f"{pid}:{k}" for pid, pl in plugins.REGISTRY.items() for k in pl.params if not help_it.text(pid, k)]
    assert not missing, missing


def test_no_stale_entries():
    known = {f"{pid}:{k}" for pid, pl in plugins.REGISTRY.items() for k in pl.params}
    assert not [k for k in help_it.HELP if k not in known]


def test_shared_texts_only_where_they_fit():
    # a SHARED text is used only by parameters without their own entry: list them to keep the meaning right
    used = sorted({f"{pid}:{k}" for pid, pl in plugins.REGISTRY.items() for k in pl.params
                   if f"{pid}:{k}" not in help_it.HELP and k in help_it.SHARED})
    allowed = {"fov_deg", "ra_hours", "dec_deg", "radius_deg", "timeout", "gpu", "aperture", "saturation",
               "neutralize_background", "correction"}
    assert all(u.split(":")[1] in allowed for u in used)
    assert all(u.split(":")[0] in ("analysis.plate_solve", "color.pcc", "color.spcc", "background.allsky",
                                   "background.dbe", "ext.graxpert_background", "ext.cc_satellite",
                                   "ext.cosmic_clarity", "ext.graxpert_background", "ext.graxpert_denoise",
                                   "stars.darkstar") for u in used), used
