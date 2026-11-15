"""Every test session gets its own empty master library and timings file (never the user's ~/.astroforge)."""
import os
import tempfile

os.environ["ASTROFORGE_MASTER_LIBRARY"] = tempfile.mkdtemp(prefix="astroforge_masters_")
# stage timings of the test runs (tiny frames) must not end up in the user's estimates
os.environ["ASTROFORGE_TIMINGS"] = os.path.join(tempfile.mkdtemp(prefix="astroforge_timings_"), "timings.json")
os.environ["ASTROFORGE_ATLAS"] = tempfile.mkdtemp(prefix="astroforge_atlas_")       # never the user's atlas
os.environ["ASTROFORGE_ATLAS_SOLVE"] = "0"       # a finished test job does not start a plate solve in the background
os.environ["ASTROFORGE_TOOL_UPDATES"] = os.path.join(tempfile.mkdtemp(prefix="astroforge_upd_"), "tool_updates.json")
