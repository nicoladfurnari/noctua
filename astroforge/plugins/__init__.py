"""Importing this package registers every built-in plugin."""
from . import annotate, background_dbe, contrast, core, cosmetic, external, gradient_ref, look, pi_extra, spcc, stars, stretch_extra, tools, video_workflow, workflow  # noqa: F401
from .base import REGISTRY, Context, Param, Plugin, get, register  # noqa: F401
