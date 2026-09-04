"""WESAD stress — real protocol labels (stress vs non-stress)."""
from __future__ import annotations

from data_loaders import load_task


def load(cfg=None, **kwargs):
    return load_task("wesad_stress", cfg=cfg, **kwargs)
