"""DEAP affect — real SAM labels; anxiety/arousal sit at chance and are excluded from claims."""
from __future__ import annotations

from data_loaders import load_task


def load_anxiety(cfg=None, **kwargs):
    return load_task("deap_anxiety", cfg=cfg, **kwargs)


def load_arousal(cfg=None, **kwargs):
    return load_task("deap_arousal", cfg=cfg, **kwargs)
