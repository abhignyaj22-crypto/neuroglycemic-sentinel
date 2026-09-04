"""Mumtaz MDD — real diagnosis labels. Identity-leakage-confounded; not a product claim."""
from __future__ import annotations

from data_loaders import load_task


def load(cfg=None, **kwargs):
    return load_task("mumtaz_depression", cfg=cfg, **kwargs)
