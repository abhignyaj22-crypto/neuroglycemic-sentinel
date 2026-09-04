"""EEGMAT cognitive workload — real rest-vs-arithmetic labels."""
from __future__ import annotations

from data_loaders import load_task


def load(cfg=None, **kwargs):
    return load_task("eegmat_workload", cfg=cfg, **kwargs)
