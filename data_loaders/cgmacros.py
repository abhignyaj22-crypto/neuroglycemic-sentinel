"""CGMacros 30-min glucose forecast — real future CGM. Never pair with unmatched EEG."""
from __future__ import annotations

from data_loaders import load_task
from data_loaders.pairing import refuse_unmatched_eeg_glucose


def load(cfg=None, eeg_source: str = "", **kwargs):
    refuse_unmatched_eeg_glucose(eeg_source, "cgmacros")
    return load_task("cgmacros_glucose", cfg=cfg, **kwargs)
