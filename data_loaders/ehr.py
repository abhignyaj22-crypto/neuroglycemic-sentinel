"""EHR / clinical-notes loader — Goal 1 modality 2.

MTSamples surgery-vs-rest is a public notes task. MIMIC-IV mortality stays
behind a DUA and is not invoked from this extract by default.
"""
from __future__ import annotations

from typing import Any, Dict

from data_loaders import load_task


def load(cfg: Dict[str, Any] | None = None, **kwargs):
    return load_task("clinical_notes_surgery", cfg=cfg, **kwargs)
