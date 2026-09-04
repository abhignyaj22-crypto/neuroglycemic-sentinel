"""Public-cohort loaders. Thin wrappers over the lab ``dvxr.bench.tasks`` builders.

Raw data stays in the private lab (or a local ``data/real`` fetch). This package
never ships PhysioNet/MIMIC/device recordings.
"""
from __future__ import annotations

from typing import Any, Dict

from utils.paths import ensure_lab_on_path, lab_data, load_config

TASK_NAMES = (
    "wesad_stress",
    "eegmat_workload",
    "mumtaz_depression",
    "deap_anxiety",
    "deap_arousal",
    "cgmacros_glucose",
    "clinical_notes_surgery",
)


def _builders(cfg: Dict[str, Any] | None = None):
    cfg = cfg or load_config()
    ensure_lab_on_path(cfg)
    from dvxr.bench.tasks import TASK_BUILDERS
    return TASK_BUILDERS


def load_task(name: str, cfg: Dict[str, Any] | None = None, **kwargs):
    import inspect

    cfg = cfg or load_config()
    builders = _builders(cfg)
    if name not in builders:
        raise KeyError(f"unknown task {name!r}; known: {TASK_NAMES}")
    builder = builders[name]
    data_dir = kwargs.pop("data_dir", None)
    if data_dir is None:
        rel = {
            "wesad_stress": "WESAD",
            "eegmat_workload": "eegmat",
            "mumtaz_depression": "mumtaz_mdd",
            "deap_anxiety": "deap",
            "deap_arousal": "deap",
            "cgmacros_glucose": "cgmacros",
            "clinical_notes_surgery": "mtsamples",
        }.get(name)
        if rel:
            data_dir = str(lab_data(cfg, rel))
    params = inspect.signature(builder).parameters
    if data_dir is not None and ("data_dir" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())):
        kwargs["data_dir"] = data_dir
    from utils.trace import now, trace, trace_task
    t0 = now()
    trace(f"load_task start name={name} data_dir={kwargs.get('data_dir', data_dir)}")
    task = builder(**kwargs)
    trace_task(task, t0=t0)
    return task


def load_tasks(names, cfg: Dict[str, Any] | None = None) -> dict:
    return {name: load_task(name, cfg=cfg) for name in names}
