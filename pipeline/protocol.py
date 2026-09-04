"""Repeated grouped CV — thin wrap of the lab protocol. CACMF stays CPU."""
from __future__ import annotations

from typing import Any, Dict, List

from utils.trace import now, trace


def run_task_protocol(task, *, repeats: int, folds: int, seed: int, no_sota: bool,
                      representations: List[str] | None, out_dir: str,
                      cfg: Dict[str, Any]):
    from utils.paths import ensure_lab_on_path
    ensure_lab_on_path(cfg)
    from dvxr.bench.run import run_task
    from dvxr.bench.runtime_config import RuntimeConfig

    # Device is always CPU for this extract. CACMF GPU training is locked off.
    runtime = RuntimeConfig(device="cpu")
    t0 = now()
    trace(
        f"protocol start task={task.name} repeats={repeats} folds={folds} "
        f"seed={seed} no_sota={no_sota} reps={representations} device=cpu"
    )
    result = run_task(
        task,
        n_repeats=repeats,
        n_folds=folds,
        seed=seed,
        include_sota=not no_sota,
        representations=representations,
        out_dir=out_dir,
        runtime_config=runtime,
    )
    n_folds = getattr(result, "n_folds", "n/a")
    trace(
        f"protocol done task={getattr(result, 'task', getattr(task, 'name', '?'))} "
        f"metric={getattr(result, 'metric', '?')} "
        f"best_baseline={getattr(result, 'best_baseline', '?')} n_folds={n_folds}",
        t0=t0,
    )
    return result
