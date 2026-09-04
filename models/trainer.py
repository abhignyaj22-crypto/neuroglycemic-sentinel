"""Train / evaluate one task through the lab protocol (CPU)."""
from __future__ import annotations

from typing import Any, Dict, List

from pipeline.protocol import run_task_protocol


def train_and_eval(task, *, repeats: int, folds: int, seed: int, no_sota: bool,
                   representations: List[str] | None, out_dir: str,
                   cfg: Dict[str, Any]):
    return run_task_protocol(
        task,
        repeats=repeats,
        folds=folds,
        seed=seed,
        no_sota=no_sota,
        representations=representations,
        out_dir=out_dir,
        cfg=cfg,
    )
