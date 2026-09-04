"""Write the relativity scoreboard. Fusion is never assumed to win."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def write_scoreboard(results: list, out_dir: str, meta: Dict[str, Any] | None = None) -> dict:
    from utils.paths import ensure_lab_on_path, load_config
    cfg = load_config()
    ensure_lab_on_path(cfg)
    from dvxr.bench.scoreboard import write_scoreboard as _write
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return _write(results, out_dir=out_dir, meta=meta)


def evaluate_all(results: list) -> List[dict]:
    rows = []
    for r in results:
        means = r.config_means()
        rows.append({
            "task": r.task,
            "metric": r.metric,
            "best_baseline": r.best_baseline,
            "means": {k: float(v) for k, v in means.items()},
            "proposed_wins": False,  # never assumed; scoreboard computes RER
        })
    return rows
