"""Shape / dtype / wall-time prints for POW pipeline walks."""
from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from utils.logger import log


def now() -> float:
    return time.perf_counter()


def elapsed_s(t0: float) -> float:
    return time.perf_counter() - t0


def _shape(obj: Any) -> str:
    if obj is None:
        return "None"
    if hasattr(obj, "shape"):
        return str(tuple(obj.shape))
    if isinstance(obj, (list, tuple)):
        return f"len={len(obj)}"
    return type(obj).__name__


def _dtype(obj: Any) -> str:
    if obj is None:
        return "None"
    dt = getattr(obj, "dtype", None)
    if dt is not None:
        return str(dt)
    return type(obj).__name__


def trace(msg: str, *, t0: Optional[float] = None) -> None:
    suffix = f" elapsed_s={elapsed_s(t0):.3f}" if t0 is not None else ""
    log(f"[trace] {msg}{suffix}")


def _minmax(arr: Any) -> str:
    try:
        import numpy as np
    except ImportError:
        return ""
    if not isinstance(arr, np.ndarray) or not arr.size:
        return ""
    if np.issubdtype(arr.dtype, np.floating):
        return f" min={float(np.nanmin(arr)):.4g} max={float(np.nanmax(arr)):.4g}"
    if np.issubdtype(arr.dtype, np.integer):
        return f" min={int(np.min(arr))} max={int(np.max(arr))}"
    return ""


def trace_array(name: str, arr: Any, *, t0: Optional[float] = None) -> None:
    trace(f"{name} shape={_shape(arr)} dtype={_dtype(arr)}{_minmax(arr)}", t0=t0)


def trace_task(task: Any, *, t0: Optional[float] = None) -> None:
    try:
        import numpy as np
        n_subj = len(np.unique(task.subject_ids)) if getattr(task, "subject_ids", None) is not None else "?"
    except Exception:
        n_subj = "?"
    trace(
        f"task={task.name} kind={task.kind} n={task.n} n_subjects={n_subj} "
        f"metric={task.metric} modalities={list(task.modalities)}",
        t0=t0,
    )
    trace_array(f"{task.name}.y", task.y)
    trace_array(f"{task.name}.subject_ids", task.subject_ids)
    feats: Mapping[str, Any] = task.features
    widths = {}
    for mod, x in feats.items():
        trace_array(f"{task.name}.features[{mod}]", x)
        arr_shape = getattr(x, "shape", None)
        if arr_shape is not None and len(arr_shape) == 2:
            widths[mod] = int(arr_shape[1])
        else:
            widths[mod] = int(getattr(x, "size", 0) or 0)
    concat_w = sum(widths.values())
    trace(
        f"{task.name} early-concat geometry n={task.n} width={concat_w} "
        f"(per-modality widths={widths}) — shape only; not a claimed win"
    )
