"""Locate this extract and the private lab (config / env)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load config.json, then apply DVXR_LAB_ROOT env override if set."""
    cfg_path = path or (ROOT / "config.json")
    if not cfg_path.is_file():
        example = ROOT / "config.example.json"
        if example.is_file():
            cfg_path = example
        else:
            raise FileNotFoundError(
                f"No config at {cfg_path}; copy config.example.json → config.json"
            )
    with cfg_path.open() as f:
        cfg = json.load(f)
    env_root = os.environ.get("DVXR_LAB_ROOT", "").strip()
    if env_root:
        cfg["lab_root"] = env_root
    return cfg


def lab_root(cfg: Dict[str, Any]) -> Path:
    raw = cfg.get("lab_root") or os.environ.get("DVXR_LAB_ROOT")
    if not raw:
        raise RuntimeError(
            "lab_root unset. Export DVXR_LAB_ROOT=/path/to/pipelinedvxr "
            "or set lab_root in config.json (see config.example.json)."
        )
    return Path(raw).expanduser().resolve()


def lab_src(cfg: Dict[str, Any]) -> Path:
    return lab_root(cfg) / "src"


def lab_data(cfg: Dict[str, Any], *parts: str) -> Path:
    return lab_root(cfg) / "data" / "real" / Path(*parts)


def ensure_lab_on_path(cfg: Dict[str, Any]) -> Path:
    """Insert the lab ``src/`` so ``import dvxr`` resolves without vendoring."""
    src = lab_src(cfg)
    if not src.is_dir():
        raise FileNotFoundError(f"lab src not found: {src}")
    src_s = str(src)
    if src_s not in sys.path:
        sys.path.insert(0, src_s)
    return src
