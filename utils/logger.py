"""Tiny logger matching the RxFusion cadence."""
from __future__ import annotations


def log(msg: str, level: str = "info") -> None:
    print(f"[{level}] {msg}", flush=True)
