#!/usr/bin/env python3
"""Reproduce Tables IV-VI from raw public data — distinct from ``validate_paper_claims.py``.

Two different questions, kept structurally separate (critique item 19's concern, now
addressed directly):

  * ``python3 scripts/validate_paper_claims.py`` — ARTIFACT VALIDATION. Checks the
    already-committed scoreboards under ``outputs/scoreboards/`` against the manuscript's
    numbers. No lab required, no retraining, runs in seconds.
  * ``python3 scripts/reproduce_from_raw.py`` (this script) — REPRODUCTION. Retrains
    from raw public cohorts through the lab (requires ``DVXR_LAB_ROOT`` /
    ``lab_root`` in config.json, same as the ``smoke``/``mh``/``glucose``/``pow``
    profiles already in ``main.py``) and writes NEW artifacts under
    ``outputs/scoreboards/regenerated/`` — it never overwrites the committed evidence.

This chains, in order:
  1. The three new generator scripts (glucose ladder, glycemic warning, meal ablation).
  2. ``main.py --profile mh`` and ``--profile glucose`` — the existing, already-real
     retrain profiles, for the classification tasks (WESAD/DEAP/EEGMAT/Mumtaz) and the
     CGMacros fusion-comparator board.

After running, ``python3 scripts/validate_paper_claims.py`` will pick up the regenerated
hypo/hyper CSV automatically (see that script's ``check_glucose_ladder_and_warnings``).

Usage::

    DVXR_LAB_ROOT=/path/to/pipelinedvxr python3 scripts/reproduce_from_raw.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paths import ensure_lab_on_path, load_config  # noqa: E402


def _run(cmd: list[str]) -> None:
    print(f"\n[reproduce-from-raw] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    cfg = load_config()
    try:
        lab_src = ensure_lab_on_path(cfg)
    except Exception as exc:                                          # noqa: BLE001
        print(f"[reproduce-from-raw] lab not available: {exc}", file=sys.stderr)
        print("[reproduce-from-raw] set DVXR_LAB_ROOT or config.json's lab_root — "
             "reproduction (unlike paper-validate) requires the lab.", file=sys.stderr)
        return 2
    print(f"[reproduce-from-raw] lab src on path: {lab_src}")

    for script in ("build_glucose_model_ladder.py", "build_glycemic_warning_table.py",
                  "build_meal_ablation_table.py"):
        _run([sys.executable, str(ROOT / "scripts" / script)])

    for profile in ("mh", "glucose"):
        _run([sys.executable, str(ROOT / "main.py"), "--profile", profile])

    print("\n[reproduce-from-raw] done. New artifacts under "
         "outputs/scoreboards/regenerated/ and outputs/_scratch/. "
         "Run `python3 scripts/validate_paper_claims.py` next to check them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
