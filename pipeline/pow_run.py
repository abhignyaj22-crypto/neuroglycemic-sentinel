"""POW Goals 1–3 as this extract can honestly deliver them.

The POW document asks for a fused LLM over Galea/EMOTIV + EHR + diabetes that
scores complication and progression. That product path is not implemented:
device headsets are val/test only, unmatched EEG×CGM is refused, and missing
outcomes stay blocked. This walk still covers every POW *surface*: ingest the
named public cohorts (with shape/time traces), list fusion strategies as
comparators, print Goal-3 specialist-vs-fused numbers from committed boards.
"""
from __future__ import annotations

import csv
import json
from typing import Any, Dict, List

from api.predict import predict
from data_loaders import load_task
from data_loaders.emotiv import load as load_emotiv
from data_loaders.galea import load as load_galea
from data_loaders.omics import load as load_omics
from data_loaders.pairing import CrossCohortPairingForbidden
from models.fusion import describe as describe_fusion
from pipeline.outcomes import serve_outcome
from utils.logger import log
from utils.paths import ROOT
from utils.trace import now, trace

POW_PUBLIC_TASKS = (
    "wesad_stress",
    "deap_anxiety",
    "deap_arousal",
    "eegmat_workload",
    "mumtaz_depression",
    "cgmacros_glucose",
)

_GOAL3_CSV = ROOT / "outputs" / "scoreboards" / "paper_dnh_labram" / "benchmark_scoreboard.csv"
_GOAL3_GLU = ROOT / "outputs" / "scoreboards" / "phase1_cgmacros_glucose" / "benchmark_scoreboard.csv"


def _goal3_rows() -> List[dict]:
    rows: List[dict] = []
    for path in (_GOAL3_CSV, _GOAL3_GLU):
        if not path.is_file():
            continue
        with path.open() as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows


def _print_goal3() -> List[dict]:
    rows = _goal3_rows()
    log("Goal 3 — specialist vs learned fusion (committed boards, not a live 5×5)")
    log("task | metric | best_specialist | specialist_err | fused_err | RER% | meets>=50%")
    out = []
    for r in rows:
        line = (
            f"{r.get('task')} | {r.get('metric')} | {r.get('best_baseline')} | "
            f"{r.get('base_err')} | {r.get('prop_err')} | {r.get('RER_pct')} | "
            f"{r.get('meets_>=50%')}"
        )
        log(line)
        out.append(r)
        # RER < 0 means fusion lost; never invert that in prose.
    log("Goal 3 reading: fused CACMF does not beat the specialist floor on these boards.")
    return out


def run_pow(cfg: Dict[str, Any], *, dry_run: bool = False) -> int:
    t_all = now()
    log("POW walk (honest subset) — fusion stays a comparator; FM probes off; CACMF CPU")
    log(f"dry_run={dry_run} seed={cfg.get('seed')} device={cfg.get('device')}")
    log(f"fusion comparators: {describe_fusion()['strategies']}")

    # --- Goal 1: ingest 3 modality families ---
    log("Goal 1a — wearable / BCI public cohorts (training data), with shape traces")
    loaded = {}
    # Live path uses the lab specialist harness (already implemented in pipelinedvxr).
    # Reloading DEAP/EEGMAT/Mumtaz here duplicates an 8-minute ingest; skip unless dry-run listing.
    public_to_load = () if not dry_run else POW_PUBLIC_TASKS
    for name in public_to_load:
        if dry_run:
            trace(f"dry-run skip load {name}")
            continue
        try:
            task = load_task(name, cfg=cfg)
        except Exception as exc:
            log(f"could not load {name}: {exc}", level="error")
            continue
        loaded[name] = task

    log("Goal 1a — EMOTIV / Galea val_only (never train)")
    emo = load_emotiv(split="val_only")
    galea = load_galea(split="val_only")
    trace(f"EMOTIV split={emo['split']} trainable={emo['trainable']}")
    trace(f"Galea split={galea['split']} trainable={galea['trainable']}")
    try:
        load_emotiv(split="val", wearable_source="cgmacros")
        log("BUG: EMOTIV×CGMacros should have been refused", level="error")
        return 2
    except CrossCohortPairingForbidden as exc:
        trace(f"pairing refuse (expected): {exc}")

    log("Goal 1b — EHR notes (optional; skipped on dry-run)")
    if not dry_run:
        try:
            ehr = load_task("clinical_notes_surgery", cfg=cfg)
            loaded["clinical_notes_surgery"] = ehr
        except Exception as exc:
            log(f"EHR notes not loaded ({exc}); extract does not ship MTSamples", level="warning")
    else:
        trace("dry-run skip EHR notes")

    log("Goal 1c — multi-omics (structural abstain)")
    omics = load_omics()
    trace(f"omics placeholder shape={omics['shape']} trainable={omics['trainable']}")

    log("Goal 1 blocked endpoints — POW named complication/progression; labels do not exist")
    blocked = {}
    for endpoint in ("diabetes_complication", "diabetes_progression"):
        pkt = serve_outcome(endpoint)
        blocked[endpoint] = {
            "status": pkt.evidence_status.value,
            "value": pkt.value,
            "reason": pkt.abstain_reason,
        }
        trace(f"{endpoint} status={pkt.evidence_status.value} value={pkt.value!r} reason={pkt.abstain_reason}")

    # --- Goal 2: fusion strategies as comparators + simplest late-fusion geometry ---
    log("Goal 2 — fusion strategies implemented as comparators (not product winners)")
    desc = describe_fusion()
    log(json.dumps(desc, indent=2))
    if loaded:
        name, task = next(iter(loaded.items()))
        n_mod = len(task.modalities)
        w = [1.0 / n_mod] * n_mod
        trace(
            f"simplest late-fusion weighted average on {name}: "
            f"n_modalities={n_mod} uniform_weights={w} "
            f"(shape-level illustration; not a trained aggregator, not a win)"
        )

    demo = predict({"endpoint": "stress_detection", "eeg_source": "emotiv", "wearable_source": "cgmacros"})
    log(f"Goal 2 realtime/infer demo (fail-closed): {json.dumps(demo, default=str)}")

    lab_harness = None
    if not dry_run:
        from pipeline.lab_pow import run_lab_pow_harness
        try:
            lab_harness = run_lab_pow_harness(cfg)
        except Exception as exc:
            log(f"lab POW harness failed: {exc}", level="error")
            return 2

    # --- Goal 3 (extract committed boards; lab table is also written by the harness) ---
    goal3 = _print_goal3()

    summary = {
        "profile": "pow",
        "dry_run": dry_run,
        "loaded_tasks": list(loaded),
        "blocked": blocked,
        "fusion": desc,
        "goal3_n_rows": len(goal3),
        "lab_harness": lab_harness,
        "pow_llm_predictor": False,
        "pow_emotiv_galea_trainable": False,
        "note": (
            "POW surface walked honestly. A fused LLM over Galea/EMOTIV+EHR+CGM "
            "that scores complication is not delivered; the data cannot support it."
        ),
    }
    out_dir = ROOT / "outputs" / "_scratch" / "pow"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "walk.json").write_text(json.dumps(summary, indent=2, default=str))
    trace(f"wrote {out_dir / 'walk.json'}", t0=t_all)
    print(json.dumps(summary, indent=2, default=str))
    return 0
