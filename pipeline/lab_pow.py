"""Call the lab's already-built POW harness (Goals 1–3) from this extract.

Does not vendor ``src/dvxr``. Galea ingest here is n=1 schema/validation, not training.
LLM specialists report ``predicts: false``. Complication/progression stay blocked in
``pipeline.outcomes`` — the lab harness does not un-block them.
"""
from __future__ import annotations

from typing import Any, Dict

from utils.logger import log
from utils.paths import ROOT, ensure_lab_on_path, lab_root
from utils.trace import now, trace

HARNESS_TASKS = (
    "wesad_stress",
    "mumtaz_depression",
    "cgmacros_glucose",
    "mimic_ehr",
    "mimic_eeg_glucose",
    "mimic_ecg_glucose",
)


def run_lab_pow_harness(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ensure_lab_on_path(cfg)
    lab = lab_root(cfg)
    from dvxr.serve.specialists import run_specialist_prediction
    from dvxr.serve.specialists.goal3 import write_pow_goal3_table
    from dvxr.serve.specialists.ingest import ingest_goal1_real

    t0 = now()
    log("Lab POW harness — ingest Goal 1 named modalities (BCI / wearable / EHR / CGM)")
    ingest = ingest_goal1_real()
    for key, body in (ingest.get("modalities") or {}).items():
        trace(
            f"goal1[{key}] n_rows={body.get('n_rows', body.get('n_eeg_rows'))} "
            f"n_channels={body.get('n_channels')} n_subjects={body.get('n_subjects')} "
            f"source={body.get('source')}"
        )
    trace(f"goal1 cross_joins={ingest.get('cross_joins')} pivot={ingest.get('pivot')}", t0=t0)

    t1 = now()
    rejected = run_specialist_prediction(
        {"task": "auto", "eeg_source": "wesad", "wearable_source": "cgmacros"}
    )
    trace(
        f"unmatched WESAD×CGMacros status={rejected.get('status')} "
        f"reason={rejected.get('reason_codes')}",
        t0=t1,
    )

    specialists: Dict[str, Any] = {}
    for task in HARNESS_TASKS:
        t2 = now()
        body = run_specialist_prediction({"task": task})
        specialists[task] = {
            "status": body.get("status"),
            "same_person": body.get("same_person"),
            "fused_with_cgmacros": body.get("fused_with_cgmacros"),
            "predicts": (body.get("explanation") or {}).get("predicts"),
            "reason_codes": body.get("reason_codes"),
            "family": body.get("family"),
            "orchestration": body.get("orchestration"),
        }
        trace(
            f"specialist[{task}] status={specialists[task]['status']} "
            f"predicts={specialists[task]['predicts']} "
            f"family={specialists[task]['family']}",
            t0=t2,
        )

    dest = ROOT / "outputs" / "_scratch" / "pow" / "pow_goal3_ablation.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = write_pow_goal3_table(repo=lab, destination=dest)
    trace(f"Goal 3 table wrote {dest} chars={len(text)}")
    log(text[:2000])

    paper = lab / "paper" / "main.pdf"
    tex = lab / "paper" / "main.tex"
    goal4 = {
        "ieee_pdf": str(paper) if paper.is_file() else None,
        "ieee_tex": str(tex) if tex.is_file() else None,
        "title": "When Learned Cross-Modal Fusion Helps and When It Harms",
        "in_this_extract": False,
        "note": "Goal 4 lives in the lab paper/; this extract does not re-claim Galea+LLM fusion as a product.",
    }
    trace(f"Goal 4 paper pdf_exists={paper.is_file()} tex_exists={tex.is_file()}")

    return {
        "goal1_ingest": ingest,
        "rejected_unmatched": {
            "status": rejected.get("status"),
            "reason_codes": rejected.get("reason_codes"),
        },
        "specialists": specialists,
        "goal3_table": str(dest),
        "goal4": goal4,
        "orchestration": (specialists.get("wesad_stress") or {}).get("orchestration"),
    }
