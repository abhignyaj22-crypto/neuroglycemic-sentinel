#!/usr/bin/env python3
"""Regenerate separate 30-minute hypoglycemia/hyperglycemia warning AUROC rows from real
CGMacros data, via the lab.

This is the generator this extract was missing: ``best_models_summary.csv``'s
"0.976 hypo / 0.981 hyper" row is committed as evidence, but ``validate_paper_claims.py``
only checked those numbers by string-searching the summary file. This script recomputes
them, and writes a structured CSV that the validator can compare against real numbers
instead (see ``validate_paper_claims.py::check_glucose_ladder_and_warnings``).

Reuses the lab's already-tested causal excursion labeling
(``dvxr.targets.excursion.build_excursion_labels``) twice with a one-sided threshold each
time (the other bound set to +/-inf) — no new labeling logic, only reuse.

Usage::

    DVXR_LAB_ROOT=/path/to/pipelinedvxr python3 scripts/build_glycemic_warning_table.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paths import ensure_lab_on_path, lab_data, load_config  # noqa: E402

SEED = 7
HORIZON_MINUTES = 30
DIRECTIONS = {
    "hypoglycemia_warning": dict(high_mg_dl=float("inf")),
    "hyperglycemia_warning": dict(low_mg_dl=float("-inf")),
}
FEATURE_COLS = ["cgm_last", "cgm_mean", "cgm_std", "cgm_min", "cgm_max", "cgm_range",
               "cgm_slope_per_min", "cgm_tir", "cgm_frac_hyper", "cgm_frac_hypo", "cgm_n_samples"]


def _one_direction(cgm, base_thr, overrides, build_cgm_feature_matrix, build_excursion_labels,
                   subject_holdout_split) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    thr = replace(base_thr, **overrides, horizons_minutes=(HORIZON_MINUTES,))
    anchors = []
    for sid, g in cgm.groupby("subject_id"):
        t = pd.to_datetime(g["timestamp"]).sort_values()
        anchors += list(t.iloc[thr.history_minutes // 5::8])[:60]
    examples = build_excursion_labels(cgm, thresholds=thr, anchors=sorted(set(anchors)),
                                      subject_col="subject_id")
    examples = examples[examples["censored"] == False].reset_index(drop=True)  # noqa: E712
    if examples.empty or examples["label"].nunique() < 2:
        return {"status": "insufficient_data", "n": int(len(examples))}

    feats = build_cgm_feature_matrix(cgm, examples, thresholds=thr)
    feats = feats.dropna(subset=FEATURE_COLS)
    if feats.empty or feats["label"].nunique() < 2:
        return {"status": "insufficient_data", "n": int(len(feats))}

    subjects = feats["subject_id"].unique()
    train_idx, test_idx = subject_holdout_split(subjects, test_frac=0.3, seed=SEED)
    train_subj = set(subjects[train_idx].tolist())
    test_subj = set(subjects[test_idx].tolist())
    assert train_subj.isdisjoint(test_subj), "patient split must be disjoint"
    is_train = feats["subject_id"].isin(train_subj).to_numpy()
    is_test = feats["subject_id"].isin(test_subj).to_numpy()

    X = feats[FEATURE_COLS].to_numpy(dtype=float)
    y = feats["label"].to_numpy(dtype=float)
    if len(np.unique(y[is_train])) < 2 or len(np.unique(y[is_test])) < 2:
        return {"status": "insufficient_data_for_auroc",
                "n_train": int(is_train.sum()), "n_test": int(is_test.sum())}

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X[is_train], y[is_train])
    p = model.predict_proba(X[is_test])[:, 1]
    auroc = float(roc_auc_score(y[is_test], p))
    return {"status": "ok", "auroc": round(auroc, 3),
            "n_train": int(is_train.sum()), "n_test": int(is_test.sum()),
            "positive_rate_test": round(float(y[is_test].mean()), 3)}


def main() -> int:
    cfg = load_config()
    ensure_lab_on_path(cfg)
    from dvxr.bench.tasks import assert_no_fabrication
    from dvxr.eval.glucose_ablation import load_cgmacros
    from dvxr.eval.splits import subject_holdout_split
    from dvxr.prediction.service import build_cgm_feature_matrix
    from dvxr.targets import ExcursionThresholds, build_excursion_labels

    assert_no_fabrication()
    cgm = load_cgmacros(root=str(lab_data(cfg, "cgmacros")), max_subjects=None)

    rows = []
    for name, overrides in DIRECTIONS.items():
        result = _one_direction(cgm, ExcursionThresholds(), overrides, build_cgm_feature_matrix,
                                build_excursion_labels, subject_holdout_split)
        rows.append({"endpoint": name, "horizon_minutes": HORIZON_MINUTES, "metric": "auroc", **result})

    out = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "scoreboards" / "regenerated" / "glycemic_warning.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
