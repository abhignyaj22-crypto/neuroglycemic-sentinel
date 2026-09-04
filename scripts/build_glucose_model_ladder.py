#!/usr/bin/env python3
"""Regenerate the 4-horizon (30/60/90/120 min) classical CGM forecast ladder from real
CGMacros data, via the lab (``DVXR_LAB_ROOT`` / ``lab_root`` in config.json).

This is the generator this extract was missing: ``outputs/scoreboards/glucose_ablation/
glucose_model_ladder.csv`` was previously committed as evidence with no script here that
produces it. This script regenerates the classical-model half of that table (persistence,
ridge, decision tree, random forest, gradient boosting, MLP) using only functions already
real and audited in the lab:

  * ``dvxr.loaders.load_cgmacros_dataset`` — the real CGMacros loader.
  * ``dvxr.features.build_glucose_forecast_table`` — already horizon-parametric causal
    feature builder (no new causal-leakage logic here).
  * ``dvxr.eval.splits.subject_holdout_split`` — the existing patient-disjoint split.

Writes to ``outputs/scoreboards/regenerated/glucose_model_ladder.csv`` — a NEW path,
never overwriting the committed evidence file, so committed-vs-regenerated is always
diffable. Numbers will not match to the decimal (independent retraining), but should be
in the same range; see docs/PAPER_MAP.md.

Usage::

    DVXR_LAB_ROOT=/path/to/pipelinedvxr python3 scripts/build_glucose_model_ladder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paths import ensure_lab_on_path, lab_data, load_config  # noqa: E402

HORIZONS_MINUTES = (30, 60, 90, 120)
HISTORY_MINUTES = 40
SEED = 7


def _build_models():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor

    return {
        "linear_ridge": lambda: Ridge(alpha=1.0, random_state=SEED),
        "decision_tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=SEED),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=12, n_jobs=1, random_state=SEED),
        "gradient_boosting": lambda: GradientBoostingRegressor(random_state=SEED),
        "mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=SEED),
    }


def main() -> int:
    cfg = load_config()
    ensure_lab_on_path(cfg)
    from dvxr.bench.tasks import assert_no_fabrication
    from dvxr.eval.clinical_metrics import mae, rmse
    from dvxr.eval.splits import subject_holdout_split
    from dvxr.features import build_glucose_forecast_table, feature_columns
    from dvxr.loaders import load_cgmacros_dataset

    assert_no_fabrication()
    data_dir = str(lab_data(cfg, "cgmacros"))
    events = load_cgmacros_dataset(data_dir, subjects=12, include_bio=False)
    cgm = events[events["modality"] == "cgm"]
    if "glucose_source" in cgm.columns:
        cgm = cgm[cgm["glucose_source"] == "dexcom"]
    cgm = cgm.sort_values(["subject_id", "session_id", "timestamp_utc"])
    order = cgm.groupby(["subject_id", "session_id"]).cumcount()
    cgm = cgm[order % 10 == 0]

    all_subjects = cgm["subject_id"].unique()
    train_idx, test_idx = subject_holdout_split(all_subjects, test_frac=0.3, seed=SEED)
    train_subj = set(all_subjects[train_idx].tolist())
    test_subj = set(all_subjects[test_idx].tolist())
    assert train_subj.isdisjoint(test_subj), "patient split must be disjoint"

    models = _build_models()
    rows: list[dict] = []
    for h in HORIZONS_MINUTES:
        tbl = build_glucose_forecast_table(cgm, history_minutes=HISTORY_MINUTES, horizon_minutes=h)
        tbl = tbl.dropna(subset=["target_glucose", "glucose_now"]).reset_index(drop=True)
        is_train = tbl["subject_id"].isin(train_subj).to_numpy()
        is_test = tbl["subject_id"].isin(test_subj).to_numpy()
        cols = [c for c in feature_columns(tbl) if c != "target_glucose"]
        X = tbl[cols].to_numpy(dtype=float)
        y = tbl["target_glucose"].to_numpy(dtype=float)
        current = tbl["glucose_now"].to_numpy(dtype=float)

        X_tr, y_tr = X[is_train], y[is_train]
        X_te, y_te, cur_te = X[is_test], y[is_test], current[is_test]

        rows.append({"model": "persistence", "horizon_minutes": h,
                    "rmse_mg_dl": round(rmse(y_te, cur_te), 3), "mae_mg_dl": round(mae(y_te, cur_te), 3)})
        for name, factory in models.items():
            model = factory()
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            rows.append({"model": name, "horizon_minutes": h,
                        "rmse_mg_dl": round(rmse(y_te, pred), 3), "mae_mg_dl": round(mae(y_te, pred), 3)})

    out = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "scoreboards" / "regenerated" / "glucose_model_ladder.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
