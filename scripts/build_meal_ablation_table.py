#!/usr/bin/env python3
"""Regenerate the 30-minute CGM+meal context ablation from real CGMacros data, via the lab.

This is the generator this extract was missing: ``outputs/scoreboards/glucose_ablation/
leave_one_modality_out_cgmacros.csv`` is committed as evidence, but nothing here
re-derives it. Three retrained arms on the SAME patient-disjoint split:

  * ``full`` — causal CGM-history features + causal meal-carbohydrate covariate.
  * ``no_meal`` — causal CGM-history features only.
  * ``no_cgm_history`` — meal covariate only (every CGM-derived column dropped).

The causal meal-carbohydrate covariate construction (real ``behavior``/``meal_carbs``
CGMacros channel, causal lookback, never touching a sample after the prediction cutoff)
mirrors ``dvxr.bench.tasks.cgmacros_glucose_covariate_v2_task`` exactly, reimplemented
standalone here so this ablation has no dependency on that task's (currently
structural-only) LLM-candidate registration.

Usage::

    DVXR_LAB_ROOT=/path/to/pipelinedvxr python3 scripts/build_meal_ablation_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.paths import ensure_lab_on_path, lab_data, load_config  # noqa: E402

SEED = 7
HORIZON_MINUTES = 30
LOOKBACK_MINUTES = 40


def _meal_carbs_cumulative(tbl, data_dir, load_cgmacros_dataset) -> np.ndarray:
    full_events = load_cgmacros_dataset(data_dir, subjects=12, include_bio=False)
    meal = full_events[(full_events["modality"] == "behavior")
                       & (full_events["channel"] == "meal_carbs")]
    lookback = pd.Timedelta(minutes=LOOKBACK_MINUTES)
    n = len(tbl)
    out = np.zeros(n, dtype=np.float64)
    by_key = {k: g.sort_values("timestamp_utc") for k, g in
             meal.groupby(["subject_id", "session_id"], sort=False)}
    for i, row in enumerate(tbl.itertuples(index=False)):
        key = (row.subject_id, row.session_id)
        cutoff = row.timestamp_utc
        g = by_key.get(key)
        if g is not None:
            win = g[(g["timestamp_utc"] > cutoff - lookback) & (g["timestamp_utc"] <= cutoff)]
            if len(win):
                out[i] = float(win["value"].sum())
    return out


def main() -> int:
    cfg = load_config()
    ensure_lab_on_path(cfg)
    from dvxr.bench.tasks import assert_no_fabrication, cgmacros_glucose_task
    from dvxr.eval.clinical_metrics import mae, rmse
    from dvxr.eval.splits import subject_holdout_split
    from dvxr.features import feature_columns
    from dvxr.loaders import load_cgmacros_dataset
    from sklearn.ensemble import GradientBoostingRegressor

    assert_no_fabrication()
    data_dir = str(lab_data(cfg, "cgmacros"))
    base = cgmacros_glucose_task(data_dir=data_dir, subjects=12, horizon_minutes=HORIZON_MINUTES)
    tbl = base.raw_windows.reset_index(drop=True)
    cgm_cols = [c for c in feature_columns(tbl) if c != "target_glucose"]
    meal_col = _meal_carbs_cumulative(tbl, data_dir, load_cgmacros_dataset)

    y = tbl["target_glucose"].to_numpy(dtype=float)
    valid = ~np.isnan(y)
    tbl, y, meal_col = tbl[valid].reset_index(drop=True), y[valid], meal_col[valid]

    all_subjects = tbl["subject_id"].unique()
    train_idx, test_idx = subject_holdout_split(all_subjects, test_frac=0.3, seed=SEED)
    train_subj = set(all_subjects[train_idx].tolist())
    test_subj = set(all_subjects[test_idx].tolist())
    assert train_subj.isdisjoint(test_subj), "patient split must be disjoint"
    is_train = tbl["subject_id"].isin(train_subj).to_numpy()
    is_test = tbl["subject_id"].isin(test_subj).to_numpy()

    X_cgm = tbl[cgm_cols].to_numpy(dtype=float)
    arms = {
        "full": np.column_stack([X_cgm, meal_col]),
        "no_meal": X_cgm,
        "no_cgm_history": meal_col.reshape(-1, 1),
    }

    rows = []
    for arm_name, X in arms.items():
        model = GradientBoostingRegressor(random_state=SEED)
        model.fit(X[is_train], y[is_train])
        pred = model.predict(X[is_test])
        rows.append({
            "arm": arm_name, "horizon_minutes": HORIZON_MINUTES,
            "rmse_mg_dl": round(rmse(y[is_test], pred), 3),
            "mae_mg_dl": round(mae(y[is_test], pred), 3),
            "n_train": int(is_train.sum()), "n_test": int(is_test.sum()),
        })

    out = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "scoreboards" / "regenerated" / "meal_ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
