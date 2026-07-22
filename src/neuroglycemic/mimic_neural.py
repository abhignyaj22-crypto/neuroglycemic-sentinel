"""Adapt the causal v3 MIMIC-IV demo cohort to the neural data contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .neural_dataset import target_column, target_time_column


MIMIC_NEURAL_HORIZON_MINUTES = 360
MIMIC_NEURAL_COHORT = "MIMIC-IV-demo-2.2"

_SOURCE_FEATURES = (
    "age_years",
    "sex_female",
    "hours_since_admission",
    "current_glucose_mg_dl",
    "previous_glucose_mg_dl",
    "hours_since_previous_glucose",
    "glucose_mean_24h",
    "glucose_std_24h",
    "glucose_min_24h",
    "glucose_max_24h",
    "glucose_slope_24h_mg_dl_per_hour",
    "glucose_count_24h",
    "ehr_creatinine_last",
    "ehr_creatinine_age_hours",
    "ehr_creatinine_available",
    "ehr_urea_nitrogen_last",
    "ehr_urea_nitrogen_age_hours",
    "ehr_urea_nitrogen_available",
    "ehr_sodium_last",
    "ehr_sodium_age_hours",
    "ehr_sodium_available",
    "ehr_potassium_last",
    "ehr_potassium_age_hours",
    "ehr_potassium_available",
    "ehr_bicarbonate_last",
    "ehr_bicarbonate_age_hours",
    "ehr_bicarbonate_available",
    "ehr_lactate_last",
    "ehr_lactate_age_hours",
    "ehr_lactate_available",
    "ehr_hemoglobin_last",
    "ehr_hemoglobin_age_hours",
    "ehr_hemoglobin_available",
    "ehr_white_blood_cells_last",
    "ehr_white_blood_cells_age_hours",
    "ehr_white_blood_cells_available",
    "ehr_albumin_last",
    "ehr_albumin_age_hours",
    "ehr_albumin_available",
    "ehr_magnesium_last",
    "ehr_magnesium_age_hours",
    "ehr_magnesium_available",
)

MIMIC_NEURAL_FEATURES = tuple(
    name if name.startswith("ehr_") else f"ehr_{name}" for name in _SOURCE_FEATURES
)


def prepare_mimic_neural_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Create an EHR-only neural cohort without changing the recorded endpoint."""

    required = {
        "patient_id",
        "anchor_time",
        "target_charttime",
        "target_glucose_mg_dl",
        "target_delta_hours",
        "max_feature_storetime",
        "feature_cutoff_verified",
        "target_after_anchor_verified",
        *_SOURCE_FEATURES,
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"MIMIC neural source is missing columns: {sorted(missing)}")
    frame = source.copy()
    anchor = pd.to_datetime(frame["anchor_time"], utc=True, errors="coerce")
    target_time = pd.to_datetime(frame["target_charttime"], utc=True, errors="coerce")
    available_time = pd.to_datetime(
        frame["max_feature_storetime"], utc=True, errors="coerce"
    )
    target = pd.to_numeric(frame["target_glucose_mg_dl"], errors="coerce")
    delta_hours = pd.to_numeric(frame["target_delta_hours"], errors="coerce")
    verified_cutoff = pd.to_numeric(frame["feature_cutoff_verified"], errors="coerce").eq(1)
    verified_target = pd.to_numeric(
        frame["target_after_anchor_verified"], errors="coerce"
    ).eq(1)
    valid = (
        anchor.notna()
        & target_time.notna()
        & available_time.notna()
        & target.gt(0)
        & delta_hours.between(3.0, 9.0)
        & available_time.le(anchor)
        & target_time.gt(anchor)
        & verified_cutoff
        & verified_target
    )
    frame = frame.loc[valid].copy()
    anchor, target_time, available_time, target = (
        values.loc[valid] for values in (anchor, target_time, available_time, target)
    )
    if frame.empty:
        raise ValueError("No causal MIMIC-IV demo rows survived the neural adapter.")

    result = pd.DataFrame(
        {
            "patient_id": frame["patient_id"].astype(str),
            "cohort_id": MIMIC_NEURAL_COHORT,
            "anchor_time": anchor,
            "ehr_available": True,
            "ehr_quality": frame[list(_SOURCE_FEATURES)].notna().mean(axis=1).astype(float),
            "ehr_staleness_minutes": (
                (anchor - available_time).dt.total_seconds() / 60.0
            ).clip(lower=0.0),
            "ehr_clock_uncertainty_ms": 0.0,
            "ehr_patient_id": frame["patient_id"].astype(str),
            "ehr_cohort_id": MIMIC_NEURAL_COHORT,
            "ehr_anchor_time": anchor,
            "ehr_available_time": available_time,
            target_column(MIMIC_NEURAL_HORIZON_MINUTES): target.astype(float),
            target_time_column(MIMIC_NEURAL_HORIZON_MINUTES): target_time,
        }
    )
    for source_name, destination_name in zip(
        _SOURCE_FEATURES, MIMIC_NEURAL_FEATURES, strict=True
    ):
        result[destination_name] = pd.to_numeric(frame[source_name], errors="coerce")
    result = result.sort_values(["patient_id", "anchor_time"], ignore_index=True)
    if result.duplicated(["cohort_id", "patient_id", "anchor_time"]).any():
        raise ValueError("MIMIC neural windows duplicate a patient-time anchor.")
    if result["patient_id"].nunique() < 10:
        raise ValueError("MIMIC neural demonstration needs at least ten patients.")
    if not np.isfinite(result["ehr_quality"]).all():
        raise ValueError("MIMIC neural quality values must be finite.")
    return result


def prepare_mimic_neural_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        source = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".gz"}:
        source = pd.read_csv(path)
    else:
        raise ValueError("MIMIC neural source must be CSV, CSV.GZ, or Parquet.")
    return prepare_mimic_neural_frame(source)
