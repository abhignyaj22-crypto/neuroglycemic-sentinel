from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EHRGlucoseConfig


LAB_CONCEPT_LABELS: dict[str, tuple[str, ...]] = {
    "creatinine": ("creatinine",),
    "urea_nitrogen": ("urea nitrogen",),
    "sodium": ("sodium",),
    "potassium": ("potassium",),
    "bicarbonate": ("bicarbonate",),
    "lactate": ("lactate",),
    "hemoglobin": ("hemoglobin",),
    "white_blood_cells": ("white blood cells",),
    "albumin": ("albumin",),
    "magnesium": ("magnesium",),
}

LAB_CONCEPT_UNITS: dict[str, tuple[str, ...]] = {
    "creatinine": ("mg/dl",),
    "urea_nitrogen": ("mg/dl",),
    "sodium": ("meq/l",),
    "potassium": ("meq/l",),
    "bicarbonate": ("meq/l",),
    "lactate": ("mmol/l",),
    "hemoglobin": ("g/dl",),
    "white_blood_cells": ("k/ul",),
    "albumin": ("g/dl",),
    "magnesium": ("mg/dl",),
}

BASE_FEATURES = (
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
)

EHR_FEATURES = tuple(
    [*BASE_FEATURES]
    + [f"ehr_{concept}_last" for concept in LAB_CONCEPT_LABELS]
    + [f"ehr_{concept}_age_hours" for concept in LAB_CONCEPT_LABELS]
    + [f"ehr_{concept}_available" for concept in LAB_CONCEPT_LABELS]
)

LAB_USE_COLUMNS = [
    "subject_id",
    "hadm_id",
    "itemid",
    "charttime",
    "storetime",
    "valuenum",
    "valueuom",
]


def _resolve_csv(root: Path, stem: str) -> Path:
    candidates = (root / f"{stem}.csv.gz", root / f"{stem}.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing MIMIC table {stem}. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def inspect_mimic_tables(config: EHRGlucoseConfig) -> dict[str, pd.DataFrame]:
    return {
        "patients": pd.read_csv(_resolve_csv(config.raw_dir, "patients"), nrows=5),
        "admissions": pd.read_csv(_resolve_csv(config.raw_dir, "admissions"), nrows=5),
        "d_labitems": pd.read_csv(_resolve_csv(config.raw_dir, "d_labitems"), nrows=5),
        "labevents": pd.read_csv(_resolve_csv(config.raw_dir, "labevents"), nrows=5),
    }


def _read_selected_labs(
    path: Path, selected_itemids: set[int], *, chunk_size: int = 500_000
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    reader = pd.read_csv(path, usecols=LAB_USE_COLUMNS, chunksize=chunk_size)
    for chunk in reader:
        keep = chunk.loc[
            chunk["itemid"].isin(selected_itemids)
            & chunk["valuenum"].notna()
            & chunk["hadm_id"].notna()
            & chunk["storetime"].notna()
            & chunk["charttime"].notna()
        ].copy()
        if not keep.empty:
            selected.append(keep)
    if not selected:
        raise ValueError("No selected numeric glucose/EHR laboratory events were found.")
    result = pd.concat(selected, ignore_index=True)
    result["hadm_id"] = result["hadm_id"].astype("int64")
    result["charttime"] = pd.to_datetime(result["charttime"], errors="coerce")
    result["storetime"] = pd.to_datetime(result["storetime"], errors="coerce")
    result["valuenum"] = pd.to_numeric(result["valuenum"], errors="coerce")
    return result.dropna(subset=["charttime", "storetime", "valuenum"])


def load_mimic_tables(
    config: EHRGlucoseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = pd.read_csv(_resolve_csv(config.raw_dir, "patients"))
    admissions = pd.read_csv(
        _resolve_csv(config.raw_dir, "admissions"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
    )
    items = pd.read_csv(_resolve_csv(config.raw_dir, "d_labitems"))
    items["label_normalized"] = items["label"].str.strip().str.lower()

    glucose_labels = {"glucose", "glucose, whole blood"}
    concept_labels = {label for labels in LAB_CONCEPT_LABELS.values() for label in labels}
    selected_items = items.loc[
        (items["fluid"].str.lower() == "blood")
        & items["label_normalized"].isin(glucose_labels | concept_labels)
    ].copy()
    selected_itemids = set(selected_items["itemid"].astype(int))
    labs = _read_selected_labs(_resolve_csv(config.raw_dir, "labevents"), selected_itemids)
    labs = labs.merge(
        selected_items[["itemid", "label_normalized"]], on="itemid", how="inner", validate="many_to_one"
    )

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    return patients, admissions, items, labs


def _concept_for_label(label: str) -> str | None:
    for concept, labels in LAB_CONCEPT_LABELS.items():
        if label in labels:
            return concept
    return None


def _deduplicate_glucose(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse same-chart-time values only when every value has become available."""
    return (
        events.groupby(["subject_id", "hadm_id", "charttime"], as_index=False)
        .agg(storetime=("storetime", "max"), valuenum=("valuenum", "median"))
        .sort_values(["subject_id", "hadm_id", "storetime", "charttime"], ignore_index=True)
    )


def _latest_lab(
    lookup: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    hadm_id: int,
    concept: str,
    anchor_time: pd.Timestamp,
    lookback_hours: float,
) -> tuple[float, float, int, pd.Timestamp | None]:
    series = lookup.get((hadm_id, concept))
    if series is None:
        return float("nan"), float("nan"), 0, None
    times, values = series
    anchor64 = np.datetime64(anchor_time)
    index = int(np.searchsorted(times, anchor64, side="right") - 1)
    if index < 0:
        return float("nan"), float("nan"), 0, None
    age_hours = float((anchor64 - times[index]) / np.timedelta64(1, "h"))
    if age_hours < 0 or age_hours > lookback_hours:
        return float("nan"), float("nan"), 0, None
    return float(values[index]), age_hours, 1, pd.Timestamp(times[index])


def build_glucose_forecast_table(
    config: EHRGlucoseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal anchors for a fixed-horizon hospital glucose forecast."""
    patients, admissions, items, labs = load_mimic_tables(config)
    glucose_mask = labs["label_normalized"].isin({"glucose", "glucose, whole blood"})
    glucose = labs.loc[glucose_mask].copy()
    glucose = glucose.loc[
        glucose["valueuom"].str.lower().eq("mg/dl")
        & glucose["valuenum"].between(
            config.minimum_glucose_mg_dl, config.maximum_glucose_mg_dl
        )
    ]
    glucose = _deduplicate_glucose(glucose)

    context = labs.loc[~glucose_mask].copy()
    context["concept"] = context["label_normalized"].map(_concept_for_label)
    context = context.dropna(subset=["concept"])
    context["unit_normalized"] = context["valueuom"].astype(str).str.strip().str.lower()
    context = context.loc[
        [
            unit in LAB_CONCEPT_UNITS[str(concept)]
            for concept, unit in zip(
                context["concept"], context["unit_normalized"], strict=True
            )
        ]
    ].copy()
    context_lookup: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for (hadm_id, concept), group in context.groupby(["hadm_id", "concept"], sort=False):
        group = group.sort_values("storetime")
        context_lookup[(int(hadm_id), str(concept))] = (
            group["storetime"].to_numpy(dtype="datetime64[ns]"),
            group["valuenum"].to_numpy(dtype=float),
        )

    patient_lookup = patients.set_index("subject_id").to_dict("index")
    admission_lookup = admissions.set_index("hadm_id").to_dict("index")
    rows: list[dict[str, object]] = []
    rejected = {
        "outside_admission": 0,
        "insufficient_history": 0,
        "no_target_in_horizon": 0,
    }

    lower_horizon = config.forecast_horizon_hours - config.forecast_tolerance_hours
    upper_horizon = config.forecast_horizon_hours + config.forecast_tolerance_hours

    for (subject_id, hadm_id), group in glucose.groupby(["subject_id", "hadm_id"], sort=False):
        group = group.sort_values("storetime").reset_index(drop=True)
        admission = admission_lookup.get(int(hadm_id))
        patient = patient_lookup.get(int(subject_id))
        if admission is None or patient is None:
            continue
        admit_time = pd.Timestamp(admission["admittime"])
        discharge_time = pd.Timestamp(admission["dischtime"])

        for anchor_index, anchor in group.iterrows():
            anchor_time = pd.Timestamp(anchor["storetime"])
            if not (admit_time <= anchor_time <= discharge_time):
                rejected["outside_admission"] += 1
                continue

            history = group.loc[
                (group["storetime"] <= anchor_time)
                & (
                    group["storetime"]
                    >= anchor_time - timedelta(hours=float(config.lookback_hours))
                )
            ].copy()
            if len(history) < config.min_glucose_history:
                rejected["insufficient_history"] += 1
                continue

            future = group.loc[
                (group.index > anchor_index)
                & (group["storetime"] > anchor_time)
            ].copy()
            future["delta_hours"] = (
                future["charttime"] - anchor_time
            ).dt.total_seconds() / 3600.0
            future = future.loc[future["delta_hours"].between(lower_horizon, upper_horizon)]
            if future.empty:
                rejected["no_target_in_horizon"] += 1
                continue
            target_index = (future["delta_hours"] - config.forecast_horizon_hours).abs().idxmin()
            target = future.loc[target_index]

            history = history.sort_values("storetime")
            history_values = history["valuenum"].to_numpy(dtype=float)
            history_times = history["storetime"].to_numpy(dtype="datetime64[ns]")
            elapsed = float((history_times[-1] - history_times[0]) / np.timedelta64(1, "h"))
            slope = (
                float((history_values[-1] - history_values[0]) / elapsed)
                if elapsed > 0
                else 0.0
            )
            previous = history.iloc[-2]
            age_years = float(patient["anchor_age"] + (anchor_time.year - patient["anchor_year"]))

            row: dict[str, object] = {
                "patient_id": f"mimic_{int(subject_id)}",
                "subject_id": int(subject_id),
                "hadm_id": int(hadm_id),
                "anchor_time": anchor_time,
                "target_charttime": pd.Timestamp(target["charttime"]),
                "target_storetime": pd.Timestamp(target["storetime"]),
                "target_delta_hours": float(target["delta_hours"]),
                "target_glucose_mg_dl": float(target["valuenum"]),
                "target_hyperglycemia": int(
                    float(target["valuenum"]) > config.hyperglycemia_threshold_mg_dl
                ),
                "age_years": age_years,
                "sex_female": int(str(patient["gender"]).upper() == "F"),
                "hours_since_admission": float((anchor_time - admit_time).total_seconds() / 3600.0),
                "current_glucose_mg_dl": float(history_values[-1]),
                "previous_glucose_mg_dl": float(previous["valuenum"]),
                "hours_since_previous_glucose": float(
                    (anchor_time - pd.Timestamp(previous["storetime"])).total_seconds() / 3600.0
                ),
                "glucose_mean_24h": float(np.mean(history_values)),
                "glucose_std_24h": float(np.std(history_values)),
                "glucose_min_24h": float(np.min(history_values)),
                "glucose_max_24h": float(np.max(history_values)),
                "glucose_slope_24h_mg_dl_per_hour": slope,
                "glucose_count_24h": int(len(history_values)),
                "target_after_anchor_verified": int(
                    pd.Timestamp(target["charttime"]) > anchor_time
                    and pd.Timestamp(target["storetime"]) > anchor_time
                ),
            }
            latest_feature_storetime = pd.Timestamp(history["storetime"].max())
            for concept in LAB_CONCEPT_LABELS:
                value, age_hours, available, event_time = _latest_lab(
                    context_lookup,
                    int(hadm_id),
                    concept,
                    anchor_time,
                    config.lookback_hours,
                )
                row[f"ehr_{concept}_last"] = value
                row[f"ehr_{concept}_age_hours"] = age_hours
                row[f"ehr_{concept}_available"] = available
                if event_time is not None:
                    latest_feature_storetime = max(latest_feature_storetime, event_time)
            row["max_feature_storetime"] = latest_feature_storetime
            row["feature_cutoff_verified"] = int(latest_feature_storetime <= anchor_time)
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No causal fixed-horizon glucose forecast anchors could be built.")
    if not frame["feature_cutoff_verified"].eq(1).all():
        raise AssertionError("A feature event occurs after at least one prediction anchor.")
    if not frame["target_after_anchor_verified"].eq(1).all():
        raise AssertionError("At least one target is not strictly after its prediction anchor.")
    frame = frame.sort_values(["patient_id", "anchor_time"], ignore_index=True)

    audit = pd.DataFrame(
        [
            {"measure": "selected_lab_rows", "value": len(labs)},
            {"measure": "glucose_rows_after_quality_filter", "value": len(glucose)},
            {"measure": "forecast_anchors", "value": len(frame)},
            {"measure": "forecast_patients", "value": frame["patient_id"].nunique()},
            {"measure": "hyperglycemia_targets", "value": int(frame["target_hyperglycemia"].sum())},
            *({"measure": key, "value": value} for key, value in rejected.items()),
        ]
    )
    return frame, audit
