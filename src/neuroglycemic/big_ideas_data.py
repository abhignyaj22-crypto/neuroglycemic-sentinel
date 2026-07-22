"""Causal wearable-to-CGM windows for the Big Ideas longitudinal dataset.

The adapter never interpolates future glucose and never exposes current CGM to
the neural feature registry.  Current CGM is retained only as the persistence
baseline required to establish whether a learned model adds value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import math
import numpy as np
import pandas as pd

from .neural_dataset import target_column, target_time_column


BIG_IDEAS_COHORT_ID = "big-ideas-glycemic-wearable-1.1.3"
BIG_IDEAS_WEARABLE_FEATURES = (
    "wearable_heart_rate_mean_bpm",
    "wearable_heart_rate_std_bpm",
    "wearable_heart_rate_min_bpm",
    "wearable_heart_rate_max_bpm",
    "wearable_heart_rate_slope_bpm_per_minute",
    "wearable_ibi_mean_seconds",
    "wearable_ibi_std_seconds",
    "wearable_ibi_rmssd_seconds",
    "wearable_meal_carbohydrate_g_240m",
    "wearable_minutes_since_meal",
)


@dataclass(frozen=True)
class BigIdeasBuildConfig:
    source_timezone: str
    horizons_minutes: tuple[int, ...] = (30, 60)
    lookback_minutes: int = 120
    stride_minutes: int = 15
    target_tolerance_minutes: float = 5.0
    minimum_hr_minutes: int = 12
    clock_uncertainty_ms: float = 60_000.0
    emit_meal_lag_basis: bool = True
    meal_lag_centers_minutes: tuple[float, ...] = (
        0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0
    )
    meal_lag_width_minutes: float = 20.0

    def __post_init__(self) -> None:
        if not self.source_timezone.strip():
            raise ValueError("source_timezone is required for timezone-naive device files.")
        if not self.horizons_minutes or any(value <= 0 for value in self.horizons_minutes):
            raise ValueError("At least one positive forecast horizon is required.")
        if len(set(self.horizons_minutes)) != len(self.horizons_minutes):
            raise ValueError("Forecast horizons must be unique.")
        if self.lookback_minutes <= 0 or self.stride_minutes <= 0:
            raise ValueError("lookback_minutes and stride_minutes must be positive.")
        if not math.isfinite(self.target_tolerance_minutes) or self.target_tolerance_minutes <= 0:
            raise ValueError("target_tolerance_minutes must be finite and positive.")
        if self.minimum_hr_minutes <= 0:
            raise ValueError("minimum_hr_minutes must be positive.")
        if not math.isfinite(self.clock_uncertainty_ms) or self.clock_uncertainty_ms < 0:
            raise ValueError("clock_uncertainty_ms must be finite and non-negative.")
        if self.emit_meal_lag_basis:
            if not self.meal_lag_centers_minutes or any(
                not math.isfinite(value) or value < 0
                for value in self.meal_lag_centers_minutes
            ):
                raise ValueError("Meal-lag centers must be finite and non-negative.")
            if (
                not math.isfinite(self.meal_lag_width_minutes)
                or self.meal_lag_width_minutes <= 0
            ):
                raise ValueError("meal_lag_width_minutes must be finite and positive.")


@dataclass(frozen=True)
class BigIdeasPatientFiles:
    patient_id: str
    dexcom: Path
    heart_rate: Path
    ibi: Path | None
    food_log: Path | None


@dataclass(frozen=True)
class BigIdeasPatientAudit:
    patient_id: str
    cgm_rows: int
    heart_rate_rows: int
    ibi_rows: int
    meal_rows: int
    aligned_windows: int
    start_utc: str | None
    stop_utc: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def big_ideas_feature_registry() -> dict[str, tuple[str, ...]]:
    return {"wearable": BIG_IDEAS_WEARABLE_FEATURES}


def discover_big_ideas_patients(root: Path) -> tuple[BigIdeasPatientFiles, ...]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    discovered: list[BigIdeasPatientFiles] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        patient = directory.name.strip()
        dexcom = directory / f"Dexcom_{patient}.csv"
        heart_rate = directory / f"HR_{patient}.csv"
        if not dexcom.is_file() or not heart_rate.is_file():
            continue
        ibi = directory / f"IBI_{patient}.csv"
        food = directory / f"Food_Log_{patient}.csv"
        discovered.append(
            BigIdeasPatientFiles(
                patient_id=patient,
                dexcom=dexcom,
                heart_rate=heart_rate,
                ibi=ibi if ibi.is_file() else None,
                food_log=food if food.is_file() else None,
            )
        )
    if not discovered:
        raise FileNotFoundError(
            f"No participant directories with Dexcom_<id>.csv and HR_<id>.csv found in {root}."
        )
    return tuple(discovered)


def _to_utc(values: pd.Series, timezone_name: str, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if parsed.dt.tz is None:
        try:
            parsed = parsed.dt.tz_localize(
                timezone_name, ambiguous="infer", nonexistent="NaT"
            )
        except Exception:
            # If sequence-based inference cannot resolve a fallback transition,
            # reject only those ambiguous rows rather than inventing an offset.
            try:
                parsed = parsed.dt.tz_localize(
                    timezone_name, ambiguous="NaT", nonexistent="NaT"
                )
            except Exception as exc:
                raise ValueError(f"Invalid source timezone {timezone_name!r}.") from exc
    return parsed.dt.tz_convert("UTC")


def _read_patient(
    files: BigIdeasPatientFiles, *, source_timezone: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cgm = pd.read_csv(files.dexcom, low_memory=False)
    cgm.columns = [str(value).lstrip("\ufeff").strip() for value in cgm.columns]
    timestamp_name = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    glucose_name = "Glucose Value (mg/dL)"
    required = {timestamp_name, glucose_name, "Event Type"}
    if not required.issubset(cgm.columns):
        raise ValueError(f"{files.dexcom.name} is missing {sorted(required - set(cgm))}.")
    cgm = cgm.loc[cgm["Event Type"].astype(str).str.strip().eq("EGV")].copy()
    cgm = cgm.loc[cgm[timestamp_name].notna(), [timestamp_name, glucose_name]]
    cgm["time"] = _to_utc(cgm[timestamp_name], source_timezone, name="Dexcom timestamp")
    cgm["glucose_mg_dl"] = pd.to_numeric(cgm[glucose_name], errors="coerce")
    cgm = cgm.loc[cgm["glucose_mg_dl"].between(20.0, 600.0), ["time", "glucose_mg_dl"]]
    cgm = cgm.groupby("time", as_index=False)["glucose_mg_dl"].median().sort_values("time")

    heart_rate = pd.read_csv(files.heart_rate)
    heart_rate.columns = [str(value).lstrip("\ufeff").strip().lower() for value in heart_rate]
    if not {"datetime", "hr"}.issubset(heart_rate.columns):
        raise ValueError(f"{files.heart_rate.name} requires datetime and hr columns.")
    heart_rate["time"] = _to_utc(
        heart_rate["datetime"], source_timezone, name="heart-rate timestamp"
    )
    heart_rate["value"] = pd.to_numeric(heart_rate["hr"], errors="coerce")
    heart_rate = heart_rate.loc[
        heart_rate["value"].between(25.0, 240.0), ["time", "value"]
    ].sort_values("time")

    ibi = pd.DataFrame(columns=["time", "value"])
    if files.ibi is not None:
        ibi = pd.read_csv(files.ibi)
        ibi.columns = [str(value).lstrip("\ufeff").strip().lower() for value in ibi]
        if not {"datetime", "ibi"}.issubset(ibi.columns):
            raise ValueError(f"{files.ibi.name} requires datetime and ibi columns.")
        ibi["time"] = _to_utc(ibi["datetime"], source_timezone, name="IBI timestamp")
        ibi["value"] = pd.to_numeric(ibi["ibi"], errors="coerce")
        ibi = ibi.loc[ibi["value"].between(0.25, 2.5), ["time", "value"]].sort_values("time")

    meals = pd.DataFrame(columns=["time", "total_carb"])
    if files.food_log is not None:
        meals = pd.read_csv(files.food_log)
        meals.columns = [str(value).lstrip("\ufeff").strip().lower() for value in meals]
        if {"time_begin", "total_carb"}.issubset(meals.columns):
            meals["time"] = _to_utc(
                meals["time_begin"], source_timezone, name="food-log timestamp"
            )
            meals["total_carb"] = pd.to_numeric(meals["total_carb"], errors="coerce")
            meals = meals.loc[
                meals["total_carb"].ge(0.0), ["time", "total_carb"]
            ].sort_values("time")
        else:
            meals = pd.DataFrame(columns=["time", "total_carb"])
    if cgm.empty or heart_rate.empty:
        raise ValueError(f"Patient {files.patient_id} has no valid CGM or HR rows.")
    return cgm, heart_rate, ibi, meals


def _slope(values: np.ndarray, times: pd.Series) -> float:
    if len(values) < 2:
        return float("nan")
    elapsed = (times - times.iloc[0]).dt.total_seconds().to_numpy(float) / 60.0
    if np.ptp(elapsed) <= 0:
        return float("nan")
    return float(np.polyfit(elapsed, values, 1)[0])


def _nearest(
    frame: pd.DataFrame,
    expected: pd.Timestamp,
    *,
    tolerance_minutes: float,
    require_future_of: pd.Timestamp | None = None,
) -> tuple[float, pd.Timestamp] | None:
    if frame.empty:
        return None
    deltas = (frame["time"] - expected).abs().dt.total_seconds().to_numpy(float) / 60.0
    index = int(np.argmin(deltas))
    observed_time = pd.Timestamp(frame.iloc[index]["time"])
    if deltas[index] > tolerance_minutes:
        return None
    if require_future_of is not None and observed_time <= require_future_of:
        return None
    return float(frame.iloc[index]["glucose_mg_dl"]), observed_time


def _attach_meal_lag_basis(
    frame: pd.DataFrame,
    meals: pd.DataFrame,
    patient_id: str,
    config: BigIdeasBuildConfig,
) -> pd.DataFrame:
    """Attach the causal carbohydrate lag basis consumed by the response kernel.

    The columns (``meal_lag_carbohydrate_g_<center>m``) are raw, amount-weighted
    radial-basis values over event age — the same contract the LSL window
    builder emits — so the learned response kernel (contribution 4) can engage
    on Big IDEAS without any training-side changes.  They are *not* registry
    features and never enter the encoder inputs.

    Logging caveat: Big IDEAS food logs carry only ``time_begin``, so the event
    time doubles as the availability time.  This assumes meals were logged
    when eaten; a retrospective logging habit would soften, not fabricate, the
    learned kernel.
    """

    if not config.emit_meal_lag_basis or frame.empty:
        return frame
    from .meal_context import MealLagSpec, build_causal_meal_lag_features

    spec = MealLagSpec(
        centers_minutes=tuple(config.meal_lag_centers_minutes),
        width_minutes=float(config.meal_lag_width_minutes),
        lookback_minutes=240.0,
        value_columns=("carbohydrate_g",),
    )
    events = pd.DataFrame(
        {
            "patient_id": pd.Series([], dtype=str),
            "event_time": pd.Series([], dtype="datetime64[ns, UTC]"),
            "available_time": pd.Series([], dtype="datetime64[ns, UTC]"),
            "carbohydrate_g": pd.Series([], dtype=float),
        }
    )
    if not meals.empty:
        events = pd.DataFrame(
            {
                "patient_id": patient_id,
                "event_time": meals["time"],
                "available_time": meals["time"],
                "carbohydrate_g": meals["total_carb"],
            }
        )
    lag = build_causal_meal_lag_features(
        frame[["patient_id", "anchor_time"]].copy(), events, spec=spec
    )
    lag_columns = [column for column in lag.columns if column.startswith("meal_lag_")]
    merged = frame.merge(
        lag[["patient_id", "anchor_time", *lag_columns]],
        on=["patient_id", "anchor_time"],
        how="left",
        validate="one_to_one",
    )
    merged[lag_columns] = merged[lag_columns].fillna(0.0)
    return merged


def build_big_ideas_patient_windows(
    files: BigIdeasPatientFiles, *, config: BigIdeasBuildConfig
) -> tuple[pd.DataFrame, BigIdeasPatientAudit]:
    cgm, heart_rate, ibi, meals = _read_patient(
        files, source_timezone=config.source_timezone
    )
    start = max(
        heart_rate["time"].min() + pd.Timedelta(minutes=config.lookback_minutes),
        cgm["time"].min(),
    ).ceil(f"{config.stride_minutes}min")
    stop = min(
        heart_rate["time"].max(),
        cgm["time"].max() - pd.Timedelta(minutes=max(config.horizons_minutes)),
    ).floor(f"{config.stride_minutes}min")
    anchors = (
        pd.date_range(start, stop, freq=f"{config.stride_minutes}min")
        if stop >= start
        else pd.DatetimeIndex([])
    )
    rows: list[dict[str, object]] = []
    for anchor in anchors:
        window_start = anchor - pd.Timedelta(minutes=config.lookback_minutes)
        hr_window = heart_rate.loc[
            heart_rate["time"].gt(window_start) & heart_rate["time"].le(anchor)
        ]
        hr_minutes = int(hr_window["time"].dt.floor("min").nunique())
        if hr_minutes < config.minimum_hr_minutes:
            continue
        ibi_window = ibi.loc[ibi["time"].gt(window_start) & ibi["time"].le(anchor)]
        hr_values = hr_window["value"].to_numpy(float)
        ibi_values = ibi_window["value"].to_numpy(float)
        latest = max(
            hr_window["time"].max(),
            ibi_window["time"].max() if not ibi_window.empty else hr_window["time"].max(),
        )
        meal_window = meals.loc[
            meals["time"].gt(anchor - pd.Timedelta(hours=4))
            & meals["time"].le(anchor)
        ]
        current = _nearest(
            cgm.loc[cgm["time"].le(anchor)],
            anchor,
            tolerance_minutes=15.0,
        )
        row: dict[str, object] = {
            "patient_id": files.patient_id,
            "cohort_id": BIG_IDEAS_COHORT_ID,
            "anchor_time": anchor,
            "wearable_available": True,
            "wearable_quality": min(1.0, hr_minutes / config.lookback_minutes),
            "wearable_staleness_minutes": max(
                0.0, (anchor - latest).total_seconds() / 60.0
            ),
            "wearable_available_time": latest,
            "wearable_clock_uncertainty_ms": config.clock_uncertainty_ms,
            "wearable_patient_id": files.patient_id,
            "wearable_cohort_id": BIG_IDEAS_COHORT_ID,
            "wearable_anchor_time": anchor,
            "reference_current_glucose_mg_dl": current[0] if current else np.nan,
            "wearable_heart_rate_mean_bpm": float(np.mean(hr_values)),
            "wearable_heart_rate_std_bpm": float(np.std(hr_values)),
            "wearable_heart_rate_min_bpm": float(np.min(hr_values)),
            "wearable_heart_rate_max_bpm": float(np.max(hr_values)),
            "wearable_heart_rate_slope_bpm_per_minute": _slope(
                hr_values, hr_window["time"]
            ),
            "wearable_ibi_mean_seconds": (
                float(np.mean(ibi_values)) if len(ibi_values) else np.nan
            ),
            "wearable_ibi_std_seconds": (
                float(np.std(ibi_values)) if len(ibi_values) else np.nan
            ),
            "wearable_ibi_rmssd_seconds": (
                float(np.sqrt(np.mean(np.diff(ibi_values) ** 2)))
                if len(ibi_values) >= 2
                else np.nan
            ),
            "wearable_meal_carbohydrate_g_240m": (
                float(meal_window["total_carb"].sum())
                if not meal_window.empty
                else 0.0
            ),
            "wearable_minutes_since_meal": (
                float((anchor - meal_window["time"].max()).total_seconds() / 60.0)
                if not meal_window.empty
                else np.nan
            ),
        }
        target_count = 0
        for horizon in config.horizons_minutes:
            target = _nearest(
                cgm,
                anchor + pd.Timedelta(minutes=horizon),
                tolerance_minutes=config.target_tolerance_minutes,
                require_future_of=anchor,
            )
            row[target_column(horizon)] = target[0] if target else np.nan
            row[target_time_column(horizon)] = target[1] if target else pd.NaT
            target_count += int(target is not None)
        if target_count:
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame = _attach_meal_lag_basis(frame, meals, files.patient_id, config)
    audit = BigIdeasPatientAudit(
        patient_id=files.patient_id,
        cgm_rows=len(cgm),
        heart_rate_rows=len(heart_rate),
        ibi_rows=len(ibi),
        meal_rows=len(meals),
        aligned_windows=len(frame),
        start_utc=str(cgm["time"].min()) if not cgm.empty else None,
        stop_utc=str(cgm["time"].max()) if not cgm.empty else None,
    )
    return frame, audit


def build_big_ideas_dataset(
    patients: Sequence[BigIdeasPatientFiles], *, config: BigIdeasBuildConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for patient in patients:
        frame, audit = build_big_ideas_patient_windows(patient, config=config)
        audits.append(audit.as_dict())
        if not frame.empty:
            windows.append(frame)
    if not windows:
        raise ValueError("Big Ideas preparation produced no causal forecast windows.")
    combined = pd.concat(windows, ignore_index=True).sort_values(
        ["patient_id", "anchor_time"], ignore_index=True
    )
    if combined.duplicated(["patient_id", "anchor_time"]).any():
        raise ValueError("Big Ideas preparation produced duplicate patient-time windows.")
    return combined, pd.DataFrame(audits)


def big_ideas_build_manifest(
    frame: pd.DataFrame,
    *,
    source_root: Path,
    config: BigIdeasBuildConfig,
) -> Mapping[str, object]:
    return {
        "schema_version": "neuroglycemic-big-ideas-builder-v1",
        "cohort_id": BIG_IDEAS_COHORT_ID,
        "source_root": str(Path(source_root).resolve()),
        "patients": int(frame["patient_id"].nunique()),
        "windows": int(len(frame)),
        "horizons_minutes": list(config.horizons_minutes),
        "source_timezone": config.source_timezone,
        "clock_uncertainty_ms": config.clock_uncertainty_ms,
        "feature_registry": {
            name: list(values) for name, values in big_ideas_feature_registry().items()
        },
        "input_cgm_used_as_feature": False,
        "reference_current_glucose_retained_for_persistence_only": True,
    }