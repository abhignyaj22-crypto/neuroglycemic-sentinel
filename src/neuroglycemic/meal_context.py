"""Causal meal/insulin timing features for neural glucose prediction, representing event
age with smooth radial-basis functions for magnitude, sign, and timing of associations.
"""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_EVENT_VALUES = (
    "carbohydrate_g",
    "protein_g",
    "fat_g",
    "fiber_g",
    "bolus_insulin_units",
)


@dataclass(frozen=True)
class MealLagSpec:
    """Small, explicit temporal basis for previously available meal events."""

    centers_minutes: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0)
    width_minutes: float = 20.0
    lookback_minutes: float = 240.0
    value_columns: tuple[str, ...] = DEFAULT_EVENT_VALUES

    def __post_init__(self) -> None:
        if not self.centers_minutes:
            raise ValueError("At least one meal-lag center is required.")
        if any(not math.isfinite(value) or value < 0 for value in self.centers_minutes):
            raise ValueError("Meal-lag centers must be finite and non-negative.")
        if len(set(self.centers_minutes)) != len(self.centers_minutes):
            raise ValueError("Meal-lag centers must be unique.")
        if not math.isfinite(self.width_minutes) or self.width_minutes <= 0:
            raise ValueError("width_minutes must be finite and positive.")
        if not math.isfinite(self.lookback_minutes) or self.lookback_minutes <= 0:
            raise ValueError("lookback_minutes must be finite and positive.")
        if max(self.centers_minutes) > self.lookback_minutes:
            raise ValueError("Meal-lag centers must fall inside the lookback window.")
        if not self.value_columns or any(not value.strip() for value in self.value_columns):
            raise ValueError("At least one non-empty meal-event value column is required.")


def meal_lag_feature_names(spec: MealLagSpec = MealLagSpec()) -> tuple[str, ...]:
    lagged = tuple(
        f"meal_lag_{column}_{center:g}m"
        for column in spec.value_columns
        for center in spec.centers_minutes
    )
    return (*lagged, "meal_event_count", "minutes_since_last_meal", "meal_context_available")


def _as_utc(values: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{name} contains missing or invalid timestamps.")
    return parsed


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def build_causal_meal_lag_features(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    *,
    spec: MealLagSpec = MealLagSpec(),
) -> pd.DataFrame:
    """Return one causal meal-context row for every patient-time anchor.

    ``event_time`` is when food/insulin occurred. ``available_time`` is when the
    system learned about it.  Both must be no later than the prediction anchor;
    this prevents retrospective food logging from leaking into an earlier
    prediction.  All amounts must use the units named in ``value_columns``.
    """

    _validate_columns(anchors, ("patient_id", "anchor_time"), name="anchors")
    _validate_columns(
        events,
        ("patient_id", "event_time", "available_time", *spec.value_columns),
        name="events",
    )
    anchor_frame = anchors[["patient_id", "anchor_time"]].copy()
    anchor_frame["anchor_time"] = _as_utc(anchor_frame["anchor_time"], name="anchor_time")
    event_frame = events[
        ["patient_id", "event_time", "available_time", *spec.value_columns]
    ].copy()
    event_frame["event_time"] = _as_utc(event_frame["event_time"], name="event_time")
    event_frame["available_time"] = _as_utc(
        event_frame["available_time"], name="available_time"
    )
    for column in spec.value_columns:
        event_frame[column] = pd.to_numeric(event_frame[column], errors="coerce")
        if (event_frame[column].dropna() < 0).any():
            raise ValueError(f"{column} must be non-negative when observed.")

    rows: list[dict[str, object]] = []
    width = float(spec.width_minutes)
    for anchor_index, anchor in anchor_frame.iterrows():
        same_patient = event_frame["patient_id"].eq(anchor["patient_id"])
        known = event_frame["available_time"].le(anchor["anchor_time"])
        occurred = event_frame["event_time"].le(anchor["anchor_time"])
        selected = event_frame.loc[same_patient & known & occurred].copy()
        selected["age_minutes"] = (
            anchor["anchor_time"] - selected["event_time"]
        ).dt.total_seconds() / 60.0
        selected = selected.loc[
            selected["age_minutes"].between(0.0, spec.lookback_minutes, inclusive="both")
        ]

        result: dict[str, object] = {
            "anchor_index": anchor_index,
            "patient_id": anchor["patient_id"],
            "anchor_time": anchor["anchor_time"],
        }
        ages = selected["age_minutes"].to_numpy(dtype=float)
        for column in spec.value_columns:
            amounts = selected[column].fillna(0.0).to_numpy(dtype=float)
            for center in spec.centers_minutes:
                basis = np.exp(-0.5 * ((ages - float(center)) / width) ** 2)
                result[f"meal_lag_{column}_{center:g}m"] = float(np.sum(amounts * basis))
        meal_rows = selected.loc[selected["carbohydrate_g"].fillna(0.0).gt(0.0)]
        result["meal_event_count"] = int(len(meal_rows))
        result["minutes_since_last_meal"] = (
            float(meal_rows["age_minutes"].min()) if not meal_rows.empty else np.nan
        )
        result["meal_context_available"] = int(not selected.empty)
        rows.append(result)

    result = pd.DataFrame(rows).set_index("anchor_index").reindex(anchor_frame.index)
    ordered = ["patient_id", "anchor_time", *meal_lag_feature_names(spec)]
    return result[ordered].reset_index(drop=True)
