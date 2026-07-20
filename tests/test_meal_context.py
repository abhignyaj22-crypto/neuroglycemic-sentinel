import numpy as np
import pandas as pd
import pytest

from src.neuroglycemic.meal_context import (
    MealLagSpec,
    build_causal_meal_lag_features,
    meal_lag_feature_names,
)


def _event(patient, event_time, available_time, carbs):
    return {
        "patient_id": patient,
        "event_time": event_time,
        "available_time": available_time,
        "carbohydrate_g": carbs,
        "protein_g": 0.0,
        "fat_g": 0.0,
        "fiber_g": 0.0,
        "bolus_insulin_units": 0.0,
    }


def test_meal_age_changes_continuous_lag_features():
    anchors = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "anchor_time": ["2026-01-01T12:30:00Z", "2026-01-01T13:30:00Z"],
        }
    )
    events = pd.DataFrame(
        [_event("p1", "2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z", 50.0)]
    )
    result = build_causal_meal_lag_features(anchors, events)

    assert result.loc[0, "meal_lag_carbohydrate_g_30m"] > result.loc[1, "meal_lag_carbohydrate_g_30m"]
    assert result.loc[1, "meal_lag_carbohydrate_g_90m"] > result.loc[0, "meal_lag_carbohydrate_g_90m"]
    assert result["minutes_since_last_meal"].tolist() == [30.0, 90.0]


def test_future_and_retrospectively_logged_events_are_excluded():
    anchors = pd.DataFrame(
        {"patient_id": ["p1"], "anchor_time": ["2026-01-01T12:00:00Z"]}
    )
    events = pd.DataFrame(
        [
            _event("p1", "2026-01-01T11:30:00Z", "2026-01-01T12:30:00Z", 30.0),
            _event("p1", "2026-01-01T12:30:00Z", "2026-01-01T11:00:00Z", 40.0),
            _event("different", "2026-01-01T11:30:00Z", "2026-01-01T11:30:00Z", 60.0),
        ]
    )
    result = build_causal_meal_lag_features(anchors, events)

    assert result.loc[0, "meal_context_available"] == 0
    assert result.loc[0, "meal_event_count"] == 0
    assert np.isnan(result.loc[0, "minutes_since_last_meal"])
    lag_values = result.loc[0, [name for name in meal_lag_feature_names() if name.startswith("meal_lag_")]]
    assert np.allclose(lag_values.to_numpy(dtype=float), 0.0)


def test_invalid_amounts_and_temporal_basis_are_rejected():
    with pytest.raises(ValueError, match="inside the lookback"):
        MealLagSpec(centers_minutes=(0.0, 300.0), lookback_minutes=240.0)

    anchors = pd.DataFrame(
        {"patient_id": ["p1"], "anchor_time": ["2026-01-01T12:00:00Z"]}
    )
    events = pd.DataFrame(
        [_event("p1", "2026-01-01T11:30:00Z", "2026-01-01T11:30:00Z", -1.0)]
    )
    with pytest.raises(ValueError, match="carbohydrate_g"):
        build_causal_meal_lag_features(anchors, events)
