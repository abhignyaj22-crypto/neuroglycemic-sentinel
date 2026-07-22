"""XDF-to-window contract tests; generated signals are never used for research training."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.neuroglycemic.lsl_windowing import (
    CGMReference,
    EEGSource,
    LSLWindowConfig,
    LSL_EEG_FEATURES,
    WearableSummary,
    build_lsl_glucose_windows,
)
from src.neuroglycemic.neural_dataset import load_aligned_window_frame


def _recorded_shape_fixture():
    eeg_time = np.arange(0.0, 601.0, 0.01)
    eeg = pd.DataFrame(
        {
            "lsl_timestamp": eeg_time,
            "AF3": 20.0 * np.sin(2.0 * np.pi * 10.0 * eeg_time),
            "AF4": 15.0 * np.sin(2.0 * np.pi * 10.0 * eeg_time + 0.2),
        }
    )
    pulse_time = np.arange(0.0, 601.0, 1.0)
    pulse = pd.DataFrame(
        {
            "lsl_timestamp": pulse_time,
            "heart_rate": 72.0 + np.sin(pulse_time / 20.0),
        }
    )
    cgm_time = np.arange(0.0, 601.0, 30.0)
    cgm = pd.DataFrame(
        {"lsl_timestamp": cgm_time, "glucose": 100.0 + cgm_time / 60.0}
    )
    audit = pd.DataFrame(
        [
            {"source_id": "emotiv-1", "nominal_rate_hz": 100.0},
            {"source_id": "pulse-1", "nominal_rate_hz": 1.0},
            {"source_id": "cgm-1", "nominal_rate_hz": 1.0 / 30.0},
        ]
    )
    frames = {"emotiv-1": eeg, "pulse-1": pulse, "cgm-1": cgm}
    return audit, frames


def _config() -> LSLWindowConfig:
    return LSLWindowConfig(
        patient_id="P001",
        cohort_id="dvxr-prospective",
        session_id="S001",
        session_start_utc="2026-01-01T00:00:00+00:00",
        horizons_minutes=(1, 2),
        lookback_seconds=30,
        stride_seconds=30,
        target_tolerance_minutes=0.1,
        minimum_eeg_coverage=0.9,
        eeg_sources=(EEGSource("emotiv-1", ("AF3", "AF4"), 0.5),),
        wearable_summaries=(
            WearableSummary(
                "pulse-1", "heart_rate", "wearable_heart_rate_mean_bpm", "mean", 1.0
            ),
        ),
        cgm_reference=CGMReference("cgm-1", "glucose", "mg/dL", 2.0),
    )


def test_lsl_window_builder_extracts_nonzero_eeg_and_exact_future_labels(
    tmp_path: Path,
) -> None:
    audit, frames = _recorded_shape_fixture()
    windows, build_audit = build_lsl_glucose_windows(audit, frames, _config())
    assert len(windows) > 5
    assert build_audit["eeg_channel_map"] == {
        "emotiv-1": ["AF3", "AF4"]
    }
    assert build_audit["eeg_channel_counts"] == {"emotiv-1": 2}
    assert windows["eeg_alpha_mean"].median() > 0.8
    assert windows["eeg_beta_mean"].median() < windows["eeg_alpha_mean"].median()
    first = windows.iloc[0]
    assert first["target_glucose_1m_mg_dl"] == 101.5
    assert pd.Timestamp(first["target_glucose_1m_time"]) - pd.Timestamp(
        first["anchor_time"]
    ) == pd.Timedelta(minutes=1)
    assert not any("cgm" in value for value in build_audit["feature_registry"]["eeg"])

    destination = tmp_path / "windows.csv"
    windows.to_csv(destination, index=False)
    loaded, features = load_aligned_window_frame(
        destination,
        (1, 2),
        modalities=("eeg", "wearable"),
        feature_registry={
            "eeg": LSL_EEG_FEATURES,
            "wearable": ("wearable_heart_rate_mean_bpm",),
        },
        input_cgm=False,
    )
    assert len(loaded) == len(windows)
    assert features["eeg"] == LSL_EEG_FEATURES


def test_lsl_window_builder_keeps_missing_eeg_as_missing_not_zero() -> None:
    audit, frames = _recorded_shape_fixture()
    frames["emotiv-1"] = frames["emotiv-1"].loc[
        lambda value: value["lsl_timestamp"] < 60
    ]
    windows, _ = build_lsl_glucose_windows(audit, frames, _config())
    missing = windows.loc[~windows["eeg_available"]]
    assert not missing.empty
    assert missing[list(LSL_EEG_FEATURES)].isna().all(axis=None)
    assert missing["wearable_available"].all()


def test_xdf_frames_are_joined_by_source_id_not_dictionary_order() -> None:
    audit, frames = _recorded_shape_fixture()
    reversed_frames = {
        "cgm-1": frames["cgm-1"],
        "pulse-1": frames["pulse-1"],
        "emotiv-1": frames["emotiv-1"],
    }
    windows, _ = build_lsl_glucose_windows(audit, reversed_frames, _config())
    assert windows["eeg_alpha_mean"].median() > 0.8
    assert windows["wearable_heart_rate_mean_bpm"].notna().any()
    assert windows["eeg_available_time"].le(windows["anchor_time"]).all()
    assert windows["wearable_available_time"].le(windows["anchor_time"]).all()
