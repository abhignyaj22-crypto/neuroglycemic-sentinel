"""Big Ideas adapter tests use generated schema fixtures, never study training data."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.neuroglycemic.big_ideas_data import (
    BIG_IDEAS_WEARABLE_FEATURES,
    BigIdeasBuildConfig,
    build_big_ideas_dataset,
    discover_big_ideas_patients,
)
from src.neuroglycemic.neural_dataset import (
    TrainOnlyFeatureStandardizer,
    load_aligned_window_frame,
    make_neural_batches,
)


def _write_patient(root: Path, patient_id: str) -> None:
    directory = root / patient_id
    directory.mkdir(parents=True)
    start = pd.Timestamp("2026-01-01 08:00:00")
    cgm_times = pd.date_range(start, periods=73, freq="5min")
    dexcom = pd.DataFrame(
        {
            "Index": np.arange(len(cgm_times)),
            "Timestamp (YYYY-MM-DDThh:mm:ss)": cgm_times.astype(str),
            "Event Type": "EGV",
            "Glucose Value (mg/dL)": 100.0 + np.sin(np.arange(len(cgm_times)) / 8) * 15,
        }
    )
    dexcom.to_csv(directory / f"Dexcom_{patient_id}.csv", index=False)

    hr_times = pd.date_range(start, periods=361, freq="1min")
    pd.DataFrame(
        {"datetime": hr_times.astype(str), " hr": 72.0 + np.sin(np.arange(361) / 12)}
    ).to_csv(directory / f"HR_{patient_id}.csv", index=False)
    pd.DataFrame(
        {
            "datetime": hr_times.astype(str),
            " ibi": 0.82 + np.sin(np.arange(361) / 15) * 0.04,
        }
    ).to_csv(directory / f"IBI_{patient_id}.csv", index=False)
    pd.DataFrame(
        {
            "time_begin": [str(start + pd.Timedelta(hours=2))],
            "total_carb": [35.0],
        }
    ).to_csv(directory / f"Food_Log_{patient_id}.csv", index=False)


def test_big_ideas_build_is_causal_and_keeps_cgm_out_of_features(tmp_path: Path) -> None:
    source = tmp_path / "big-ideas"
    _write_patient(source, "001")
    _write_patient(source, "002")
    patients = discover_big_ideas_patients(source)
    config = BigIdeasBuildConfig(source_timezone="UTC")
    windows, audit = build_big_ideas_dataset(patients, config=config)

    assert len(audit) == 2
    assert windows["wearable_available_time"].le(windows["anchor_time"]).all()
    assert windows["reference_current_glucose_mg_dl"].notna().any()
    assert all("glucose" not in feature for feature in BIG_IDEAS_WEARABLE_FEATURES)
    for horizon in config.horizons_minutes:
        labelled = windows[f"target_glucose_{horizon}m_mg_dl"].notna()
        assert labelled.any()
        assert windows.loc[labelled, f"target_glucose_{horizon}m_time"].gt(
            windows.loc[labelled, "anchor_time"]
        ).all()

    destination = tmp_path / "aligned.csv"
    windows.loc[windows.index[1], "wearable_available_time"] = (
        pd.Timestamp(windows.loc[windows.index[1], "anchor_time"])
        - pd.Timedelta(microseconds=123456)
    )
    windows.to_csv(destination, index=False)
    loaded, features = load_aligned_window_frame(
        destination,
        config.horizons_minutes,
        modalities=("wearable",),
        feature_registry={"wearable": BIG_IDEAS_WEARABLE_FEATURES},
        input_cgm=False,
    )
    assert set(loaded["patient_id"].astype(str)) == {"001", "002"}
    standardizer = TrainOnlyFeatureStandardizer.fit(loaded, features)
    batch = make_neural_batches(
        loaded,
        standardizer,
        config.horizons_minutes,
        batch_size=len(loaded),
    )[0]
    assert bool(torch.isfinite(batch["persistence_glucose"]).any())
