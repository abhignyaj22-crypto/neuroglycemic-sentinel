"""Contract tests only; these fixtures are never used as research data."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.neuroglycemic.neural_dataset import (  # noqa: E402
    TrainOnlyFeatureStandardizer,
    load_aligned_window_frame,
    make_neural_batches,
    patient_grouped_split,
)


def _aligned_fixture() -> pd.DataFrame:
    rows = []
    for patient_index in range(6):
        patient = f"p{patient_index}"
        for window in range(2):
            anchor = pd.Timestamp("2026-01-01T08:00:00Z") + pd.Timedelta(
                days=patient_index, minutes=15 * window
            )
            eeg_available = not (patient_index == 0 and window == 1)
            rows.append(
                {
                    "patient_id": patient,
                    "cohort_id": "paired_bridge_cohort",
                    "anchor_time": anchor,
                    "eeg_available": eeg_available,
                    "eeg_quality": 0.9 if eeg_available else 0.0,
                    "eeg_staleness_minutes": 1.0 if eeg_available else 0.0,
                    "eeg_patient_id": patient if eeg_available else None,
                    "eeg_cohort_id": "paired_bridge_cohort" if eeg_available else None,
                    "eeg_anchor_time": anchor if eeg_available else None,
                    "eeg_delta_power": 1.0 + patient_index + window if eeg_available else np.nan,
                    "wearable_available": True,
                    "wearable_quality": 0.8,
                    "wearable_staleness_minutes": 2.0,
                    "wearable_patient_id": patient,
                    "wearable_cohort_id": "paired_bridge_cohort",
                    "wearable_anchor_time": anchor,
                    "wearable_hr_bpm": 60.0 + patient_index + window,
                    "ehr_available": True,
                    "ehr_quality": 1.0,
                    "ehr_staleness_minutes": 30.0,
                    "ehr_patient_id": patient,
                    "ehr_cohort_id": "paired_bridge_cohort",
                    "ehr_anchor_time": anchor,
                    "ehr_age_years": 30.0 + patient_index,
                    "meal_carbohydrate_g": float(10 * window),
                    "target_glucose_30m_mg_dl": 90.0 + 3 * patient_index + window,
                    "target_glucose_30m_time": anchor + pd.Timedelta(minutes=30),
                    "target_glucose_60m_mg_dl": 95.0 + 4 * patient_index + window,
                    "target_glucose_60m_time": anchor + pd.Timedelta(minutes=60),
                }
            )
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_aligned_loader_patient_split_and_batch_masks(tmp_path: Path) -> None:
    frame, feature_names = load_aligned_window_frame(
        _write(_aligned_fixture(), tmp_path / "aligned.csv"), (30, 60)
    )
    split_frame, split = patient_grouped_split(
        frame, seed=42, train_fraction=0.67, validation_fraction=0.17
    )
    assert not (set(split.train) & set(split.validation))
    assert not (set(split.train) & set(split.test))
    assert split_frame.groupby("patient_id")["split"].nunique().max() == 1

    train = split_frame.loc[split_frame["split"] == "train"].copy()
    standardizer = TrainOnlyFeatureStandardizer.fit(train, feature_names)
    batches = make_neural_batches(
        split_frame, standardizer, (30, 60), batch_size=5
    )
    assert batches[0]["targets"].shape == (5, 2)
    assert tuple(batches[0]["features"]) == ("eeg", "wearable", "ehr")
    for batch in batches:
        unavailable = ~batch["availability"][:, 0]
        assert not batch["feature_masks"]["eeg"][unavailable].any()


def test_cross_cohort_or_patient_modality_provenance_is_rejected(tmp_path: Path) -> None:
    frame = _aligned_fixture()
    frame.loc[0, "eeg_cohort_id"] = "unrelated_eeg_cohort"
    with pytest.raises(ValueError, match="Cross-patient and cross-cohort"):
        load_aligned_window_frame(_write(frame, tmp_path / "bad.csv"), (30, 60))


def test_feature_statistics_are_fit_on_training_rows_only(tmp_path: Path) -> None:
    frame, feature_names = load_aligned_window_frame(
        _write(_aligned_fixture(), tmp_path / "aligned.csv"), (30, 60)
    )
    frame["split"] = "test"
    frame.loc[frame["patient_id"].isin(["p0", "p1", "p2", "p3"]), "split"] = "train"
    test_rows = frame["split"].eq("test")
    frame.loc[test_rows, "wearable_hr_bpm"] = 1_000_000.0
    training = frame.loc[frame["split"] == "train"].copy()
    standardizer = TrainOnlyFeatureStandardizer.fit(training, feature_names)
    heart_rate_index = standardizer.feature_names["wearable"].index("wearable_hr_bpm")
    assert standardizer.means["wearable"][heart_rate_index] < 100.0
    assert standardizer.fit_split == "train"


def test_future_or_wrong_horizon_target_time_is_rejected(tmp_path: Path) -> None:
    frame = _aligned_fixture()
    frame.loc[0, "target_glucose_30m_time"] = frame.loc[0, "anchor_time"]
    with pytest.raises(ValueError, match="future timestamp"):
        load_aligned_window_frame(_write(frame, tmp_path / "bad_target.csv"), (30, 60))


def test_unavailable_modality_must_not_be_zero_filled(tmp_path: Path) -> None:
    frame = _aligned_fixture()
    unavailable = ~frame["eeg_available"]
    frame.loc[unavailable, "eeg_delta_power"] = 0.0
    with pytest.raises(ValueError, match="must leave modality features missing"):
        load_aligned_window_frame(_write(frame, tmp_path / "zero_filled.csv"), (30, 60))


def test_full_neural_data_path_updates_the_real_mixture_model_and_reloads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the production neural classes; fixture results are not study metrics."""

    from main import run_neural_evaluate, run_neural_train

    data_path = _write(_aligned_fixture(), tmp_path / "aligned.csv")
    project_root = Path(__file__).resolve().parents[1]
    config_values = json.loads(
        (project_root / "config" / "neural_glucose.json").read_text(encoding="utf-8")
    )
    config_values.update(
        {
            "epochs": 12,
            "learning_rate": 0.01,
            "early_stopping_patience": 12,
            "minimum_delta": 0.0,
            "checkpoint_relative_path": "outputs/models/integration.pt",
        }
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "neural_glucose.json"
    config_path.write_text(json.dumps(config_values), encoding="utf-8")
    checkpoint = tmp_path / "neural.pt"
    output_dir = tmp_path / "outputs"

    run_neural_train(
        data_path=data_path,
        config_path=config_path,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        batch_size=8,
        train_fraction=0.67,
        validation_fraction=0.17,
    )
    history = pd.read_csv(output_dir / "training_losses.csv")
    assert checkpoint.exists()
    assert history["train_loss"].iloc[1:].min() < history["train_loss"].iloc[0]

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["metadata"]["model_version"] == "neuroglycemic-neural-v1"
    assert payload["metadata"]["model_spec"]["min_scale"] == pytest.approx(0.05)
    assert payload["metadata"]["feature_schema"]["fit_split"] == "train"
    run_neural_evaluate(
        data_path=data_path,
        config_path=config_path,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        batch_size=8,
    )
    metrics_path = output_dir / "reloaded_test_metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for horizon_metrics in metrics["by_horizon"].values():
        assert np.isfinite(horizon_metrics["gaussian_mixture_nll"])
        assert horizon_metrics["prediction_coverage"] == pytest.approx(1.0)
        assert np.isfinite(horizon_metrics["hypoglycemia_event"]["brier_score"])
        assert np.isfinite(horizon_metrics["hyperglycemia_event"]["brier_score"])
    prediction_columns = pd.read_csv(
        output_dir / "reloaded_test_predictions.csv", nrows=0
    ).columns
    assert "prediction_lower_mg_dl" in prediction_columns
    assert "expert_mean_eeg_mg_dl" in prediction_columns
    assert "hypoglycemia_probability" in prediction_columns
    ablation = pd.read_csv(
        output_dir / "reloaded_missing_modality_ablation.csv"
    )
    all_unavailable = ablation.loc[ablation["scenario"].eq("all_unavailable")]
    assert not all_unavailable.empty
    assert all_unavailable["abstention_rate"].eq(1.0).all()
    assert "NEURAL TRAINING AND VALIDATION LOSSES" in capsys.readouterr().out
