"""Structural contract tests; values below are not research observations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.neuroglycemic.diatrend import (
    DiaTrendBuildConfig,
    build_diatrend_patient_windows,
    diatrend_feature_registry,
)
from src.neuroglycemic.evaluation import gaussian_mixture_quantile
from src.neuroglycemic.lsl import replay_numeric_table
from src.neuroglycemic.interoperability import convert_measurement_unit
from src.neuroglycemic.fhir import (
    CGM_MASS_PROFILE,
    cgm_sensor_observation,
    neural_forecast_observation,
)
from src.neuroglycemic.neural_dataset import (
    glucose_forecast_metrics,
    load_aligned_window_frame,
)
from src.neuroglycemic.neural_model import NeuroGlycemicNet, neuroglycemic_loss
from src.neuroglycemic.workspace import ResearchWorkspace


def _published_shape_frames():
    times = pd.date_range("2026-01-01T00:00:00Z", periods=97, freq="5min")
    cgm = pd.DataFrame(
        {
            "date": times,
            "mg/dL": 120.0 + 8.0 * np.sin(np.linspace(0.0, 4.0, len(times))),
        }
    )
    bolus = pd.DataFrame(
        {
            "date": [times[24], times[60]],
            "normal": [2.0, 1.5],
            "carbInput": [30.0, 20.0],
            "insulinOnBoard": [1.0, 0.5],
        }
    )
    basal = pd.DataFrame(
        {
            "date": [times[12], times[48]],
            "duration": [30.0 * 60.0 * 1000.0, 15.0 * 60.0 * 1000.0],
            "rate": [0.8, 0.9],
        }
    )
    return cgm, bolus, basal


def test_diatrend_builder_creates_causal_exact_future_targets(tmp_path: Path) -> None:
    cgm, bolus, basal = _published_shape_frames()
    config = DiaTrendBuildConfig(
        source_timezone="UTC",
        horizons_minutes=(30, 60),
        history_minutes=120,
        anchor_stride_minutes=15,
        minimum_history_coverage=1.0,
    )
    windows = build_diatrend_patient_windows(
        "S-contract", cgm, bolus, basal, config=config
    )
    assert not windows.empty
    assert set(diatrend_feature_registry()) == {"cgm", "events"}
    assert windows["cgm_available_time"].le(windows["anchor_time"]).all()
    assert windows["target_glucose_30m_time"].sub(
        windows["anchor_time"]
    ).eq(pd.Timedelta(minutes=30)).all()
    basal_start = cgm.iloc[48]["date"]
    basal_rows = windows.loc[
        windows["anchor_time"].between(
            basal_start, basal_start + pd.Timedelta(minutes=10)
        )
    ]
    assert not basal_rows.empty
    assert basal_rows["events_current_basal_units_per_hour"].eq(0.9).all()
    after_basal = windows.loc[
        windows["anchor_time"].eq(basal_start + pd.Timedelta(minutes=30))
    ]
    assert not after_basal.empty
    assert after_basal["events_current_basal_units_per_hour"].isna().all()

    source = cgm.set_index("date")["mg/dL"]
    first = windows.iloc[0]
    assert first["target_glucose_30m_mg_dl"] == pytest.approx(
        source.loc[first["target_glucose_30m_time"]]
    )
    path = tmp_path / "diatrend.csv"
    windows.to_csv(path, index=False)
    loaded, features = load_aligned_window_frame(
        path,
        (30, 60),
        modalities=("cgm", "events"),
        feature_registry=diatrend_feature_registry(),
        input_cgm=True,
    )
    assert len(loaded) == len(windows)
    assert features == diatrend_feature_registry()


def test_explicit_feature_registry_blocks_unlisted_prefix_leakage(tmp_path: Path) -> None:
    cgm, bolus, basal = _published_shape_frames()
    windows = build_diatrend_patient_windows(
        "S-contract",
        cgm,
        bolus,
        basal,
        config=DiaTrendBuildConfig(
            source_timezone="UTC",
            horizons_minutes=(30,),
            history_minutes=120,
        ),
    )
    windows["cgm_future_answer_mg_dl"] = windows["target_glucose_30m_mg_dl"]
    path = tmp_path / "leakage.csv"
    windows.to_csv(path, index=False)
    _, features = load_aligned_window_frame(
        path,
        (30,),
        modalities=("cgm", "events"),
        feature_registry=diatrend_feature_registry(),
        input_cgm=True,
    )
    assert "cgm_future_answer_mg_dl" not in features["cgm"]


def test_off_grid_readings_are_never_backdated_into_earlier_forecasts() -> None:
    cgm, bolus, basal = _published_shape_frames()
    cgm = cgm.copy()
    cgm["date"] = cgm["date"] + pd.Timedelta(minutes=4)
    bolus = bolus.copy()
    bolus["date"] = bolus["date"] + pd.Timedelta(minutes=4)
    basal = basal.copy()
    basal["date"] = basal["date"] + pd.Timedelta(minutes=4)
    windows = build_diatrend_patient_windows(
        "off-grid",
        cgm,
        bolus,
        basal,
        config=DiaTrendBuildConfig(
            source_timezone="UTC",
            horizons_minutes=(30,),
            history_minutes=120,
            anchor_stride_minutes=5,
            minimum_history_coverage=1.0,
        ),
    )
    assert not windows.empty
    assert windows["cgm_available_time"].lt(windows["anchor_time"]).all()
    # Actual reference times are retained. With readings at hh:mm+4 and
    # right-labeled anchors at the next five-minute mark, the realized horizon
    # is 29 minutes, within the declared five-minute matching tolerance.
    realized = windows["target_glucose_30m_time"] - windows["anchor_time"]
    assert realized.eq(pd.Timedelta(minutes=29)).all()
    assert windows["target_glucose_30m_time"].isin(cgm["date"]).all()


def test_horizon_specific_gate_and_clock_uncertainty_are_active() -> None:
    torch.manual_seed(4)
    model = NeuroGlycemicNet(
        {"cgm": 2, "events": 2},
        horizons_minutes=(30, 60, 90),
        hidden_dim=8,
        embedding_dim=6,
        dropout=0.0,
        modality_dropout_probability=0.25,
    ).eval()
    features = {name: torch.ones(4, 2) for name in model.modalities}
    masks = {name: torch.ones(4, 2, dtype=torch.bool) for name in model.modalities}
    availability = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.bool)
    quality = availability.float()
    staleness = torch.zeros(4, 2)
    clock = torch.tensor([[0.2, 4.0], [0.2, 0.0], [0.0, 4.0], [0.0, 0.0]])
    outputs = model(features, masks, availability, quality, staleness, clock)
    weights = outputs["fusion_weights_by_horizon"]
    assert weights.shape == (4, 2, 3)
    assert torch.allclose(weights[:3].sum(dim=1), torch.ones(3, 3))
    assert torch.equal(weights[3], torch.zeros(2, 3))


def test_affective_heads_learn_only_from_observed_validated_labels() -> None:
    torch.manual_seed(9)
    model = NeuroGlycemicNet(
        {"eeg": 2, "wearable": 2},
        horizons_minutes=(30,),
        hidden_dim=8,
        embedding_dim=6,
        dropout=0.0,
        auxiliary_task_kinds={
            "acute_stress": "binary",
            "gad7_score": "continuous",
            "phq9_score": "continuous",
        },
    )
    features = {name: torch.randn(3, 2) for name in model.modalities}
    masks = {name: torch.ones(3, 2, dtype=torch.bool) for name in model.modalities}
    availability = torch.ones(3, 2, dtype=torch.bool)
    outputs = model(
        features,
        masks,
        availability,
        torch.ones(3, 2),
        torch.zeros(3, 2),
    )
    loss = neuroglycemic_loss(
        outputs,
        torch.tensor([[0.1], [0.2], [0.3]]),
        auxiliary_targets={
            "acute_stress": torch.tensor([0.0, 1.0, float("nan")]),
            "gad7_score": torch.tensor([4.0, float("nan"), 8.0]),
            "phq9_score": torch.tensor([float("nan"), 3.0, 7.0]),
        },
        auxiliary_task_kinds=model.auxiliary_task_kinds,
        auxiliary_loss_weights={
            "acute_stress": 0.2,
            "gad7_score": 0.05,
            "phq9_score": 0.05,
        },
    )
    loss["loss"].backward()
    assert torch.isfinite(loss["loss"])
    for head in model.auxiliary_heads.values():
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0


def test_gaussian_mixture_interval_uses_actual_mixture_quantiles() -> None:
    lower = gaussian_mixture_quantile([0.0], [1.0], [1.0], 0.025)
    upper = gaussian_mixture_quantile([0.0], [1.0], [1.0], 0.975)
    assert lower == pytest.approx(-1.95996, abs=1e-4)
    assert upper == pytest.approx(1.95996, abs=1e-4)

    mixed_upper = gaussian_mixture_quantile(
        [0.0, 6.0], [0.5, 2.0], [0.8, 0.2], 0.975
    )
    mixture_mean = 1.2
    mixture_variance = 0.8 * (0.5**2 + 0.0**2) + 0.2 * (2.0**2 + 6.0**2) - mixture_mean**2
    gaussian_approximation = mixture_mean + 1.96 * np.sqrt(mixture_variance)
    assert mixed_upper != pytest.approx(gaussian_approximation, abs=0.1)


def test_auxiliary_heads_receive_label_specific_held_out_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2"],
            "anchor_time": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "horizon_minutes": [30] * 4,
            "target_glucose_mg_dl": [65.0, 100.0, 190.0, 120.0],
            "predicted_glucose_mg_dl": [70.0, 105.0, 180.0, 115.0],
            "prediction_lower_mg_dl": [50.0, 80.0, 150.0, 90.0],
            "prediction_upper_mg_dl": [90.0, 130.0, 210.0, 140.0],
            "target_hypoglycemia": [1, 0, 0, 0],
            "hypoglycemia_probability": [0.7, 0.1, 0.05, 0.1],
            "target_hyperglycemia": [0, 0, 1, 0],
            "hyperglycemia_probability": [0.05, 0.1, 0.7, 0.1],
            "persistence_glucose_mg_dl": [68.0, 98.0, 185.0, 121.0],
            "weight_eeg": [1.0] * 4,
            "expert_mean_eeg_mg_dl": [70.0, 105.0, 180.0, 115.0],
            "expert_sd_eeg_mg_dl": [10.0] * 4,
            "auxiliary_stress": [0.8, 0.2, 0.7, 0.1],
            "target_auxiliary_stress": [1.0, 0.0, 1.0, 0.0],
            "auxiliary_kind_stress": ["binary"] * 4,
        }
    )
    metrics = glucose_forecast_metrics(predictions)
    assert metrics["auxiliary_tasks"]["stress"]["n_event_predictions"] == 4
    assert metrics["auxiliary_tasks"]["stress"]["brier_score"] < 0.1


def test_workspace_is_external_and_replay_rejects_targets_before_lsl_import(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside"):
        ResearchWorkspace.create(repository / "runtime", repository_root=repository)
    with pytest.raises(ValueError, match="disjoint sibling"):
        ResearchWorkspace.create(repository.parent, repository_root=repository)
    workspace = ResearchWorkspace.create(tmp_path / "runtime", repository_root=repository)
    assert workspace.models.is_dir()
    frame = pd.DataFrame(
        {
            "anchor_time": ["2026-01-01T00:00:00Z"],
            "target_glucose_30m_mg_dl": [110.0],
        }
    )
    with pytest.raises(ValueError, match="Future target"):
        replay_numeric_table(
            frame,
            timestamp_column="anchor_time",
            channel_columns=["target_glucose_30m_mg_dl"],
            stream_name="test",
            stream_type="CGM",
            source_id="contract-test",
        )


def test_clinical_unit_conversion_changes_values_not_only_labels() -> None:
    assert convert_measurement_unit(
        10.0, source_unit="mmol/L", target_unit="mg/dL"
    ) == pytest.approx(180.182)
    assert convert_measurement_unit(
        98.6, source_unit="fahrenheit", target_unit="celsius"
    ) == pytest.approx(37.0)
    with pytest.raises(ValueError, match="Unsupported unit conversion"):
        convert_measurement_unit(1.0, source_unit="unknown", target_unit="mg/dL")


def test_fhir_keeps_sensor_measurements_distinct_from_model_forecasts() -> None:
    measured = cgm_sensor_observation(
        patient_reference="Patient/123",
        effective_time="2026-01-01T12:00:00-06:00",
        glucose_mg_dl=121.0,
        source_identifier="device-reading-1",
        device_reference="Device/cgm-1",
    )
    assert measured["meta"]["profile"] == [CGM_MASS_PROFILE]
    assert measured["code"]["coding"][0]["code"] == "99504-3"

    forecast = neural_forecast_observation(
        {
            "anchor_time": "2026-01-01T12:00:00-06:00",
            "horizon_minutes": 60,
            "model_version": "neuroglycemic-neural-v2",
            "predicted_glucose_mg_dl": 135.0,
            "prediction_lower_mg_dl": 100.0,
            "prediction_upper_mg_dl": 175.0,
            "abstained": False,
        },
        patient_reference="Patient/123",
    )
    assert "meta" not in forecast or CGM_MASS_PROFILE not in forecast["meta"].get(
        "profile", []
    )
    assert forecast["status"] == "preliminary"
    assert forecast["code"]["coding"][0]["code"] == "future-glucose"

    with pytest.raises(ValueError, match="Patient/<id>"):
        cgm_sensor_observation(
            patient_reference="Device/not-a-patient",
            effective_time="2026-01-01T12:00:00-06:00",
            glucose_mg_dl=121.0,
            source_identifier="invalid-subject",
        )
    with pytest.raises(ValueError, match="lower <= prediction <= upper"):
        neural_forecast_observation(
            {
                "anchor_time": "2026-01-01T12:00:00-06:00",
                "horizon_minutes": 60,
                "predicted_glucose_mg_dl": 135.0,
                "prediction_lower_mg_dl": 150.0,
                "prediction_upper_mg_dl": 170.0,
                "abstained": False,
            },
            patient_reference="Patient/123",
        )
