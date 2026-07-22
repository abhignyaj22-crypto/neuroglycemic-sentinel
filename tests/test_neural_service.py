from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.neuroglycemic.architecture import run_neural_architecture_case  # noqa: E402
from src.neuroglycemic.neural_model import NeuroGlycemicNet  # noqa: E402
from src.neuroglycemic.neural_training import (  # noqa: E402
    GlucoseTargetStandardizer,
    load_neural_training_config,
    save_neural_checkpoint,
)
from src.neuroglycemic.service import (  # noqa: E402
    NEURAL_FEATURE_SCHEMA,
    NeuralGlucoseForecastRequest,
    NeuralGlucoseService,
    build_neural_checkpoint_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(tmp_path: Path, *, metadata: bool = True) -> Path:
    torch.manual_seed(29)
    model = NeuroGlycemicNet(
        {"eeg": 2, "wearable": 2, "ehr": 1},
        horizons_minutes=(30, 60),
        hidden_dim=6,
        embedding_dim=4,
        dropout=0.0,
        min_scale=0.05,
    )
    config = replace(
        load_neural_training_config(ROOT / "config" / "neural_glucose.json"),
        project_root=tmp_path,
        checkpoint_relative_path="outputs/models/test.pt",
        model={
            "hidden_dim": 6,
            "embedding_dim": 4,
            "dropout": 0.0,
            "min_scale": 0.05,
        },
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    serving_metadata = {}
    if metadata:
        serving_metadata = build_neural_checkpoint_metadata(
            model,
            feature_names={
                "eeg": ["eeg_delta_mean", "eeg_beta_mean"],
                "wearable": ["heart_rate_bpm", "step_count"],
                "ehr": ["age_years"],
            },
            feature_means={
                "eeg": [1.0, 2.0],
                "wearable": [70.0, 1000.0],
                "ehr": [50.0],
            },
            feature_scales={
                "eeg": [0.5, 0.5],
                "wearable": [10.0, 500.0],
                "ehr": [15.0],
            },
            hidden_dim=6,
            embedding_dim=4,
            dropout=0.0,
            min_scale=0.05,
            model_version="test-neural-v1",
        )
    save_neural_checkpoint(
        config.checkpoint_path,
        model,
        optimizer,
        epoch=1,
        validation_loss=1.0,
        config=config,
        target_standardizer=GlucoseTargetStandardizer(
            horizons_minutes=(30, 60),
            means_mg_dl=(105.0, 115.0),
            scales_mg_dl=(12.0, 18.0),
            valid_counts=(20, 20),
        ),
        metadata=serving_metadata,
    )
    return config.checkpoint_path


def _request(
    *,
    horizon_minutes: int = 60,
    schema: str = NEURAL_FEATURE_SCHEMA,
    availability: dict[str, bool] | None = None,
) -> NeuralGlucoseForecastRequest:
    return NeuralGlucoseForecastRequest(
        patient_id="prospective-001",
        anchor_time="2026-07-20T12:00:00-05:00",
        horizon_minutes=horizon_minutes,
        feature_schema_version=schema,
        features={
            "eeg": {"eeg_delta_mean": 1.2, "eeg_beta_mean": 2.1},
            "wearable": {"heart_rate_bpm": 82.0, "step_count": None},
            "ehr": {"age_years": 46.0},
        },
        availability=availability or {"eeg": False, "wearable": True, "ehr": True},
        quality={"eeg": 0.0, "wearable": 0.8, "ehr": 1.0},
        staleness_minutes={"eeg": 0.0, "wearable": 2.0, "ehr": 240.0},
        clock_uncertainty_ms={"eeg": 0.0, "wearable": 1.0, "ehr": 0.0},
    )


def test_checkpoint_service_runs_neural_forward_and_masks_missing_eeg(
    tmp_path: Path,
) -> None:
    service = NeuralGlucoseService.from_checkpoint(_checkpoint(tmp_path))
    response = service.forecast(_request())

    assert response.abstained is False
    assert response.prediction_target == "future_cgm_glucose_mg_dl"
    assert response.horizon_minutes == 60
    assert response.predicted_glucose_mg_dl is not None
    assert response.prediction_sd_mg_dl is not None
    assert response.prediction_sd_mg_dl > 0
    assert response.hypoglycemia_probability is not None
    assert response.hyperglycemia_probability is not None
    assert 0.0 <= response.hypoglycemia_probability <= 1.0
    assert 0.0 <= response.hyperglycemia_probability <= 1.0
    forecasts = {item.modality: item for item in response.modality_forecasts}
    assert forecasts["eeg"].learned_weight == 0.0
    assert forecasts["eeg"].predicted_glucose_mg_dl is None
    assert sum(
        item.learned_weight for item in response.modality_forecasts
    ) == pytest.approx(1.0)
    assert response.as_dict()["learned_weights"]["wearable"] > 0


def test_service_rejects_untrained_contracts_and_caller_selected_horizons(
    tmp_path: Path,
) -> None:
    service = NeuralGlucoseService.from_checkpoint(_checkpoint(tmp_path))
    with pytest.raises(ValueError, match="Unsupported horizon"):
        service.forecast(_request(horizon_minutes=45))
    with pytest.raises(ValueError, match="feature schema version"):
        service.forecast(_request(schema="wrong-schema"))

    incomplete = _checkpoint(tmp_path / "incomplete", metadata=False)
    with pytest.raises(ValueError, match="model_spec"):
        NeuralGlucoseService.from_checkpoint(incomplete)


def test_service_abstains_instead_of_returning_a_default_glucose(
    tmp_path: Path,
) -> None:
    service = NeuralGlucoseService.from_checkpoint(_checkpoint(tmp_path))
    response = service.forecast(
        _request(availability={"eeg": False, "wearable": False, "ehr": False})
    )
    assert response.abstained is True
    assert response.predicted_glucose_mg_dl is None
    assert response.prediction_lower_mg_dl is None
    assert response.hypoglycemia_probability is None
    assert response.hyperglycemia_probability is None
    assert all(item.learned_weight == 0 for item in response.modality_forecasts)


def test_neural_case_study_uses_checkpoint_not_cached_prediction_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    request = _request()
    expected = NeuralGlucoseService.from_checkpoint(checkpoint).forecast(request)

    result = run_neural_architecture_case(
        tmp_path,
        checkpoint_path=checkpoint.relative_to(tmp_path),
        request=request,
    )

    assert result["architecture_status"]["neural_forward_pass_executed"] is True
    assert result["architecture_status"]["cached_prediction_artifacts_read"] is False
    assert result["neural_forecast"]["predicted_glucose_mg_dl"] == pytest.approx(
        expected.predicted_glucose_mg_dl
    )
    assert result["neural_forecast"]["learned_weights"] == pytest.approx(
        expected.as_dict()["learned_weights"]
    )
    assert (
        result["health_agent"]["evidence_packet"]["model_output"]["abstained"] is False
    )
    assert (tmp_path / "outputs" / "architecture" / "neural_case_study.json").exists()


def test_neural_case_accepts_disjoint_external_runtime_workspace(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "software"
    workspace = tmp_path / "runtime"
    repository.mkdir()
    workspace.mkdir()
    checkpoint = _checkpoint(workspace)
    output = workspace / "runs" / "case.json"

    result = run_neural_architecture_case(
        repository,
        checkpoint_path=checkpoint,
        request=_request(),
        workspace_root=workspace,
        output_path=output,
    )

    assert output.exists()
    assert result["health_agent"]["evidence_packet"]["metadata"]["input_cgm"] is False
