from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from src.neuroglycemic.neural_training import (
    GlucoseTargetStandardizer,
    LossOutput,
    inverse_transform_neuroglycemic_outputs,
    load_neural_checkpoint,
    load_neural_training_config,
    make_neuroglycemic_loss_step,
    train_with_early_stopping,
)
from src.neuroglycemic.neural_model import NeuroGlycemicNet


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    (
        "neural_glucose.json",
        "neural_glucose_physio.json",
        "big_ideas_neural.json",
        "diatrend_glucose.json",
        "mimic_neural_glucose.json",
    ),
)
def test_every_shipped_neural_config_loads(name: str) -> None:
    config = load_neural_training_config(ROOT / "config" / name)
    assert config.feature_registry
    assert config.forecast_horizons_minutes


class TinyRegressor(nn.Module):
    """A unit-test model; study artifacts are never trained on these fixtures."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def regression_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> LossOutput:
    prediction = model(batch["features"])
    mse = torch.mean((prediction - batch["target"]) ** 2)
    return LossOutput(total=mse, components={"mse": mse.detach()})


def _batches() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "features": torch.tensor([[-1.0], [0.0], [1.0], [2.0]]),
            "target": torch.tensor([-1.0, 1.0, 3.0, 5.0]),
        }
    ]


def _test_config(tmp_path: Path):
    loaded = load_neural_training_config(ROOT / "config" / "neural_glucose.json")
    return replace(
        loaded,
        epochs=40,
        learning_rate=0.05,
        weight_decay=0.0,
        early_stopping_patience=10,
        minimum_delta=0.0,
        checkpoint_relative_path="model.pt",
        project_root=tmp_path,
    )


def _target_standardizer() -> GlucoseTargetStandardizer:
    return GlucoseTargetStandardizer(
        horizons_minutes=(30, 60),
        means_mg_dl=(100.0, 120.0),
        scales_mg_dl=(10.0, 20.0),
        valid_counts=(20, 18),
    )


def test_loss_backpropagates_finite_nonzero_gradients() -> None:
    torch.manual_seed(7)
    model = TinyRegressor()
    output = regression_loss(model, _batches()[0])
    output.total.backward()
    assert model.linear.weight.grad is not None
    assert torch.isfinite(model.linear.weight.grad).all()
    assert float(torch.linalg.vector_norm(model.linear.weight.grad).item()) > 0.0


def test_training_updates_parameters_and_reduces_loss(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = TinyRegressor()
    before = model.linear.weight.detach().clone()
    initial_loss = float(regression_loss(model, _batches()[0]).total.item())
    result = train_with_early_stopping(
        model,
        _batches(),
        _batches(),
        regression_loss,
        _test_config(tmp_path),
        target_standardizer=_target_standardizer(),
    )
    final_loss = float(regression_loss(model, _batches()[0]).total.item())
    assert not torch.equal(model.linear.weight.detach(), before)
    assert final_loss < initial_loss
    assert result.best_epoch >= 1
    assert result.checkpoint_path.exists()
    assert all(float(row["train_gradient_norm"]) > 0.0 for row in result.history)


def test_checkpoint_restores_weights_and_enforces_target_contract(tmp_path: Path) -> None:
    model = TinyRegressor()
    config = _test_config(tmp_path)
    result = train_with_early_stopping(
        model,
        _batches(),
        _batches(),
        regression_loss,
        config,
        target_standardizer=_target_standardizer(),
    )
    expected_weight = model.linear.weight.detach().clone()
    expected_bias = model.linear.bias.detach().clone()
    with torch.no_grad():
        model.linear.weight.add_(100.0)
        model.linear.bias.sub_(100.0)

    payload = load_neural_checkpoint(
        result.checkpoint_path,
        model,
        expected_prediction_target=config.prediction_target,
        expected_horizons_minutes=config.forecast_horizons_minutes,
    )
    assert payload["epoch"] == result.best_epoch
    assert payload["training_config"]["forecast_mode"] == "ambient_no_cgm"
    assert payload["training_config"]["input_cgm"] is False
    assert payload["training_config"]["meal_context"]["lookback_minutes"] == 240
    assert payload["target_standardizer"]["fit_split"] == "train"
    assert payload["target_standardizer"]["unit"] == "mg/dL"
    assert torch.equal(model.linear.weight.detach(), expected_weight)
    assert torch.equal(model.linear.bias.detach(), expected_bias)

    with pytest.raises(ValueError, match="prediction target"):
        load_neural_checkpoint(
            result.checkpoint_path,
            model,
            expected_prediction_target="different_target",
        )


def test_target_standardization_round_trip_and_output_units() -> None:
    raw = torch.tensor([[80.0, 100.0], [100.0, 140.0], [120.0, float("nan")]])
    standardizer = GlucoseTargetStandardizer.fit(raw, (30, 60))
    standardized = standardizer.transform(raw)
    restored = standardizer.inverse_mean(standardized)
    assert torch.allclose(restored[:2], raw[:2])
    assert torch.isnan(restored[2, 1])

    outputs = {
        "expert_mean": torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]]),
        "expert_scale": torch.ones(1, 2, 2),
        "mixture_mean": torch.tensor([[0.0, 1.0]]),
        "mixture_variance": torch.ones(1, 2),
        "fusion_weights": torch.tensor([[0.4, 0.6]]),
    }
    converted = inverse_transform_neuroglycemic_outputs(outputs, standardizer)
    assert torch.allclose(converted["mixture_mean"], torch.tensor([[100.0, 140.0]]))
    assert torch.allclose(converted["expert_scale"][0, 0], torch.tensor([16.3299, 20.0]), atol=1e-4)
    assert torch.allclose(
        converted["mixture_variance"], torch.tensor([[16.3299**2, 20.0**2]]), atol=1e-2
    )
    assert converted["fusion_weights"] is outputs["fusion_weights"]


def test_neural_model_loss_adapter_accepts_raw_mg_dl_targets() -> None:
    model = NeuroGlycemicNet(
        {"eeg": 2, "wearable": 2},
        horizons_minutes=(30, 60),
        hidden_dim=4,
        embedding_dim=4,
        dropout=0.0,
        min_scale=0.1,
    )
    batch = {
        "features": {
            "eeg": torch.tensor([[0.1, -0.2], [0.3, 0.4]]),
            "wearable": torch.tensor([[0.2, 0.1], [-0.1, 0.5]]),
        },
        "feature_masks": {
            "eeg": torch.ones(2, 2, dtype=torch.bool),
            "wearable": torch.ones(2, 2, dtype=torch.bool),
        },
        "availability": torch.ones(2, 2, dtype=torch.bool),
        "quality": torch.ones(2, 2),
        "staleness": torch.zeros(2, 2),
        "targets": torch.tensor([[90.0, 100.0], [110.0, 140.0]]),
    }
    loss = make_neuroglycemic_loss_step(0.25, _target_standardizer())(model, batch)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
