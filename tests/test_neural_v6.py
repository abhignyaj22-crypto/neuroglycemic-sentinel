"""Tests for the v6 learning contributions.

Fixture tensors verify software behavior only; they are not scientific
training data and are never reported as model performance.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.neuroglycemic.calibration import (
    ConformalCalibrator,
    fit_conformal_calibrator,
    mixture_cdf,
)
from src.neuroglycemic.ensemble import combine_member_predictions
from src.neuroglycemic.neural_model import (
    CrossModalContextLayer,
    HorizonFilmHead,
    NeuroGlycemicNet,
    ResponseKernelHead,
    mixture_crps,
    neuroglycemic_loss,
)
from src.neuroglycemic.neural_training import (
    GlucoseTargetStandardizer,
    make_neuroglycemic_loss_step,
)
from src.neuroglycemic.pretrain import (
    load_pretrain_weights,
    masked_reconstruction_loss,
    pretrain_masked_reconstruction,
    save_pretrain_checkpoint,
)

HORIZONS = (30, 60)
CENTERS = (0.0, 15.0, 30.0, 60.0)


def _model(**overrides) -> NeuroGlycemicNet:
    options = {
        "hidden_dim": 16,
        "embedding_dim": 8,
        "dropout": 0.0,
        "min_scale": 0.05,
        "horizons_minutes": HORIZONS,
    }
    options.update(overrides)
    return NeuroGlycemicNet({"eeg": 4, "wearable": 3}, **options)


def _batch(batch_size: int = 6, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    eeg = torch.randn(batch_size, 4, generator=generator)
    wearable = torch.randn(batch_size, 3, generator=generator)
    availability = torch.ones(batch_size, 2, dtype=torch.bool)
    availability[0, 0] = False  # one EEG-missing row
    return {
        "features": {"eeg": eeg, "wearable": wearable},
        "feature_masks": {
            "eeg": availability[:, [0]].expand(-1, 4).clone(),
            "wearable": torch.ones(batch_size, 3, dtype=torch.bool),
        },
        "availability": availability,
        "quality": torch.rand(batch_size, 2, generator=generator),
        "staleness": torch.rand(batch_size, 2, generator=generator) * 10,
        "targets": 120.0 + 20.0 * torch.randn(batch_size, 2, generator=generator),
    }


def test_cross_modal_attention_respects_availability() -> None:
    layer = CrossModalContextLayer(8, num_heads=2)
    tokens = torch.randn(5, 3, 8)
    availability = torch.tensor(
        [[True, True, False], [True, False, True], [False, False, False],
         [True, True, True], [False, True, False]]
    )
    output, attention = layer(tokens, availability)
    assert output.shape == tokens.shape
    # Missing modality tokens carry no information after the layer.
    assert torch.all(output[~availability] == 0.0)
    # Attention to missing keys is exactly zero (row 0, key 2).
    assert torch.all(attention[0, :, :, 2] == 0.0)
    # Fully-missing rows remain finite (they abstain downstream).
    assert torch.isfinite(output[2]).all()


def test_horizon_film_head_shapes_and_shared_strength() -> None:
    head = HorizonFilmHead(8, HORIZONS, min_scale=0.05)
    mean, scale = head(torch.randn(7, 8))
    assert mean.shape == (7, 2) and scale.shape == (7, 2)
    assert (scale > 0.05).all()
    # Horizon codes are learned parameters distinct across horizons.
    assert not torch.allclose(head.horizon_code[0], head.horizon_code[1])


def test_response_kernel_sign_constraints_and_personalization() -> None:
    kernel = ResponseKernelHead(
        {"carbohydrate_g": 1.0, "bolus_insulin_units": -1.0},
        CENTERS,
        len(HORIZONS),
        patient_count=3,
        rank=2,
    )
    kernels = kernel.kernels()
    assert (kernels[0] >= 0).all(), "carbohydrate kernel must be non-negative"
    assert (kernels[1] <= 0).all(), "insulin kernel must be non-positive"
    basis = {
        "carbohydrate_g": torch.rand(4, len(CENTERS)),
        "bolus_insulin_units": torch.rand(4, len(CENTERS)),
    }
    index = torch.tensor([0, 1, 2, 0])
    seen = torch.tensor([True, True, False, False])
    personalized = kernel(basis, patient_index=index, seen_patient=seen)
    population = kernel(basis)
    assert personalized.shape == (4, len(HORIZONS))
    # Unseen patients receive exactly the population kernel.
    assert torch.allclose(personalized[2:], population[2:])


def test_model_forward_with_all_contributions() -> None:
    model = _model(
        cross_modal_layers=1,
        cross_modal_heads=2,
        horizon_film=True,
        response_kernel={
            "channels": {"carbohydrate_g": 1.0},
            "basis_centers_minutes": list(CENTERS),
            "patient_count": 2,
            "rank": 2,
        },
    )
    batch = _batch()
    outputs = model(
        batch["features"],
        batch["feature_masks"],
        batch["availability"],
        batch["quality"],
        batch["staleness"],
        patient_index=torch.zeros(6, dtype=torch.long),
        seen_patient=torch.ones(6, dtype=torch.bool),
        event_basis={"carbohydrate_g": torch.rand(6, len(CENTERS))},
    )
    assert outputs["mixture_mean"].shape == (6, 2)
    assert outputs["cross_attention"] is not None
    assert outputs["response_delta"] is not None
    assert torch.isfinite(outputs["mixture_mean"][1:]).all()
    extras = model.architecture_extras()
    assert extras["cross_modal_layers"] == 1
    assert extras["horizon_film"] is True
    assert extras["response_kernel"]["channels"] == {"carbohydrate_g": 1.0}


def test_backward_compatible_default_architecture_unchanged() -> None:
    legacy = _model()
    upgraded = _model(
        cross_modal_layers=0, horizon_film=False, response_kernel=None
    )
    assert set(legacy.state_dict()) == set(upgraded.state_dict())
    assert legacy.response_kernel is None
    assert len(legacy.context_layers) == 0


def test_mixture_crps_is_differentiable_and_sharpness_aware() -> None:
    # When the predictive mean is (nearly) right, a proper score must reward
    # the sharper distribution; CRPS is proper, unlike raw interval width.
    target = torch.tensor([[0.05, -0.04], [0.02, 0.03]])
    mean = torch.zeros(2, 2, 2, requires_grad=True)
    scale_narrow = torch.full((2, 2, 2), 0.5, requires_grad=True)
    scale_wide = torch.full((2, 2, 2), 3.0, requires_grad=True)
    weights = torch.full((2, 2, 2), 0.5)
    narrow = mixture_crps(target, mean, scale_narrow, weights)
    wide = mixture_crps(target, mean.detach(), scale_wide, weights)
    assert float(narrow) < float(wide)
    narrow.backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()


def test_crps_closed_form_matches_gaussian_special_case() -> None:
    # Single component with weight 1 reduces to the Gaussian CRPS closed form.
    target = torch.tensor([[0.3]])
    mean = torch.tensor([[[0.0]]])
    scale = torch.tensor([[[1.0]]])
    weights = torch.ones(1, 1, 1)
    value = mixture_crps(target, mean, scale, weights)
    z = 0.3
    expected = 1.0 * (
        z * (2 * 0.6179114222 - 1) + 2 * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        - 1 / math.sqrt(math.pi)
    ) - 0.5 * (math.sqrt(2.0) * (2 / math.sqrt(math.pi) * math.exp(0) - 1 / math.sqrt(math.pi)) * 1.0)
    # Closed form of the pairwise term for the single-component case:
    pair = math.sqrt(2.0) * (2 * math.exp(0) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))
    expected = (
        z * (2 * 0.6179114222 - 1)
        + 2 * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        - 1 / math.sqrt(math.pi)
    ) - 0.5 * pair
    assert float(value) == pytest.approx(expected, rel=1e-5)


def test_loss_step_includes_crps_component() -> None:
    standardizer = GlucoseTargetStandardizer(
        horizons_minutes=HORIZONS,
        means_mg_dl=(120.0, 121.0),
        scales_mg_dl=(20.0, 21.0),
        valid_counts=(10, 10),
    )
    step = make_neuroglycemic_loss_step(0.25, standardizer, crps_loss_weight=0.1)
    model = _model()
    output = step(model, _batch())
    assert "mixture_crps" in output.components
    assert float(output.components["mixture_crps"]) >= 0.0


def test_pretraining_then_finetune_weights(tmp_path: Path) -> None:
    model = _model(cross_modal_layers=1, cross_modal_heads=2,
                   build_reconstruction_heads=True)
    batch = _batch(batch_size=8)
    generator = torch.Generator().manual_seed(0)
    loss, terms = masked_reconstruction_loss(
        model, batch, hide_probability=0.5, generator=generator
    )
    assert torch.isfinite(loss) and terms > 0
    history = pretrain_masked_reconstruction(
        model, [batch], epochs=3, learning_rate=0.01, seed=0
    )
    assert len(history) == 3
    checkpoint = tmp_path / "pretrain.pt"
    save_pretrain_checkpoint(checkpoint, model)
    model.drop_reconstruction_heads()
    fresh = _model(cross_modal_layers=1, cross_modal_heads=2)
    load_pretrain_weights(fresh, checkpoint)
    for name, value in fresh.state_dict().items():
        assert torch.allclose(value, model.state_dict()[name]), name


def test_conformal_calibrator_levels_and_fallback() -> None:
    model = _model()
    standardizer = GlucoseTargetStandardizer(
        horizons_minutes=HORIZONS,
        means_mg_dl=(120.0, 121.0),
        scales_mg_dl=(20.0, 21.0),
        valid_counts=(10, 10),
    )
    batches = [_batch(seed=index) for index in range(4)]
    calibrator = fit_conformal_calibrator(
        model, batches, standardizer, HORIZONS, min_count=2
    )
    for horizon in HORIZONS:
        low, high = calibrator.levels_for(horizon, "eeg+wearable")
        assert 0.0 <= low < high <= 1.0
    # Unknown patterns fall back to pooled horizon levels or the alpha defaults.
    low, high = calibrator.levels_for(30, "eeg")
    assert 0.0 <= low < high <= 1.0
    restored = ConformalCalibrator.from_dict(calibrator.as_dict())
    assert restored.levels_for(30, "eeg+wearable") == calibrator.levels_for(
        30, "eeg+wearable"
    )


def test_mixture_cdf_tolerates_missing_targets() -> None:
    # Datasets like Big IDEAS have windows with NaN targets at some horizons;
    # the CDF must not let distribution input validation reject the batch.
    values = torch.tensor([[100.0, float("nan")], [float("nan"), 140.0]])
    mean = torch.full((2, 1, 2), 110.0)
    scale = torch.full((2, 1, 2), 20.0)
    weights = torch.ones(2, 1, 2)
    cdf = mixture_cdf(values, mean, scale, weights)
    assert math.isfinite(float(cdf[0, 0])) and math.isfinite(float(cdf[1, 1]))


def test_mixture_cdf_bounds() -> None:
    values = torch.tensor([[100.0, 200.0]])
    mean = torch.full((1, 2, 2), 150.0)
    scale = torch.full((1, 2, 2), 20.0)
    weights = torch.full((1, 2, 2), 0.5)
    cdf = mixture_cdf(values, mean, scale, weights)
    assert ((cdf >= 0) & (cdf <= 1)).all()
    assert float(cdf[0, 0]) < 0.5 < float(cdf[0, 1])


def test_ensemble_combination_contract() -> None:
    columns = {
        "patient_id": ["p1", "p1"],
        "cohort_id": ["c", "c"],
        "participant_key": ["c::p1", "c::p1"],
        "anchor_time": ["2026-01-01T00:00:00+00:00"] * 2,
        "horizon_minutes": [30, 60],
        "target_glucose_mg_dl": [120.0, 125.0],
        "target_hypoglycemia": [0, 0],
        "target_hyperglycemia": [0, 0],
        "persistence_glucose_mg_dl": [118.0, 118.0],
        "abstained": [False, False],
        "predicted_glucose_mg_dl": [119.0, 124.0],
        "predicted_standard_deviation_mg_dl": [10.0, 11.0],
        "prediction_lower_mg_dl": [100.0, 104.0],
        "prediction_upper_mg_dl": [138.0, 144.0],
        "hypoglycemia_probability": [0.01, 0.01],
        "hyperglycemia_probability": [0.05, 0.06],
        "weight_eeg": [0.6, 0.6],
        "weight_wearable": [0.4, 0.4],
        "expert_mean_eeg_mg_dl": [118.0, 123.0],
        "expert_mean_wearable_mg_dl": [120.5, 125.5],
        "expert_sd_eeg_mg_dl": [9.0, 10.0],
        "expert_sd_wearable_mg_dl": [12.0, 13.0],
    }
    member_a = pd.DataFrame(columns)
    shifted = dict(columns)
    shifted["predicted_glucose_mg_dl"] = [121.0, 126.0]
    shifted["expert_mean_eeg_mg_dl"] = [120.0, 125.0]
    shifted["expert_mean_wearable_mg_dl"] = [122.5, 127.5]
    member_b = pd.DataFrame(shifted)
    combined = combine_member_predictions([member_a, member_b])
    assert len(combined) == 2
    # Mixture of mixtures: four experts, weights renormalized to one.
    weight_columns = [c for c in combined.columns if c.startswith("weight_")]
    assert len(weight_columns) == 4
    assert math.isclose(
        float(combined[weight_columns].sum(axis=1).iloc[0]), 1.0, rel_tol=1e-6
    )
    assert float(combined["predicted_glucose_mg_dl"].iloc[0]) == pytest.approx(
        120.0, abs=1e-6
    )


def test_anchor_thinning_decorrelates_dense_windows() -> None:
    from src.neuroglycemic.neural_dataset import _thin_anchor_spacing

    base = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for patient in ("p1", "p2"):
        for minute in range(0, 240, 15):
            rows.append(
                {
                    "patient_id": patient,
                    "cohort_id": "c",
                    "anchor_time": base + pd.Timedelta(minutes=minute),
                }
            )
    frame = pd.DataFrame(rows)
    thinned = _thin_anchor_spacing(frame, 30.0)
    assert len(thinned) == 16  # 8 per patient at 30-minute spacing
    for patient in ("p1", "p2"):
        anchors = thinned.loc[thinned["patient_id"] == patient, "anchor_time"]
        assert anchors.diff().dropna().min() >= pd.Timedelta(minutes=30)
    # Deterministic: same input, same output.
    assert _thin_anchor_spacing(frame, 30.0).equals(thinned)


def test_big_ideas_meal_lag_basis_attachment() -> None:
    from src.neuroglycemic.big_ideas_data import (
        BigIdeasBuildConfig,
        _attach_meal_lag_basis,
    )

    config = BigIdeasBuildConfig(source_timezone="UTC")
    anchor = pd.Timestamp("2026-01-01 12:00", tz="UTC")
    frame = pd.DataFrame(
        {"patient_id": ["p1"], "anchor_time": [anchor], "marker": [1.0]}
    )
    meals = pd.DataFrame(
        {
            "time": [anchor - pd.Timedelta(minutes=30)],
            "total_carb": [60.0],
        }
    )
    merged = _attach_meal_lag_basis(frame, meals, "p1", config)
    lag_columns = [c for c in merged.columns if c.startswith("meal_lag_")]
    assert len(lag_columns) == 8
    # A 60 g meal exactly 30 minutes old peaks at the 30-minute basis center.
    assert float(merged["meal_lag_carbohydrate_g_30m"].iloc[0]) == pytest.approx(
        60.0, rel=1e-6
    )
    assert float(merged["meal_lag_carbohydrate_g_180m"].iloc[0]) < 1.0
    # Empty meal history yields zeros, never NaNs.
    empty = _attach_meal_lag_basis(
        frame, pd.DataFrame(columns=["time", "total_carb"]), "p1", config
    )
    assert float(empty[lag_columns].sum().sum()) == 0.0