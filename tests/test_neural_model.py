import pytest
import torch

from src.neuroglycemic.neural_model import (  
    NeuroGlycemicNet,
    mixture_gaussian_nll,
    neuroglycemic_loss,
)


def _inputs():
    features = {
        "eeg": torch.tensor([[0.2, float("nan")], [0.3, 0.4], [0.0, 0.0]]),
        "wearable": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0]]),
        "ehr": torch.tensor([[0.5], [0.6], [0.0]]),
    }
    masks = {name: torch.isfinite(values) for name, values in features.items()}
    availability = torch.tensor(
        [[True, True, True], [False, True, True], [False, False, False]]
    )
    quality = torch.tensor([[0.8, 0.9, 1.0], [0.0, 0.7, 0.8], [0.0, 0.0, 0.0]])
    staleness = torch.tensor([[0.0, 1.0, 5.0], [0.0, 2.0, 7.0], [0.0, 0.0, 0.0]])
    return features, masks, availability, quality, staleness


def test_forward_masks_missing_modalities_and_abstains_when_everything_is_missing():
    torch.manual_seed(3)
    model = NeuroGlycemicNet(
        {"eeg": 2, "wearable": 3, "ehr": 1}, dropout=0.0
    )
    outputs = model(*_inputs())

    assert outputs["expert_mean"].shape == (3, 3, 2)
    assert outputs["expert_scale"].shape == (3, 3, 2)
    assert torch.all(outputs["expert_scale"] > 0.05)
    assert outputs["fusion_weights"][1, 0].item() == 0.0
    assert outputs["fusion_weights"][0].sum().item() == pytest.approx(1.0)
    assert outputs["fusion_weights"][1].sum().item() == pytest.approx(1.0)
    assert torch.equal(outputs["fusion_weights"][2], torch.zeros(3))
    assert outputs["abstained"].tolist() == [False, False, True]
    assert torch.isnan(outputs["mixture_mean"][2]).all()
    assert torch.isnan(outputs["mixture_variance"][2]).all()


def test_loss_backpropagates_into_experts_and_learned_gate():
    torch.manual_seed(5)
    model = NeuroGlycemicNet(
        {"eeg": 2, "wearable": 3, "ehr": 1}, dropout=0.0
    )
    features, masks, availability, quality, staleness = _inputs()
    # Use only non-abstained examples; the third label is intentionally missing.
    target = torch.tensor([[105.0, 112.0], [118.0, 126.0], [float("nan"), float("nan")]])
    outputs = model(features, masks, availability, quality, staleness)
    losses = neuroglycemic_loss(outputs, target)
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert losses["loss"].item() > 0.0
    assert model.fusion.modality_bias.grad is not None
    assert model.fusion.modality_bias.grad.abs().sum().item() > 0.0
    for name in model.modalities:
        head_gradient = model.glucose_heads[name].projection.weight.grad
        assert head_gradient is not None
        assert torch.isfinite(head_gradient).all()
        assert head_gradient.abs().sum().item() > 0.0


def test_optimizer_step_changes_predictions_and_reduces_loss_on_tiny_batch():
    torch.manual_seed(11)
    model = NeuroGlycemicNet(
        {"eeg": 2, "wearable": 3, "ehr": 1},
        hidden_dim=12,
        embedding_dim=8,
        dropout=0.0,
        min_scale=0.5,
    )
    features, masks, availability, quality, staleness = _inputs()
    # Exclude the abstention row from this tiny optimization check.
    features = {name: values[:2] for name, values in features.items()}
    masks = {name: values[:2] for name, values in masks.items()}
    availability, quality, staleness = (
        availability[:2], quality[:2], staleness[:2]
    )
    target = torch.tensor([[1.0, 1.5], [2.0, 2.5]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)

    with torch.no_grad():
        initial_output = model(features, masks, availability, quality, staleness)
        initial_prediction = initial_output["mixture_mean"].clone()
        initial_loss = neuroglycemic_loss(initial_output, target)["loss"].item()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        output = model(features, masks, availability, quality, staleness)
        loss = neuroglycemic_loss(output, target)["loss"]
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_output = model(features, masks, availability, quality, staleness)
        final_prediction = final_output["mixture_mean"]
        final_loss = neuroglycemic_loss(final_output, target)["loss"].item()

    assert not torch.allclose(initial_prediction, final_prediction)
    assert final_loss < initial_loss


def test_mixture_nll_matches_single_selected_expert_and_rejects_bad_shapes():
    target = torch.tensor([[10.0]])
    means = torch.tensor([[[10.0], [50.0]]])
    scales = torch.ones_like(means)
    weights = torch.tensor([[1.0, 0.0]])
    loss = mixture_gaussian_nll(target, means, scales, weights)
    assert loss.item() == pytest.approx(0.5 * torch.log(torch.tensor(2.0 * torch.pi)).item())

    with pytest.raises(ValueError, match="target"):
        mixture_gaussian_nll(torch.zeros(2, 1), means, scales, weights)


def test_observed_nonfinite_value_is_rejected_instead_of_silently_imputed():
    model = NeuroGlycemicNet({"eeg": 1}, dropout=0.0)
    features = {"eeg": torch.tensor([[float("nan")]])}
    masks = {"eeg": torch.tensor([[True]])}
    one = torch.ones(1, 1)
    with pytest.raises(ValueError, match="finite"):
        model(features, masks, one.bool(), one, torch.zeros_like(one))
