import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ModalityEncoder(nn.Module):
    """Encode standardized features together with their observation mask."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, embedding_dim) <= 0:
            raise ValueError("Encoder dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(2 * input_dim),
            nn.Linear(2 * input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features: Tensor, observed: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected features shaped [batch, {self.input_dim}], "
                f"received {tuple(features.shape)}."
            )
        if observed.shape != features.shape:
            raise ValueError("The feature observation mask must match the features.")
        observed = observed.to(device=features.device, dtype=torch.bool)
        if not torch.isfinite(features[observed]).all():
            raise ValueError("Observed feature values must be finite.")

        # Zero is the training-mean value after train-only standardization.  The
        # concatenated mask allows the network to distinguish it from a real zero.
        values = torch.where(observed, features, torch.zeros_like(features))
        inputs = torch.cat((values, observed.to(features.dtype)), dim=-1)
        return self.network(inputs)


class GaussianGlucoseHead(nn.Module):
    """Predict mean and aleatoric scale for each glucose horizon."""

    def __init__(
        self, embedding_dim: int, horizon_count: int, *, min_scale: float = 0.05
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or horizon_count <= 0:
            raise ValueError("Head dimensions must be positive.")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive.")
        self.horizon_count = int(horizon_count)
        self.min_scale = float(min_scale)
        self.projection = nn.Linear(embedding_dim, 2 * horizon_count)

    def forward(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        mean, raw_scale = self.projection(embedding).chunk(2, dim=-1)
        scale = F.softplus(raw_scale) + self.min_scale
        return mean, scale


class LearnedMaskedFusion(nn.Module):
    """Learn sample-specific expert weights while enforcing availability."""

    def __init__(self, modality_count: int, embedding_dim: int) -> None:
        super().__init__()
        if modality_count <= 0 or embedding_dim <= 0:
            raise ValueError("Fusion dimensions must be positive.")
        gate_hidden = max(4, embedding_dim // 2)
        self.modality_count = int(modality_count)
        self.gate = nn.Sequential(
            nn.LayerNorm(embedding_dim + 2),
            nn.Linear(embedding_dim + 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.modality_bias = nn.Parameter(torch.zeros(modality_count))

    def forward(
        self,
        embeddings: Tensor,
        availability: Tensor,
        quality: Tensor,
        staleness: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if embeddings.ndim != 3 or embeddings.shape[1] != self.modality_count:
            raise ValueError("embeddings must be [batch, modality, embedding].")
        expected = embeddings.shape[:2]
        if any(value.shape != expected for value in (availability, quality, staleness)):
            raise ValueError("availability, quality, and staleness must be [batch, modality].")

        availability = availability.to(device=embeddings.device, dtype=torch.bool)
        quality = quality.to(device=embeddings.device, dtype=embeddings.dtype)
        staleness = staleness.to(device=embeddings.device, dtype=embeddings.dtype)
        if not torch.isfinite(quality).all() or not torch.isfinite(staleness).all():
            raise ValueError("Quality and staleness values must be finite.")
        if bool((quality < 0).any()) or bool((quality > 1).any()):
            raise ValueError("Quality must be in [0, 1].")
        if bool((staleness < 0).any()):
            raise ValueError("Staleness must be non-negative.")

        gate_features = torch.cat(
            (embeddings, quality.unsqueeze(-1), torch.log1p(staleness).unsqueeze(-1)),
            dim=-1,
        )
        logits = self.gate(gate_features).squeeze(-1) + self.modality_bias
        masked_logits = logits.masked_fill(~availability, torch.finfo(logits.dtype).min)
        weights = torch.softmax(masked_logits, dim=-1) * availability.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        abstained = ~availability.any(dim=-1)
        return weights, abstained


class NeuroGlycemicNet(nn.Module):
    """Neural modality experts with quality-aware, learned late fusion.

    Input tensors use ``model.modalities`` order for availability, quality, and
    staleness.  Feature values must already be standardized using training-only
    statistics.  Missing feature values may be NaN only where ``feature_masks``
    is false.
    """

    def __init__(
        self,
        input_dims: Mapping[str, int],
        *,
        horizons_minutes: Sequence[int] = (30, 60),
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        dropout: float = 0.1,
        min_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("At least one modality is required.")
        if any(int(value) <= 0 for value in input_dims.values()):
            raise ValueError("Every modality input dimension must be positive.")
        if not horizons_minutes or any(int(value) <= 0 for value in horizons_minutes):
            raise ValueError("At least one positive forecast horizon is required.")
        if len(set(horizons_minutes)) != len(horizons_minutes):
            raise ValueError("Forecast horizons must be unique.")

        self.modalities = tuple(input_dims)
        self.horizons_minutes = tuple(int(value) for value in horizons_minutes)
        self.input_dims = {name: int(input_dims[name]) for name in self.modalities}
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(
                    self.input_dims[name],
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                )
                for name in self.modalities
            }
        )
        self.glucose_heads = nn.ModuleDict(
            {
                name: GaussianGlucoseHead(
                    embedding_dim, len(self.horizons_minutes), min_scale=min_scale
                )
                for name in self.modalities
            }
        )
        self.fusion = LearnedMaskedFusion(len(self.modalities), embedding_dim)

    def forward(
        self,
        features: Mapping[str, Tensor],
        feature_masks: Mapping[str, Tensor],
        availability: Tensor,
        quality: Tensor,
        staleness: Tensor,
    ) -> dict[str, Tensor]:
        if availability.ndim != 2 or availability.shape[1] != len(self.modalities):
            raise ValueError(
                f"availability must be [batch, {len(self.modalities)}] in "
                f"modality order {self.modalities}."
            )
        batch_size = availability.shape[0]
        embeddings: list[Tensor] = []
        expert_means: list[Tensor] = []
        expert_scales: list[Tensor] = []
        for name in self.modalities:
            if name not in features or name not in feature_masks:
                raise KeyError(f"Missing features or feature mask for modality {name!r}.")
            values = features[name]
            mask = feature_masks[name]
            if values.shape[0] != batch_size:
                raise ValueError("Every modality must use the availability batch size.")
            embedding = self.encoders[name](values, mask)
            mean, scale = self.glucose_heads[name](embedding)
            embeddings.append(embedding)
            expert_means.append(mean)
            expert_scales.append(scale)

        embedding_tensor = torch.stack(embeddings, dim=1)
        expert_mean = torch.stack(expert_means, dim=1)
        expert_scale = torch.stack(expert_scales, dim=1)
        availability = availability.to(device=embedding_tensor.device, dtype=torch.bool)
        weights, abstained = self.fusion(
            embedding_tensor, availability, quality, staleness
        )

        expanded_weights = weights.unsqueeze(-1)
        mixture_mean = (expanded_weights * expert_mean).sum(dim=1)
        second_moment = (
            expanded_weights * (expert_scale.square() + expert_mean.square())
        ).sum(dim=1)
        mixture_variance = (second_moment - mixture_mean.square()).clamp_min(0.0)
        # An all-missing row is not a zero-glucose prediction.  NaN plus an
        # explicit flag makes accidental downstream clinical use much harder.
        mixture_mean = mixture_mean.masked_fill(abstained.unsqueeze(-1), torch.nan)
        mixture_variance = mixture_variance.masked_fill(
            abstained.unsqueeze(-1), torch.nan
        )
        return {
            "expert_mean": expert_mean,
            "expert_scale": expert_scale,
            "fusion_weights": weights,
            "mixture_mean": mixture_mean,
            "mixture_variance": mixture_variance,
            "availability": availability,
            "abstained": abstained,
        }


def gaussian_nll(
    target: Tensor,
    mean: Tensor,
    scale: Tensor,
    *,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Gaussian negative log likelihood with optional missing-label mask."""

    if target.shape != mean.shape or scale.shape != mean.shape:
        raise ValueError("target, mean, and scale must have identical shapes.")
    if not bool((scale > 0).all()) or not torch.isfinite(scale).all():
        raise ValueError("Gaussian scales must be finite and positive.")
    valid = torch.isfinite(target) if mask is None else mask.to(dtype=torch.bool)
    valid = valid & torch.isfinite(target)
    # Sanitizing before arithmetic is essential: masking a NaN loss afterward
    # still lets NaN gradients flow through autograd.
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    values = 0.5 * (
        math.log(2.0 * math.pi)
        + 2.0 * torch.log(scale)
        + ((safe_target - mean) / scale).square()
    )
    if reduction == "none":
        return values.masked_fill(~valid, torch.nan)
    selected = torch.where(valid, values, torch.zeros_like(values))
    if reduction == "sum":
        return selected.sum()
    if reduction == "mean":
        return selected.sum() / valid.sum().clamp_min(1)
    raise ValueError("reduction must be 'none', 'sum', or 'mean'.")


def mixture_gaussian_nll(
    target: Tensor,
    expert_mean: Tensor,
    expert_scale: Tensor,
    fusion_weights: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Mean NLL under the learned Gaussian mixture, excluding invalid rows."""

    if expert_mean.ndim != 3 or expert_scale.shape != expert_mean.shape:
        raise ValueError("Expert tensors must be [batch, modality, horizon].")
    if target.shape != (expert_mean.shape[0], expert_mean.shape[2]):
        raise ValueError("target must be [batch, horizon].")
    if fusion_weights.shape != expert_mean.shape[:2]:
        raise ValueError("fusion_weights must be [batch, modality].")
    if not bool((expert_scale > 0).all()):
        raise ValueError("Expert scales must be positive.")

    has_expert = fusion_weights.sum(dim=1) > 0
    valid = torch.isfinite(target) if mask is None else mask.to(dtype=torch.bool)
    valid = valid & torch.isfinite(target) & has_expert.unsqueeze(-1)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    expanded_target = safe_target.unsqueeze(1)
    log_probability = -0.5 * (
        math.log(2.0 * math.pi)
        + 2.0 * torch.log(expert_scale)
        + ((expanded_target - expert_mean) / expert_scale).square()
    )
    # Give all-missing rows a harmless dummy expert for the internal logsumexp.
    # Those rows remain excluded by ``valid``.  logsumexp([-inf, ...]) has an
    # undefined gradient, so masking only after it is too late for autograd.
    dummy = torch.zeros_like(fusion_weights)
    dummy[:, 0] = (~has_expert).to(fusion_weights.dtype)
    effective_weights = fusion_weights + dummy
    negative_infinity = torch.full_like(effective_weights, -torch.inf)
    safe_weights = torch.where(
        effective_weights > 0, effective_weights, torch.ones_like(effective_weights)
    )
    log_weights = torch.where(
        effective_weights > 0, torch.log(safe_weights), negative_infinity
    )
    mixture_nll = -torch.logsumexp(log_weights.unsqueeze(-1) + log_probability, dim=1)
    selected = torch.where(valid, mixture_nll, torch.zeros_like(mixture_nll))
    return selected.sum() / valid.sum().clamp_min(1)


def neuroglycemic_loss(
    outputs: Mapping[str, Tensor],
    target: Tensor,
    *,
    target_mask: Tensor | None = None,
    expert_loss_weight: float = 0.25,
) -> dict[str, Tensor]:
    """Train the fused mixture and every available unimodal expert."""

    if expert_loss_weight < 0:
        raise ValueError("expert_loss_weight must be non-negative.")
    expert_mean = outputs["expert_mean"]
    expert_scale = outputs["expert_scale"]
    weights = outputs["fusion_weights"]
    availability = outputs["availability"].to(dtype=torch.bool)
    label_mask = torch.isfinite(target) if target_mask is None else target_mask.bool()
    label_mask = label_mask & torch.isfinite(target)

    mixture_nll = mixture_gaussian_nll(
        target,
        expert_mean,
        expert_scale,
        weights,
        mask=label_mask,
    )
    expanded_target = target.unsqueeze(1).expand_as(expert_mean)
    expert_mask = availability.unsqueeze(-1) & label_mask.unsqueeze(1)
    expert_nll = gaussian_nll(
        expanded_target,
        expert_mean,
        expert_scale,
        mask=expert_mask,
    )
    total = mixture_nll + float(expert_loss_weight) * expert_nll
    return {"loss": total, "mixture_nll": mixture_nll, "expert_nll": expert_nll}
