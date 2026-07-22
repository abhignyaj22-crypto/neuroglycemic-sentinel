"""Small neural mixture-of-experts for patient-level glucose forecasting.

The module deliberately contains no clinical threshold rules.  Signal quality,
staleness, and modality availability are inputs to a learned fusion gate; only
availability is enforced as a hard mask because a missing measurement cannot be
used safely.  Deterministic signal processing and unit normalization belong
upstream of this model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

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


class CrossModalContextLayer(nn.Module):
    """One availability-masked self-attention exchange between modality tokens.

    Experts in the baseline architecture commit to their predictions without
    ever seeing each other; this layer lets every modality token attend to the
    other *available* tokens before the Gaussian heads and the fusion gate
    consume the embeddings.  Missing modalities are excluded as keys/values,
    and their own tokens are zeroed on the residual path, so sensor dropout can
    never inject fabricated cross-modal information.  The returned attention
    matrix is a per-sample interpretability artifact.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be positive and divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, tokens: Tensor, availability: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.embedding_dim:
            raise ValueError("tokens must be [batch, modality, embedding].")
        if availability.shape != tokens.shape[:2]:
            raise ValueError("availability must be [batch, modality].")
        availability = availability.to(device=tokens.device, dtype=torch.bool)
        key_padding_mask = ~availability
        fully_missing = key_padding_mask.all(dim=1)
        if bool(fully_missing.any()):
            # MultiheadAttention produces NaN when a query has no valid keys.
            # Fully-missing rows abstain downstream; unmask them only to keep
            # the forward pass finite.
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_missing] = False
        normalized = self.norm1(tokens)
        attended, attention = self.attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))
        # Unavailable modality tokens carry no information downstream.
        tokens = tokens.masked_fill(~availability.unsqueeze(-1), 0.0)
        return tokens, attention


class HorizonFilmHead(nn.Module):
    """Horizon-conditioned Gaussian head sharing strength across horizons.

    A learned per-horizon code (initialized from ``log(minutes)``) modulates
    the expert embedding through feature-wise scale-and-shift (FiLM) before
    the mean/scale projection.  Adjacent horizons therefore partially pool
    statistical strength instead of learning independent linear maps from the
    same embedding.
    """

    def __init__(
        self,
        embedding_dim: int,
        horizons_minutes: Sequence[int],
        *,
        min_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or not horizons_minutes:
            raise ValueError("Head dimensions must be positive.")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive.")
        self.horizons_minutes = tuple(int(value) for value in horizons_minutes)
        self.min_scale = float(min_scale)
        initial = torch.log(
            torch.tensor(self.horizons_minutes, dtype=torch.float32)
        ).unsqueeze(-1)
        self.horizon_code = nn.Parameter(
            initial.repeat(1, embedding_dim) * 0.02
        )
        self.film = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.out = nn.Linear(embedding_dim, 2)

    def forward(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        # embedding: [batch, embedding] -> conditioned: [batch, horizon, embedding]
        conditioned = embedding.unsqueeze(1) + self.horizon_code.unsqueeze(0)
        gamma, beta = self.film(conditioned).chunk(2, dim=-1)
        conditioned = torch.tanh(gamma) * conditioned + beta
        mean, raw_scale = self.out(conditioned).chunk(2, dim=-1)
        scale = F.softplus(raw_scale) + self.min_scale
        return mean.squeeze(-1), scale.squeeze(-1)


class ResponseKernelHead(nn.Module):
    """Learned, sign-constrained delayed event-to-glucose response operator.

    Each event channel (for example carbohydrate or bolus insulin) acts on
    glucose through a learned non-negative-magnitude kernel over the causal
    lag basis produced upstream (``meal_context``).  Physiological direction
    is enforced by a signed softplus on the kernel coefficients, never by a
    branch on model outputs: carbohydrate-family channels are constrained to
    raise glucose and insulin channels to lower it, while the *magnitude* and
    *timing* of the response are learned from data.

    A low-rank patient adapter personalizes the response gain.  Patients that
    were unseen during training (``seen_patient == False`` or a null
    ``patient_index``) receive exactly the population kernel, so
    personalization degrades gracefully instead of fabricating a patient
    identity.
    """

    def __init__(
        self,
        channels: Mapping[str, float],
        basis_centers_minutes: Sequence[float],
        horizon_count: int,
        *,
        patient_count: int = 0,
        rank: int = 4,
        max_gain_deviation: float = 0.25,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("The response kernel needs at least one event channel.")
        if any(sign not in (-1.0, 1.0) for sign in channels.values()):
            raise ValueError("Response channel signs must be +1.0 or -1.0.")
        if not basis_centers_minutes or any(
            value < 0 for value in basis_centers_minutes
        ):
            raise ValueError("Kernel basis centers must be non-negative minutes.")
        if horizon_count <= 0 or rank <= 0 or patient_count < 0:
            raise ValueError("Invalid kernel dimensions.")
        if not 0 < max_gain_deviation < 1:
            raise ValueError("max_gain_deviation must be in (0, 1).")
        self.channels = tuple(str(name) for name in channels)
        self.basis_centers_minutes = tuple(
            float(value) for value in basis_centers_minutes
        )
        self.horizon_count = int(horizon_count)
        self.patient_count = int(patient_count)
        self.max_gain_deviation = float(max_gain_deviation)
        signs = torch.tensor(
            [float(channels[name]) for name in self.channels], dtype=torch.float32
        )
        self.register_buffer("channel_signs", signs)
        # Kernel *shape* is learned in softplus space (gradient 0.5 at the zero
        # init, so Adam moves it immediately); kernel *scale* is learned in log
        # space and starts near zero, so the operator is effectively inactive
        # until the data support a response.  This avoids both the large
        # spurious excursion of a raw zero init (softplus(0) ~= 0.69 on
        # gram-scaled inputs) and the frozen gradients of a very negative
        # init (sigmoid(-6) ~= 0.002).
        self.raw_kernel = nn.Parameter(
            torch.zeros(len(self.channels), len(self.basis_centers_minutes), horizon_count)
        )
        self.log_gain = nn.Parameter(
            torch.full((1, 1, horizon_count), math.log(1e-3))
        )
        if self.patient_count > 0:
            self.patient_embed = nn.Embedding(self.patient_count, rank)
            nn.init.zeros_(self.patient_embed.weight)
            self.gain_adapter = nn.Linear(rank, horizon_count)
            nn.init.zeros_(self.gain_adapter.weight)
            nn.init.zeros_(self.gain_adapter.bias)
        else:
            self.patient_embed = None
            self.gain_adapter = None

    def kernels(self) -> Tensor:
        """Signed, magnitude-learned kernels shaped [channel, basis, horizon]."""

        shape = self.channel_signs.view(-1, 1, 1) * F.softplus(self.raw_kernel)
        return shape * torch.exp(self.log_gain)

    def forward(
        self,
        event_basis: Mapping[str, Tensor],
        patient_index: Tensor | None = None,
        seen_patient: Tensor | None = None,
    ) -> Tensor:
        missing = [name for name in self.channels if name not in event_basis]
        if missing:
            raise KeyError(f"event_basis is missing channels: {missing}")
        basis = torch.stack(
            [event_basis[name] for name in self.channels], dim=1
        ).to(dtype=self.raw_kernel.dtype)
        if basis.ndim != 3 or basis.shape[2] != len(self.basis_centers_minutes):
            raise ValueError(
                "Each event channel must be [batch, basis_centers]."
            )
        if bool((basis < 0).any()) or not torch.isfinite(basis).all():
            raise ValueError("Event basis values must be finite and non-negative.")
        # [batch, channel, basis] @ [channel, basis, horizon] -> [batch, horizon]
        delta = torch.einsum("bck,ckh->bh", basis, self.kernels().to(basis.dtype))
        if self.patient_embed is not None and patient_index is not None:
            if seen_patient is None:
                raise ValueError("seen_patient is required when patient_index is given.")
            patient_index = patient_index.to(
                device=delta.device, dtype=torch.long
            ).clamp_(0, self.patient_count - 1)
            seen = seen_patient.to(device=delta.device, dtype=torch.bool)
            raw_gain = self.gain_adapter(self.patient_embed(patient_index))
            gain = 1.0 + self.max_gain_deviation * torch.tanh(raw_gain)
            gain = torch.where(
                seen.unsqueeze(-1), gain, torch.ones_like(gain)
            )
            delta = delta * gain
        return delta


class LearnedMaskedFusion(nn.Module):
    """Learn sample- and horizon-specific weights while enforcing availability."""

    def __init__(
        self, modality_count: int, embedding_dim: int, *, horizon_count: int = 1
    ) -> None:
        super().__init__()
        if modality_count <= 0 or embedding_dim <= 0 or horizon_count <= 0:
            raise ValueError("Fusion dimensions must be positive.")
        gate_hidden = max(4, embedding_dim // 2)
        self.modality_count = int(modality_count)
        self.horizon_count = int(horizon_count)
        self.gate = nn.Sequential(
            nn.LayerNorm(embedding_dim + 3),
            nn.Linear(embedding_dim + 3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, horizon_count),
        )
        self.modality_bias = nn.Parameter(torch.zeros(modality_count, horizon_count))

    def forward(
        self,
        embeddings: Tensor,
        availability: Tensor,
        quality: Tensor,
        staleness: Tensor,
        clock_uncertainty: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if embeddings.ndim != 3 or embeddings.shape[1] != self.modality_count:
            raise ValueError("embeddings must be [batch, modality, embedding].")
        expected = embeddings.shape[:2]
        if any(value.shape != expected for value in (availability, quality, staleness)):
            raise ValueError("availability, quality, and staleness must be [batch, modality].")

        availability = availability.to(device=embeddings.device, dtype=torch.bool)
        quality = quality.to(device=embeddings.device, dtype=embeddings.dtype)
        staleness = staleness.to(device=embeddings.device, dtype=embeddings.dtype)
        if clock_uncertainty is None:
            clock_uncertainty = torch.zeros_like(staleness)
        if clock_uncertainty.shape != expected:
            raise ValueError("clock_uncertainty must be [batch, modality].")
        clock_uncertainty = clock_uncertainty.to(
            device=embeddings.device, dtype=embeddings.dtype
        )
        if not torch.isfinite(quality).all() or not torch.isfinite(staleness).all():
            raise ValueError("Quality and staleness values must be finite.")
        if not torch.isfinite(clock_uncertainty).all():
            raise ValueError("Clock uncertainty values must be finite.")
        if bool((quality < 0).any()) or bool((quality > 1).any()):
            raise ValueError("Quality must be in [0, 1].")
        if bool((staleness < 0).any()):
            raise ValueError("Staleness must be non-negative.")
        if bool((clock_uncertainty < 0).any()):
            raise ValueError("Clock uncertainty must be non-negative.")

        gate_features = torch.cat(
            (
                embeddings,
                quality.unsqueeze(-1),
                torch.log1p(staleness).unsqueeze(-1),
                torch.log1p(clock_uncertainty).unsqueeze(-1),
            ),
            dim=-1,
        )
        logits = self.gate(gate_features) + self.modality_bias
        expanded_availability = availability.unsqueeze(-1)
        masked_logits = logits.masked_fill(
            ~expanded_availability, torch.finfo(logits.dtype).min
        )
        weights = torch.softmax(masked_logits, dim=1) * expanded_availability.to(
            logits.dtype
        )
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
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
        modality_dropout_probability: float = 0.0,
        auxiliary_task_kinds: Mapping[str, str] | None = None,
        cross_modal_layers: int = 0,
        cross_modal_heads: int = 4,
        horizon_film: bool = False,
        response_kernel: Mapping[str, Any] | None = None,
        build_reconstruction_heads: bool = False,
    ) -> None:
        super().__init__()
        if cross_modal_layers < 0 or cross_modal_heads <= 0:
            raise ValueError("cross_modal_layers must be >= 0 and heads positive.")
        if not input_dims:
            raise ValueError("At least one modality is required.")
        if any(int(value) <= 0 for value in input_dims.values()):
            raise ValueError("Every modality input dimension must be positive.")
        if not horizons_minutes or any(int(value) <= 0 for value in horizons_minutes):
            raise ValueError("At least one positive forecast horizon is required.")
        if len(set(horizons_minutes)) != len(horizons_minutes):
            raise ValueError("Forecast horizons must be unique.")
        if not 0.0 <= modality_dropout_probability < 1.0:
            raise ValueError("modality_dropout_probability must be in [0, 1).")

        self.modalities = tuple(input_dims)
        self.horizons_minutes = tuple(int(value) for value in horizons_minutes)
        self.input_dims = {name: int(input_dims[name]) for name in self.modalities}
        self.modality_dropout_probability = float(modality_dropout_probability)
        auxiliary_task_kinds = dict(auxiliary_task_kinds or {})
        if any(not name.isidentifier() for name in auxiliary_task_kinds):
            raise ValueError("Auxiliary task names must be Python identifiers.")
        if any(kind not in {"binary", "continuous"} for kind in auxiliary_task_kinds.values()):
            raise ValueError("Auxiliary task kind must be binary or continuous.")
        self.auxiliary_task_kinds = auxiliary_task_kinds
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
        self.cross_modal_layers = int(cross_modal_layers)
        self.cross_modal_heads = int(cross_modal_heads)
        self.horizon_film = bool(horizon_film)
        if self.horizon_film:
            self.glucose_heads = nn.ModuleDict(
                {
                    name: HorizonFilmHead(
                        embedding_dim,
                        self.horizons_minutes,
                        min_scale=min_scale,
                    )
                    for name in self.modalities
                }
            )
        else:
            self.glucose_heads = nn.ModuleDict(
                {
                    name: GaussianGlucoseHead(
                        embedding_dim, len(self.horizons_minutes), min_scale=min_scale
                    )
                    for name in self.modalities
                }
            )
        self.context_layers = nn.ModuleList(
            CrossModalContextLayer(
                embedding_dim, num_heads=self.cross_modal_heads, dropout=dropout
            )
            for _ in range(self.cross_modal_layers)
        )
        # Sign-constrained delayed event response (for example meal/insulin).
        # ``response_kernel`` expects: channels {name: +1.0|-1.0},
        # basis_centers_minutes, optional patient_count and rank.
        if response_kernel is not None:
            kernel_spec = dict(response_kernel)
            self.response_kernel = ResponseKernelHead(
                {str(k): float(v) for k, v in dict(kernel_spec["channels"]).items()},
                tuple(float(v) for v in kernel_spec["basis_centers_minutes"]),
                len(self.horizons_minutes),
                patient_count=int(kernel_spec.get("patient_count", 0)),
                rank=int(kernel_spec.get("rank", 4)),
                max_gain_deviation=float(
                    kernel_spec.get("max_gain_deviation", 0.25)
                ),
            )
        else:
            self.response_kernel = None
        # Reconstruction decoders exist only for self-supervised pretraining;
        # they are dropped before the forecasting checkpoint is serialized.
        self.has_reconstruction_heads = bool(build_reconstruction_heads)
        self.reconstruction_heads = (
            nn.ModuleDict(
                {
                    name: nn.Linear(embedding_dim, self.input_dims[name])
                    for name in self.modalities
                }
            )
            if build_reconstruction_heads
            else nn.ModuleDict()
        )
        self.fusion = LearnedMaskedFusion(
            len(self.modalities),
            embedding_dim,
            horizon_count=len(self.horizons_minutes),
        )
        self.auxiliary_heads = nn.ModuleDict(
            {name: nn.Linear(embedding_dim, 1) for name in auxiliary_task_kinds}
        )

    def _drop_modalities(self, availability: Tensor) -> Tensor:
        if not self.training or self.modality_dropout_probability <= 0:
            return availability
        keep = torch.rand(
            availability.shape, device=availability.device
        ) >= self.modality_dropout_probability
        dropped = availability & keep
        # A row that originally had a sensor must never become an artificial
        # abstention during optimization. Retain its first available modality.
        needs_one = availability.any(dim=1) & ~dropped.any(dim=1)
        if bool(needs_one.any()):
            first_available = availability.to(torch.int64).argmax(dim=1)
            dropped[needs_one, first_available[needs_one]] = True
        return dropped

    def architecture_extras(self) -> dict[str, Any]:
        """Serializable non-default architecture choices for serving contracts."""

        kernel_spec: dict[str, Any] | None = None
        if self.response_kernel is not None:
            kernel_spec = {
                "channels": {
                    name: float(sign)
                    for name, sign in zip(
                        self.response_kernel.channels,
                        self.response_kernel.channel_signs.tolist(),
                    )
                },
                "basis_centers_minutes": list(
                    self.response_kernel.basis_centers_minutes
                ),
                "patient_count": self.response_kernel.patient_count,
                "rank": (
                    int(self.response_kernel.patient_embed.embedding_dim)
                    if self.response_kernel.patient_embed is not None
                    else 4
                ),
                "max_gain_deviation": self.response_kernel.max_gain_deviation,
            }
        return {
            "cross_modal_layers": self.cross_modal_layers,
            "cross_modal_heads": self.cross_modal_heads,
            "horizon_film": self.horizon_film,
            "response_kernel": kernel_spec,
        }

    def drop_reconstruction_heads(self) -> None:
        """Remove pretraining decoders before a forecasting checkpoint is saved."""

        self.reconstruction_heads = nn.ModuleDict()
        self.has_reconstruction_heads = False

    def encode_contextualized(
        self,
        features: Mapping[str, Tensor],
        feature_masks: Mapping[str, Tensor],
        availability: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Shared encoder + cross-modal context path used by both tasks."""

        embeddings = torch.stack(
            [
                self.encoders[name](features[name], feature_masks[name])
                for name in self.modalities
            ],
            dim=1,
        )
        attention: Tensor | None = None
        for layer in self.context_layers:
            embeddings, attention = layer(embeddings, availability)
        return embeddings, attention

    def forward(
        self,
        features: Mapping[str, Tensor],
        feature_masks: Mapping[str, Tensor],
        availability: Tensor,
        quality: Tensor,
        staleness: Tensor,
        clock_uncertainty: Tensor | None = None,
        patient_index: Tensor | None = None,
        seen_patient: Tensor | None = None,
        event_basis: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        if availability.ndim != 2 or availability.shape[1] != len(self.modalities):
            raise ValueError(
                f"availability must be [batch, {len(self.modalities)}] in "
                f"modality order {self.modalities}."
            )
        batch_size = availability.shape[0]
        for name in self.modalities:
            if name not in features or name not in feature_masks:
                raise KeyError(f"Missing features or feature mask for modality {name!r}.")
            if features[name].shape[0] != batch_size:
                raise ValueError("Every modality must use the availability batch size.")
        availability = availability.to(
            device=features[self.modalities[0]].device, dtype=torch.bool
        )
        expert_availability = availability
        gate_availability = self._drop_modalities(availability)
        embedding_tensor, cross_attention = self.encode_contextualized(
            features, feature_masks, gate_availability
        )
        expert_means: list[Tensor] = []
        expert_scales: list[Tensor] = []
        for index, name in enumerate(self.modalities):
            mean, scale = self.glucose_heads[name](embedding_tensor[:, index])
            expert_means.append(mean)
            expert_scales.append(scale)

        expert_mean = torch.stack(expert_means, dim=1)
        expert_scale = torch.stack(expert_scales, dim=1)
        response_delta: Tensor | None = None
        if self.response_kernel is not None:
            if event_basis is None:
                # No recorded pre-anchor events: the operator contributes zero.
                # This is semantically "no events", never imputed event data.
                event_basis = {
                    name: torch.zeros(
                        batch_size,
                        len(self.response_kernel.basis_centers_minutes),
                        device=expert_mean.device,
                        dtype=expert_mean.dtype,
                    )
                    for name in self.response_kernel.channels
                }
            response_delta = self.response_kernel(
                event_basis, patient_index=patient_index, seen_patient=seen_patient
            )
            expert_mean = expert_mean + response_delta.unsqueeze(1)
        weights, abstained = self.fusion(
            embedding_tensor,
            gate_availability,
            quality,
            staleness,
            clock_uncertainty,
        )

        mixture_mean = (weights * expert_mean).sum(dim=1)
        second_moment = (
            weights * (expert_scale.square() + expert_mean.square())
        ).sum(dim=1)
        mixture_variance = (second_moment - mixture_mean.square()).clamp_min(0.0)
        mean_weights = weights.mean(dim=-1)
        fused_embedding = (mean_weights.unsqueeze(-1) * embedding_tensor).sum(dim=1)
        auxiliary_outputs = {
            name: head(fused_embedding).squeeze(-1).masked_fill(abstained, torch.nan)
            for name, head in self.auxiliary_heads.items()
        }
        # An all-missing row is not a zero-glucose prediction.  NaN plus an
        # explicit flag makes accidental downstream clinical use much harder.
        mixture_mean = mixture_mean.masked_fill(abstained.unsqueeze(-1), torch.nan)
        mixture_variance = mixture_variance.masked_fill(
            abstained.unsqueeze(-1), torch.nan
        )
        return {
            "expert_mean": expert_mean,
            "expert_scale": expert_scale,
            # The mean is retained for v3 result readers. Scientific evaluation
            # and serving use the horizon-specific tensor below.
            "fusion_weights": weights.mean(dim=-1),
            "fusion_weights_by_horizon": weights,
            "mixture_mean": mixture_mean,
            "mixture_variance": mixture_variance,
            "availability": gate_availability,
            "expert_availability": expert_availability,
            "abstained": abstained,
            "auxiliary_outputs": auxiliary_outputs,
            "cross_attention": cross_attention,
            "response_delta": response_delta,
        }


def gaussian_nll(
    target: Tensor,
    mean: Tensor,
    scale: Tensor,
    *,
    mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
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
    weights = torch.ones_like(values)
    if sample_weight is not None:
        if sample_weight.shape != (target.shape[0],):
            raise ValueError("sample_weight must be [batch].")
        if not torch.isfinite(sample_weight).all() or bool((sample_weight <= 0).any()):
            raise ValueError("sample_weight must be finite and positive.")
        weights = sample_weight.to(values).view(-1, *([1] * (values.ndim - 1))).expand_as(values)
    selected = torch.where(valid, values * weights, torch.zeros_like(values))
    if reduction == "sum":
        return selected.sum()
    if reduction == "mean":
        denominator = torch.where(valid, weights, torch.zeros_like(weights)).sum()
        return selected.sum() / denominator.clamp_min(1.0)
    raise ValueError("reduction must be 'none', 'sum', or 'mean'.")


def mixture_gaussian_nll(
    target: Tensor,
    expert_mean: Tensor,
    expert_scale: Tensor,
    fusion_weights: Tensor,
    *,
    mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Mean NLL under the learned Gaussian mixture, excluding invalid rows."""

    if expert_mean.ndim != 3 or expert_scale.shape != expert_mean.shape:
        raise ValueError("Expert tensors must be [batch, modality, horizon].")
    if target.shape != (expert_mean.shape[0], expert_mean.shape[2]):
        raise ValueError("target must be [batch, horizon].")
    if fusion_weights.shape == expert_mean.shape[:2]:
        fusion_weights = fusion_weights.unsqueeze(-1).expand_as(expert_mean)
    elif fusion_weights.shape != expert_mean.shape:
        raise ValueError(
            "fusion_weights must be [batch, modality] or [batch, modality, horizon]."
        )
    if not bool((expert_scale > 0).all()):
        raise ValueError("Expert scales must be positive.")

    has_expert = fusion_weights.sum(dim=1) > 0
    valid = torch.isfinite(target) if mask is None else mask.to(dtype=torch.bool)
    valid = valid & torch.isfinite(target) & has_expert
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
    dummy[:, 0, :] = (~has_expert).to(fusion_weights.dtype)
    effective_weights = fusion_weights + dummy
    negative_infinity = torch.full_like(effective_weights, -torch.inf)
    safe_weights = torch.where(
        effective_weights > 0, effective_weights, torch.ones_like(effective_weights)
    )
    log_weights = torch.where(
        effective_weights > 0, torch.log(safe_weights), negative_infinity
    )
    mixture_nll = -torch.logsumexp(log_weights + log_probability, dim=1)
    row_weights = torch.ones_like(mixture_nll)
    if sample_weight is not None:
        if sample_weight.shape != (target.shape[0],):
            raise ValueError("sample_weight must be [batch].")
        if not torch.isfinite(sample_weight).all() or bool((sample_weight <= 0).any()):
            raise ValueError("sample_weight must be finite and positive.")
        row_weights = sample_weight.to(mixture_nll).unsqueeze(-1).expand_as(mixture_nll)
    selected = torch.where(valid, mixture_nll * row_weights, torch.zeros_like(mixture_nll))
    denominator = torch.where(valid, row_weights, torch.zeros_like(row_weights)).sum()
    return selected.sum() / denominator.clamp_min(1.0)


def mixture_crps(
    target: Tensor,
    expert_mean: Tensor,
    expert_scale: Tensor,
    fusion_weights: Tensor,
    *,
    mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Closed-form, differentiable CRPS of the learned Gaussian mixture.

    NLL alone is minimized by inflating the predicted scales; CRPS scores the
    full distribution against the observation and penalizes interval width and
    location error simultaneously.  For a Gaussian mixture the score has the
    closed form (Grimit et al., 2006)::

        CRPS(F, y) = sum_m w_m A(mu_m, sigma_m, y)
                     - 0.5 sum_m sum_n w_m w_n A(mu_m, sigma_m, mu_n, sigma_n)

    with ``A(mu, sigma, y) = sigma * [z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)]``
    and ``z = (y - mu) / sigma``.  The pairwise term uses the fact that the
    difference of two Gaussians is Gaussian with ``sqrt(sigma_m^2 + sigma_n^2)``.
    """

    if expert_mean.ndim != 3 or expert_scale.shape != expert_mean.shape:
        raise ValueError("Expert tensors must be [batch, modality, horizon].")
    if target.shape != (expert_mean.shape[0], expert_mean.shape[2]):
        raise ValueError("target must be [batch, horizon].")
    if fusion_weights.shape == expert_mean.shape[:2]:
        fusion_weights = fusion_weights.unsqueeze(-1).expand_as(expert_mean)
    elif fusion_weights.shape != expert_mean.shape:
        raise ValueError(
            "fusion_weights must be [batch, modality] or [batch, modality, horizon]."
        )
    if not bool((expert_scale > 0).all()):
        raise ValueError("Expert scales must be positive.")

    standard_normal = torch.distributions.Normal(
        torch.zeros((), device=expert_mean.device, dtype=expert_mean.dtype),
        torch.ones((), device=expert_mean.device, dtype=expert_mean.dtype),
    )

    def _a(mean: Tensor, scale: Tensor, value: Tensor) -> Tensor:
        z = (value - mean) / scale
        return scale * (
            z * (2.0 * standard_normal.cdf(z) - 1.0)
            + 2.0 * torch.exp(standard_normal.log_prob(z))
            - 1.0 / math.sqrt(math.pi)
        )

    has_expert = fusion_weights.sum(dim=1) > 0
    valid = torch.isfinite(target) if mask is None else mask.to(dtype=torch.bool)
    valid = valid & torch.isfinite(target) & has_expert
    safe_target = torch.where(valid, target, torch.zeros_like(target))

    expanded_target = safe_target.unsqueeze(1).expand_as(expert_mean)
    first_term = (fusion_weights * _a(expert_mean, expert_scale, expanded_target)).sum(1)

    modality_count = expert_mean.shape[1]
    mean_i = expert_mean.unsqueeze(2).expand(-1, -1, modality_count, -1)
    scale_i = expert_scale.unsqueeze(2).expand(-1, -1, modality_count, -1)
    mean_j = expert_mean.unsqueeze(1).expand(-1, modality_count, -1, -1)
    scale_j = expert_scale.unsqueeze(1).expand(-1, modality_count, -1, -1)
    # The difference of two independent Gaussians is Gaussian.
    pair_scale = torch.sqrt(scale_i.square() + scale_j.square())
    pair_term = _a(mean_i, pair_scale, mean_j)
    weight_pairs = fusion_weights.unsqueeze(2) * fusion_weights.unsqueeze(1)
    second_term = (weight_pairs * pair_term).sum(dim=(1, 2))

    crps = first_term - 0.5 * second_term
    row_weights = torch.ones_like(crps)
    if sample_weight is not None:
        if sample_weight.shape != (target.shape[0],):
            raise ValueError("sample_weight must be [batch].")
        if not torch.isfinite(sample_weight).all() or bool((sample_weight <= 0).any()):
            raise ValueError("sample_weight must be finite and positive.")
        row_weights = sample_weight.to(crps).unsqueeze(-1).expand_as(crps)
    selected = torch.where(valid, crps * row_weights, torch.zeros_like(crps))
    denominator = torch.where(valid, row_weights, torch.zeros_like(row_weights)).sum()
    return selected.sum() / denominator.clamp_min(1.0)


def neuroglycemic_loss(
    outputs: Mapping[str, Tensor],
    target: Tensor,
    *,
    target_mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
    expert_loss_weight: float = 0.25,
    crps_loss_weight: float = 0.0,
    auxiliary_targets: Mapping[str, Tensor] | None = None,
    auxiliary_masks: Mapping[str, Tensor] | None = None,
    auxiliary_task_kinds: Mapping[str, str] | None = None,
    auxiliary_loss_weights: Mapping[str, float] | None = None,
) -> dict[str, Tensor]:
    """Train the fused mixture and every available unimodal expert."""

    if expert_loss_weight < 0:
        raise ValueError("expert_loss_weight must be non-negative.")
    if not math.isfinite(crps_loss_weight) or crps_loss_weight < 0:
        raise ValueError("crps_loss_weight must be finite and non-negative.")
    expert_mean = outputs["expert_mean"]
    expert_scale = outputs["expert_scale"]
    weights = outputs.get("fusion_weights_by_horizon", outputs["fusion_weights"])
    availability = outputs.get(
        "expert_availability", outputs["availability"]
    ).to(dtype=torch.bool)
    label_mask = torch.isfinite(target) if target_mask is None else target_mask.bool()
    label_mask = label_mask & torch.isfinite(target)

    mixture_nll = mixture_gaussian_nll(
        target,
        expert_mean,
        expert_scale,
        weights,
        mask=label_mask,
        sample_weight=sample_weight,
    )
    expert_mask = availability.unsqueeze(-1) & label_mask.unsqueeze(1)
    # Normalize each modality separately so a frequently observed stream cannot
    # dominate the expert objective solely through coverage.
    per_modality = []
    for modality_index in range(expert_mean.shape[1]):
        modality_mask = expert_mask[:, modality_index, :]
        if bool(modality_mask.any()):
            per_modality.append(
                gaussian_nll(
                    target,
                    expert_mean[:, modality_index, :],
                    expert_scale[:, modality_index, :],
                    mask=modality_mask,
                    sample_weight=sample_weight,
                )
            )
    expert_nll = (
        torch.stack(per_modality).mean()
        if per_modality
        else mixture_nll.new_zeros(())
    )
    total = mixture_nll + float(expert_loss_weight) * expert_nll
    result = {"mixture_nll": mixture_nll, "expert_nll": expert_nll}
    if crps_loss_weight > 0:
        crps = mixture_crps(
            target,
            expert_mean,
            expert_scale,
            weights,
            mask=label_mask,
            sample_weight=sample_weight,
        )
        total = total + float(crps_loss_weight) * crps
        result["mixture_crps"] = crps
    task_outputs = outputs.get("auxiliary_outputs", {})
    targets = auxiliary_targets or {}
    masks = auxiliary_masks or {}
    kinds = auxiliary_task_kinds or {}
    weights_by_task = auxiliary_loss_weights or {}
    if set(task_outputs) != set(kinds):
        if task_outputs or kinds:
            raise ValueError("Model auxiliary outputs do not match configured tasks.")
    for name, prediction in task_outputs.items():
        if name not in targets:
            raise KeyError(f"Missing auxiliary target for {name!r}.")
        target_value = targets[name].to(device=prediction.device, dtype=prediction.dtype)
        valid = masks.get(name, torch.isfinite(target_value)).to(
            device=prediction.device, dtype=torch.bool
        ) & torch.isfinite(target_value)
        if target_value.shape != prediction.shape or valid.shape != prediction.shape:
            raise ValueError("Auxiliary predictions, targets, and masks must match.")
        safe_target = torch.where(valid, target_value, torch.zeros_like(target_value))
        safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
        if kinds[name] == "binary":
            if bool(((safe_target[valid] < 0) | (safe_target[valid] > 1)).any()):
                raise ValueError("Binary auxiliary targets must be in [0, 1].")
            per_row = F.binary_cross_entropy_with_logits(
                safe_prediction, safe_target, reduction="none"
            )
        elif kinds[name] == "continuous":
            per_row = (safe_prediction - safe_target).square()
        else:
            raise ValueError(f"Unsupported auxiliary task kind {kinds[name]!r}.")
        task_weight = torch.ones_like(per_row)
        if sample_weight is not None:
            if sample_weight.shape != prediction.shape:
                raise ValueError("sample_weight must match one-dimensional auxiliary targets.")
            task_weight = sample_weight.to(per_row)
        task_loss = torch.where(
            valid, per_row * task_weight, torch.zeros_like(per_row)
        ).sum() / torch.where(
            valid, task_weight, torch.zeros_like(task_weight)
        ).sum().clamp_min(1.0)
        weight = float(weights_by_task.get(name, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Auxiliary loss weights must be finite and non-negative.")
        total = total + weight * task_loss
        result[f"auxiliary_{name}"] = task_loss
    return {"loss": total, **result}