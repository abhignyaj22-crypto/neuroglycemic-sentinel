"""Split-conformal interval calibration for the neural glucose forecaster.

The Gaussian mixture produces model-based intervals, but their coverage is
only as good as the learned scales.  This module recalibrates the *quantile
levels* used for prediction intervals on the validation split only, separately
for every (horizon, availability-pattern) combination:

1. run the trained model over the validation batches;
2. compute the mixture CDF value ``u = F_hat(y)`` of each observed target;
3. take the empirical alpha/2 and 1-alpha/2 quantiles of ``u`` as the
   corrected levels.

Requesting the corrected levels from the same mixture CDF then yields
intervals with ~1-alpha empirical coverage on validation data, per sensor
combination.  Patterns with too few validation rows fall back to the pooled
per-horizon correction; the calibrator never fabricates coverage it cannot
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

CONFORMAL_SCHEMA = "neuroglycemic-conformal-v1"
DEFAULT_LEVELS = (0.025, 0.975)


def mixture_cdf(
    values: Tensor,
    expert_mean: Tensor,
    expert_scale: Tensor,
    fusion_weights: Tensor,
) -> Tensor:
    """CDF of the weighted Gaussian mixture evaluated at ``values``."""

    if expert_mean.ndim != 3 or expert_scale.shape != expert_mean.shape:
        raise ValueError("Expert tensors must be [batch, modality, horizon].")
    if fusion_weights.shape == expert_mean.shape[:2]:
        fusion_weights = fusion_weights.unsqueeze(-1).expand_as(expert_mean)
    # Missing targets (NaN) are excluded by the caller after the fact, but the
    # distribution's CDF validates its input, so sanitize them here first.
    expanded = values.unsqueeze(1).expand_as(expert_mean)
    safe_values = torch.where(
        torch.isfinite(expanded), expanded, torch.zeros_like(expanded)
    )
    z = (safe_values - expert_mean) / expert_scale
    normal = torch.distributions.Normal(
        torch.zeros((), device=expert_mean.device, dtype=expert_mean.dtype),
        torch.ones((), device=expert_mean.device, dtype=expert_mean.dtype),
    )
    return (fusion_weights * normal.cdf(z)).sum(dim=1)


def availability_pattern(modalities: Sequence[str], availability_row: Sequence[bool]) -> str:
    """Stable key describing which modalities were observed for one row."""

    observed = [name for name, flag in zip(modalities, availability_row) if flag]
    return "+".join(observed) if observed else "none"


@dataclass(frozen=True)
class ConformalCalibrator:
    """Corrected interval levels per horizon and availability pattern."""

    alpha: float
    horizon_levels: Mapping[str, tuple[float, float]]
    pattern_levels: Mapping[str, Mapping[str, tuple[float, float]]]
    counts: Mapping[str, int]
    min_count: int
    schema_version: str = CONFORMAL_SCHEMA

    def levels_for(self, horizon_minutes: int, pattern: str) -> tuple[float, float]:
        """Corrected (lower, upper) quantile levels with honest fallbacks.

        Levels are clamped strictly inside (0, 1): a severely miscalibrated
        model can produce empirical levels of exactly 0 or 1, which are not
        valid quantile requests.
        """

        by_pattern = self.pattern_levels.get(str(int(horizon_minutes)), {})
        if pattern in by_pattern:
            levels = by_pattern[pattern]
        elif str(int(horizon_minutes)) in self.horizon_levels:
            levels = self.horizon_levels[str(int(horizon_minutes))]
        else:
            levels = (self.alpha / 2.0, 1.0 - self.alpha / 2.0)
        epsilon = 1e-3
        lower = min(max(levels[0], epsilon), 1.0 - 2 * epsilon)
        upper = max(min(levels[1], 1.0 - epsilon), lower + epsilon)
        return (lower, upper)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "alpha": self.alpha,
            "min_count": self.min_count,
            "horizon_levels": {
                key: list(value) for key, value in self.horizon_levels.items()
            },
            "pattern_levels": {
                horizon: {pattern: list(levels) for pattern, levels in entries.items()}
                for horizon, entries in self.pattern_levels.items()
            },
            "counts": dict(self.counts),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ConformalCalibrator":
        if values.get("schema_version") != CONFORMAL_SCHEMA:
            raise ValueError("Unsupported conformal calibration schema.")
        return cls(
            alpha=float(values["alpha"]),
            horizon_levels={
                str(key): (float(pair[0]), float(pair[1]))
                for key, pair in values["horizon_levels"].items()
            },
            pattern_levels={
                str(horizon): {
                    str(pattern): (float(pair[0]), float(pair[1]))
                    for pattern, pair in entries.items()
                }
                for horizon, entries in values.get("pattern_levels", {}).items()
            },
            counts={str(key): int(value) for key, value in values["counts"].items()},
            min_count=int(values["min_count"]),
        )


def _conformal_levels(samples: list[float], alpha: float) -> tuple[float, float]:
    """Empirical alpha/2 and 1-alpha/2 quantiles with conformal inflation."""

    ordered = sorted(samples)
    n = len(ordered)

    def quantile(probability: float) -> float:
        # Conformal-style conservative index: ceil((n + 1) * p) - 1, clamped.
        index = math.ceil((n + 1) * probability) - 1
        index = min(max(index, 0), n - 1)
        return ordered[index]

    return (quantile(alpha / 2.0), quantile(1.0 - alpha / 2.0))


@torch.no_grad()
def fit_conformal_calibrator(
    model: torch.nn.Module,
    validation_batches: Sequence[Mapping[str, Any]],
    target_standardizer: Any,
    horizons_minutes: Sequence[int],
    *,
    alpha: float = 0.05,
    min_count: int = 20,
) -> ConformalCalibrator:
    """Fit corrected interval levels on validation batches only.

    ``min_count`` is the minimum number of observed labels required before a
    pattern-specific correction is trusted; smaller groups fall back to the
    pooled per-horizon correction.
    """

    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5).")
    if min_count < 2:
        raise ValueError("min_count must be at least 2.")
    from .neural_training import inverse_transform_neuroglycemic_outputs

    model.eval()
    device = next(model.parameters()).device
    modalities = tuple(model.modalities)
    per_horizon: dict[str, list[float]] = {str(int(h)): [] for h in horizons_minutes}
    per_pattern: dict[str, dict[str, list[float]]] = {
        str(int(h)): {} for h in horizons_minutes
    }
    for batch in validation_batches:
        outputs = model(
            {name: value.to(device) for name, value in batch["features"].items()},
            {
                name: value.to(device)
                for name, value in batch["feature_masks"].items()
            },
            batch["availability"].to(device),
            batch["quality"].to(device),
            batch["staleness"].to(device),
            batch.get("clock_uncertainty", torch.zeros_like(batch["staleness"])).to(device),
            patient_index=batch.get("patient_index"),
            seen_patient=batch.get("seen_patient"),
            event_basis=(
                {k: v.to(device) for k, v in batch["event_basis"].items()}
                if batch.get("event_basis") is not None
                else None
            ),
        )
        converted = inverse_transform_neuroglycemic_outputs(outputs, target_standardizer)
        weights = outputs.get(
            "fusion_weights_by_horizon",
            outputs["fusion_weights"].unsqueeze(-1).expand_as(converted["expert_mean"]),
        )
        targets = batch["targets"].to(device)
        cdf_values = mixture_cdf(
            targets,
            converted["expert_mean"],
            converted["expert_scale"],
            weights,
        )
        availability = batch["availability"].to(device)
        for row_index in range(targets.shape[0]):
            if bool(outputs["abstained"][row_index]):
                continue
            pattern = availability_pattern(
                modalities, availability[row_index].detach().cpu().tolist()
            )
            for horizon_index, horizon in enumerate(horizons_minutes):
                value = float(cdf_values[row_index, horizon_index].detach().cpu())
                target_value = float(targets[row_index, horizon_index].detach().cpu())
                if not math.isfinite(target_value):
                    continue
                key = str(int(horizon))
                per_horizon[key].append(value)
                per_pattern[key].setdefault(pattern, []).append(value)

    horizon_levels: dict[str, tuple[float, float]] = {}
    pattern_levels: dict[str, dict[str, tuple[float, float]]] = {}
    counts: dict[str, int] = {}
    for horizon_key, samples in per_horizon.items():
        counts[horizon_key] = len(samples)
        if len(samples) >= min_count:
            horizon_levels[horizon_key] = _conformal_levels(samples, alpha)
        pattern_entries: dict[str, tuple[float, float]] = {}
        for pattern, pattern_samples in per_pattern[horizon_key].items():
            if len(pattern_samples) >= min_count:
                pattern_entries[pattern] = _conformal_levels(pattern_samples, alpha)
        if pattern_entries:
            pattern_levels[horizon_key] = pattern_entries
    return ConformalCalibrator(
        alpha=alpha,
        horizon_levels=horizon_levels,
        pattern_levels=pattern_levels,
        counts=counts,
        min_count=min_count,
    )