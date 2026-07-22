"""Small, explicit PyTorch training and checkpoint utilities.

The neural model owns its forward pass and scientific loss.  This module owns
only the mechanics that every experiment needs: seeded optimization, finite
gradient checks, validation-only checkpoint selection, early stopping, and an
auditable checkpoint.  Keeping this boundary narrow makes the trainer usable
for a simple baseline as well as a later multimodal model.

Expected loss-step interface::

    def loss_step(model, batch) -> LossOutput:
        outputs = model(batch)
        return glucose_multitask_loss(outputs, batch)

The batch may contain any tensors required by the model.  Patient-disjoint data
splitting must happen before batches are passed here; this module deliberately
does not guess how patient records should be partitioned.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Any, TypeAlias

import torch
from torch import Tensor, nn


CHECKPOINT_SCHEMA = "neural-glucose-checkpoint-v2"
Batch: TypeAlias = Mapping[str, Any]


@dataclass(frozen=True)
class NeuralTrainingConfig:
    """Validated optimizer and checkpoint settings for one target definition."""

    schema_version: str
    prediction_target: str
    forecast_horizons_minutes: tuple[int, ...]
    target_standardization: str
    seed: int
    epochs: int
    learning_rate: float
    weight_decay: float
    expert_loss_weight: float
    model: dict[str, Any]
    forecast_mode: str
    input_cgm: bool
    risk_thresholds_mg_dl: dict[str, float]
    meal_context: dict[str, Any]
    feature_registry: dict[str, tuple[str, ...]]
    auxiliary_tasks: dict[str, dict[str, Any]]
    gradient_clip_norm: float
    early_stopping_patience: int
    minimum_delta: float
    device: str
    checkpoint_relative_path: str
    project_root: Path
    horizon_tolerance_minutes: float = 5.0

    @property
    def checkpoint_path(self) -> Path:
        return self.project_root / self.checkpoint_relative_path

    def checkpoint_values(self) -> dict[str, Any]:
        """Return serializable settings without a machine-specific root path."""

        values = asdict(self)
        values.pop("project_root")
        values["forecast_horizons_minutes"] = list(self.forecast_horizons_minutes)
        return values


@dataclass(frozen=True)
class GlucoseTargetStandardizer:
    """Train-only, per-horizon glucose target statistics and their provenance."""

    horizons_minutes: tuple[int, ...]
    means_mg_dl: tuple[float, ...]
    scales_mg_dl: tuple[float, ...]
    valid_counts: tuple[int, ...]
    fit_split: str = "train"
    unit: str = "mg/dL"
    method: str = "zscore_per_horizon"

    def __post_init__(self) -> None:
        size = len(self.horizons_minutes)
        if size == 0 or any(
            len(values) != size
            for values in (self.means_mg_dl, self.scales_mg_dl, self.valid_counts)
        ):
            raise ValueError("Target statistics must match the configured horizons.")
        if any(value <= 0 for value in self.horizons_minutes):
            raise ValueError("Target horizons must be positive minutes.")
        if any(not math.isfinite(value) for value in self.means_mg_dl):
            raise ValueError("Target means must be finite.")
        if any(not math.isfinite(value) or value <= 0 for value in self.scales_mg_dl):
            raise ValueError("Target scales must be finite and positive.")
        if any(value < 2 for value in self.valid_counts):
            raise ValueError("Every target horizon needs at least two valid training labels.")
        if self.fit_split != "train":
            raise ValueError("Target statistics must be fit on the training split only.")
        if self.unit != "mg/dL" or self.method != "zscore_per_horizon":
            raise ValueError("Unsupported glucose target unit or standardization method.")

    @classmethod
    def fit(
        cls,
        targets_mg_dl: Tensor,
        horizons_minutes: tuple[int, ...],
        *,
        target_mask: Tensor | None = None,
    ) -> "GlucoseTargetStandardizer":
        """Fit only on a training tensor shaped ``[example, horizon]``."""

        if targets_mg_dl.ndim != 2 or targets_mg_dl.shape[1] != len(horizons_minutes):
            raise ValueError("targets_mg_dl must be [example, configured horizon].")
        valid = torch.isfinite(targets_mg_dl)
        if target_mask is not None:
            if target_mask.shape != targets_mg_dl.shape:
                raise ValueError("target_mask must match targets_mg_dl.")
            valid = valid & target_mask.to(device=targets_mg_dl.device, dtype=torch.bool)

        means: list[float] = []
        scales: list[float] = []
        counts: list[int] = []
        for index in range(targets_mg_dl.shape[1]):
            selected = targets_mg_dl[:, index][valid[:, index]]
            if selected.numel() < 2:
                raise ValueError("Each horizon needs at least two valid training labels.")
            mean = selected.mean()
            scale = selected.std(unbiased=False)
            if not torch.isfinite(mean) or not torch.isfinite(scale) or float(scale.item()) <= 0:
                raise ValueError("Training targets must have finite, non-zero variation.")
            means.append(float(mean.item()))
            scales.append(float(scale.item()))
            counts.append(int(selected.numel()))
        return cls(
            horizons_minutes=tuple(int(value) for value in horizons_minutes),
            means_mg_dl=tuple(means),
            scales_mg_dl=tuple(scales),
            valid_counts=tuple(counts),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizons_minutes": list(self.horizons_minutes),
            "means_mg_dl": list(self.means_mg_dl),
            "scales_mg_dl": list(self.scales_mg_dl),
            "valid_counts": list(self.valid_counts),
            "fit_split": self.fit_split,
            "unit": self.unit,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GlucoseTargetStandardizer":
        return cls(
            horizons_minutes=tuple(int(value) for value in values["horizons_minutes"]),
            means_mg_dl=tuple(float(value) for value in values["means_mg_dl"]),
            scales_mg_dl=tuple(float(value) for value in values["scales_mg_dl"]),
            valid_counts=tuple(int(value) for value in values["valid_counts"]),
            fit_split=str(values["fit_split"]),
            unit=str(values["unit"]),
            method=str(values["method"]),
        )

    def _statistics(self, values: Tensor) -> tuple[Tensor, Tensor]:
        if values.ndim == 0 or values.shape[-1] != len(self.horizons_minutes):
            raise ValueError("The final tensor dimension must match the target horizons.")
        means = values.new_tensor(self.means_mg_dl)
        scales = values.new_tensor(self.scales_mg_dl)
        return means, scales

    def transform(self, targets_mg_dl: Tensor) -> Tensor:
        means, scales = self._statistics(targets_mg_dl)
        return (targets_mg_dl - means) / scales

    def inverse_mean(self, standardized_mean: Tensor) -> Tensor:
        means, scales = self._statistics(standardized_mean)
        return standardized_mean * scales + means

    def inverse_scale(self, standardized_scale: Tensor) -> Tensor:
        _, scales = self._statistics(standardized_scale)
        return standardized_scale * scales

    def inverse_variance(self, standardized_variance: Tensor) -> Tensor:
        _, scales = self._statistics(standardized_variance)
        return standardized_variance * scales.square()


def inverse_transform_neuroglycemic_outputs(
    outputs: Mapping[str, Tensor], standardizer: GlucoseTargetStandardizer
) -> dict[str, Tensor]:
    """Return model outputs in mg/dL while leaving masks and weights unchanged."""

    required = {"expert_mean", "expert_scale", "mixture_mean", "mixture_variance"}
    missing = required - set(outputs)
    if missing:
        raise KeyError(f"NeuroGlycemic outputs are missing fields: {sorted(missing)}")
    transformed = dict(outputs)
    transformed["expert_mean"] = standardizer.inverse_mean(outputs["expert_mean"])
    transformed["expert_scale"] = standardizer.inverse_scale(outputs["expert_scale"])
    transformed["mixture_mean"] = standardizer.inverse_mean(outputs["mixture_mean"])
    transformed["mixture_variance"] = standardizer.inverse_variance(
        outputs["mixture_variance"]
    )
    return transformed


@dataclass(frozen=True)
class LossOutput:
    """A differentiable total loss plus detached reporting components."""

    total: Tensor
    components: Mapping[str, Tensor | float]


LossStep: TypeAlias = Callable[[nn.Module, Batch], LossOutput]


@dataclass(frozen=True)
class NeuralTrainingResult:
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    stopped_early: bool
    checkpoint_path: Path
    history: tuple[dict[str, float | int], ...]


def load_neural_training_config(path: Path) -> NeuralTrainingConfig:
    """Load and validate the small JSON training contract."""

    path = path.resolve()
    values = json.loads(path.read_text(encoding="utf-8"))
    config = NeuralTrainingConfig(
        schema_version=str(values["schema_version"]),
        prediction_target=str(values["prediction_target"]),
        forecast_horizons_minutes=tuple(
            int(value) for value in values["forecast_horizons_minutes"]
        ),
        target_standardization=str(values["target_standardization"]),
        seed=int(values["seed"]),
        epochs=int(values["epochs"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        expert_loss_weight=float(values["expert_loss_weight"]),
        model=dict(values["model"]),
        forecast_mode=str(values["forecast_mode"]),
        input_cgm=bool(values["input_cgm"]),
        risk_thresholds_mg_dl={
            str(name): float(value)
            for name, value in values["risk_thresholds_mg_dl"].items()
        },
        meal_context=dict(values["meal_context"]),
        feature_registry={
            str(modality): tuple(str(name) for name in names)
            for modality, names in values.get("feature_registry", {}).items()
        },
        auxiliary_tasks={
            str(name): dict(specification)
            for name, specification in values.get("auxiliary_tasks", {}).items()
        },
        gradient_clip_norm=float(values["gradient_clip_norm"]),
        early_stopping_patience=int(values["early_stopping_patience"]),
        minimum_delta=float(values["minimum_delta"]),
        device=str(values["device"]),
        checkpoint_relative_path=str(values["checkpoint_relative_path"]),
        project_root=path.parent.parent,
        horizon_tolerance_minutes=float(values.get("horizon_tolerance_minutes", 5.0)),
    )
    _validate_config(config)
    return config


def _validate_config(config: NeuralTrainingConfig) -> None:
    if config.schema_version != "neural-glucose-training-v1":
        raise ValueError(f"Unsupported neural training schema: {config.schema_version!r}.")
    if not config.prediction_target.strip():
        raise ValueError("prediction_target is required.")
    if not config.forecast_horizons_minutes:
        raise ValueError("At least one forecast horizon is required.")
    if any(value <= 0 for value in config.forecast_horizons_minutes):
        raise ValueError("Forecast horizons must be positive minutes.")
    if len(set(config.forecast_horizons_minutes)) != len(config.forecast_horizons_minutes):
        raise ValueError("Forecast horizons must be unique.")
    if (
        not math.isfinite(config.horizon_tolerance_minutes)
        or config.horizon_tolerance_minutes <= 0
    ):
        raise ValueError("horizon_tolerance_minutes must be finite and positive.")
    if config.target_standardization != "train_only_zscore_per_horizon":
        raise ValueError(
            "target_standardization must be 'train_only_zscore_per_horizon'."
        )
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative.")
    if not math.isfinite(config.expert_loss_weight) or config.expert_loss_weight < 0:
        raise ValueError("expert_loss_weight must be finite and non-negative.")
    required_model_values = {"hidden_dim", "embedding_dim", "dropout", "min_scale"}
    missing_model_values = required_model_values - set(config.model)
    if missing_model_values:
        raise ValueError(f"model is missing fields: {sorted(missing_model_values)}")
    hidden_dim = int(config.model["hidden_dim"])
    embedding_dim = int(config.model["embedding_dim"])
    dropout = float(config.model["dropout"])
    min_scale = float(config.model["min_scale"])
    modality_dropout_probability = float(
        config.model.get("modality_dropout_probability", 0.0)
    )
    if hidden_dim <= 0 or embedding_dim <= 0:
        raise ValueError("model hidden_dim and embedding_dim must be positive.")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("model dropout must be finite and in [0, 1).")
    if not math.isfinite(min_scale) or min_scale <= 0:
        raise ValueError("model min_scale must be finite and positive.")
    if (
        not math.isfinite(modality_dropout_probability)
        or not 0 <= modality_dropout_probability < 1
    ):
        raise ValueError("model modality_dropout_probability must be in [0, 1).")
    if config.forecast_mode not in {
        "ambient_no_cgm",
        "announced_meal_no_cgm",
        "cgm_augmented",
        "ehr_laboratory",
    }:
        raise ValueError(
            "forecast_mode must be ambient_no_cgm, announced_meal_no_cgm, "
            "cgm_augmented, or ehr_laboratory."
        )
    if config.input_cgm != (config.forecast_mode == "cgm_augmented"):
        raise ValueError(
            "input_cgm must be true only for an explicitly named cgm_augmented "
            "experiment and false for non-invasive experiments."
        )
    if not config.feature_registry:
        raise ValueError(
            "feature_registry is required; production training never discovers predictors by prefix."
        )
    for modality, names in config.feature_registry.items():
        if not modality.strip() or not names:
            raise ValueError("Every feature-registry modality needs features.")
        if len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise ValueError("Feature-registry names must be unique and non-empty.")
        if any(not name.startswith(f"{modality}_") for name in names):
            raise ValueError(
                f"Every {modality!r} feature must start with {modality}_ for auditability."
            )
    for name, specification in config.auxiliary_tasks.items():
        if not name.isidentifier():
            raise ValueError("Auxiliary task names must be Python identifiers.")
        required = {"target_column", "kind", "loss_weight"}
        missing = required - set(specification)
        if missing:
            raise ValueError(f"Auxiliary task {name!r} is missing: {sorted(missing)}")
        if str(specification["kind"]) not in {"binary", "continuous"}:
            raise ValueError("Auxiliary task kind must be binary or continuous.")
        if not str(specification["target_column"]).startswith("target_"):
            raise ValueError("Auxiliary target columns must start with target_.")
        weight = float(specification["loss_weight"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Auxiliary loss weights must be finite and non-negative.")
    if set(config.risk_thresholds_mg_dl) != {"hypoglycemia", "hyperglycemia"}:
        raise ValueError(
            "risk_thresholds_mg_dl requires hypoglycemia and hyperglycemia."
        )
    hypoglycemia = config.risk_thresholds_mg_dl["hypoglycemia"]
    hyperglycemia = config.risk_thresholds_mg_dl["hyperglycemia"]
    if (
        not math.isfinite(hypoglycemia)
        or not math.isfinite(hyperglycemia)
        or hypoglycemia <= 0
        or hypoglycemia >= hyperglycemia
    ):
        raise ValueError(
            "Glucose risk thresholds must be finite, positive, and ordered."
        )
    from .meal_context import MealLagSpec

    required_meal_values = {
        "lookback_minutes",
        "lag_centers_minutes",
        "lag_width_minutes",
        "value_columns",
        "require_event_and_available_time_before_anchor",
    }
    missing_meal_values = required_meal_values - set(config.meal_context)
    if missing_meal_values:
        raise ValueError(
            f"meal_context is missing fields: {sorted(missing_meal_values)}"
        )
    if config.meal_context["require_event_and_available_time_before_anchor"] is not True:
        raise ValueError("Causal meal context requires both event and availability times.")
    MealLagSpec(
        centers_minutes=tuple(
            float(value) for value in config.meal_context["lag_centers_minutes"]
        ),
        width_minutes=float(config.meal_context["lag_width_minutes"]),
        lookback_minutes=float(config.meal_context["lookback_minutes"]),
        value_columns=tuple(str(value) for value in config.meal_context["value_columns"]),
    )
    if not math.isfinite(config.gradient_clip_norm) or config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be finite and positive.")
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")
    if not math.isfinite(config.minimum_delta) or config.minimum_delta < 0:
        raise ValueError("minimum_delta must be finite and non-negative.")
    if config.device not in {"cpu", "cuda", "mps", "auto"}:
        raise ValueError("device must be one of: cpu, cuda, mps, auto.")
    checkpoint = Path(config.checkpoint_relative_path)
    if checkpoint.is_absolute() or ".." in checkpoint.parts:
        raise ValueError("checkpoint_relative_path must stay inside the project root.")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_value(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {name: _move_value(item, device) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    return value


def _move_batch(batch: Batch, device: torch.device) -> dict[str, Any]:
    return {name: _move_value(value, device) for name, value in batch.items()}


def make_neuroglycemic_loss_step(
    expert_loss_weight: float,
    target_standardizer: GlucoseTargetStandardizer,
    auxiliary_tasks: Mapping[str, Mapping[str, Any]] | None = None,
) -> LossStep:
    """Adapt the neural model API and standardize raw mg/dL labels safely."""

    if not math.isfinite(expert_loss_weight) or expert_loss_weight < 0:
        raise ValueError("expert_loss_weight must be finite and non-negative.")

    def loss_step(model: nn.Module, batch: Batch) -> LossOutput:
        from .neural_model import neuroglycemic_loss

        required = {
            "features",
            "feature_masks",
            "availability",
            "quality",
            "staleness",
            "targets",
        }
        missing = required - set(batch)
        if missing:
            raise KeyError(f"NeuroGlycemic batch is missing fields: {sorted(missing)}")
        outputs = model(
            features=batch["features"],
            feature_masks=batch["feature_masks"],
            availability=batch["availability"],
            quality=batch["quality"],
            staleness=batch["staleness"],
            clock_uncertainty=batch.get("clock_uncertainty"),
        )
        values = neuroglycemic_loss(
            outputs,
            target_standardizer.transform(batch["targets"]),
            target_mask=batch.get("target_mask"),
            sample_weight=batch.get("sample_weight"),
            expert_loss_weight=expert_loss_weight,
            auxiliary_targets=batch.get("auxiliary_targets"),
            auxiliary_masks=batch.get("auxiliary_masks"),
            auxiliary_task_kinds={
                name: str(specification["kind"])
                for name, specification in (auxiliary_tasks or {}).items()
            },
            auxiliary_loss_weights={
                name: float(specification["loss_weight"])
                for name, specification in (auxiliary_tasks or {}).items()
            },
        )
        return LossOutput(
            total=values["loss"],
            components={name: value for name, value in values.items() if name != "loss"},
        )

    return loss_step


def _batch_size(batch: Batch) -> int:
    for value in batch.values():
        if isinstance(value, Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("A batch must contain at least one non-scalar tensor.")


def _batch_objective_weight(batch: Batch) -> float:
    value = batch.get("sample_weight")
    if value is None:
        return float(_batch_size(batch))
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise ValueError("sample_weight must be a one-dimensional tensor.")
    result = float(value.detach().sum().cpu().item())
    if not math.isfinite(result) or result <= 0:
        raise ValueError("sample_weight sum must be finite and positive.")
    return result


def _finite_scalar(value: Tensor, *, name: str) -> float:
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor.")
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is not finite: {result}.")
    return result


def _run_epoch(
    model: nn.Module,
    batches: Iterable[Batch],
    loss_step: LossStep,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_examples = 0
    total_objective_weight = 0.0
    loss_sum = 0.0
    component_sums: dict[str, float] = {}
    gradient_norm_sum = 0.0
    batch_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in batches:
            batch = _move_batch(raw_batch, device)
            examples = _batch_size(batch)
            objective_weight = _batch_objective_weight(batch)
            output = loss_step(model, batch)
            if not isinstance(output, LossOutput):
                raise TypeError("loss_step must return neural_training.LossOutput.")
            loss_value = _finite_scalar(output.total, name="total loss")

            if training:
                optimizer.zero_grad(set_to_none=True)
                output.total.backward()
                parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not parameters:
                    raise RuntimeError("The loss produced no gradients for trainable parameters.")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, max_norm=gradient_clip_norm, error_if_nonfinite=True
                )
                gradient_norm_sum += _finite_scalar(gradient_norm, name="gradient norm")
                optimizer.step()

            total_examples += examples
            total_objective_weight += objective_weight
            batch_count += 1
            loss_sum += loss_value * objective_weight
            for name, value in output.components.items():
                if isinstance(value, Tensor):
                    component_value = _finite_scalar(value, name=f"loss component {name!r}")
                else:
                    component_value = float(value)
                    if not math.isfinite(component_value):
                        raise FloatingPointError(
                            f"Loss component {name!r} is not finite: {component_value}."
                        )
                component_sums[name] = component_sums.get(name, 0.0) + component_value * objective_weight

    if total_examples == 0:
        raise ValueError("Training and validation iterables must not be empty.")
    result = {"loss": loss_sum / total_objective_weight}
    result.update(
        {name: value / total_objective_weight for name, value in component_sums.items()}
    )
    if training:
        result["gradient_norm"] = gradient_norm_sum / batch_count
    return result


def save_neural_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    validation_loss: float,
    config: NeuralTrainingConfig,
    target_standardizer: GlucoseTargetStandardizer,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save portable model and optimizer state plus target provenance."""

    if epoch <= 0:
        raise ValueError("A trained checkpoint must have epoch >= 1.")
    if not math.isfinite(validation_loss):
        raise ValueError("validation_loss must be finite.")
    if target_standardizer.horizons_minutes != config.forecast_horizons_minutes:
        raise ValueError("Target statistics do not match the configured forecast horizons.")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "validation_loss": float(validation_loss),
        "training_config": config.checkpoint_values(),
        "target_standardizer": target_standardizer.as_dict(),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": dict(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_neural_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
    expected_prediction_target: str | None = None,
    expected_horizons_minutes: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Restore trusted project state and optionally verify its target contract."""

    resolved_device = torch.device(device)
    try:
        payload = torch.load(path, map_location=resolved_device, weights_only=True)
    except TypeError:  # PyTorch releases before ``weights_only`` was introduced.
        payload = torch.load(path, map_location=resolved_device)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported neural checkpoint schema.")
    training_config = payload.get("training_config", {})
    if "target_standardizer" not in payload:
        raise ValueError("Checkpoint is missing glucose target standardization provenance.")
    target_standardizer = GlucoseTargetStandardizer.from_dict(
        payload["target_standardizer"]
    )
    stored_horizons = tuple(
        int(value) for value in training_config.get("forecast_horizons_minutes", ())
    )
    if target_standardizer.horizons_minutes != stored_horizons:
        raise ValueError("Checkpoint target statistics do not match its forecast horizons.")
    if (
        expected_prediction_target is not None
        and training_config.get("prediction_target") != expected_prediction_target
    ):
        raise ValueError("Checkpoint prediction target does not match the requested target.")
    if expected_horizons_minutes is not None:
        if stored_horizons != tuple(expected_horizons_minutes):
            raise ValueError("Checkpoint forecast horizons do not match the requested horizons.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(resolved_device)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def train_with_early_stopping(
    model: nn.Module,
    train_batches: Iterable[Batch],
    validation_batches: Iterable[Batch],
    loss_step: LossStep,
    config: NeuralTrainingConfig,
    *,
    target_standardizer: GlucoseTargetStandardizer,
    checkpoint_path: Path | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> NeuralTrainingResult:
    """Optimize from epoch 1 and restore the best validation checkpoint.

    Unlike the legacy linear loop, an untrained epoch-0 initialization is never
    serialized as a trained model.  If the first optimizer step is invalid, the
    run fails instead of silently returning all-zero parameters.
    """

    _validate_config(config)
    if target_standardizer.horizons_minutes != config.forecast_horizons_minutes:
        raise ValueError("Target statistics do not match the configured forecast horizons.")
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    destination = checkpoint_path or config.checkpoint_path
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    train_batch_values = list(train_batches)
    validation_batch_values = list(validation_batches)
    if not train_batch_values or not validation_batch_values:
        raise ValueError("Training and validation batches must not be empty.")
    for epoch in range(1, config.epochs + 1):
        epoch_batches = list(train_batch_values)
        random.Random(config.seed + epoch).shuffle(epoch_batches)
        train_values = _run_epoch(
            model,
            epoch_batches,
            loss_step,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        validation_values = _run_epoch(
            model,
            validation_batch_values,
            loss_step,
            device=device,
            optimizer=None,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_values.items()},
            **{f"validation_{name}": value for name, value in validation_values.items()},
        }
        history.append(row)
        validation_loss = validation_values["loss"]
        if validation_loss < best_loss - config.minimum_delta:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_neural_checkpoint(
                destination,
                model,
                optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                config=config,
                target_standardizer=target_standardizer,
                metadata=checkpoint_metadata,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_epoch == 0 or not destination.exists():
        raise RuntimeError("Training ended without a finite validation checkpoint.")
    load_neural_checkpoint(
        destination,
        model,
        device=device,
        expected_prediction_target=config.prediction_target,
        expected_horizons_minutes=config.forecast_horizons_minutes,
    )
    return NeuralTrainingResult(
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        epochs_completed=len(history),
        stopped_early=len(history) < config.epochs,
        checkpoint_path=destination,
        history=tuple(history),
    )
