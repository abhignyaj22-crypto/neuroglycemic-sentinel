from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .model import ProbabilisticGlucoseModel
from .neural_model import NeuroGlycemicNet
from .neural_training import (
    CHECKPOINT_SCHEMA,
    GlucoseTargetStandardizer,
    inverse_transform_neuroglycemic_outputs,
    load_neural_checkpoint,
)
from .release import load_release_manifest


@dataclass(frozen=True)
class GlucoseForecastRequest:
    patient_id: str
    anchor_time: str
    features: dict[str, float | int | None]


@dataclass(frozen=True)
class GlucoseForecastResponse:
    patient_id: str
    anchor_time: str
    horizon_hours: float
    predicted_glucose_mg_dl: float
    prediction_lower_mg_dl: float
    prediction_upper_mg_dl: float
    hyperglycemia_probability: float
    abstained: bool
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "anchor_time": self.anchor_time,
            "horizon_hours": self.horizon_hours,
            "predicted_glucose_mg_dl": self.predicted_glucose_mg_dl,
            "prediction_lower_mg_dl": self.prediction_lower_mg_dl,
            "prediction_upper_mg_dl": self.prediction_upper_mg_dl,
            "hyperglycemia_probability": self.hyperglycemia_probability,
            "abstained": self.abstained,
            "warnings": list(self.warnings),
        }


def forecast_one(
    model: ProbabilisticGlucoseModel,
    request: GlucoseForecastRequest,
    *,
    horizon_hours: float,
) -> GlucoseForecastResponse:
    unknown = set(request.features) - set(model.feature_names)
    if unknown:
        raise ValueError(f"Unknown EHR feature fields: {sorted(unknown)}")
    missing_required = [
        name
        for name in ("current_glucose_mg_dl", "previous_glucose_mg_dl")
        if request.features.get(name) is None
    ]
    if missing_required:
        return GlucoseForecastResponse(
            patient_id=request.patient_id,
            anchor_time=request.anchor_time,
            horizon_hours=horizon_hours,
            predicted_glucose_mg_dl=float("nan"),
            prediction_lower_mg_dl=float("nan"),
            prediction_upper_mg_dl=float("nan"),
            hyperglycemia_probability=float("nan"),
            abstained=True,
            warnings=(
                f"Missing required glucose history: {', '.join(missing_required)}",
            ),
        )

    row = {name: request.features.get(name, np.nan) for name in model.feature_names}
    prediction = model.predict(pd.DataFrame([row])).iloc[0]
    warnings = (
        "Hospital laboratory glucose is intermittent and is not continuous glucose monitoring.",
    )
    return GlucoseForecastResponse(
        patient_id=request.patient_id,
        anchor_time=request.anchor_time,
        horizon_hours=horizon_hours,
        predicted_glucose_mg_dl=float(prediction["predicted_glucose_mg_dl"]),
        prediction_lower_mg_dl=float(prediction["prediction_lower_mg_dl"]),
        prediction_upper_mg_dl=float(prediction["prediction_upper_mg_dl"]),
        hyperglycemia_probability=float(prediction["hyperglycemia_probability"]),
        abstained=False,
        warnings=warnings,
    )


NEURAL_FEATURE_SCHEMA = "neuroglycemic-aligned-window-v1"
NEURAL_PREDICTION_TARGET = "future_cgm_glucose_mg_dl"


def _parse_anchor_time(value: str) -> None:
    """Validate an ISO-8601 prediction anchor without changing its identity."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("anchor_time must be a valid ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError("anchor_time must include a timezone offset.")


@dataclass(frozen=True)
class NeuralGlucoseForecastRequest:
    """One checkpoint-bound, patient-time neural inference request.

    ``features`` contains raw values in the units declared by the feature
    schema.  The service applies the training-only normalization saved in the
    checkpoint.  A missing numeric value is represented by ``None``; an
    unavailable modality must be declared in ``availability`` and cannot
    receive learned fusion weight.
    """

    patient_id: str
    anchor_time: str
    horizon_minutes: int
    feature_schema_version: str
    features: dict[str, dict[str, float | int | None]]
    availability: dict[str, bool]
    quality: dict[str, float]
    staleness_minutes: dict[str, float]
    clock_uncertainty_ms: dict[str, float] | None = None


@dataclass(frozen=True)
class NeuralModalityForecast:
    modality: str
    available: bool
    quality: float
    staleness_minutes: float
    learned_weight: float
    predicted_glucose_mg_dl: float | None
    prediction_sd_mg_dl: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "available": self.available,
            "quality": self.quality,
            "staleness_minutes": self.staleness_minutes,
            "learned_weight": self.learned_weight,
            "predicted_glucose_mg_dl": self.predicted_glucose_mg_dl,
            "prediction_sd_mg_dl": self.prediction_sd_mg_dl,
        }


@dataclass(frozen=True)
class NeuralGlucoseForecastResponse:
    patient_id: str
    anchor_time: str
    prediction_target: str
    horizon_minutes: int
    feature_schema_version: str
    checkpoint_schema_version: str
    model_version: str
    release_status: str
    predicted_glucose_mg_dl: float | None
    prediction_sd_mg_dl: float | None
    prediction_lower_mg_dl: float | None
    prediction_upper_mg_dl: float | None
    hypoglycemia_threshold_mg_dl: float
    hyperglycemia_threshold_mg_dl: float
    hypoglycemia_probability: float | None
    hyperglycemia_probability: float | None
    modality_forecasts: tuple[NeuralModalityForecast, ...]
    auxiliary_predictions: dict[str, float | None]
    abstained: bool
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "anchor_time": self.anchor_time,
            "prediction_target": self.prediction_target,
            "horizon_minutes": self.horizon_minutes,
            "feature_schema_version": self.feature_schema_version,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "model_version": self.model_version,
            "release_status": self.release_status,
            "predicted_glucose_mg_dl": self.predicted_glucose_mg_dl,
            "prediction_sd_mg_dl": self.prediction_sd_mg_dl,
            "prediction_lower_mg_dl": self.prediction_lower_mg_dl,
            "prediction_upper_mg_dl": self.prediction_upper_mg_dl,
            "hypoglycemia_threshold_mg_dl": self.hypoglycemia_threshold_mg_dl,
            "hyperglycemia_threshold_mg_dl": self.hyperglycemia_threshold_mg_dl,
            "hypoglycemia_probability": self.hypoglycemia_probability,
            "hyperglycemia_probability": self.hyperglycemia_probability,
            "modality_forecasts": [
                value.as_dict() for value in self.modality_forecasts
            ],
            "auxiliary_predictions": dict(self.auxiliary_predictions),
            "learned_weights": {
                value.modality: value.learned_weight
                for value in self.modality_forecasts
            },
            "abstained": self.abstained,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _FeatureSchema:
    version: str
    feature_names: dict[str, tuple[str, ...]]
    means: dict[str, tuple[float, ...]]
    scales: dict[str, tuple[float, ...]]


def build_neural_checkpoint_metadata(
    model: NeuroGlycemicNet,
    *,
    feature_names: dict[str, list[str] | tuple[str, ...]],
    feature_means: dict[str, list[float] | tuple[float, ...]],
    feature_scales: dict[str, list[float] | tuple[float, ...]],
    hidden_dim: int,
    embedding_dim: int,
    dropout: float,
    min_scale: float,
    modality_dropout_probability: float = 0.0,
    feature_schema_version: str = NEURAL_FEATURE_SCHEMA,
    model_version: str = "neuroglycemic-neural-v2",
) -> dict[str, Any]:
    """Build the portable metadata required to reconstruct neural inference.

    This helper belongs in the training-to-serving contract: a checkpoint that
    lacks its exact architecture and train-only feature normalization is not a
    deployable model and is rejected by :class:`NeuralGlucoseService`.
    """

    modalities = tuple(model.modalities)
    if set(feature_names) != set(modalities):
        raise ValueError("Feature names must cover exactly the model modalities.")
    if set(feature_means) != set(modalities) or set(feature_scales) != set(modalities):
        raise ValueError("Feature statistics must cover exactly the model modalities.")
    schema_modalities: dict[str, dict[str, Any]] = {}
    for modality in modalities:
        names = tuple(str(value) for value in feature_names[modality])
        means = tuple(float(value) for value in feature_means[modality])
        scales = tuple(float(value) for value in feature_scales[modality])
        expected = model.input_dims[modality]
        if len(names) != expected or len(means) != expected or len(scales) != expected:
            raise ValueError(
                f"Feature schema for {modality!r} must contain {expected} values."
            )
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError(
                f"Feature names for {modality!r} must be unique and non-empty."
            )
        if any(not math.isfinite(value) for value in means):
            raise ValueError("Feature means must be finite.")
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("Feature scales must be finite and positive.")
        schema_modalities[modality] = {
            "feature_names": list(names),
            "means": list(means),
            "scales": list(scales),
        }
    if hidden_dim <= 0 or embedding_dim <= 0 or min_scale <= 0:
        raise ValueError(
            "Neural architecture dimensions and min_scale must be positive."
        )
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    if not 0 <= modality_dropout_probability < 1:
        raise ValueError("modality_dropout_probability must be in [0, 1).")
    return {
        "model_version": str(model_version),
        "model_spec": {
            "input_dims": dict(model.input_dims),
            "horizons_minutes": list(model.horizons_minutes),
            "hidden_dim": int(hidden_dim),
            "embedding_dim": int(embedding_dim),
            "dropout": float(dropout),
            "min_scale": float(min_scale),
            "modality_dropout_probability": float(
                modality_dropout_probability
            ),
            "auxiliary_task_kinds": dict(model.auxiliary_task_kinds),
        },
        "feature_schema": {
            "version": str(feature_schema_version),
            "fit_split": "train",
            "ordered_feature_names": {
                name: values["feature_names"]
                for name, values in schema_modalities.items()
            },
            "means": {
                name: values["means"] for name, values in schema_modalities.items()
            },
            "scales": {
                name: values["scales"] for name, values in schema_modalities.items()
            },
        },
    }


def _trusted_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Neural checkpoint payload must be a mapping.")
    return payload


def _read_feature_schema(
    metadata: dict[str, Any], input_dims: dict[str, int]
) -> _FeatureSchema:
    values = metadata.get("feature_schema")
    if not isinstance(values, dict):
        raise ValueError("Checkpoint is missing its feature_schema metadata.")
    version = values.get("version")
    names_by_modality_values = values.get(
        "ordered_feature_names", values.get("feature_names")
    )
    means_by_modality_values = values.get("means")
    scales_by_modality_values = values.get("scales")
    if not isinstance(version, str) or not version:
        raise ValueError("Checkpoint feature schema requires a version.")
    if values.get("fit_split") != "train":
        raise ValueError(
            "Checkpoint feature statistics must be fit on training data only."
        )
    if (
        not isinstance(names_by_modality_values, dict)
        or not isinstance(means_by_modality_values, dict)
        or not isinstance(scales_by_modality_values, dict)
        or set(names_by_modality_values) != set(input_dims)
        or set(means_by_modality_values) != set(input_dims)
        or set(scales_by_modality_values) != set(input_dims)
    ):
        raise ValueError("Checkpoint feature schema does not match model modalities.")
    names_by_modality: dict[str, tuple[str, ...]] = {}
    means_by_modality: dict[str, tuple[float, ...]] = {}
    scales_by_modality: dict[str, tuple[float, ...]] = {}
    for modality, dimension in input_dims.items():
        names = tuple(
            str(value) for value in names_by_modality_values.get(modality, ())
        )
        means = tuple(
            float(value) for value in means_by_modality_values.get(modality, ())
        )
        scales = tuple(
            float(value) for value in scales_by_modality_values.get(modality, ())
        )
        if (
            len(names) != dimension
            or len(means) != dimension
            or len(scales) != dimension
        ):
            raise ValueError(
                f"Feature schema for {modality!r} does not match input dimension {dimension}."
            )
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError(f"Feature names for {modality!r} are invalid.")
        if any(not math.isfinite(value) for value in means):
            raise ValueError("Checkpoint feature means must be finite.")
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("Checkpoint feature scales must be finite and positive.")
        names_by_modality[modality] = names
        means_by_modality[modality] = means
        scales_by_modality[modality] = scales
    return _FeatureSchema(
        version, names_by_modality, means_by_modality, scales_by_modality
    )


class NeuralGlucoseService:
    """Checkpoint-bound neural glucose inference with strict schema checks."""

    def __init__(
        self,
        model: NeuroGlycemicNet,
        *,
        prediction_target: str,
        target_standardizer: GlucoseTargetStandardizer,
        feature_schema: _FeatureSchema,
        model_version: str,
        release_status: str,
        hypoglycemia_threshold_mg_dl: float,
        hyperglycemia_threshold_mg_dl: float,
        input_cgm: bool,
        forecast_mode: str,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.prediction_target = prediction_target
        self.target_standardizer = target_standardizer
        self.feature_schema = feature_schema
        self.model_version = model_version
        self.release_status = release_status
        self.hypoglycemia_threshold_mg_dl = hypoglycemia_threshold_mg_dl
        self.hyperglycemia_threshold_mg_dl = hyperglycemia_threshold_mg_dl
        self.input_cgm = bool(input_cgm)
        self.forecast_mode = str(forecast_mode)
        self.device = device

    @property
    def supported_horizons_minutes(self) -> tuple[int, ...]:
        return tuple(self.model.horizons_minutes)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        expected_prediction_target: str | None = None,
        expected_feature_schema_version: str = NEURAL_FEATURE_SCHEMA,
        device: str | torch.device = "cpu",
        release_manifest_path: Path | None = None,
        allow_research_only: bool = False,
    ) -> "NeuralGlucoseService":
        """Reconstruct a model only when its complete serving contract matches."""

        resolved_device = torch.device(device)
        payload = _trusted_torch_load(Path(checkpoint_path), resolved_device)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported neural checkpoint schema.")
        release = load_release_manifest(
            Path(checkpoint_path), manifest_path=release_manifest_path
        )
        if release.status == "rejected":
            raise ValueError("The model release manifest rejects this checkpoint.")
        if release.status == "research_only" and not allow_research_only:
            raise ValueError(
                "This checkpoint is research-only. Pass allow_research_only=True "
                "only from an explicitly labelled research workflow."
            )
        training_config = payload.get("training_config")
        metadata = payload.get("metadata")
        if not isinstance(training_config, dict) or not isinstance(metadata, dict):
            raise ValueError("Checkpoint is missing training or serving metadata.")
        stored_target = training_config.get("prediction_target")
        if not isinstance(stored_target, str) or not stored_target.strip():
            raise ValueError("Checkpoint prediction target is missing or invalid.")
        if (
            expected_prediction_target is not None
            and stored_target != expected_prediction_target
        ):
            raise ValueError(
                "Checkpoint prediction target does not match the service target."
            )
        prediction_target = stored_target
        forecast_mode = training_config.get("forecast_mode")
        if forecast_mode not in {
            "ambient_no_cgm",
            "announced_meal_no_cgm",
            "cgm_augmented",
            "ehr_laboratory",
        }:
            raise ValueError("Checkpoint forecast_mode is missing or invalid.")
        input_cgm = training_config.get("input_cgm")
        if not isinstance(input_cgm, bool):
            raise ValueError("Checkpoint must declare whether CGM history is an input.")
        risk_thresholds = training_config.get("risk_thresholds_mg_dl")
        if not isinstance(risk_thresholds, dict):
            raise ValueError("Checkpoint is missing glucose risk thresholds.")
        try:
            hypoglycemia_threshold = float(risk_thresholds["hypoglycemia"])
            hyperglycemia_threshold = float(risk_thresholds["hyperglycemia"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Checkpoint glucose risk thresholds are invalid.") from exc
        if (
            not math.isfinite(hypoglycemia_threshold)
            or not math.isfinite(hyperglycemia_threshold)
            or hypoglycemia_threshold <= 0
            or hypoglycemia_threshold >= hyperglycemia_threshold
        ):
            raise ValueError("Checkpoint glucose risk thresholds are invalid.")
        spec = metadata.get("model_spec")
        if not isinstance(spec, dict):
            raise ValueError("Checkpoint is missing model_spec metadata.")
        try:
            input_dims = {
                str(name): int(value)
                for name, value in dict(spec["input_dims"]).items()
            }
            horizons = tuple(int(value) for value in spec["horizons_minutes"])
            hidden_dim = int(spec["hidden_dim"])
            embedding_dim = int(spec["embedding_dim"])
            dropout = float(spec["dropout"])
            min_scale = float(spec["min_scale"])
            modality_dropout_probability = float(
                spec.get("modality_dropout_probability", 0.0)
            )
            auxiliary_task_kinds = {
                str(name): str(kind)
                for name, kind in dict(spec.get("auxiliary_task_kinds", {})).items()
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Checkpoint model_spec is incomplete or invalid.") from exc
        stored_horizons = tuple(
            int(value) for value in training_config.get("forecast_horizons_minutes", ())
        )
        if horizons != stored_horizons:
            raise ValueError(
                "Checkpoint model horizons do not match its training contract."
            )
        configured_model = training_config.get("model")
        expected_model = {
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "min_scale": min_scale,
            "modality_dropout_probability": modality_dropout_probability,
        }
        if not isinstance(configured_model, dict) or any(
            not math.isclose(
                float(
                    configured_model.get(
                        name, 0.0 if name == "modality_dropout_probability" else float("nan")
                    )
                ),
                float(value),
            )
            for name, value in expected_model.items()
        ):
            raise ValueError(
                "Checkpoint model_spec does not match its training configuration."
            )
        feature_schema = _read_feature_schema(metadata, input_dims)
        if feature_schema.version != expected_feature_schema_version:
            raise ValueError(
                "Checkpoint feature schema version does not match the service."
            )
        model = NeuroGlycemicNet(
            input_dims,
            horizons_minutes=horizons,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            min_scale=min_scale,
            modality_dropout_probability=modality_dropout_probability,
            auxiliary_task_kinds=auxiliary_task_kinds,
        )
        loaded = load_neural_checkpoint(
            Path(checkpoint_path),
            model,
            device=resolved_device,
            expected_prediction_target=prediction_target,
            expected_horizons_minutes=horizons,
        )
        standardizer = GlucoseTargetStandardizer.from_dict(
            loaded["target_standardizer"]
        )
        return cls(
            model,
            prediction_target=prediction_target,
            target_standardizer=standardizer,
            feature_schema=feature_schema,
            model_version=str(metadata.get("model_version", "unversioned")),
            release_status=release.status,
            hypoglycemia_threshold_mg_dl=hypoglycemia_threshold,
            hyperglycemia_threshold_mg_dl=hyperglycemia_threshold,
            input_cgm=input_cgm,
            forecast_mode=forecast_mode,
            device=resolved_device,
        )

    def forecast(
        self, request: NeuralGlucoseForecastRequest
    ) -> NeuralGlucoseForecastResponse:
        """Execute one real forward pass; no cached prediction artifact is read."""

        if not request.patient_id.strip():
            raise ValueError("patient_id is required.")
        _parse_anchor_time(request.anchor_time)
        if request.feature_schema_version != self.feature_schema.version:
            raise ValueError(
                "Request feature schema version does not match the checkpoint."
            )
        if request.horizon_minutes not in self.supported_horizons_minutes:
            raise ValueError(
                f"Unsupported horizon {request.horizon_minutes}; checkpoint supports "
                f"{list(self.supported_horizons_minutes)} minutes."
            )
        modalities = tuple(self.model.modalities)
        for field_name, values in (
            ("features", request.features),
            ("availability", request.availability),
            ("quality", request.quality),
            ("staleness_minutes", request.staleness_minutes),
            ("clock_uncertainty_ms", request.clock_uncertainty_ms or {}),
        ):
            unknown = set(values) - set(modalities)
            if unknown:
                raise ValueError(
                    f"Unknown modalities in {field_name}: {sorted(unknown)}"
                )

        feature_tensors: dict[str, Tensor] = {}
        mask_tensors: dict[str, Tensor] = {}
        availability_values: list[bool] = []
        quality_values: list[float] = []
        staleness_values: list[float] = []
        clock_uncertainty_values: list[float] = []
        for modality in modalities:
            names = self.feature_schema.feature_names[modality]
            supplied = request.features.get(modality, {})
            unknown_features = set(supplied) - set(names)
            if unknown_features:
                raise ValueError(
                    f"Unknown {modality} feature fields: {sorted(unknown_features)}"
                )
            available = bool(request.availability.get(modality, False))
            raw_values: list[float] = []
            masks: list[bool] = []
            for name in names:
                raw = supplied.get(name)
                if raw is None:
                    raw_values.append(0.0)
                    masks.append(False)
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError(
                        f"Feature {modality}.{name} must be finite or None."
                    )
                raw_values.append(value)
                masks.append(True)
            if available and not any(masks):
                raise ValueError(
                    f"Available modality {modality!r} must contain an observed feature."
                )
            means = self.feature_schema.means[modality]
            scales = self.feature_schema.scales[modality]
            standardized = [
                (value - mean) / scale if observed else 0.0
                for value, observed, mean, scale in zip(
                    raw_values, masks, means, scales, strict=True
                )
            ]
            feature_tensors[modality] = torch.tensor(
                [standardized], dtype=torch.float32, device=self.device
            )
            mask_tensors[modality] = torch.tensor(
                [masks], dtype=torch.bool, device=self.device
            )
            quality = float(request.quality.get(modality, 0.0))
            staleness = float(request.staleness_minutes.get(modality, 0.0))
            if not math.isfinite(quality) or not 0 <= quality <= 1:
                raise ValueError(
                    f"Quality for {modality!r} must be finite and in [0, 1]."
                )
            if not math.isfinite(staleness) or staleness < 0:
                raise ValueError(
                    f"Staleness for {modality!r} must be finite and non-negative."
                )
            availability_values.append(available)
            quality_values.append(quality)
            staleness_values.append(staleness)
            supplied_clock = request.clock_uncertainty_ms or {}
            if available and modality not in supplied_clock:
                raise ValueError(
                    f"Available modality {modality!r} requires measured clock uncertainty."
                )
            clock_uncertainty = float(supplied_clock.get(modality, 0.0))
            if not math.isfinite(clock_uncertainty) or clock_uncertainty < 0:
                raise ValueError(
                    f"Clock uncertainty for {modality!r} must be finite and non-negative."
                )
            clock_uncertainty_values.append(clock_uncertainty)

        availability = torch.tensor(
            [availability_values], dtype=torch.bool, device=self.device
        )
        quality = torch.tensor(
            [quality_values], dtype=torch.float32, device=self.device
        )
        staleness = torch.tensor(
            [staleness_values], dtype=torch.float32, device=self.device
        )
        clock_uncertainty = torch.tensor(
            [clock_uncertainty_values], dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            standardized_outputs = self.model(
                feature_tensors,
                mask_tensors,
                availability,
                quality,
                staleness,
                clock_uncertainty,
            )
            outputs = inverse_transform_neuroglycemic_outputs(
                standardized_outputs, self.target_standardizer
            )

        horizon_index = self.supported_horizons_minutes.index(request.horizon_minutes)
        abstained = bool(outputs["abstained"][0].item())
        weights = outputs.get(
            "fusion_weights_by_horizon",
            outputs["fusion_weights"].unsqueeze(-1).expand_as(outputs["expert_mean"]),
        )[0, :, horizon_index].detach().cpu()
        expert_mean = outputs["expert_mean"][0, :, horizon_index].detach().cpu()
        expert_scale = outputs["expert_scale"][0, :, horizon_index].detach().cpu()
        modality_forecasts = tuple(
            NeuralModalityForecast(
                modality=modality,
                available=availability_values[index],
                quality=quality_values[index],
                staleness_minutes=staleness_values[index],
                learned_weight=float(weights[index].item()),
                predicted_glucose_mg_dl=(
                    float(expert_mean[index].item())
                    if availability_values[index]
                    else None
                ),
                prediction_sd_mg_dl=(
                    float(expert_scale[index].item())
                    if availability_values[index]
                    else None
                ),
            )
            for index, modality in enumerate(modalities)
        )
        if abstained:
            predicted = standard_deviation = lower = upper = None
            hypoglycemia_probability = hyperglycemia_probability = None
            warnings = ("All modalities are unavailable; the neural model abstained.",)
            auxiliary_predictions = {
                name: None for name in self.model.auxiliary_task_kinds
            }
        else:
            from .evaluation import gaussian_mixture_quantile

            predicted = float(outputs["mixture_mean"][0, horizon_index].cpu().item())
            variance = float(outputs["mixture_variance"][0, horizon_index].cpu().item())
            standard_deviation = math.sqrt(max(variance, 0.0))
            interval_means = expert_mean.tolist()
            interval_scales = expert_scale.tolist()
            interval_weights = weights.tolist()
            lower = gaussian_mixture_quantile(
                interval_means, interval_scales, interval_weights, 0.025
            )
            upper = gaussian_mixture_quantile(
                interval_means, interval_scales, interval_weights, 0.975
            )
            sqrt_two = math.sqrt(2.0)
            horizon_means = outputs["expert_mean"][0, :, horizon_index]
            horizon_scales = outputs["expert_scale"][0, :, horizon_index]
            horizon_weights = outputs.get(
                "fusion_weights_by_horizon",
                outputs["fusion_weights"].unsqueeze(-1).expand_as(outputs["expert_mean"]),
            )[0, :, horizon_index]
            hypoglycemia_cdf = 0.5 * (
                1.0
                + torch.erf(
                    (self.hypoglycemia_threshold_mg_dl - horizon_means)
                    / (sqrt_two * horizon_scales)
                )
            )
            hyperglycemia_cdf = 0.5 * (
                1.0
                + torch.erf(
                    (self.hyperglycemia_threshold_mg_dl - horizon_means)
                    / (sqrt_two * horizon_scales)
                )
            )
            hypoglycemia_probability = float(
                torch.sum(horizon_weights * hypoglycemia_cdf).cpu().item()
            )
            hyperglycemia_probability = float(
                torch.sum(horizon_weights * (1.0 - hyperglycemia_cdf)).cpu().item()
            )
            if self.forecast_mode == "cgm_augmented":
                warning = (
                    "Research CGM-history forecast; it is not a non-invasive glucose measurement."
                )
            elif self.forecast_mode == "ehr_laboratory":
                warning = (
                    "Research hospital-laboratory forecast; it is not continuous glucose monitoring "
                    "or a non-invasive glucose measurement."
                )
            else:
                warning = (
                    "Research non-invasive forecast; it is not a glucose measurement or clinical alarm."
                )
            warnings = (warning,)
            auxiliary_predictions = {
                name: float(
                    (
                        torch.sigmoid(value[0])
                        if self.model.auxiliary_task_kinds[name] == "binary"
                        else value[0]
                    ).cpu().item()
                )
                for name, value in outputs.get("auxiliary_outputs", {}).items()
            }
        return NeuralGlucoseForecastResponse(
            patient_id=request.patient_id,
            anchor_time=request.anchor_time,
            prediction_target=self.prediction_target,
            horizon_minutes=request.horizon_minutes,
            feature_schema_version=self.feature_schema.version,
            checkpoint_schema_version=CHECKPOINT_SCHEMA,
            model_version=self.model_version,
            release_status=self.release_status,
            predicted_glucose_mg_dl=predicted,
            prediction_sd_mg_dl=standard_deviation,
            prediction_lower_mg_dl=lower,
            prediction_upper_mg_dl=upper,
            hypoglycemia_threshold_mg_dl=self.hypoglycemia_threshold_mg_dl,
            hyperglycemia_threshold_mg_dl=self.hyperglycemia_threshold_mg_dl,
            hypoglycemia_probability=hypoglycemia_probability,
            hyperglycemia_probability=hyperglycemia_probability,
            modality_forecasts=modality_forecasts,
            auxiliary_predictions=auxiliary_predictions,
            abstained=abstained,
            warnings=warnings,
        )
