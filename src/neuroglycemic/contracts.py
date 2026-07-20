"""Canonical prediction contracts and target-safe late fusion.

The central invariant is deliberately strict: scores may only be averaged when
they describe the same patient, outcome, anchor time, and forecast horizon.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Iterable


class IncompatibleEvidenceError(ValueError):
    """Raised when modality outputs do not describe one prediction problem."""


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class PredictionEvidence:
    """One calibrated modality-head output for one patient-time target."""

    patient_id: str
    anchor_time: str
    prediction_target: str
    horizon_hours: float
    modality: str
    probability: float | None
    available: bool
    quality: float
    source_cohort: str
    source_device: str
    model_version: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.patient_id.strip():
            raise ValueError("patient_id is required.")
        _parse_time(self.anchor_time)
        if not self.prediction_target.strip():
            raise ValueError("prediction_target is required.")
        if not math.isfinite(self.horizon_hours) or self.horizon_hours < 0:
            raise ValueError("horizon_hours must be finite and non-negative.")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1].")
        if self.available:
            if self.probability is None or not math.isfinite(self.probability):
                raise ValueError("Available evidence requires a finite probability.")
            if not 0.0 <= self.probability <= 1.0:
                raise ValueError("probability must be in [0, 1].")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusedRisk:
    """Auditable result of availability-aware, non-negative late fusion."""

    patient_id: str
    anchor_time: str
    prediction_target: str
    horizon_hours: float
    probability: float | None
    abstained: bool
    used_modalities: tuple[str, ...]
    missing_modalities: tuple[str, ...]
    low_quality_modalities: tuple[str, ...]
    zero_weight_modalities: tuple[str, ...]
    normalized_weights: dict[str, float]
    numerator: float
    denominator: float
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AvailabilityAwareFusion:
    """Fuse compatible probabilities and renormalize around missing inputs.

    For availability indicator ``a_m``, non-negative validation-fit weight
    ``w_m``, and modality probability ``p_m``::

        p_fused = sum_m a_m w_m p_m / sum_m a_m w_m

    When the denominator is zero the system abstains instead of substituting a
    fabricated patient score.
    """

    weights: dict[str, float]
    expected_modalities: tuple[str, ...] = ("eeg", "wearable", "ehr")
    minimum_quality: float = 0.5
    anchor_tolerance_seconds: float = 1.0

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(self.expected_modalities)
        if unknown:
            raise ValueError(f"Weights contain unexpected modalities: {sorted(unknown)}")
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("Fusion weights must be finite and non-negative.")
        if not 0.0 <= self.minimum_quality <= 1.0:
            raise ValueError("minimum_quality must be in [0, 1].")

    def fuse(self, evidence: Iterable[PredictionEvidence]) -> FusedRisk:
        items = tuple(evidence)
        if not items:
            raise ValueError("At least one evidence record is required.")
        self._validate_context(items)
        by_modality: dict[str, PredictionEvidence] = {}
        for item in items:
            if item.modality in by_modality:
                raise ValueError(f"Duplicate evidence for modality {item.modality!r}.")
            if item.modality not in self.expected_modalities:
                raise ValueError(f"Unexpected modality {item.modality!r}.")
            by_modality[item.modality] = item

        used: list[str] = []
        missing: list[str] = []
        low_quality: list[str] = []
        zero_weight: list[str] = []
        numerator = 0.0
        denominator = 0.0
        for modality in self.expected_modalities:
            item = by_modality.get(modality)
            if item is None or not item.available:
                missing.append(modality)
                continue
            if item.quality < self.minimum_quality:
                low_quality.append(modality)
                continue
            weight = float(self.weights.get(modality, 0.0))
            if weight <= 0:
                zero_weight.append(modality)
                continue
            assert item.probability is not None
            numerator += weight * item.probability
            denominator += weight
            used.append(modality)

        normalized = {
            modality: float(self.weights.get(modality, 0.0) / denominator)
            for modality in used
        } if denominator > 0 else {}
        first = items[0]
        abstained = denominator <= 0
        warnings = tuple(
            warning for item in items for warning in item.warnings
        )
        if abstained:
            warnings += ("No compatible modality passed availability and quality gates.",)
        return FusedRisk(
            patient_id=first.patient_id,
            anchor_time=first.anchor_time,
            prediction_target=first.prediction_target,
            horizon_hours=first.horizon_hours,
            probability=None if abstained else numerator / denominator,
            abstained=abstained,
            used_modalities=tuple(used),
            missing_modalities=tuple(missing),
            low_quality_modalities=tuple(low_quality),
            zero_weight_modalities=tuple(zero_weight),
            normalized_weights=normalized,
            numerator=numerator,
            denominator=denominator,
            warnings=warnings,
        )

    def _validate_context(self, items: tuple[PredictionEvidence, ...]) -> None:
        first = items[0]
        first_time = _parse_time(first.anchor_time)
        for item in items[1:]:
            mismatches: list[str] = []
            if item.patient_id != first.patient_id:
                mismatches.append("patient_id")
            if item.prediction_target != first.prediction_target:
                mismatches.append("prediction_target")
            if not math.isclose(item.horizon_hours, first.horizon_hours, abs_tol=1e-9):
                mismatches.append("horizon_hours")
            seconds = abs((_parse_time(item.anchor_time) - first_time).total_seconds())
            if seconds > self.anchor_tolerance_seconds:
                mismatches.append("anchor_time")
            if mismatches:
                raise IncompatibleEvidenceError(
                    "Cannot fuse evidence with mismatched " + ", ".join(mismatches) + "."
                )


@dataclass(frozen=True)
class HealthEvidencePacket:
    """Frozen, model-produced facts passed to the HealthAgent wording layer."""

    patient_id: str
    anchor_time: str
    task: str
    model_output: dict[str, Any]
    modality_evidence: tuple[dict[str, Any], ...]
    interpretation_features: tuple[dict[str, Any], ...] = ()
    unsupported_outcomes: tuple[str, ...] = ("stress", "anxiety", "depression")
    limitations: tuple[str, ...] = ()
    release_status: str = "research_only_do_not_deploy"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
