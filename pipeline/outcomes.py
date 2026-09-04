"""Endpoint registry — blocked packets cannot carry a score.

Standalone copy of the lab contract so this extract's tests do not need the
private tree. Complication and progression stay structurally absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class EvidenceStatus(str, Enum):
    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    BLOCKED = "blocked"
    ABSTAINED = "abstained"


class ValueType(str, Enum):
    PROBABILITY = "probability"
    FORECAST = "forecast"
    INDEX = "index"


class OutcomeFitError(RuntimeError):
    """Raised when a caller tries to fit a reportable model on a non-reportable endpoint."""


@dataclass(frozen=True)
class OutcomeSpec:
    endpoint_id: str
    label_kind: str
    default_status: EvidenceStatus
    required_modalities: Tuple[str, ...]
    real_label_task: Optional[str] = None
    structural_label_absent: bool = False
    real_label_is_reportable: bool = False
    caveat: str = ""

    @property
    def may_fit_reportable(self) -> bool:
        return (
            self.label_kind == "real"
            and self.real_label_is_reportable
            and not self.structural_label_absent
        )


@dataclass(frozen=True)
class OutcomeResult:
    endpoint: str
    value: Optional[float]
    value_type: ValueType
    evidence_status: EvidenceStatus
    abstain_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.evidence_status in (EvidenceStatus.BLOCKED, EvidenceStatus.ABSTAINED):
            if self.value is not None:
                raise ValueError(
                    f"{self.endpoint}: {self.evidence_status.value} packets cannot carry a score"
                )
            if not self.abstain_reason:
                raise ValueError(
                    f"{self.endpoint}: {self.evidence_status.value} packets require abstain_reason"
                )


def _spec(**kwargs) -> OutcomeSpec:
    return OutcomeSpec(**kwargs)


OUTCOMES: Dict[str, OutcomeSpec] = {
    "stress_detection": _spec(
        endpoint_id="stress_detection",
        label_kind="real",
        default_status=EvidenceStatus.EXPERIMENTAL,
        required_modalities=("eda", "ppg", "resp", "temp", "motion"),
        real_label_task="wesad_stress",
        real_label_is_reportable=True,
        caveat="WESAD specialists remain the validated path.",
    ),
    "anxiety_prediction": _spec(
        endpoint_id="anxiety_prediction",
        label_kind="proxy",
        default_status=EvidenceStatus.EXPERIMENTAL,
        required_modalities=("eda", "ppg"),
        real_label_task="deap_anxiety",
        caveat="DEAP anxiety remains excluded/at-chance.",
    ),
    "depression_risk": _spec(
        endpoint_id="depression_risk",
        label_kind="real",
        default_status=EvidenceStatus.EXPERIMENTAL,
        required_modalities=("eeg",),
        real_label_task="mumtaz_depression",
        caveat="Mumtaz identity-leakage confound; not a product claim.",
    ),
    "cognitive_overload": _spec(
        endpoint_id="cognitive_overload",
        label_kind="real",
        default_status=EvidenceStatus.EXPERIMENTAL,
        required_modalities=("eeg",),
        real_label_task="eegmat_workload",
        real_label_is_reportable=True,
        caveat="EEGMAT rest-vs-task labels are real.",
    ),
    "glucose_forecast": _spec(
        endpoint_id="glucose_forecast",
        label_kind="real",
        default_status=EvidenceStatus.EXPERIMENTAL,
        required_modalities=("cgm",),
        real_label_task="cgmacros_glucose",
        real_label_is_reportable=True,
        caveat="ridge_raw_sequence is the current floor.",
    ),
    "diabetes_complication": _spec(
        endpoint_id="diabetes_complication",
        label_kind="structural_absent",
        default_status=EvidenceStatus.BLOCKED,
        required_modalities=("ehr", "cgm"),
        structural_label_absent=True,
        caveat="No dated complication labels; blocked.",
    ),
    "diabetes_progression": _spec(
        endpoint_id="diabetes_progression",
        label_kind="structural_absent",
        default_status=EvidenceStatus.BLOCKED,
        required_modalities=("ehr",),
        structural_label_absent=True,
        caveat="No incident-progression timestamps; blocked.",
    ),
}


def get_outcome(endpoint_id: str) -> OutcomeSpec:
    try:
        return OUTCOMES[endpoint_id]
    except KeyError as exc:
        raise KeyError(f"Unknown endpoint {endpoint_id!r}") from exc


def may_fit_reportable(endpoint_id: str, *, raise_on_deny: bool = False) -> bool:
    spec = get_outcome(endpoint_id)
    if spec.may_fit_reportable:
        return True
    if raise_on_deny:
        raise OutcomeFitError(
            f"{endpoint_id} cannot fit a reportable model "
            f"(label_kind={spec.label_kind}, structural_absent={spec.structural_label_absent})"
        )
    return False


def serve_outcome(endpoint_id: str, *, value: Optional[float] = None) -> OutcomeResult:
    spec = get_outcome(endpoint_id)
    if spec.structural_label_absent or spec.default_status == EvidenceStatus.BLOCKED:
        return OutcomeResult(
            endpoint=endpoint_id,
            value=None,
            value_type=ValueType.PROBABILITY,
            evidence_status=EvidenceStatus.BLOCKED,
            abstain_reason=spec.caveat or "structural label absence",
        )
    if value is None:
        return OutcomeResult(
            endpoint=endpoint_id,
            value=None,
            value_type=ValueType.PROBABILITY,
            evidence_status=EvidenceStatus.ABSTAINED,
            abstain_reason=spec.caveat or "no score",
        )
    return OutcomeResult(
        endpoint=endpoint_id,
        value=value,
        value_type=ValueType.PROBABILITY,
        evidence_status=spec.default_status,
        abstain_reason=None,
    )
