"""Auditable product artifacts for a trained NeuroGlycemic research model."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_CARD_SCHEMA = "neuroglycemic-model-card-v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_neural_model_card(
    *,
    prediction_target: str,
    horizons_minutes: Sequence[int],
    modalities: Sequence[str],
    cohorts: Sequence[str],
    split_counts: Mapping[str, Mapping[str, int]],
    metrics: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    checkpoint_path: Path,
    release_manifest_path: Path,
    data_sha256: str,
) -> dict[str, Any]:
    """Build a factual model card from immutable run outputs.

    The card intentionally distinguishes a functioning research software
    product from a clinically released medical model.  It never promotes a run
    based only on optimizer convergence.
    """

    return {
        "schema_version": MODEL_CARD_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "product": "NeuroGlycemic Sentinel",
        "model_family": "availability-aware neural Gaussian mixture-of-experts",
        "prediction_target": str(prediction_target),
        "forecast_horizons_minutes": [int(value) for value in horizons_minutes],
        "modalities": [str(value) for value in modalities],
        "evaluated_cohorts": [str(value) for value in cohorts],
        "patient_disjoint_split": {
            str(name): {
                "rows": int(values["rows"]),
                "patients": int(values["patients"]),
            }
            for name, values in split_counts.items()
        },
        "data_sha256": str(data_sha256),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "release_manifest_path": str(Path(release_manifest_path).resolve()),
        "held_out_metrics": dict(metrics),
        "assurance": dict(acceptance),
        "intended_use": (
            "Research evaluation of future glucose forecasting on the explicitly "
            "named cohorts and horizons."
        ),
        "prohibited_use": [
            "glucose measurement replacement",
            "diagnosis",
            "hypoglycemia alarm",
            "insulin or medication dosing",
            "clinical deployment without prospective validation and regulatory review",
        ],
        "interpretation_contract": (
            "HealthAgent may explain immutable numerical outputs but cannot alter "
            "predictions, learned fusion weights, uncertainty, or release status."
        ),
        "release_status": (
            "clinical_release_ready"
            if bool(acceptance.get("clinical_release_ready"))
            else "research_only"
        ),
    }


def write_neural_model_card(path: Path, card: Mapping[str, Any]) -> Path:
    """Atomically write a strict-JSON model card."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(dict(card)), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
