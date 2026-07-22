"""FHIR R4 export for measured CGM readings and research forecasts.

Measured CGM values follow the HL7 CGM STU1 mass-per-volume Observation
profile. Neural forecasts use a separate local code so a prediction cannot be
mistaken for a sensor measurement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import math
from typing import Any, Mapping, Sequence


CGM_MASS_PROFILE = (
    "http://hl7.org/fhir/uv/cgm/StructureDefinition/"
    "cgm-sensor-reading-mass-per-volume"
)
LOINC_CGM_READING_MASS = "99504-3"
FORECAST_CODE_SYSTEM = "https://neuroglycemic.example/fhir/CodeSystem/forecast"


def _patient_reference(value: str) -> str:
    reference = str(value).strip()
    if not reference.startswith("Patient/") or len(reference) <= len("Patient/"):
        raise ValueError("patient_reference must be a non-empty Patient/<id> reference.")
    return reference


def _iso(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("FHIR timestamps must be valid ISO-8601 values.") from exc
    if parsed.tzinfo is None:
        raise ValueError("FHIR timestamps must include a timezone offset.")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _quantity_mg_dl(value: float) -> dict[str, Any]:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("FHIR glucose quantities must be finite.")
    return {
        "value": numeric,
        "unit": "mg/dl",
        "system": "http://unitsofmeasure.org",
        "code": "mg/dL",
    }


def cgm_sensor_observation(
    *,
    patient_reference: str,
    effective_time: str,
    glucose_mg_dl: float,
    source_identifier: str,
    device_reference: str | None = None,
) -> dict[str, Any]:
    patient_reference = _patient_reference(patient_reference)
    if not source_identifier.strip():
        raise ValueError("source_identifier is required for deduplication.")
    observation: dict[str, Any] = {
        "resourceType": "Observation",
        "meta": {"profile": [CGM_MASS_PROFILE]},
        "identifier": [
            {
                "system": "https://neuroglycemic.example/identifier/cgm-reading",
                "value": source_identifier,
            }
        ],
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": LOINC_CGM_READING_MASS,
                    "display": "Glucose [Mass/volume] in interstitial fluid by continuous glucose monitor",
                }
            ]
        },
        "subject": {"reference": patient_reference},
        "effectiveDateTime": _iso(effective_time),
        "valueQuantity": _quantity_mg_dl(glucose_mg_dl),
    }
    if device_reference:
        if not device_reference.startswith("Device/"):
            raise ValueError("device_reference must start with Device/.")
        observation["device"] = {"reference": device_reference}
    return observation


def neural_forecast_observation(
    forecast: Mapping[str, Any], *, patient_reference: str
) -> dict[str, Any]:
    """Export a model forecast without labeling it as a CGM measurement."""

    if bool(forecast.get("abstained")):
        raise ValueError("An abstained model response cannot become a FHIR value.")
    patient_reference = _patient_reference(patient_reference)
    anchor = datetime.fromisoformat(str(forecast["anchor_time"]).replace("Z", "+00:00"))
    if anchor.tzinfo is None:
        raise ValueError("Forecast anchor_time must include a timezone offset.")
    horizon = int(forecast["horizon_minutes"])
    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")
    effective = anchor + timedelta(minutes=horizon)
    predicted = float(forecast["predicted_glucose_mg_dl"])
    lower = float(forecast["prediction_lower_mg_dl"])
    upper = float(forecast["prediction_upper_mg_dl"])
    if not all(math.isfinite(value) for value in (predicted, lower, upper)):
        raise ValueError("Forecast values and interval bounds must be finite.")
    if lower > predicted or predicted > upper:
        raise ValueError("Forecast interval must satisfy lower <= prediction <= upper.")
    identity = "|".join(
        (
            patient_reference,
            anchor.isoformat(),
            str(horizon),
            str(forecast.get("model_version", "unknown")),
        )
    )
    identifier = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return {
        "resourceType": "Observation",
        "identifier": [
            {
                "system": "https://neuroglycemic.example/identifier/model-forecast",
                "value": identifier,
            }
        ],
        "status": "preliminary",
        "code": {
            "coding": [
                {
                    "system": FORECAST_CODE_SYSTEM,
                    "code": "future-glucose",
                    "display": "Research neural glucose forecast",
                }
            ],
            "text": "Research neural glucose forecast; not a CGM sensor reading",
        },
        "subject": {"reference": patient_reference},
        "effectiveDateTime": effective.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "issued": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "valueQuantity": _quantity_mg_dl(predicted),
        "component": [
            {
                "code": {
                    "coding": [
                        {"system": FORECAST_CODE_SYSTEM, "code": "lower-95"}
                    ]
                },
                "valueQuantity": _quantity_mg_dl(lower),
            },
            {
                "code": {
                    "coding": [
                        {"system": FORECAST_CODE_SYSTEM, "code": "upper-95"}
                    ]
                },
                "valueQuantity": _quantity_mg_dl(upper),
            },
            {
                "code": {
                    "coding": [
                        {"system": FORECAST_CODE_SYSTEM, "code": "horizon-minutes"}
                    ]
                },
                "valueQuantity": {
                    "value": horizon,
                    "unit": "minute",
                    "system": "http://unitsofmeasure.org",
                    "code": "min",
                },
            },
        ],
        "method": {
            "coding": [
                {
                    "system": FORECAST_CODE_SYSTEM,
                    "code": str(forecast.get("model_version", "unknown")),
                }
            ]
        },
        "note": [
            {
                "text": "Research use only. This predicted value is not a sensor measurement, diagnosis, alarm, or dosing recommendation."
            }
        ],
    }


def observation_transaction_bundle(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not observations:
        raise ValueError("At least one Observation is required.")
    entries = []
    for observation in observations:
        if observation.get("resourceType") != "Observation":
            raise ValueError("Only Observation resources are accepted.")
        identifier = observation.get("identifier", [{}])[0]
        system = str(identifier.get("system", ""))
        value = str(identifier.get("value", ""))
        if not system or not value:
            raise ValueError("Every Observation needs a stable identifier.")
        entries.append(
            {
                "resource": dict(observation),
                "request": {
                    "method": "POST",
                    "url": "Observation",
                    "ifNoneExist": f"identifier={system}|{value}",
                },
            }
        )
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}
