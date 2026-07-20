from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .model import ProbabilisticGlucoseModel


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
            warnings=(f"Missing required glucose history: {', '.join(missing_required)}",),
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

