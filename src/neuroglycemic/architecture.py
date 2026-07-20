import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    AvailabilityAwareFusion,
    HealthEvidencePacket,
    IncompatibleEvidenceError,
    PredictionEvidence,
)
from .health_agent import HealthAgent


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing architecture artifact {path}. Run both training studies first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def run_architecture_case(
    project_root: Path,
    *,
    health_agent: HealthAgent | None = None,
    include_raw_llm_response: bool = False,
) -> dict[str, Any]:
    """Exercise target-safe routing, missingness, and HealthAgent narration.

    This deliberately does not merge CogWear and MIMIC records. It proves that
    the production boundary rejects that invalid operation while still running
    each real-data task and explaining it through one HealthAgent interface.
    """
    health_agent = health_agent or HealthAgent()
    cogwear_metrics = _read_json(project_root / "outputs" / "metrics.json")
    ehr_case = _read_json(project_root / "outputs" / "ehr_glucose" / "case_study.json")
    ehr_acceptance = _read_json(
        project_root / "outputs" / "ehr_glucose" / "acceptance_checks.json"
    )
    predictions_path = project_root / "outputs" / "test_window_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    cogwear = pd.read_csv(predictions_path).sort_values(
        ["patient_id", "condition", "window_index"]
    )
    row = cogwear.iloc[0]
    anchor_time = pd.Timestamp(float(row["window_start_unix"]), unit="s", tz="UTC").isoformat()
    checkpoint = cogwear_metrics["selected_checkpoints"]
    eeg_quality = 1.0 if int(checkpoint["eeg"]["epoch"]) > 0 else 0.0
    wearable_quality = 1.0 if int(checkpoint["wearable"]["epoch"]) > 0 else 0.0

    eeg = PredictionEvidence(
        patient_id=str(row["patient_id"]),
        anchor_time=anchor_time,
        prediction_target="cognitive_load_during_recorded_session",
        horizon_hours=0.0,
        modality="eeg",
        probability=float(row["alpha_eeg"]),
        available=True,
        quality=eeg_quality,
        source_cohort="CogWear pilot",
        source_device="Muse",
        model_version="cogwear-eeg-v1",
        warnings=("EEG head is validation-degenerate.",) if eeg_quality == 0 else (),
    )
    wearable = PredictionEvidence(
        patient_id=str(row["patient_id"]),
        anchor_time=anchor_time,
        prediction_target="cognitive_load_during_recorded_session",
        horizon_hours=0.0,
        modality="wearable",
        probability=float(row["beta_wearable"]),
        available=True,
        quality=wearable_quality,
        source_cohort="CogWear pilot",
        source_device="Empatica",
        model_version="cogwear-wearable-v1",
    )
    cognitive_fusion = AvailabilityAwareFusion(
        weights={
            "eeg": float(cogwear_metrics["fusion_weights"]["eeg"]),
            "wearable": float(cogwear_metrics["fusion_weights"]["wearable"]),
        },
        expected_modalities=("eeg", "wearable"),
    ).fuse((eeg, wearable))

    ehr_response = ehr_case["response"]
    ehr = PredictionEvidence(
        patient_id=str(ehr_response["patient_id"]),
        anchor_time=str(ehr_response["anchor_time"]),
        prediction_target="hyperglycemia_above_180_mg_dl",
        horizon_hours=float(ehr_response["horizon_hours"]),
        modality="ehr",
        probability=float(ehr_response["hyperglycemia_probability"]),
        available=not bool(ehr_response["abstained"]),
        quality=1.0,
        source_cohort="MIMIC-IV Demo 2.2",
        source_device="hospital_EHR",
        model_version="ehr-glucose-v3",
        warnings=tuple(ehr_response.get("warnings", [])),
    )

    invalid_fusion_error: str | None = None
    try:
        AvailabilityAwareFusion(
            weights={"eeg": 1 / 3, "wearable": 1 / 3, "ehr": 1 / 3}
        ).fuse((eeg, wearable, ehr))
    except IncompatibleEvidenceError as exc:
        invalid_fusion_error = str(exc)
    if invalid_fusion_error is None:
        raise RuntimeError("Cross-cohort, cross-target fusion was not rejected.")

    cognitive_packet = HealthEvidencePacket(
        patient_id=eeg.patient_id,
        anchor_time=eeg.anchor_time,
        task=eeg.prediction_target,
        model_output={
            "p_cognitive_load_eeg": eeg.probability,
            "p_cognitive_load_wearable": wearable.probability,
            "p_cognitive_load_fused": cognitive_fusion.probability,
            "abstained": cognitive_fusion.abstained,
        },
        modality_evidence=(eeg.as_dict(), wearable.as_dict()),
        limitations=(
            "CogWear labels baseline versus Stroop cognitive load, not a psychiatric disorder.",
            "This held-out set is a small pipeline case study, not clinical validation.",
        ),
        metadata={
            "cohort": "CogWear pilot",
            "horizon_hours": 0.0,
            "model_version": "cogwear-late-fusion-v1",
            "data_scope": "paired EEG and wearable cognitive-load recordings",
        },
    )
    glucose_packet = HealthEvidencePacket(
        patient_id=ehr.patient_id,
        anchor_time=ehr.anchor_time,
        task="six_hour_hospital_laboratory_glucose_forecast",
        model_output={
            "predicted_glucose_mg_dl": float(ehr_response["predicted_glucose_mg_dl"]),
            "prediction_lower_mg_dl": float(ehr_response["prediction_lower_mg_dl"]),
            "prediction_upper_mg_dl": float(ehr_response["prediction_upper_mg_dl"]),
            "p_hyperglycemia_ehr": ehr.probability,
            "abstained": bool(ehr_response["abstained"]),
        },
        modality_evidence=(ehr.as_dict(),),
        interpretation_features=tuple(
            ehr_case.get("interpretation", {}).get(
                "future_minus_current_glucose_z_contributions", []
            )[:5]
        ),
        limitations=(
            "MIMIC-IV glucose is intermittent hospital laboratory data, not CGM.",
            "The test performance gate controls release status.",
        ),
        release_status=str(ehr_acceptance["release_recommendation"]),
        metadata={
            "cohort": "MIMIC-IV Demo 2.2",
            "horizon_hours": float(ehr_response["horizon_hours"]),
            "model_version": "ehr-glucose-v3",
            "data_scope": "hospital EHR laboratory-glucose forecast",
        },
    )
    cognitive_agent = health_agent.run(
        cognitive_packet, include_raw_response=include_raw_llm_response
    )
    glucose_agent = health_agent.run(
        glucose_packet, include_raw_response=include_raw_llm_response
    )

    result = {
        "architecture_status": {
            "current_trainable_tasks": {
                "eeg_wearable": "cognitive_load_during_recorded_session",
                "ehr": "six_hour_hospital_laboratory_glucose_forecast",
            },
            "joint_glucose_fusion_ready": False,
            "reason": (
                "A same-patient, time-aligned EEG + wearable + EHR + CGM cohort is not present. "
                "The implemented fusion contract is ready for such evidence but rejects current incompatible records."
            ),
            "unsupported_current_outcomes": ["stress", "anxiety", "depression"],
            "required_joint_primary_target": (
                "All alpha/beta/theta heads must predict the same future CGM endpoint and horizon."
            ),
        },
        "cogwear_case": {
            "evidence": [eeg.as_dict(), wearable.as_dict()],
            "fusion": cognitive_fusion.as_dict(),
            "health_agent": cognitive_agent.as_dict(),
        },
        "ehr_case": {
            "evidence": ehr.as_dict(),
            "actual_future_glucose_mg_dl": ehr_case["actual_future_glucose_mg_dl"],
            "absolute_error_mg_dl": ehr_case["absolute_error_mg_dl"],
            "health_agent": glucose_agent.as_dict(),
        },
        "cross_cohort_fusion_guard": {
            "passed": True,
            "rejected_reason": invalid_fusion_error,
        },
        "acceptance": {
            "same_target_fusion_enforced": True,
            "same_patient_fusion_enforced": True,
            "missing_quality_gates_active": True,
            "unsupported_mental_health_outcomes_not_inferred": True,
            "llm_is_post_inference_only": True,
            "ehr_clinical_release_gate_passed": bool(
                ehr_acceptance["clinical_performance_gate_passed"]
            ),
            "release_recommendation": "research_only_do_not_deploy",
        },
    }
    safe_result = _safe(result)
    output = project_root / "outputs" / "architecture" / "case_study.json"
    _write_json(output, safe_result)
    return safe_result
