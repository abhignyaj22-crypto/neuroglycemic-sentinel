import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neuroglycemic.architecture import run_architecture_case
from src.neuroglycemic.contracts import (
    AvailabilityAwareFusion,
    HealthEvidencePacket,
    IncompatibleEvidenceError,
    PredictionEvidence,
)
from src.neuroglycemic.health_agent import HealthAgent
from src.neuroglycemic.interoperability import (
    audit_canonical_streams,
    canonicalize_wide_stream,
)
from src.neuroglycemic.lsl import audit_timestamp_array


ROOT = Path(__file__).resolve().parents[1]


def _evidence(
    modality: str,
    probability: float,
    *,
    available: bool = True,
    patient_id: str = "same-patient",
    target: str = "hyperglycemia_60m",
) -> PredictionEvidence:
    return PredictionEvidence(
        patient_id=patient_id,
        anchor_time="2026-01-01T00:00:00+00:00",
        prediction_target=target,
        horizon_hours=1.0,
        modality=modality,
        probability=probability if available else None,
        available=available,
        quality=1.0,
        source_cohort="prospective_joint_cohort",
        source_device=modality,
        model_version="test-v1",
    )


def test_all_three_modality_missingness_patterns_are_exact_or_abstain() -> None:
    weights = {"eeg": 0.2, "wearable": 0.3, "ehr": 0.5}
    probabilities = {"eeg": 0.1, "wearable": 0.4, "ehr": 0.8}
    fusion = AvailabilityAwareFusion(weights=weights)
    modalities = tuple(weights)
    for mask in range(8):
        present = {
            modality for index, modality in enumerate(modalities) if mask & (1 << index)
        }
        evidence = tuple(
            _evidence(
                modality,
                probabilities[modality],
                available=modality in present,
            )
            for modality in modalities
        )
        result = fusion.fuse(evidence)
        if not present:
            assert result.abstained is True
            assert result.probability is None
            continue
        expected = sum(weights[name] * probabilities[name] for name in present) / sum(
            weights[name] for name in present
        )
        assert result.abstained is False
        assert result.probability == pytest.approx(expected)
        assert set(result.used_modalities) == present


@pytest.mark.parametrize(
    ("field", "second"),
    [
        ("patient", _evidence("wearable", 0.4, patient_id="different-patient")),
        ("target", _evidence("wearable", 0.4, target="cognitive_load")),
    ],
)
def test_fusion_rejects_cross_patient_or_cross_target_evidence(
    field: str, second: PredictionEvidence
) -> None:
    del field
    fusion = AvailabilityAwareFusion(
        weights={"eeg": 0.5, "wearable": 0.5},
        expected_modalities=("eeg", "wearable"),
    )
    with pytest.raises(IncompatibleEvidenceError):
        fusion.fuse((_evidence("eeg", 0.2), second))


def test_health_agent_calls_injected_langchain_llm_and_hides_patient_identifier() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    captured: dict[str, str] = {}

    def respond(prompt: object) -> AIMessage:
        captured["prompt"] = str(prompt)
        return AIMessage(
            content=json.dumps(
                {
                    "interpretation": "The supplied research output is uncertain.",
                    "limitations": ["This is not a diagnosis."],
                    "research_next_step": "Validate on a larger held-out cohort.",
                }
            )
        )

    packet = HealthEvidencePacket(
        patient_id="private-patient-id",
        anchor_time="2026-01-01T00:00:00+00:00",
        task="hyperglycemia_60m",
        model_output={"probability": 0.4},
        modality_evidence=(_evidence("ehr", 0.4).as_dict(),),
    )
    result = HealthAgent(
        llm=RunnableLambda(respond), provider="fake", model_name="recording-llm"
    ).run(packet, include_raw_response=True)
    assert result.telemetry.llm_called is True
    assert result.telemetry.fallback_reason is None
    assert result.llm_interpretation == "The supplied research output is uncertain."
    assert "private-patient-id" not in captured["prompt"]
    assert "patient_reference" in captured["prompt"]
    assert result.telemetry.raw_response is not None
    assert result.evidence_packet["model_output"]["probability"] == 0.4


def test_health_agent_reports_deterministic_fallback_when_llm_is_disabled() -> None:
    packet = HealthEvidencePacket(
        patient_id="p1",
        anchor_time="2026-01-01T00:00:00+00:00",
        task="research_task",
        model_output={"probability": 0.25},
        modality_evidence=(),
    )
    result = HealthAgent().run(packet)
    assert result.telemetry.llm_called is False
    assert result.telemetry.fallback_reason is not None
    assert result.llm_interpretation is None


def test_real_empatica_rows_convert_to_canonical_interoperable_schema() -> None:
    path = ROOT / "data" / "raw" / "cogwear" / "pilot" / "0" / "baseline" / "empatica_bvp.csv"
    if not path.exists():
        pytest.skip("CogWear raw files are not present.")
    raw = pd.read_csv(path, nrows=10)
    canonical = canonicalize_wide_stream(
        raw,
        patient_id="cogwear_00",
        session_id="baseline",
        device="empatica",
        signal="bvp",
        unit="arbitrary_unit",
        timestamp_column="time",
        channel_columns={"bvp": "bvp"},
        sampling_rate_hz=64.0,
        source=str(path),
        unix_timestamps=True,
    )
    audit = audit_canonical_streams(canonical)
    assert len(canonical) == 10
    assert canonical["patient_id"].eq("cogwear_00").all()
    assert audit.iloc[0]["duplicate_timestamps"] == 0
    assert audit.iloc[0]["sampling_rate_hz"] == 64.0


def test_lsl_audit_detects_clock_failures() -> None:
    audit = audit_timestamp_array(
        timestamps=np.array([1.0, 1.1, 1.1, 1.05, 1.4]),
        name="EEG",
        stream_type="EEG",
        source_id="emotiv-test",
        channel_count=14,
        nominal_rate_hz=128.0,
    )
    assert audit.duplicate_timestamps == 1
    assert audit.backward_timestamps == 1
    assert audit.largest_gap_seconds == pytest.approx(0.35)


def test_tracked_real_artifacts_exercise_cross_cohort_guard_end_to_end(tmp_path: Path) -> None:
    del tmp_path
    required = (
        ROOT / "outputs" / "metrics.json",
        ROOT / "outputs" / "test_window_predictions.csv",
        ROOT / "outputs" / "ehr_glucose" / "case_study.json",
        ROOT / "outputs" / "ehr_glucose" / "acceptance_checks.json",
    )
    if not all(path.exists() for path in required):
        pytest.skip("Run both real-data pipelines to create architecture artifacts.")
    result = run_architecture_case(ROOT)
    assert result["cross_cohort_fusion_guard"]["passed"] is True
    assert result["architecture_status"]["joint_glucose_fusion_ready"] is False
    assert result["acceptance"]["unsupported_mental_health_outcomes_not_inferred"] is True
