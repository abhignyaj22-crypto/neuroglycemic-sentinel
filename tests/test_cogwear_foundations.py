import numpy as np
import pandas as pd
from pathlib import Path

from src.cogwear_study.config import load_config
from src.cogwear_study.data import discover_incomplete_sessions
from src.cogwear_study.features import EEG_FEATURES, WEARABLE_FEATURES
from src.cogwear_study.fusion import fit_late_fusion
from src.cogwear_study.model import TrainOnlyStandardizer, binary_cross_entropy, sigmoid
from src.cogwear_study.split import attach_split, split_patients


def test_patient_windows_never_cross_splits() -> None:
    patient_ids = [f"p{number}" for number in range(11)]
    split = split_patients(
        patient_ids, seed=42, train_fraction=0.64, validation_fraction=0.18
    )
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    rows = pd.DataFrame({"patient_id": patient_ids * 3})
    attached = attach_split(rows, split)
    assert attached.groupby("patient_id")["split"].nunique().max() == 1


def test_train_only_standardization_and_finite_bce() -> None:
    scaler = TrainOnlyStandardizer().fit(np.array([[0.0], [2.0], [np.nan]]))
    assert scaler.medians is not None
    assert scaler.medians[0] == 1.0
    assert scaler.transform(np.array([[101.0]]))[0, 0] > 50.0
    probability = sigmoid(np.array([-1000.0, 1000.0]))
    assert np.isfinite(binary_cross_entropy(np.array([0.0, 1.0]), probability))


def test_fusion_renormalizes_when_a_modality_is_missing() -> None:
    target = np.array([0, 0, 1, 1], dtype=float)
    probability = np.array(
        [[0.05, 0.45], [0.10, 0.55], [0.90, 0.45], [0.95, 0.55]]
    )
    fusion, _ = fit_late_fusion(
        probability, target, learning_rate=0.1, epochs=500, fallback_probability=0.4
    )
    availability = np.array([[False, True], [True, False], [False, False]])
    result = fusion.predict(
        np.array([[0.9, 0.2], [0.8, 0.1], [0.7, 0.3]]), availability
    )
    assert np.allclose(result, [0.2, 0.8, 0.4])


def test_fusion_can_zero_weight_a_degenerate_head() -> None:
    probability = np.array([[0.5, 0.1], [0.5, 0.9]])
    target = np.array([0.0, 1.0])
    fusion, _ = fit_late_fusion(
        probability,
        target,
        learning_rate=0.1,
        epochs=20,
        eligible_modalities=np.array([False, True]),
    )
    assert np.array_equal(fusion.weights, np.array([0.0, 1.0]))


def test_cogwear_features_do_not_claim_unobserved_clinical_inputs() -> None:
    features = set(EEG_FEATURES) | set(WEARABLE_FEATURES)
    forbidden = {"diagnosis", "medication", "glucose", "blood_pressure", "step_count", "ehr"}
    assert not any(term in feature for term in forbidden for feature in features)


def test_real_partial_participant_is_discovered_but_not_configured_for_training() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "study.json")
    if not (config.raw_dir / "3").exists():
        return
    incomplete = discover_incomplete_sessions(config)
    participant_three = incomplete.loc[incomplete["participant_number"] == 3]
    assert not participant_three.empty
    assert 3 not in config.participants
    complete = participant_three["eeg_available"].astype(bool) & participant_three[
        "wearable_available"
    ].astype(bool)
    assert (~complete).all()
