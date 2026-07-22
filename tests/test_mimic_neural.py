from pathlib import Path

import pandas as pd

from src.neuroglycemic.mimic_neural import (
    MIMIC_NEURAL_FEATURES,
    prepare_mimic_neural_frame,
)
from src.neuroglycemic.neural_dataset import load_aligned_window_frame


def test_real_mimic_demo_adapts_to_patient_disjoint_neural_contract(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[1]
    source = pd.read_csv(project / "data" / "processed" / "mimiciv_demo_glucose_6h.csv")
    windows = prepare_mimic_neural_frame(source)
    assert windows["patient_id"].nunique() >= 10
    assert windows["ehr_available"].all()
    assert windows["ehr_available_time"].le(windows["anchor_time"]).all()
    destination = tmp_path / "mimic_neural.csv"
    windows.to_csv(destination, index=False)
    loaded, features = load_aligned_window_frame(
        destination,
        (360,),
        modalities=("ehr",),
        horizon_tolerance_minutes=180.0,
        feature_registry={"ehr": MIMIC_NEURAL_FEATURES},
        input_cgm=False,
    )
    assert len(loaded) == len(windows)
    assert features["ehr"] == MIMIC_NEURAL_FEATURES
