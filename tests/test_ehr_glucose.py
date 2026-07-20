from pathlib import Path

import numpy as np
import pytest

from src.neuroglycemic.config import load_ehr_config
from src.neuroglycemic.ehr_data import build_glucose_forecast_table
from src.neuroglycemic.model import (
    MultitaskParameters,
    ProbabilisticGlucoseModel,
    Standardizer,
    multitask_loss_and_gradients,
)
from src.neuroglycemic.pipeline import patient_group_split, split_audit
from src.neuroglycemic.service import GlucoseForecastRequest, forecast_one


ROOT = Path(__file__).resolve().parents[1]


def test_multitask_regression_gradient_matches_finite_difference() -> None:
    x = np.array([[0.2, -0.4], [1.0, 0.3], [-0.5, 0.8]])
    y_reg = np.array([0.1, -0.2, 0.4])
    y_cls = np.array([0.0, 1.0, 0.0])
    parameters = MultitaskParameters(
        regression_weights=np.array([0.15, -0.05]),
        regression_bias=0.02,
        log_variance=-0.1,
        classification_weights=np.array([0.1, 0.2]),
        classification_bias=-0.03,
    )
    _, gradient = multitask_loss_and_gradients(
        x,
        y_reg,
        y_cls,
        parameters,
        positive_weight=2.0,
        classification_loss_weight=0.5,
        l2=0.01,
    )
    epsilon = 1e-6
    plus = parameters.copy()
    minus = parameters.copy()
    plus.regression_weights[0] += epsilon
    minus.regression_weights[0] -= epsilon
    plus_loss, _ = multitask_loss_and_gradients(
        x, y_reg, y_cls, plus, positive_weight=2.0, classification_loss_weight=0.5, l2=0.01
    )
    minus_loss, _ = multitask_loss_and_gradients(
        x, y_reg, y_cls, minus, positive_weight=2.0, classification_loss_weight=0.5, l2=0.01
    )
    numerical = (plus_loss["total_loss"] - minus_loss["total_loss"]) / (2 * epsilon)
    assert gradient.regression_weights[0] == pytest.approx(numerical, rel=1e-5, abs=1e-7)


def test_real_mimic_demo_cohort_is_causal_and_patient_disjoint() -> None:
    config = load_ehr_config(ROOT / "config" / "ehr_glucose.json")
    if not config.raw_dir.exists():
        pytest.skip("Run scripts/download_mimic_demo.py for the real-data integration test.")
    cohort, _ = build_glucose_forecast_table(config)
    assert len(cohort) > 0
    assert cohort["patient_id"].nunique() > 20
    assert (cohort["target_charttime"] > cohort["anchor_time"]).all()
    assert (cohort["target_storetime"] > cohort["anchor_time"]).all()
    split = patient_group_split(cohort, config)
    assert split_audit(split)["patient_disjoint"] is True


def test_service_abstains_without_required_glucose_history() -> None:
    model = ProbabilisticGlucoseModel(
        feature_names=("current_glucose_mg_dl", "previous_glucose_mg_dl"),
        standardizer=Standardizer(
            medians=np.array([100.0, 100.0]),
            means=np.array([100.0, 100.0]),
            scales=np.array([10.0, 10.0]),
        ),
        target_center=0.0,
        target_scale=10.0,
        regression_target="delta_from_reference",
        reference_feature="current_glucose_mg_dl",
        parameters=MultitaskParameters(
            regression_weights=np.zeros(2),
            regression_bias=0.0,
            log_variance=0.0,
            classification_weights=np.zeros(2),
            classification_bias=0.0,
        ),
        hyperglycemia_threshold_mg_dl=180.0,
        prediction_interval=0.9,
        best_epoch=1,
        best_validation_loss=1.0,
        training_metadata={},
    )
    response = forecast_one(
        model,
        GlucoseForecastRequest(
            patient_id="held-out",
            anchor_time="2026-01-01T00:00:00",
            features={"current_glucose_mg_dl": None, "previous_glucose_mg_dl": 100.0},
        ),
        horizon_hours=6.0,
    )
    assert response.abstained is True
    assert np.isnan(response.predicted_glucose_mg_dl)
