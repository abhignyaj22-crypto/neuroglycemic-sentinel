"""Orchestration for MIMIC-IV EHR glucose."""


import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import EHRGlucoseConfig
from .ehr_data import EHR_FEATURES, build_glucose_forecast_table, inspect_mimic_tables
from .evaluation import (
    comparison_table,
    forecast_metrics,
    hyperglycemia_metrics,
    patient_cluster_bootstrap,
    probabilistic_metrics,
    subgroup_table,
)
from .model import ProbabilisticGlucoseModel, fit_probabilistic_glucose_model
from .service import GlucoseForecastRequest, forecast_one


def _print_frame(name: str, frame: pd.DataFrame) -> None:
    print(f"\n{name}")
    print(f"shape: {frame.shape}")
    print(f"columns ({len(frame.columns)}): {frame.columns.tolist()}")
    print("head(5):")
    print(frame.head(5).to_string(index=False))


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_json(payload), indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_manifest(config: EHRGlucoseConfig) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for stem in ("patients", "admissions", "d_labitems", "labevents"):
        candidates = (config.raw_dir / f"{stem}.csv.gz", config.raw_dir / f"{stem}.csv")
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing MIMIC-IV table: {stem}")
        files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {
        "source": "MIMIC-IV Clinical Database Demo 2.2",
        "source_url": "https://physionet.org/content/mimic-iv-demo/2.2/",
        "important_scope": "Intermittent hospital laboratory glucose, not ambulatory CGM.",
        "files": files,
    }


def patient_group_split(dataset: pd.DataFrame, config: EHRGlucoseConfig) -> pd.DataFrame:
    """Assign every row from one patient to exactly one partition."""
    patient_outcome = (
        dataset.groupby("patient_id", as_index=False)["target_hyperglycemia"]
        .max()
        .rename(columns={"target_hyperglycemia": "ever_hyperglycemia"})
    )
    train_patients, remainder = train_test_split(
        patient_outcome,
        train_size=config.train_fraction,
        random_state=config.seed,
        stratify=patient_outcome["ever_hyperglycemia"],
    )
    validation_share = config.validation_fraction / (1.0 - config.train_fraction)
    validation_patients, test_patients = train_test_split(
        remainder,
        train_size=validation_share,
        random_state=config.seed + 1,
        stratify=remainder["ever_hyperglycemia"],
    )
    mapping = {
        **{patient: "train" for patient in train_patients["patient_id"]},
        **{patient: "validation" for patient in validation_patients["patient_id"]},
        **{patient: "test" for patient in test_patients["patient_id"]},
    }
    result = dataset.copy()
    result["split"] = result["patient_id"].map(mapping)
    if result["split"].isna().any():
        raise RuntimeError("At least one patient was not assigned to a partition.")
    return result


def split_audit(dataset: pd.DataFrame) -> dict[str, Any]:
    patient_sets = {
        split: set(part["patient_id"].unique())
        for split, part in dataset.groupby("split", sort=False)
    }
    overlaps = {
        "train_validation": sorted(patient_sets["train"] & patient_sets["validation"]),
        "train_test": sorted(patient_sets["train"] & patient_sets["test"]),
        "validation_test": sorted(patient_sets["validation"] & patient_sets["test"]),
    }
    return {
        "partitions": {
            split: {
                "rows": len(part),
                "patients": int(part["patient_id"].nunique()),
                "hyperglycemia_prevalence": float(part["target_hyperglycemia"].mean()),
            }
            for split, part in dataset.groupby("split", sort=False)
        },
        "patient_overlap": overlaps,
        "patient_disjoint": not any(overlaps.values()),
    }


def _attach_predictions(
    model: ProbabilisticGlucoseModel, partition: pd.DataFrame
) -> pd.DataFrame:
    result = partition.reset_index(drop=True).copy()
    prediction = model.predict(result).reset_index(drop=True)
    for column in prediction:
        result[column] = prediction[column]
    result["theta_ehr"] = result["hyperglycemia_probability"]
    return result


def _case_explanation(
    model: ProbabilisticGlucoseModel, row: pd.Series
) -> dict[str, list[dict[str, float | str]]]:
    standardized = model.standardizer.transform(
        row[list(model.feature_names)].to_numpy(dtype=float).reshape(1, -1)
    )[0]
    regression = standardized * model.parameters.regression_weights
    classification = standardized * model.parameters.classification_weights

    def largest(values: np.ndarray) -> list[dict[str, float | str]]:
        order = np.argsort(np.abs(values))[::-1][:8]
        return [
            {"feature": model.feature_names[index], "linear_contribution": float(values[index])}
            for index in order
        ]

    return {
        "future_minus_current_glucose_z_contributions": largest(regression),
        "hyperglycemia_logit_contributions": largest(classification),
    }


def run_ehr_glucose_pipeline(
    config: EHRGlucoseConfig, *, rebuild: bool = False
) -> dict[str, Any]:
    """Build, fit, evaluate, serialize, reload, and exercise one held-out case."""
    print("\nROUGH EHR MODEL SKETCH")
    print(
        "past 24h EHR + glucose history -> interpretable multitask head -> "
        "future glucose distribution + theta=P(glucose>180 mg/dL)"
    )
    print(
        "Loss = Gaussian NLL(future glucose - current glucose) + "
        f"{config.classification_loss_weight:g} * class-balanced BCE(theta) + "
        f"{config.l2:g} * L2; Adam learning_rate={config.learning_rate:g}."
    )
    print(
        f"Prediction time: {config.forecast_horizon_hours:g}h ahead "
        f"(accepted target window +/-{config.forecast_tolerance_hours:g}h)."
    )
    print("Scope: hospital laboratory glucose, not a CGM or treatment recommendation.")

    for name, frame in inspect_mimic_tables(config).items():
        _print_frame(f"RAW MIMIC-IV DEMO TABLE: {name}", frame)

    if config.processed_path.exists() and not rebuild:
        print(f"\nLoading previously built real EHR cohort: {config.processed_path}")
        dataset = pd.read_csv(
            config.processed_path,
            parse_dates=["anchor_time", "target_charttime", "target_storetime"],
        )
        build_audit = pd.DataFrame([{"measure": "loaded_from_cache", "value": 1}])
    else:
        dataset, build_audit = build_glucose_forecast_table(config)
        config.processed_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(config.processed_path, index=False)

    _print_frame("CAUSAL EHR GLUCOSE FORECAST COHORT", dataset)
    _print_frame("COHORT BUILD AUDIT", build_audit)
    print(f"\nMODEL FEATURES ({len(EHR_FEATURES)}):")
    print(list(EHR_FEATURES))
    print("\nFeature missingness fractions:")
    print(dataset[list(EHR_FEATURES)].isna().mean().sort_values(ascending=False).to_string())

    dataset = patient_group_split(dataset, config)
    audit = split_audit(dataset)
    print("\nPATIENT-LEVEL TRAIN / VALIDATION / TEST AUDIT")
    print(json.dumps(_safe_json(audit), indent=2))
    if not audit["patient_disjoint"]:
        raise RuntimeError("Patient leakage detected between partitions.")

    train = dataset.loc[dataset["split"] == "train"].copy()
    validation = dataset.loc[dataset["split"] == "validation"].copy()
    test = dataset.loc[dataset["split"] == "test"].copy()
    print(
        f"\nTRAIN: {len(train)} anchors/{train['patient_id'].nunique()} patients; "
        f"VALIDATION: {len(validation)} anchors/{validation['patient_id'].nunique()} patients; "
        f"TEST: {len(test)} anchors/{test['patient_id'].nunique()} patients."
    )

    model, history = fit_probabilistic_glucose_model(
        train,
        validation,
        EHR_FEATURES,
        target_regression_column="target_glucose_mg_dl",
        target_classification_column="target_hyperglycemia",
        hyperglycemia_threshold_mg_dl=config.hyperglycemia_threshold_mg_dl,
        prediction_interval=config.prediction_interval,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        l2=config.l2,
        classification_loss_weight=config.classification_loss_weight,
        training_metadata={
            "cohort": "MIMIC-IV Demo 2.2",
            "forecast_horizon_hours": config.forecast_horizon_hours,
            "training_rows": len(train),
            "training_patients": int(train["patient_id"].nunique()),
        },
    )
    _print_frame("TRAINING LOSSES", history)
    print(
        f"Selected epoch {model.best_epoch} with validation total loss "
        f"{model.best_validation_loss:.6f}."
    )
    print(
        "Validation-selected residual weight versus persistence: "
        f"{model.regression_blend_weight:.2f}."
    )
    coefficients = model.feature_coefficients()
    coefficients["max_absolute_coefficient"] = coefficients[
        [
            "future_minus_current_glucose_z_coefficient",
            "hyperglycemia_logit_coefficient",
        ]
    ].abs().max(axis=1)
    coefficients = coefficients.sort_values("max_absolute_coefficient", ascending=False)
    _print_frame("INTERPRETABLE STANDARDIZED COEFFICIENTS", coefficients)

    train_predictions = _attach_predictions(model, train)
    validation_predictions = _attach_predictions(model, validation)
    test_predictions = _attach_predictions(model, test)
    _print_frame(
        "HELD-OUT TEST PREDICTIONS",
        test_predictions[
            [
                "patient_id",
                "anchor_time",
                "target_glucose_mg_dl",
                "predicted_glucose_mg_dl",
                "prediction_lower_mg_dl",
                "prediction_upper_mg_dl",
                "target_hyperglycemia",
                "theta_ehr",
            ]
        ],
    )

    metrics = {
        **forecast_metrics(test_predictions, "predicted_glucose_mg_dl"),
        **probabilistic_metrics(test_predictions),
        **hyperglycemia_metrics(test_predictions),
    }
    comparison = comparison_table(test_predictions)
    persistence_row = comparison.loc[
        comparison["model"] == "persistence_current_glucose"
    ].iloc[0]
    model_row = comparison.loc[
        comparison["model"] == "validation_shrunk_multitask_ehr"
    ].iloc[0]
    metrics["mase_vs_persistence"] = float(
        model_row["patient_macro_mae_mg_dl"]
        / persistence_row["patient_macro_mae_mg_dl"]
    )
    subgroups = subgroup_table(test_predictions)
    bootstrap = patient_cluster_bootstrap(
        test_predictions,
        "predicted_glucose_mg_dl",
        seed=config.seed,
        replicates=config.bootstrap_replicates,
    )
    print("\nHELD-OUT TEST METRICS")
    print(json.dumps(_safe_json(metrics), indent=2))
    _print_frame("PERSISTENCE BASELINE COMPARISON", comparison)
    if not subgroups.empty:
        _print_frame("DESCRIPTIVE SUBGROUP CHECK", subgroups)
    print("\nPatient-cluster bootstrap 95% intervals:")
    print(json.dumps(_safe_json(bootstrap), indent=2))

    model_path = config.output_dir / "model.json"
    model.save(model_path)
    reloaded = ProbabilisticGlucoseModel.load(model_path)
    reloaded_predictions = reloaded.predict(test.reset_index(drop=True))
    serialization_difference = float(
        np.max(
            np.abs(
                reloaded_predictions["predicted_glucose_mg_dl"].to_numpy()
                - test_predictions["predicted_glucose_mg_dl"].to_numpy()
            )
        )
    )

    case = test_predictions.sort_values(["patient_id", "anchor_time"]).iloc[0]
    request = GlucoseForecastRequest(
        patient_id=str(case["patient_id"]),
        anchor_time=pd.Timestamp(case["anchor_time"]).isoformat(),
        features={
            feature: None if pd.isna(case[feature]) else float(case[feature])
            for feature in EHR_FEATURES
        },
    )
    response = forecast_one(reloaded, request, horizon_hours=config.forecast_horizon_hours)
    case_study = {
        "request": {
            "patient_id": request.patient_id,
            "anchor_time": request.anchor_time,
        },
        "response": response.as_dict(),
        "actual_future_glucose_mg_dl": float(case["target_glucose_mg_dl"]),
        "actual_target_time": pd.Timestamp(case["target_charttime"]).isoformat(),
        "absolute_error_mg_dl": abs(
            response.predicted_glucose_mg_dl - float(case["target_glucose_mg_dl"])
        ),
        "interpretation": _case_explanation(reloaded, case),
    }

    first_validation_loss = float(history.iloc[0]["validation_total_loss"])
    model_beats_persistence = bool(
        model_row["patient_macro_mae_mg_dl"]
        < persistence_row["patient_macro_mae_mg_dl"]
    )
    acceptance = {
        "patient_disjoint_split": audit["patient_disjoint"],
        "feature_cutoff_flags_valid": bool(dataset["feature_cutoff_verified"].eq(1).all()),
        "future_target_flags_valid": bool(dataset["target_after_anchor_verified"].eq(1).all()),
        "training_selected_nonzero_epoch": model.best_epoch > 0,
        "validation_loss_improved": model.best_validation_loss < first_validation_loss,
        "finite_test_predictions": bool(
            np.isfinite(test_predictions["predicted_glucose_mg_dl"]).all()
        ),
        "ordered_prediction_intervals": bool(
            (
                test_predictions["prediction_lower_mg_dl"]
                <= test_predictions["predicted_glucose_mg_dl"]
            ).all()
            and (
                test_predictions["predicted_glucose_mg_dl"]
                <= test_predictions["prediction_upper_mg_dl"]
            ).all()
        ),
        "probabilities_in_unit_interval": bool(test_predictions["theta_ehr"].between(0, 1).all()),
        "serialization_max_absolute_difference": serialization_difference,
        "serialization_round_trip": serialization_difference < 1e-10,
        "case_study_not_abstained": not response.abstained,
        "clinical_performance_gate_passed": model_beats_persistence,
        "regression_blend_weight": model.regression_blend_weight,
        "release_recommendation": (
            "continue_research_validation"
            if model_beats_persistence
            else "research_only_do_not_deploy"
        ),
    }
    gate_keys = [
        key
        for key in acceptance
        if key
        not in {
            "serialization_max_absolute_difference",
            "regression_blend_weight",
            "clinical_performance_gate_passed",
            "release_recommendation",
        }
    ]
    if not all(acceptance[key] is True for key in gate_keys):
        raise RuntimeError(f"Production acceptance checks failed: {acceptance}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [train_predictions, validation_predictions, test_predictions], ignore_index=True
    ).to_csv(config.output_dir / "predictions.csv", index=False)
    history.to_csv(config.output_dir / "training_losses.csv", index=False)
    coefficients.to_csv(config.output_dir / "coefficients.csv", index=False)
    comparison.to_csv(config.output_dir / "baseline_comparison.csv", index=False)
    subgroups.to_csv(config.output_dir / "subgroup_check.csv", index=False)
    build_audit.to_csv(config.output_dir / "cohort_build_audit.csv", index=False)
    _write_json(config.output_dir / "data_manifest.json", _data_manifest(config))
    _write_json(config.output_dir / "split_audit.json", audit)
    _write_json(config.output_dir / "test_metrics.json", metrics)
    _write_json(config.output_dir / "bootstrap_intervals.json", bootstrap)
    _write_json(config.output_dir / "case_study.json", case_study)
    _write_json(config.output_dir / "acceptance_checks.json", acceptance)

    print("\nHELD-OUT PATIENT CASE STUDY")
    print(json.dumps(_safe_json(case_study), indent=2))
    print("\nENGINEERING ACCEPTANCE AND CLINICAL RELEASE CHECKS")
    print(json.dumps(_safe_json(acceptance), indent=2))
    print(f"\nModel artifact: {model_path}")
    print(f"Study artifacts: {config.output_dir}")
    return {
        "metrics": metrics,
        "comparison": comparison.to_dict(orient="records"),
        "bootstrap": bootstrap,
        "case_study": case_study,
        "acceptance_checks": acceptance,
    }
