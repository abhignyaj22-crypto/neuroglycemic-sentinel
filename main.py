import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.cogwear_study.config import load_config
from src.cogwear_study.data import (
    build_session_index,
    discover_incomplete_sessions,
    inspect_first_session,
    raw_file_sizes,
)
from src.cogwear_study.features import (
    EEG_FEATURES,
    WEARABLE_FEATURES,
    build_paired_feature_table,
    build_real_missing_modality_cases,
)
from src.cogwear_study.fusion import fit_late_fusion, missing_modality_scenarios
from src.cogwear_study.health_agent import build_explanation_payload, deterministic_health_agent
from src.cogwear_study.model import (
    classification_metrics,
    fit_logistic_head,
    patient_session_predictions,
)
from src.cogwear_study.split import attach_split, split_patients


TARGET = "target_cognitive_load"


def print_frame(name: str, frame: pd.DataFrame) -> None:
    print(f"\n{name}")
    print(f"shape: {frame.shape}")
    print(f"columns ({len(frame.columns)}): {frame.columns.tolist()}")
    print("head(5):")
    print(frame.head(5).to_string(index=False))


def print_raw_inspection(sessions: pd.DataFrame) -> None:
    first = sessions.iloc[0]
    print(
        f"\nRAW DATA CHECK: {first['patient_id']} / {first['condition']} "
        "(the complete cohort is used later)"
    )
    print(raw_file_sizes(first).to_string(index=False))
    for stream_name, head in inspect_first_session(first).items():
        print_frame(stream_name, head)

    print(
        "\nWearable variables considered for this study: "
        "BVP - A PPG-derived pulse rate and pulse-interval variability, EDA - Electrodermal Activity, and skin temperature."
    )
    print(
        "TBD: step count, blood pressure, SpO2, glucose, diagnoses, medications, and EHR fields."
    )

def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _save_json(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(values), indent=2, allow_nan=False), encoding="utf-8"
    )


def _metrics_for_column(frame: pd.DataFrame, probability_column: str) -> dict[str, dict[str, float]]:
    window = classification_metrics(frame[TARGET].to_numpy(), frame[probability_column].to_numpy())
    session = patient_session_predictions(frame, probability_column)
    session_metrics = classification_metrics(
        session[TARGET].to_numpy(), session["probability"].to_numpy()
    )
    return {"window_level": window, "patient_session_level": session_metrics}


def _ablation_table(test: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, column in (
        ("EEG only (alpha)", "alpha_eeg"),
        ("wearable only (beta)", "beta_wearable"),
        ("learned late fusion", "combined_probability"),
    ):
        metrics = _metrics_for_column(test, column)["patient_session_level"]
        rows.append({"model_or_scenario": name, **metrics})

    for scenario, group in missing.groupby("scenario", sort=False):
        available = group.loc[np.isfinite(group["combined_probability"])].copy()
        if available.empty:
            metrics = {
                "auroc": float("nan"),
                "log_loss": float("nan"),
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
            }
        else:
            session = patient_session_predictions(available, "combined_probability")
            metrics = classification_metrics(session[TARGET], session["probability"])
        rows.append(
            {
                "model_or_scenario": f"missingness: {scenario}",
                **metrics,
                "abstention_rate": float(group["combined_probability"].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def main(
    config_path: Path | None = None, *, rebuild_features: bool = False
) -> dict[str, object]:
    pd.set_option("display.max_columns", 50)
    config_path = config_path or PROJECT_ROOT / "config" / "study.json"
    config = load_config(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.processed_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Discover only co-registered sessions, then show their actual schemas and values.
    sessions = build_session_index(config)
    print_frame("SESSION INDEX (one patient-condition row)", sessions)
    print_raw_inspection(sessions)

    # 2. Extract fixed, interpretable 30-second features. No estimator is fit here.
    if rebuild_features or not config.processed_path.exists():
        paired, audit = build_paired_feature_table(sessions, config)
        paired.to_csv(config.processed_path, index=False)
        audit.to_csv(config.output_dir / "raw_session_audit.csv", index=False)
    else:
        print(f"\nLoading previously extracted fixed features: {config.processed_path}")
        paired = pd.read_csv(config.processed_path)
        audit_path = config.output_dir / "raw_session_audit.csv"
        audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()

    print_frame("PAIRED EEG + WEARABLE FEATURE TABLE", paired)
    if not audit.empty:
        print_frame("RAW SESSION AUDIT", audit)

    # 3. Split patient IDs once. Windows from a patient can never cross partitions.
    patient_split = split_patients(
        paired["patient_id"],
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    paired = attach_split(paired, patient_split)
    split_frame = patient_split.as_frame()
    print_frame("PATIENT-LEVEL TRAIN / VALIDATION / TEST SPLIT", split_frame)
    print("\nRows by split and label:")
    print(pd.crosstab(paired["split"], paired[TARGET]).to_string())
    print(
        "\nEvaluation warning: the held-out test set contains only "
        f"{len(patient_split.test)} participants. Metrics are pipeline checks, not stable clinical estimates."
    )

    train = paired.loc[paired["split"] == "train"].copy()
    validation = paired.loc[paired["split"] == "validation"].copy()
    test = paired.loc[paired["split"] == "test"].copy()

    # 4. Train the EEG head first: forward pass -> BCE loss -> gradients -> update.
    print(
        f"\nTRAIN EEG HEAD: {len(EEG_FEATURES)} features, learning_rate={config.head_learning_rate}, "
        f"epochs={config.head_epochs}, L2={config.head_l2}"
    )
    eeg_head, eeg_history = fit_logistic_head(
        train,
        validation,
        EEG_FEATURES,
        target_column=TARGET,
        learning_rate=config.head_learning_rate,
        epochs=config.head_epochs,
        l2=config.head_l2,
    )
    print(eeg_history.to_string(index=False))
    print(
        f"EEG checkpoint selected at epoch {eeg_head.best_epoch} "
        f"(validation_loss={eeg_head.best_validation_loss:.6f})."
    )
    print("EEG standardized coefficients (largest magnitude first):")
    print(eeg_head.coefficient_table().to_string(index=False))

    # 5. Add the wearable head only after the EEG baseline is explicit.
    print(
        f"\nTRAIN WEARABLE HEAD: {len(WEARABLE_FEATURES)} features, "
        f"learning_rate={config.head_learning_rate}, epochs={config.head_epochs}, L2={config.head_l2}"
    )
    wearable_head, wearable_history = fit_logistic_head(
        train,
        validation,
        WEARABLE_FEATURES,
        target_column=TARGET,
        learning_rate=config.head_learning_rate,
        epochs=config.head_epochs,
        l2=config.head_l2,
    )
    print(wearable_history.to_string(index=False))
    print(
        f"Wearable checkpoint selected at epoch {wearable_head.best_epoch} "
        f"(validation_loss={wearable_head.best_validation_loss:.6f})."
    )
    print("Wearable standardized coefficients (largest magnitude first):")
    print(wearable_head.coefficient_table().to_string(index=False))

    for partition in (train, validation, test):
        partition["alpha_eeg"] = eeg_head.predict_proba(partition)
        partition["beta_wearable"] = wearable_head.predict_proba(partition)

    # 6. Learn the weighted average on validation patients, never on test patients.
    print(
        f"\nTRAIN LATE FUSION: learning_rate={config.fusion_learning_rate}, "
        f"epochs={config.fusion_epochs}"
    )
    eligible_modalities = np.array(
        [eeg_head.best_epoch > 0, wearable_head.best_epoch > 0], dtype=bool
    )
    if not eligible_modalities.all():
        excluded = [
            name
            for name, eligible in zip(("eeg", "wearable"), eligible_modalities, strict=True)
            if not eligible
        ]
        print(
            "Fusion safeguard: zero-weighting validation-degenerate heads selected at epoch 0: "
            + ", ".join(excluded)
        )
    fusion, fusion_history = fit_late_fusion(
        validation[["alpha_eeg", "beta_wearable"]].to_numpy(),
        validation[TARGET].to_numpy(),
        learning_rate=config.fusion_learning_rate,
        epochs=config.fusion_epochs,
        fallback_probability=float(train[TARGET].mean()),
        eligible_modalities=eligible_modalities,
    )
    print(fusion_history.to_string(index=False))
    print(
        "Learned weights: "
        + ", ".join(
            f"{name}={weight:.4f}"
            for name, weight in zip(fusion.modality_names, fusion.weights, strict=True)
        )
    )

    test["combined_probability"] = fusion.predict(
        test[["alpha_eeg", "beta_wearable"]].to_numpy()
    )
    print_frame(
        "HELD-OUT TEST PREDICTIONS",
        test[
            [
                "patient_id",
                "condition",
                "window_index",
                TARGET,
                "alpha_eeg",
                "beta_wearable",
                "combined_probability",
            ]
        ],
    )

    # 7. Evaluate both single modalities and explicit missing-modality scenarios.
    missing = missing_modality_scenarios(test, fusion)
    ablation = _ablation_table(test, missing)
    print_frame("PATIENT-SESSION TEST ABLATION", ablation)

    # Exercise genuine source-data missingness without letting the incomplete
    # participant influence fitting, validation, or headline test metrics.
    incomplete_sessions = discover_incomplete_sessions(config)
    real_missing = pd.DataFrame()
    real_missing_audit = pd.DataFrame()
    if not incomplete_sessions.empty:
        print_frame("DISCOVERED REAL INCOMPLETE SESSIONS (INFERENCE ONLY)", incomplete_sessions)
        real_missing, real_missing_audit = build_real_missing_modality_cases(
            incomplete_sessions, config
        )
        if not real_missing.empty:
            real_missing["alpha_eeg"] = np.nan
            real_missing["beta_wearable"] = np.nan
            eeg_rows = real_missing["eeg_available"].eq(1)
            wearable_rows = real_missing["wearable_available"].eq(1)
            if eeg_rows.any():
                real_missing.loc[eeg_rows, "alpha_eeg"] = eeg_head.predict_proba(
                    real_missing.loc[eeg_rows]
                )
            if wearable_rows.any():
                real_missing.loc[wearable_rows, "beta_wearable"] = wearable_head.predict_proba(
                    real_missing.loc[wearable_rows]
                )
            real_missing["combined_probability"] = fusion.predict(
                real_missing[["alpha_eeg", "beta_wearable"]].to_numpy(float),
                real_missing[["eeg_available", "wearable_available"]].to_numpy(bool),
            )
            print_frame(
                "REAL MISSING-MODALITY CASE PREDICTIONS (NOT A PERFORMANCE ESTIMATE)",
                real_missing[
                    [
                        "patient_id",
                        "condition",
                        "window_index",
                        TARGET,
                        "eeg_available",
                        "wearable_available",
                        "alpha_eeg",
                        "beta_wearable",
                        "combined_probability",
                    ]
                ],
            )
        if not real_missing_audit.empty:
            print_frame("REAL MISSING-MODALITY EXTRACTION AUDIT", real_missing_audit)

    metrics = {
        "eeg": _metrics_for_column(test, "alpha_eeg"),
        "wearable": _metrics_for_column(test, "beta_wearable"),
        "late_fusion": _metrics_for_column(test, "combined_probability"),
        "fusion_weights": {
            name: float(weight)
            for name, weight in zip(fusion.modality_names, fusion.weights, strict=True)
        },
        "selected_checkpoints": {
            "eeg": {
                "epoch": eeg_head.best_epoch,
                "validation_loss": eeg_head.best_validation_loss,
            },
            "wearable": {
                "epoch": wearable_head.best_epoch,
                "validation_loss": wearable_head.best_validation_loss,
            },
        },
        "split_patients": {
            "train": list(patient_split.train),
            "validation": list(patient_split.validation),
            "test": list(patient_split.test),
        },
        "scope_warning": "CogWear predicts recorded cognitive-load condition, not clinical disease.",
    }

    # 8. Produce one grounded HealthAgent explanation after numerical inference.
    example = test.iloc[0]
    explanation_payload = build_explanation_payload(
        example, eeg_head, wearable_head, fusion
    )
    print("\nHEALTHAGENT EXAMPLE (deterministic, grounded wording)")
    print(deterministic_health_agent(explanation_payload))
    print(json.dumps(explanation_payload, indent=2))

    # 9. Save only derived artifacts and exact split membership for reproducibility.
    split_frame.to_csv(config.output_dir / "patient_split.csv", index=False)
    eeg_history.to_csv(config.output_dir / "eeg_training_curve.csv", index=False)
    wearable_history.to_csv(config.output_dir / "wearable_training_curve.csv", index=False)
    fusion_history.to_csv(config.output_dir / "fusion_training_curve.csv", index=False)
    eeg_head.coefficient_table().to_csv(config.output_dir / "eeg_coefficients.csv", index=False)
    wearable_head.coefficient_table().to_csv(
        config.output_dir / "wearable_coefficients.csv", index=False
    )
    test.to_csv(config.output_dir / "test_window_predictions.csv", index=False)
    missing.to_csv(config.output_dir / "missing_modality_predictions.csv", index=False)
    if not real_missing.empty:
        real_missing.to_csv(
            config.output_dir / "real_missing_modality_predictions.csv", index=False
        )
    if not real_missing_audit.empty:
        real_missing_audit.to_csv(
            config.output_dir / "real_missing_modality_audit.csv", index=False
        )
    ablation.to_csv(config.output_dir / "ablation.csv", index=False)
    _save_json(config.output_dir / "metrics.json", metrics)
    _save_json(config.output_dir / "example_explanation.json", explanation_payload)

    model_dir = config.output_dir / "models"
    eeg_path = model_dir / "eeg_head.json"
    wearable_path = model_dir / "wearable_head.json"
    fusion_path = model_dir / "cognitive_load_fusion.json"
    eeg_head.save(eeg_path)
    wearable_head.save(wearable_path)
    fusion.save(fusion_path)
    reloaded_eeg = type(eeg_head).load(eeg_path)
    reloaded_wearable = type(wearable_head).load(wearable_path)
    reloaded_fusion = type(fusion).load(fusion_path)
    reload_difference = max(
        float(np.max(np.abs(reloaded_eeg.predict_proba(test) - test["alpha_eeg"]))),
        float(
            np.max(
                np.abs(reloaded_wearable.predict_proba(test) - test["beta_wearable"])
            )
        ),
        float(
            np.max(
                np.abs(
                    reloaded_fusion.predict(
                        test[["alpha_eeg", "beta_wearable"]].to_numpy(float)
                    )
                    - test["combined_probability"]
                )
            )
        ),
    )
    acceptance = {
        "patient_disjoint_split": not (
            set(patient_split.train) & set(patient_split.validation)
            or set(patient_split.train) & set(patient_split.test)
            or set(patient_split.validation) & set(patient_split.test)
        ),
        "eeg_head_learned_nonzero_epoch": eeg_head.best_epoch > 0,
        "wearable_head_learned_nonzero_epoch": wearable_head.best_epoch > 0,
        "degenerate_head_zero_weighted": bool(
            eeg_head.best_epoch > 0 or np.isclose(fusion.weights[0], 0.0)
        ),
        "all_modalities_missing_abstains": bool(
            missing.loc[missing["scenario"] == "both_missing", "combined_probability"]
            .isna()
            .all()
        ),
        "serialization_max_absolute_difference": reload_difference,
        "serialization_round_trip": reload_difference < 1e-10,
        "clinical_release_ready": False,
        "release_recommendation": "research_only_do_not_deploy",
    }
    _save_json(config.output_dir / "acceptance_checks.json", acceptance)
    print("\nCOGWEAR ENGINEERING AND RESEARCH GATES")
    print(json.dumps(acceptance, indent=2))
    print(f"\nSaved derived study artifacts to: {config.output_dir}")
    return {"metrics": metrics, "acceptance_checks": acceptance}


def _neural_output_dir(value: Path | None) -> Path:
    return value or PROJECT_ROOT / "outputs" / "neural_glucose"


def _load_neural_case_request(path: Path):
    from src.neuroglycemic.service import NeuralGlucoseForecastRequest

    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Neural case request JSON must contain an object.")
    required = {
        "patient_id",
        "anchor_time",
        "horizon_minutes",
        "feature_schema_version",
        "features",
        "availability",
        "quality",
        "staleness_minutes",
    }
    missing = required - set(values)
    if missing:
        raise ValueError(f"Neural case request is missing fields: {sorted(missing)}")
    unknown = set(values) - required
    if unknown:
        raise ValueError(f"Neural case request has unknown fields: {sorted(unknown)}")
    return NeuralGlucoseForecastRequest(**values)


def _load_checkpoint_payload(path: Path) -> dict[str, object]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Neural checkpoint payload must be a dictionary.")
    return payload


def _neural_model_from_spec(spec: dict[str, object]):
    from src.neuroglycemic.neural_model import NeuroGlycemicNet

    required = {
        "input_dims",
        "horizons_minutes",
        "hidden_dim",
        "embedding_dim",
        "dropout",
        "min_scale",
    }
    missing = required - set(spec)
    if missing:
        raise ValueError(f"Checkpoint model_spec is missing: {sorted(missing)}")
    return NeuroGlycemicNet(
        {str(name): int(value) for name, value in dict(spec["input_dims"]).items()},
        horizons_minutes=tuple(int(value) for value in spec["horizons_minutes"]),
        hidden_dim=int(spec["hidden_dim"]),
        embedding_dim=int(spec["embedding_dim"]),
        dropout=float(spec["dropout"]),
        min_scale=float(spec["min_scale"]),
    )


def run_neural_train(
    *,
    data_path: Path,
    config_path: Path,
    checkpoint_path: Path | None,
    output_dir: Path,
    batch_size: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, object]:
    """Fit the neural model on one real, pre-aligned patient-level table."""

    import torch

    from src.neuroglycemic.neural_dataset import (
        TrainOnlyFeatureStandardizer,
        data_sha256,
        glucose_forecast_metrics,
        load_aligned_window_frame,
        make_neural_batches,
        modality_ablation_predictions,
        modality_ablation_table,
        patient_grouped_split,
        predict_neural_batches,
        target_column,
    )
    from src.neuroglycemic.neural_model import NeuroGlycemicNet
    from src.neuroglycemic.training import (
        GlucoseTargetStandardizer,
        load_neural_training_config,
        make_neuroglycemic_loss_step,
        train_with_early_stopping,
    )
    from src.neuroglycemic.service import build_neural_checkpoint_metadata

    config = load_neural_training_config(config_path)
    frame, feature_names = load_aligned_window_frame(
        data_path,
        config.forecast_horizons_minutes,
        modalities=config.active_modalities,
    )
    print_frame("ALIGNED SAME-PATIENT NEURAL GLUCOSE WINDOWS", frame)
    print(
        "\nAlignment contract: each available EEG, wearable, and EHR record has "
        "matching patient_id, cohort_id, and anchor_time provenance. No cross-cohort join is performed."
    )
    split_frame, split = patient_grouped_split(
        frame,
        seed=config.seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    split_audit = (
        split_frame.groupby("split", sort=False)
        .agg(rows=("patient_id", "size"), patients=("patient_id", "nunique"))
        .reset_index()
    )
    print_frame("PATIENT-GROUPED NEURAL TRAIN / VALIDATION / TEST SPLIT", split_audit)
    print("\nPatient assignments:")
    print(split.as_frame().to_string(index=False))

    train = split_frame.loc[split_frame["split"] == "train"].copy()
    validation = split_frame.loc[split_frame["split"] == "validation"].copy()
    test = split_frame.loc[split_frame["split"] == "test"].copy()
    feature_standardizer = TrainOnlyFeatureStandardizer.fit(train, feature_names)
    target_values = torch.tensor(
        train[
            [target_column(value) for value in config.forecast_horizons_minutes]
        ].to_numpy(float).tolist(),
        dtype=torch.float32,
    )
    target_standardizer = GlucoseTargetStandardizer.fit(
        target_values, config.forecast_horizons_minutes
    )
    print("\nTRAIN-ONLY FEATURE STANDARDIZATION")
    print(json.dumps(feature_standardizer.as_dict(), indent=2))
    print("\nTRAIN-ONLY TARGET STANDARDIZATION")
    print(json.dumps(target_standardizer.as_dict(), indent=2))

    train_batches = make_neural_batches(
        train,
        feature_standardizer,
        config.forecast_horizons_minutes,
        batch_size=batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_batches = make_neural_batches(
        validation,
        feature_standardizer,
        config.forecast_horizons_minutes,
        batch_size=batch_size,
    )
    test_batches = make_neural_batches(
        test,
        feature_standardizer,
        config.forecast_horizons_minutes,
        batch_size=batch_size,
    )
    hidden_dim = int(config.model["hidden_dim"])
    embedding_dim = int(config.model["embedding_dim"])
    dropout = float(config.model["dropout"])
    # Targets are standardized, so the configured 0.05 default is a small
    # numerical floor rather than an irreducible one-standard-deviation floor.
    min_scale = float(config.model["min_scale"])
    torch.manual_seed(config.seed)
    model = NeuroGlycemicNet(
        feature_standardizer.input_dims,
        horizons_minutes=config.forecast_horizons_minutes,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
        min_scale=min_scale,
    )
    loss_step = make_neuroglycemic_loss_step(
        config.expert_loss_weight, target_standardizer
    )
    destination = checkpoint_path or config.checkpoint_path
    source_digest = data_sha256(data_path)
    serving_metadata = build_neural_checkpoint_metadata(
        model,
        feature_names={
            name: list(values)
            for name, values in feature_standardizer.feature_names.items()
        },
        feature_means={
            name: list(values) for name, values in feature_standardizer.means.items()
        },
        feature_scales={
            name: list(values) for name, values in feature_standardizer.scales.items()
        },
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
        min_scale=min_scale,
    )
    # The serving helper establishes the shared schema version and ordered
    # feature contract.  Valid counts additionally let the research path audit
    # exactly how many training observations supported every statistic.
    serving_feature_schema = serving_metadata["feature_schema"]
    serving_feature_schema.update(
        {
            "fit_split": feature_standardizer.fit_split,
            "ordered_feature_names": {
                name: list(values)
                for name, values in feature_standardizer.feature_names.items()
            },
            "feature_names": {
                name: list(values)
                for name, values in feature_standardizer.feature_names.items()
            },
            "means": {
                name: list(values)
                for name, values in feature_standardizer.means.items()
            },
            "scales": {
                name: list(values)
                for name, values in feature_standardizer.scales.items()
            },
            "valid_counts": {
                name: list(values)
                for name, values in feature_standardizer.valid_counts.items()
            },
        }
    )
    checkpoint_metadata = {
        **serving_metadata,
        "patient_split": {
            "train": list(split.train),
            "validation": list(split.validation),
            "test": list(split.test),
        },
        "data_sha256": source_digest,
        "data_file_name": data_path.name,
        "alignment_contract": "same_patient_same_cohort_same_anchor",
        "active_modalities": list(config.active_modalities),
        "fusion_calibrated": len(config.active_modalities) > 1,
    }
    print(
        "\nTRAIN NEURAL MIXTURE-OF-EXPERTS: "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}, "
        f"learning_rate={config.learning_rate:g}, epochs={config.epochs}, "
        f"batch_size={batch_size}, horizons={config.forecast_horizons_minutes}"
    )
    result = train_with_early_stopping(
        model,
        train_batches,
        validation_batches,
        loss_step,
        config,
        target_standardizer=target_standardizer,
        checkpoint_path=destination,
        checkpoint_metadata=checkpoint_metadata,
    )
    history = pd.DataFrame(result.history)
    print_frame("NEURAL TRAINING AND VALIDATION LOSSES", history)
    print(history.to_string(index=False))
    print(
        f"\nSelected neural checkpoint: epoch={result.best_epoch}, "
        f"validation_loss={result.best_validation_loss:.6f}, path={result.checkpoint_path}"
    )
    predictions = predict_neural_batches(
        model,
        test_batches,
        target_standardizer,
        config.forecast_horizons_minutes,
        hypoglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl[
            "hypoglycemia"
        ],
        hyperglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl[
            "hyperglycemia"
        ],
    )
    metrics = glucose_forecast_metrics(predictions)
    ablation_scenarios = modality_ablation_predictions(
        model,
        test_batches,
        target_standardizer,
        config.forecast_horizons_minutes,
        hypoglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl[
            "hypoglycemia"
        ],
        hyperglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl[
            "hyperglycemia"
        ],
    )
    ablation = modality_ablation_table(ablation_scenarios)
    print_frame("HELD-OUT PATIENT NEURAL GLUCOSE PREDICTIONS", predictions)
    print("\nHELD-OUT NEURAL GLUCOSE METRICS")
    print(json.dumps(_json_safe(metrics), indent=2, allow_nan=False))
    print_frame("PAIRED HELD-OUT MISSING-MODALITY ABLATIONS", ablation)

    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "training_losses.csv", index=False)
    split.as_frame().to_csv(output_dir / "patient_split.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    ablation.to_csv(output_dir / "missing_modality_ablation.csv", index=False)
    _save_json(output_dir / "test_metrics.json", metrics)
    _save_json(output_dir / "feature_schema.json", serving_feature_schema)
    print(f"\nSaved neural study artifacts to: {output_dir}")
    return {"metrics": metrics, "checkpoint": str(result.checkpoint_path)}


def run_neural_evaluate(
    *,
    data_path: Path,
    config_path: Path,
    checkpoint_path: Path | None,
    output_dir: Path,
    batch_size: int,
) -> dict[str, object]:
    """Reproduce held-out evaluation using checkpoint-recorded splits and scalers."""

    from src.neuroglycemic.neural_dataset import (
        TrainOnlyFeatureStandardizer,
        attach_recorded_split,
        data_sha256,
        glucose_forecast_metrics,
        load_aligned_window_frame,
        make_neural_batches,
        modality_ablation_predictions,
        modality_ablation_table,
        predict_neural_batches,
    )
    from src.neuroglycemic.training import (
        GlucoseTargetStandardizer,
        load_neural_checkpoint,
        load_neural_training_config,
    )

    config = load_neural_training_config(config_path)
    destination = checkpoint_path or config.checkpoint_path
    payload = _load_checkpoint_payload(destination)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint is missing neural dataset metadata.")
    if data_sha256(data_path) != metadata.get("data_sha256"):
        raise ValueError("Evaluation data SHA-256 does not match the training dataset.")
    model_spec = metadata.get("model_spec")
    feature_schema = metadata.get("feature_schema")
    patient_split = metadata.get("patient_split")
    if not isinstance(model_spec, dict) or not isinstance(feature_schema, dict) or not isinstance(patient_split, dict):
        raise ValueError("Checkpoint is missing model, feature, or patient-split provenance.")
    model = _neural_model_from_spec(model_spec)
    load_neural_checkpoint(
        destination,
        model,
        expected_prediction_target=config.prediction_target,
        expected_horizons_minutes=config.forecast_horizons_minutes,
    )
    target_standardizer = GlucoseTargetStandardizer.from_dict(
        payload["target_standardizer"]
    )
    feature_standardizer = TrainOnlyFeatureStandardizer.from_dict(feature_schema)
    frame, discovered = load_aligned_window_frame(
        data_path,
        config.forecast_horizons_minutes,
        modalities=model.modalities,
    )
    if tuple(config.active_modalities) != tuple(model.modalities):
        raise ValueError(
            "Evaluation active_modalities differ from the trained checkpoint."
        )
    if {name: tuple(values) for name, values in discovered.items()} != dict(
        feature_standardizer.feature_names
    ):
        raise ValueError("Evaluation feature order/schema differs from the training checkpoint.")
    frame = attach_recorded_split(frame, patient_split)
    test = frame.loc[frame["split"] == "test"].copy()
    print_frame("CHECKPOINT-MATCHED ALIGNED EVALUATION WINDOWS", test)
    batches = make_neural_batches(
        test,
        feature_standardizer,
        config.forecast_horizons_minutes,
        batch_size=batch_size,
    )
    predictions = predict_neural_batches(
        model,
        batches,
        target_standardizer,
        config.forecast_horizons_minutes,
        hypoglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl[
            "hypoglycemia"
        ],
        hyperglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl[
            "hyperglycemia"
        ],
    )
    metrics = glucose_forecast_metrics(predictions)
    ablation_scenarios = modality_ablation_predictions(
        model,
        batches,
        target_standardizer,
        config.forecast_horizons_minutes,
        hypoglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl[
            "hypoglycemia"
        ],
        hyperglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl[
            "hyperglycemia"
        ],
    )
    ablation = modality_ablation_table(ablation_scenarios)
    print_frame("RELOADED HELD-OUT NEURAL PREDICTIONS", predictions)
    print("\nRELOADED CHECKPOINT METRICS")
    print(json.dumps(_json_safe(metrics), indent=2, allow_nan=False))
    print_frame("RELOADED PAIRED MISSING-MODALITY ABLATIONS", ablation)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "reloaded_test_predictions.csv", index=False)
    ablation.to_csv(
        output_dir / "reloaded_missing_modality_ablation.csv", index=False
    )
    _save_json(output_dir / "reloaded_test_metrics.json", metrics)
    return {"metrics": metrics, "checkpoint": str(destination)}


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        nargs="?",
        choices=(
            "eeg-wearable",
            "ehr-glucose",
            "architecture",
            "lsl-audit",
            "prepare-neural-data",
            "validate-neural-data",
            "train-neural",
            "evaluate-neural",
            "neural-case",
            "run-neural-e2e",
        ),
        default="eeg-wearable",
        help="Study to execute. The default preserves the original EEG/wearable run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config. A study-specific default is used when omitted.",
    )
    parser.add_argument(
        "--rebuild",
        "--rebuild-features",
        dest="rebuild",
        action="store_true",
        help="Rebuild the processed cohort from the real raw files.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Call the optional LangChain LLM inside HealthAgent after numerical inference.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("HEALTHAGENT_LLM_MODEL"),
        help="LLM model name. Required with --use-llm (or set HEALTHAGENT_LLM_MODEL).",
    )
    parser.add_argument(
        "--print-llm-raw",
        action="store_true",
        help="Include the raw LLM JSON response in HealthAgent telemetry.",
    )
    parser.add_argument(
        "--xdf",
        type=Path,
        default=None,
        help="LabRecorder XDF file to inspect with the lsl-audit command.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Pre-aligned same-patient CSV/Parquet required by neural commands.",
    )
    parser.add_argument(
        "--source",
        choices=("physiocgm",),
        default="physiocgm",
        help="Real dataset adapter used by neural data preparation.",
    )
    parser.add_argument(
        "--input-dir",
        "--raw-dir",
        dest="input_dir",
        type=Path,
        default=None,
        help="PhysioCGM processed root containing subject/*.pkl clips.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Prepared aligned CSV/Parquet destination.",
    )
    parser.add_argument(
        "--source-timezone",
        default="UTC",
        help="Timezone assigned only to naive source timestamps before UTC conversion.",
    )
    parser.add_argument(
        "--horizon-tolerance-minutes", type=float, default=5.0
    )
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="Allow executable pickle input after verifying official PhysioCGM provenance.",
    )
    parser.add_argument(
        "--request",
        type=Path,
        default=None,
        help="JSON request for the checkpoint-backed neural-case command.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional neural checkpoint path; otherwise the neural config is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional neural artifact directory.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    arguments = parser.parse_args()
    if arguments.study == "eeg-wearable":
        config_path = arguments.config or PROJECT_ROOT / "config" / "study.json"
        main(config_path, rebuild_features=arguments.rebuild)
        return


    if arguments.study == "lsl-audit":
        from src.neuroglycemic.lsl import audit_xdf, discover_streams

        if arguments.xdf is None:
            print_frame("DISCOVERED LSL STREAMS", discover_streams())
        else:
            audit, _ = audit_xdf(arguments.xdf)
            print_frame("LABRECORDER XDF STREAM AUDIT", audit)
        return

    if arguments.study in {
        "prepare-neural-data",
        "validate-neural-data",
        "run-neural-e2e",
    }:
        from src.neuroglycemic.neural_dataset import load_aligned_window_frame
        from src.neuroglycemic.physiocgm_data import (
            build_physiocgm_aligned_windows,
            write_physiocgm_build,
        )
        from src.neuroglycemic.training import load_neural_training_config

        neural_config_path = (
            arguments.config
            or PROJECT_ROOT / "config" / "neural_glucose_physio.json"
        )
        neural_config = load_neural_training_config(neural_config_path)
        prepared_path = (
            arguments.output
            or arguments.data
            or PROJECT_ROOT / "data" / "processed" / "physiocgm_aligned_windows.parquet"
        )
        if arguments.study in {"prepare-neural-data", "run-neural-e2e"}:
            if arguments.input_dir is None:
                parser.error(f"{arguments.study} requires --input-dir/--raw-dir.")
            if arguments.source != "physiocgm":
                parser.error("Only the physiocgm builder is currently implemented.")
            prepared = build_physiocgm_aligned_windows(
                arguments.input_dir,
                horizons_minutes=neural_config.forecast_horizons_minutes,
                horizon_tolerance_minutes=arguments.horizon_tolerance_minutes,
                source_timezone=arguments.source_timezone,
                trust_pickle=arguments.trust_pickle,
            )
            write_physiocgm_build(
                prepared,
                prepared_path,
                input_dir=arguments.input_dir,
                horizons_minutes=neural_config.forecast_horizons_minutes,
                source_timezone=arguments.source_timezone,
            )
            print_frame("PREPARED PHYSIOCGM CAUSAL WINDOWS", prepared.frame)
            print_frame("PHYSIOCGM ALIGNMENT AUDIT", prepared.audit)
            print(f"\nSaved prepared neural data to: {prepared_path}")
        validated, feature_names = load_aligned_window_frame(
            prepared_path,
            neural_config.forecast_horizons_minutes,
            modalities=neural_config.active_modalities,
            horizon_tolerance_minutes=arguments.horizon_tolerance_minutes,
        )
        print_frame("VALIDATED NEURAL DATASET", validated)
        print("\nACTIVE MODALITY FEATURE CONTRACT")
        print(json.dumps({key: list(value) for key, value in feature_names.items()}, indent=2))
        if arguments.study == "run-neural-e2e":
            neural_outputs = _neural_output_dir(arguments.output_dir)
            trained = run_neural_train(
                data_path=prepared_path,
                config_path=neural_config_path,
                checkpoint_path=arguments.checkpoint,
                output_dir=neural_outputs,
                batch_size=arguments.batch_size,
                train_fraction=arguments.train_fraction,
                validation_fraction=arguments.validation_fraction,
            )
            run_neural_evaluate(
                data_path=prepared_path,
                config_path=neural_config_path,
                checkpoint_path=Path(str(trained["checkpoint"])),
                output_dir=neural_outputs,
                batch_size=arguments.batch_size,
            )
        return

    if arguments.study in {"train-neural", "evaluate-neural"}:
        if arguments.data is None:
            parser.error(f"{arguments.study} requires --data with a real aligned table.")
        neural_config = arguments.config or PROJECT_ROOT / "config" / "neural_glucose.json"
        neural_outputs = _neural_output_dir(arguments.output_dir)
        if arguments.study == "train-neural":
            run_neural_train(
                data_path=arguments.data,
                config_path=neural_config,
                checkpoint_path=arguments.checkpoint,
                output_dir=neural_outputs,
                batch_size=arguments.batch_size,
                train_fraction=arguments.train_fraction,
                validation_fraction=arguments.validation_fraction,
            )
        else:
            run_neural_evaluate(
                data_path=arguments.data,
                config_path=neural_config,
                checkpoint_path=arguments.checkpoint,
                output_dir=neural_outputs,
                batch_size=arguments.batch_size,
            )
        return

    if arguments.study == "neural-case":
        if arguments.request is None:
            parser.error("neural-case requires --request with a JSON request.")
        neural_config_path = (
            arguments.config or PROJECT_ROOT / "config" / "neural_glucose.json"
        )
        from src.neuroglycemic.architecture import run_neural_architecture_case
        from src.neuroglycemic.health_agent import HealthAgent, build_openai_llm
        from src.neuroglycemic.training import load_neural_training_config

        neural_config = load_neural_training_config(neural_config_path)
        checkpoint = arguments.checkpoint or neural_config.checkpoint_path
        llm = None
        if arguments.use_llm:
            if not arguments.llm_model:
                parser.error(
                    "--use-llm requires --llm-model or HEALTHAGENT_LLM_MODEL."
                )
            llm = build_openai_llm(model=arguments.llm_model)
        agent = HealthAgent(
            llm=llm,
            provider="openai" if llm is not None else None,
            model_name=arguments.llm_model,
        )
        case = run_neural_architecture_case(
            PROJECT_ROOT,
            checkpoint_path=checkpoint,
            request=_load_neural_case_request(arguments.request),
            health_agent=agent,
            include_raw_llm_response=arguments.print_llm_raw,
            output_path=_neural_output_dir(arguments.output_dir) / "case_study.json",
        )
        print("\nCHECKPOINT-BACKED NEURAL ARCHITECTURE CASE STUDY")
        print(json.dumps(case, indent=2))
        return

    from src.neuroglycemic.config import load_ehr_config
    from src.neuroglycemic.pipeline import run_ehr_glucose_pipeline

    config_path = arguments.config or PROJECT_ROOT / "config" / "ehr_glucose.json"
    if arguments.study == "ehr-glucose":
        run_ehr_glucose_pipeline(load_ehr_config(config_path), rebuild=arguments.rebuild)
        return

    # Full current architecture run: train/evaluate each scientifically supported
    # task, then prove that unrelated patient/target records cannot be fused.
    main(PROJECT_ROOT / "config" / "study.json", rebuild_features=arguments.rebuild)
    run_ehr_glucose_pipeline(
        load_ehr_config(PROJECT_ROOT / "config" / "ehr_glucose.json"),
        rebuild=arguments.rebuild,
    )
    from src.neuroglycemic.architecture import run_architecture_case
    from src.neuroglycemic.health_agent import HealthAgent, build_openai_llm

    llm = None
    if arguments.use_llm:
        if not arguments.llm_model:
            parser.error("--use-llm requires --llm-model or HEALTHAGENT_LLM_MODEL.")
        llm = build_openai_llm(model=arguments.llm_model)
    agent = HealthAgent(
        llm=llm,
        provider="openai" if llm is not None else None,
        model_name=arguments.llm_model,
    )
    architecture = run_architecture_case(
        PROJECT_ROOT,
        health_agent=agent,
        include_raw_llm_response=arguments.print_llm_raw,
    )
    print("\nEND-TO-END ARCHITECTURE CASE STUDY")
    print(json.dumps(architecture, indent=2))


if __name__ == "__main__":
    cli()
