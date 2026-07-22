import argparse
import difflib
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


def _require_external_runtime_path(
    path: Path,
    *,
    label: str,
    parser: argparse.ArgumentParser,
) -> Path:
    """Reject protected data and generated artifacts placed below the code tree."""

    resolved = path.expanduser().resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        parser.error(f"{label} must be outside the software repository: {resolved}")
    return resolved


def _missing_file_message(path: Path, *, label: str) -> str:
    """Return an actionable missing-file error, including a close sibling match.

    The CLI previously reported only the invalid path.  A one-character typo such
    as ``mimiv`` instead of ``mimiciv`` therefore looked like a model failure even
    though training artifacts were valid.  Suggestions are restricted to the
    requested parent directory; the CLI never searches protected data globally.
    """

    message = f"{label} does not exist or is not a file: {path}"
    parent = path.parent
    if not parent.is_dir():
        return message
    candidates = sorted(value.name for value in parent.iterdir() if value.is_file())
    matches = difflib.get_close_matches(path.name, candidates, n=1, cutoff=0.65)
    if matches:
        message += f"\nDid you mean: {parent / matches[0]}"
    return message


def _atomic_write_csv(frame: pd.DataFrame, destination: Path, **kwargs: object) -> None:
    """Write a table atomically so interrupted cohort builds cannot look complete."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False, **kwargs)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

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

NEURAL_AND_INTEROPERABILITY_COMMANDS = (
    "lsl-audit",
    "lsl-replay",
    "lsl-session-replay",
    "prepare-lsl-glucose",
    "prepare-mimic-neural",
    "prepare-diatrend",
    "prepare-big-ideas",
    "prepare-physiocgm",
    "train-neural",
    "evaluate-neural",
    "neural-case",
)

# These reproduce the original v3 experiments. They intentionally remain
# available for comparison, but they are not the neural CGM product path.
LEGACY_RESEARCH_COMMANDS = (
    "eeg-wearable",
    "ehr-glucose",
    "architecture",
)


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
    optional = {"clock_uncertainty_ms"}
    unknown = set(values) - required - optional
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
        modality_dropout_probability=float(
            spec.get("modality_dropout_probability", 0.0)
        ),
        auxiliary_task_kinds={
            str(name): str(kind)
            for name, kind in dict(spec.get("auxiliary_task_kinds", {})).items()
        },
        cross_modal_layers=int(spec.get("cross_modal_layers", 0)),
        cross_modal_heads=int(spec.get("cross_modal_heads", 4)),
        horizon_film=bool(spec.get("horizon_film", False)),
        response_kernel=spec.get("response_kernel"),
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
    pretrain_epochs: int = 0,
    pretrain_checkpoint: Path | None = None,
    init_from_pretrain: Path | None = None,
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
    from src.neuroglycemic.neural_training import (
        GlucoseTargetStandardizer,
        load_neural_training_config,
        make_neuroglycemic_loss_step,
        train_with_early_stopping,
    )
    from src.neuroglycemic.service import build_neural_checkpoint_metadata

    config = load_neural_training_config(config_path)
    # ``optimization_seed`` (optional, ensemble training) reseeds model
    # initialization and batch shuffling while the patient split stays keyed to
    # the config seed, so ensemble members share one identical split.
    raw_config_values = json.loads(config_path.read_text(encoding="utf-8"))
    optimization_seed = int(raw_config_values.get("optimization_seed", config.seed))
    split_seed = config.seed
    if optimization_seed != config.seed:
        from dataclasses import replace as _replace_config

        config = _replace_config(config, seed=optimization_seed)
    # Optional anchor thinning: decorrelates near-duplicate windows from
    # densely anchored datasets (for example 15-minute Big IDEAS anchors).
    min_anchor_spacing = raw_config_values.get("min_anchor_spacing_minutes")
    if min_anchor_spacing is not None:
        min_anchor_spacing = float(min_anchor_spacing)
    modalities = tuple(config.feature_registry) or None
    frame, feature_names = load_aligned_window_frame(
        data_path,
        config.forecast_horizons_minutes,
        modalities=modalities or ("eeg", "wearable", "ehr"),
        horizon_tolerance_minutes=config.horizon_tolerance_minutes,
        feature_registry=config.feature_registry or None,
        input_cgm=config.input_cgm,
    )
    print_frame("ALIGNED SAME-PATIENT NEURAL GLUCOSE WINDOWS", frame)
    print(
        "\nAlignment contract: every available modality has matching patient_id, "
        "cohort_id, and anchor_time provenance. No cross-cohort join is performed."
    )
    split_frame, split = patient_grouped_split(
        frame,
        seed=split_seed,
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
    availability_columns = [f"{name}_available" for name in feature_names]
    paired_training_rows = int(
        train[availability_columns].astype(bool).sum(axis=1).ge(2).sum()
    )
    if len(feature_names) > 1 and paired_training_rows == 0:
        raise ValueError(
            "Learned multimodal fusion requires at least one training row with two "
            "simultaneously observed modalities. Train modality-specific encoders "
            "separately until a same-patient bridge cohort is available."
        )
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

    # Contribution 4: sign-constrained learned event response kernel.  The
    # kernel consumes the raw (unstandardized) causal meal/event lag basis;
    # patient personalization is keyed by the training participants only.
    kernel_spec = config.model.get("response_kernel")
    event_basis_columns: dict[str, tuple[str, ...]] | None = None
    patient_to_index: dict[str, int] | None = None
    if kernel_spec is not None:
        kernel_spec = dict(kernel_spec)
        centers = tuple(
            f"{float(value):g}"
            for value in kernel_spec["basis_centers_minutes"]
        )
        event_basis_columns = {}
        for channel in kernel_spec["channels"]:
            columns = tuple(f"meal_lag_{channel}_{center}m" for center in centers)
            missing = set(columns) - set(frame.columns)
            if missing:
                raise ValueError(
                    f"response_kernel channel {channel!r} requires aligned event "
                    f"basis columns {sorted(missing)}. Build them with the causal "
                    "meal-context feature builder or disable response_kernel."
                )
            event_basis_columns[str(channel)] = columns
        train_participants = sorted(
            train["participant_key"].astype(str).unique().tolist()
        )
        patient_to_index = {
            key: index for index, key in enumerate(train_participants)
        }
        kernel_spec["patient_count"] = len(patient_to_index)
    train_batches = make_neural_batches(
        train,
        feature_standardizer,
        config.forecast_horizons_minutes,
        batch_size=batch_size,
        auxiliary_tasks=config.auxiliary_tasks,
        shuffle=True,
        seed=config.seed,
        patient_to_index=patient_to_index,
        event_basis_columns=event_basis_columns,
        min_anchor_spacing_minutes=min_anchor_spacing,
    )
    validation_batches = make_neural_batches(
        validation,
        feature_standardizer,
        config.forecast_horizons_minutes,
        batch_size=batch_size,
        auxiliary_tasks=config.auxiliary_tasks,
        patient_to_index=patient_to_index,
        event_basis_columns=event_basis_columns,
        min_anchor_spacing_minutes=min_anchor_spacing,
    )
    test_batches = make_neural_batches(
        test,
        feature_standardizer,
        config.forecast_horizons_minutes,
        batch_size=batch_size,
        auxiliary_tasks=config.auxiliary_tasks,
        patient_to_index=patient_to_index,
        event_basis_columns=event_basis_columns,
        min_anchor_spacing_minutes=min_anchor_spacing,
    )
    hidden_dim = int(config.model["hidden_dim"])
    embedding_dim = int(config.model["embedding_dim"])
    dropout = float(config.model["dropout"])
    # Targets are standardized, so the configured 0.05 default is a small
    # numerical floor rather than an irreducible one-standard-deviation floor.
    min_scale = float(config.model["min_scale"])
    modality_dropout_probability = float(
        config.model.get("modality_dropout_probability", 0.0)
    )
    torch.manual_seed(config.seed)
    model = NeuroGlycemicNet(
        feature_standardizer.input_dims,
        horizons_minutes=config.forecast_horizons_minutes,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
        min_scale=min_scale,
        modality_dropout_probability=modality_dropout_probability,
        auxiliary_task_kinds={
            name: str(specification["kind"])
            for name, specification in config.auxiliary_tasks.items()
        },
        cross_modal_layers=int(config.model.get("cross_modal_layers", 0)),
        cross_modal_heads=int(config.model.get("cross_modal_heads", 4)),
        horizon_film=bool(config.model.get("horizon_film", False)),
        response_kernel=kernel_spec,
        build_reconstruction_heads=pretrain_epochs > 0,
    )
    pretrain_provenance: dict[str, object] = {}
    if init_from_pretrain is not None:
        from src.neuroglycemic.pretrain import load_pretrain_weights

        load_pretrain_weights(model, init_from_pretrain)
        pretrain_provenance["initialized_from_pretrain"] = str(init_from_pretrain)
        print(f"Initialized encoders from pretraining checkpoint: {init_from_pretrain}")
    if pretrain_epochs > 0:
        # Contribution 2: self-supervised masked-reconstruction pretraining on
        # the training patients only.  Glucose targets are never read here.
        from src.neuroglycemic.pretrain import (
            pretrain_history_metadata,
            pretrain_masked_reconstruction,
            save_pretrain_checkpoint,
        )

        print(
            f"\nSELF-SUPERVISED PRETRAINING: epochs={pretrain_epochs}, "
            "objective=masked feature reconstruction (labels unused)"
        )
        pretrain_history = pretrain_masked_reconstruction(
            model,
            train_batches,
            epochs=pretrain_epochs,
            learning_rate=config.learning_rate,
            device=config.device,
            seed=config.seed,
        )
        pretrain_provenance.update(pretrain_history_metadata(pretrain_history))
        print(pd.DataFrame(pretrain_history).to_string(index=False))
        if pretrain_checkpoint is not None:
            save_pretrain_checkpoint(
                pretrain_checkpoint,
                model,
                metadata={
                    "feature_names": {
                        name: list(values)
                        for name, values in feature_standardizer.feature_names.items()
                    },
                    "horizons_minutes": list(config.forecast_horizons_minutes),
                },
            )
            print(f"Saved pretraining checkpoint to: {pretrain_checkpoint}")
        # Forecasting checkpoints never carry reconstruction decoders.
        model.drop_reconstruction_heads()
    loss_step = make_neuroglycemic_loss_step(
        config.expert_loss_weight,
        target_standardizer,
        config.auxiliary_tasks,
        crps_loss_weight=config.crps_loss_weight,
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
        modality_dropout_probability=modality_dropout_probability,
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
        "min_anchor_spacing_minutes": min_anchor_spacing,
        **pretrain_provenance,
    }
    if patient_to_index is not None:
        checkpoint_metadata["patient_index_map"] = dict(patient_to_index)
        checkpoint_metadata["event_basis_channels"] = {
            channel: list(columns)
            for channel, columns in (event_basis_columns or {}).items()
        }
    print(
        "\nTRAIN NEURAL MIXTURE-OF-EXPERTS: "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}, "
        f"learning_rate={config.learning_rate:g}, epochs={config.epochs}, "
        f"batch_size={batch_size}, horizons={config.forecast_horizons_minutes}"
    )
    initial_parameters = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
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
    # Contribution 5 (reliability): split-conformal interval calibration on the
    # validation split only, per horizon and per availability pattern.  The
    # corrected levels are merged into the checkpoint metadata so serving and
    # evaluation apply exactly the training-time calibration contract.
    from src.neuroglycemic.calibration import fit_conformal_calibrator

    calibrator = fit_conformal_calibrator(
        model,
        validation_batches,
        target_standardizer,
        config.forecast_horizons_minutes,
    )
    print("\nVALIDATION-ONLY CONFORMAL CALIBRATION")
    print(json.dumps(calibrator.as_dict(), indent=2))
    payload = _load_checkpoint_payload(result.checkpoint_path)
    payload_metadata = dict(payload.get("metadata") or {})
    payload_metadata["conformal_calibration"] = calibrator.as_dict()
    payload["metadata"] = payload_metadata
    temporary = result.checkpoint_path.with_suffix(
        result.checkpoint_path.suffix + ".tmp"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, result.checkpoint_path)
    finally:
        if temporary.exists():
            temporary.unlink()
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
        calibrator=calibrator,
    )
    metrics = glucose_forecast_metrics(predictions)
    parameter_delta_l2 = float(
        torch.sqrt(
            sum(
                torch.sum(
                    (value.detach().cpu() - initial_parameters[name]).square()
                )
                for name, value in model.named_parameters()
                if name in initial_parameters
            )
        ).item()
    )
    comparison_results = [
        value.get("paired_patient_bootstrap_model_minus_persistence")
        for value in metrics["by_horizon"].values()
        if "paired_patient_bootstrap_model_minus_persistence" in value
    ]
    performance_gate_passed = bool(
        comparison_results
        and all(
            int(value["patients"]) >= 5
            and np.isfinite(float(value["upper_95"]))
            and float(value["upper_95"]) < 0.0
            for value in comparison_results
        )
    )
    event_support_gate_passed = bool(
        metrics["by_horizon"]
        and all(
            bool(horizon_metrics[event_name]["minimum_support_met"])
            for horizon_metrics in metrics["by_horizon"].values()
            for event_name in ("hypoglycemia_event", "hyperglycemia_event")
        )
    )
    interval_calibration_descriptive_gate_passed = bool(
        metrics["by_horizon"]
        and all(
            abs(
                float(
                    horizon_metrics["prediction_interval_95_coverage_error"]
                )
            )
            <= 0.05
            for horizon_metrics in metrics["by_horizon"].values()
        )
    )
    learning_gate_passed = bool(
        result.best_epoch >= 1
        and result.best_validation_loss < result.initial_validation_loss
        and parameter_delta_l2 > 0.0
        and history["train_gradient_norm"].gt(0.0).all()
        and len(history) > 1
        and history["train_loss"].iloc[1:].min() < history["train_loss"].iloc[0]
    )
    multimodal_bridge_gate_passed = bool(
        len(feature_names) >= 2 and paired_training_rows > 0
    )
    prospective_validation_gate_passed = False
    release_blockers: list[str] = []
    if not performance_gate_passed:
        release_blockers.append(
            "Patient-macro superiority over persistence was not established at every horizon."
        )
    if not event_support_gate_passed:
        release_blockers.append(
            "Held-out hypo/hyperglycemia event counts are below the minimum support floor."
        )
    if not multimodal_bridge_gate_passed:
        release_blockers.append(
            "No same-patient multimodal bridge cohort was used for learned fusion."
        )
    if not prospective_validation_gate_passed:
        release_blockers.append(
            "Prospective external validation has not been completed."
        )
    acceptance = {
        "trained_checkpoint_epoch_at_least_one": result.best_epoch >= 1,
        "initial_validation_loss": result.initial_validation_loss,
        "best_validation_loss": result.best_validation_loss,
        "validation_improved_over_epoch_zero": (
            result.best_validation_loss < result.initial_validation_loss
        ),
        "paired_multimodal_training_rows": paired_training_rows,
        "parameter_delta_l2": parameter_delta_l2,
        "parameters_changed": parameter_delta_l2 > 0.0,
        "all_reported_gradient_norms_positive": bool(
            history["train_gradient_norm"].gt(0.0).all()
        ),
        "training_loss_decreased": bool(
            len(history) > 1
            and history["train_loss"].iloc[1:].min() < history["train_loss"].iloc[0]
        ),
        "learning_gate_passed": learning_gate_passed,
        "all_horizons_beat_persistence_with_patient_macro_95_ci": (
            performance_gate_passed
        ),
        # Retained for consumers of the first product contract. The estimand is
        # now explicitly patient-macro in the metrics payload.
        "all_horizons_beat_persistence_with_patient_clustered_95_ci": (
            performance_gate_passed
        ),
        "event_support_gate_passed": event_support_gate_passed,
        "interval_calibration_descriptive_gate_passed": (
            interval_calibration_descriptive_gate_passed
        ),
        "multimodal_bridge_gate_passed": multimodal_bridge_gate_passed,
        "prospective_validation_gate_passed": prospective_validation_gate_passed,
        "clinical_release_ready": False,
        "release_blockers": release_blockers,
        "release_recommendation": "research_only_not_for_clinical_use",
    }
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
        calibrator=calibrator,
    )
    ablation = modality_ablation_table(ablation_scenarios)
    print_frame("HELD-OUT PATIENT NEURAL GLUCOSE PREDICTIONS", predictions)
    print("\nHELD-OUT NEURAL GLUCOSE METRICS")
    print(json.dumps(_json_safe(metrics), indent=2, allow_nan=False))
    print_frame("PAIRED HELD-OUT MISSING-MODALITY ABLATIONS", ablation)
    print("\nNEURAL LEARNING AND RELEASE GATES")
    print(json.dumps(_json_safe(acceptance), indent=2, allow_nan=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "training_losses.csv", index=False)
    split.as_frame().to_csv(output_dir / "patient_split.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    ablation.to_csv(output_dir / "missing_modality_ablation.csv", index=False)
    _save_json(output_dir / "test_metrics.json", metrics)
    _save_json(output_dir / "feature_schema.json", serving_feature_schema)
    _save_json(output_dir / "training_acceptance.json", acceptance)
    from src.neuroglycemic.release import write_release_manifest

    release_status = (
        "approved" if acceptance["clinical_release_ready"] else "research_only"
    )
    release_manifest = write_release_manifest(
        result.checkpoint_path,
        status=release_status,
        patient_disjoint_evaluation=True,
        cohorts=tuple(sorted(frame["cohort_id"].astype(str).unique())),
        decision_reasons=(
            (
                "Neural optimization improved on the epoch-zero validation baseline."
                if acceptance["learning_gate_passed"]
                else "Neural proof-of-learning requirements were not all satisfied."
            ),
            (
                "All forecast horizons beat persistence with patient-macro "
                "95% confidence intervals."
                if acceptance[
                    "all_horizons_beat_persistence_with_patient_macro_95_ci"
                ]
                else "Persistence superiority was not established at every horizon."
            ),
            *tuple(acceptance["release_blockers"]),
        ),
        metrics_file=str((output_dir / "test_metrics.json").resolve()),
    )
    from src.neuroglycemic.reporting import (
        build_neural_model_card,
        write_neural_model_card,
    )

    model_card = build_neural_model_card(
        prediction_target=config.prediction_target,
        horizons_minutes=config.forecast_horizons_minutes,
        modalities=tuple(feature_names),
        cohorts=tuple(sorted(frame["cohort_id"].astype(str).unique())),
        split_counts={
            str(row["split"]): {
                "rows": int(row["rows"]),
                "patients": int(row["patients"]),
            }
            for row in split_audit.to_dict(orient="records")
        },
        metrics=metrics,
        acceptance=acceptance,
        checkpoint_path=result.checkpoint_path,
        release_manifest_path=release_manifest,
        data_sha256=source_digest,
    )
    model_card_path = write_neural_model_card(
        output_dir / "model_card.json", model_card
    )
    from src.neuroglycemic.figures import (
        save_forecast_figure,
        save_fusion_weight_figure,
        save_training_figure,
    )

    figure_dir = output_dir / "figures"
    save_training_figure(history, figure_dir / "training_loss.png")
    save_forecast_figure(predictions, figure_dir / "held_out_forecasts.png")
    save_fusion_weight_figure(predictions, figure_dir / "fusion_weights.png")
    print(f"\nSaved neural study artifacts to: {output_dir}")
    print(f"Saved auditable model card to: {model_card_path}")
    return {
        "metrics": metrics,
        "acceptance": acceptance,
        "checkpoint": str(result.checkpoint_path),
        "release_manifest": str(release_manifest),
        "model_card": str(model_card_path),
    }


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
    from src.neuroglycemic.neural_training import (
        GlucoseTargetStandardizer,
        load_neural_checkpoint,
        load_neural_training_config,
    )

    config = load_neural_training_config(config_path)
    destination = checkpoint_path or config.checkpoint_path
    from src.neuroglycemic.release import load_release_manifest

    release = load_release_manifest(destination)
    payload = _load_checkpoint_payload(destination)
    stored_training_config = payload.get("training_config")
    if stored_training_config != config.checkpoint_values():
        raise ValueError(
            "Evaluation config differs from the checkpoint training contract; "
            "use the exact version recorded for this run."
        )
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
    modalities = tuple(config.feature_registry) or tuple(model.modalities)
    frame, discovered = load_aligned_window_frame(
        data_path,
        config.forecast_horizons_minutes,
        modalities=modalities,
        horizon_tolerance_minutes=config.horizon_tolerance_minutes,
        feature_registry=config.feature_registry or None,
        input_cgm=config.input_cgm,
    )
    if {name: tuple(values) for name, values in discovered.items()} != dict(
        feature_standardizer.feature_names
    ):
        raise ValueError("Evaluation feature order/schema differs from the training checkpoint.")
    frame = attach_recorded_split(frame, patient_split)
    test = frame.loc[frame["split"] == "test"].copy()
    print_frame("CHECKPOINT-MATCHED ALIGNED EVALUATION WINDOWS", test)
    # Restore the response-kernel batch contract when the checkpoint learned
    # one; held-out patients are correctly marked unseen by the recorded map.
    patient_to_index = None
    if isinstance(metadata.get("patient_index_map"), dict):
        patient_to_index = {
            str(key): int(value)
            for key, value in metadata["patient_index_map"].items()
        }
    event_basis_columns = None
    if isinstance(metadata.get("event_basis_channels"), dict):
        event_basis_columns = {
            str(channel): tuple(str(column) for column in columns)
            for channel, columns in metadata["event_basis_channels"].items()
        }
    calibrator = None
    if isinstance(metadata.get("conformal_calibration"), dict):
        from src.neuroglycemic.calibration import ConformalCalibrator

        calibrator = ConformalCalibrator.from_dict(metadata["conformal_calibration"])
    # Mirror the training-time anchor thinning recorded in the checkpoint so
    # held-out metrics are computed on the same window contract.
    min_anchor_spacing = metadata.get("min_anchor_spacing_minutes")
    if min_anchor_spacing is not None:
        min_anchor_spacing = float(min_anchor_spacing)
    batches = make_neural_batches(
        test,
        feature_standardizer,
        config.forecast_horizons_minutes,
        batch_size=batch_size,
        auxiliary_tasks=config.auxiliary_tasks,
        patient_to_index=patient_to_index,
        event_basis_columns=event_basis_columns,
        min_anchor_spacing_minutes=min_anchor_spacing,
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
        calibrator=calibrator,
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
    original_predictions_path = output_dir / "test_predictions.csv"
    original_predictions_available = original_predictions_path.is_file()
    maximum_prediction_difference: float | None = None
    row_contract_matches: bool | None = None
    if original_predictions_available:
        original = pd.read_csv(original_predictions_path)
        identity_columns = [
            name
            for name in (
                "participant_key",
                "patient_id",
                "anchor_time",
                "horizon_minutes",
            )
            if name in original.columns and name in predictions.columns
        ]
        row_contract_matches = bool(
            len(original) == len(predictions)
            and identity_columns
            and original[identity_columns].astype(str).equals(
                predictions[identity_columns].astype(str)
            )
        )
        if row_contract_matches:
            maximum_prediction_difference = float(
                np.max(
                    np.abs(
                        original["predicted_glucose_mg_dl"].to_numpy(float)
                        - predictions["predicted_glucose_mg_dl"].to_numpy(float)
                    )
                )
            )
    reproducibility = {
        "checkpoint_sha256_verified": True,
        "release_status": release.status,
        "data_sha256_matches_checkpoint": True,
        "training_config_matches_checkpoint": True,
        "patient_split_restored_from_checkpoint": True,
        "original_prediction_artifact_available": original_predictions_available,
        "row_contract_matches_original": row_contract_matches,
        "maximum_absolute_prediction_difference_mg_dl": (
            maximum_prediction_difference
        ),
        "deterministic_reproduction_passed": (
            None
            if maximum_prediction_difference is None
            else maximum_prediction_difference <= 1e-6
        ),
    }
    print("\nCHECKPOINT REPRODUCIBILITY AUDIT")
    print(json.dumps(_json_safe(reproducibility), indent=2, allow_nan=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "reloaded_test_predictions.csv", index=False)
    ablation.to_csv(
        output_dir / "reloaded_missing_modality_ablation.csv", index=False
    )
    _save_json(output_dir / "reloaded_test_metrics.json", metrics)
    _save_json(output_dir / "evaluation_reproducibility.json", reproducibility)
    return {
        "metrics": metrics,
        "checkpoint": str(destination),
        "reproducibility": reproducibility,
    }


def cli() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Choose an explicit command. For neural training, use train-neural; "
            "for live/XDF inspection, use lsl-audit. The legacy research "
            "commands reproduce earlier non-product experiments only."
        ),
    )
    parser.add_argument(
        "study",
        choices=NEURAL_AND_INTEROPERABILITY_COMMANDS
        + LEGACY_RESEARCH_COMMANDS,
        help=(
            "Command to execute. No implicit study is run. eeg-wearable, "
            "ehr-glucose, and architecture are legacy research commands."
        ),
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
        "--workspace",
        type=Path,
        default=None,
        help="External runtime root for protected data, checkpoints, runs, and figures.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="External directory containing the selected source dataset.",
    )
    parser.add_argument(
        "--source-timezone",
        default=None,
        help="IANA timezone for timezone-naive source-device timestamps.",
    )
    parser.add_argument(
        "--clock-uncertainty-ms",
        type=float,
        default=60000.0,
        help="Declared retrospective source-clock uncertainty in milliseconds.",
    )
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="Allow verified official PhysioCGM pickle files to be loaded.",
    )
    parser.add_argument("--run-name", default="neuroglycemic-v4")
    parser.add_argument(
        "--channel",
        action="append",
        default=[],
        help="Numeric input column to replay through LSL; repeat for multiple channels.",
    )
    parser.add_argument("--timestamp-column", default="anchor_time")
    parser.add_argument(
        "--timestamp-format",
        choices=("iso8601", "unix_seconds", "lsl_seconds"),
        default="iso8601",
        help=(
            "Source timestamp representation for lsl-replay. CogWear device "
            "files use unix_seconds."
        ),
    )
    parser.add_argument("--stream-name", default="NeuroGlycemicReplay")
    parser.add_argument("--stream-type", default="CGMFeatures")
    parser.add_argument("--source-id", default="neuroglycemic-replay-v1")
    parser.add_argument("--replay-speed", type=float, default=60.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="External multi-device LSL replay manifest.",
    )
    parser.add_argument(
        "--startup-delay-seconds",
        type=float,
        default=5.0,
        help="Time for LabRecorder to discover replay outlets before samples start.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate replay data and metadata without opening LSL outlets.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append one validated LSL/XDF session to an existing external cohort.",
    )
    parser.add_argument(
        "--fhir-patient-reference",
        default=None,
        help="Optional Patient/<id> reference for a FHIR research-forecast export.",
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
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=0,
        help="Self-supervised masked-reconstruction epochs before fine-tuning.",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="Optional path to save the self-supervised pretraining weights.",
    )
    parser.add_argument(
        "--horizons-minutes",
        type=int,
        nargs="+",
        default=None,
        help="Forecast horizons for cohort builders that support them "
        "(for example: --horizons-minutes 30 60 90 120).",
    )
    parser.add_argument(
        "--init-from-pretrain",
        type=Path,
        default=None,
        help="Initialize encoders from a saved pretraining checkpoint.",
    )
    arguments = parser.parse_args()

    if arguments.study is None:
        parser.print_help(sys.stderr)
        parser.exit(
            2,
            "\nerror: no command selected; choose train-neural, lsl-audit, "
            "prepare-lsl-glucose, or another explicit command.\n",
        )

    if arguments.study in LEGACY_RESEARCH_COMMANDS:
        print(
            "LEGACY RESEARCH MODE: this command does not train the production "
            "neural CGM model. Use train-neural with an external aligned cohort "
            "for checkpoint-backed glucose forecasting.\n"
        )

    if arguments.study == "eeg-wearable":
        config_path = arguments.config or PROJECT_ROOT / "config" / "study.json"
        main(config_path, rebuild_features=arguments.rebuild)
        return


    if arguments.study == "lsl-audit":
        from src.neuroglycemic.lsl import audit_xdf, discover_streams

        if arguments.xdf is None:
            discovered = discover_streams()
            print_frame("DISCOVERED LSL STREAMS", discovered)
            if discovered.empty:
                print(
                    "\nNo live LSL outlets were discovered. Start the real device "
                    "outlets or a validated replay on the same reachable LSL "
                    "network, then run this command again."
                )
        else:
            xdf_path = _require_external_runtime_path(
                arguments.xdf,
                label="LabRecorder XDF",
                parser=parser,
            )
            if not xdf_path.is_file():
                parser.error(_missing_file_message(xdf_path, label="LabRecorder XDF"))
            audit, _ = audit_xdf(xdf_path)
            print_frame("LABRECORDER XDF STREAM AUDIT", audit)
        return

    if arguments.study == "prepare-lsl-glucose":
        from src.neuroglycemic.lsl_windowing import (
            build_lsl_glucose_windows_from_xdf,
            load_lsl_window_config,
        )
        from src.neuroglycemic.workspace import ResearchWorkspace

        if arguments.xdf is None or arguments.manifest is None or arguments.workspace is None:
            parser.error(
                "prepare-lsl-glucose requires --xdf, --manifest, and --workspace."
            )
        xdf_path = _require_external_runtime_path(
            arguments.xdf,
            label="LabRecorder XDF",
            parser=parser,
        )
        manifest_path = _require_external_runtime_path(
            arguments.manifest,
            label="LSL window manifest",
            parser=parser,
        )
        if not xdf_path.is_file():
            parser.error(_missing_file_message(xdf_path, label="LabRecorder XDF"))
        if not manifest_path.is_file():
            parser.error(
                _missing_file_message(manifest_path, label="LSL window manifest")
            )
        workspace = ResearchWorkspace.create(
            arguments.workspace, repository_root=PROJECT_ROOT
        )
        print(f"Software repository: {PROJECT_ROOT}")
        print(f"External workspace: {workspace.root}")
        lsl_config = load_lsl_window_config(manifest_path)
        windows, stream_audit, build_audit = build_lsl_glucose_windows_from_xdf(
            xdf_path, lsl_config
        )
        print_frame("LABRECORDER XDF STREAM AUDIT", stream_audit)
        print_frame("CAUSAL LSL EEG/WEARABLE GLUCOSE WINDOWS", windows)
        destination = workspace.aligned / "lsl_glucose_patient_windows.csv.gz"
        if destination.exists():
            if not arguments.append:
                parser.error(
                    f"{destination} already exists; use --append for another session."
                )
            existing = pd.read_csv(destination)
            if set(existing.columns) != set(windows.columns):
                parser.error(
                    "The appended session has a different feature schema; use a separate cohort."
                )
            windows = windows[existing.columns]
            combined = pd.concat((existing, windows), ignore_index=True)
            duplicate = combined.duplicated(
                ["cohort_id", "patient_id", "session_id", "anchor_time"]
            )
            if duplicate.any():
                parser.error("The appended session duplicates existing patient-time windows.")
            windows_to_save = combined
        else:
            windows_to_save = windows
        windows_to_save.to_csv(destination, index=False, compression="gzip")
        audit_directory = workspace.run_directory(arguments.run_name)
        stream_audit.to_csv(
            audit_directory / "xdf_stream_audit.csv",
            index=False,
        )
        _save_json(
            audit_directory / "window_build_audit.json",
            build_audit,
        )
        neural_config_path = arguments.config or PROJECT_ROOT / "config" / "neural_glucose.json"
        neural_values = json.loads(neural_config_path.read_text(encoding="utf-8"))
        neural_values["forecast_horizons_minutes"] = list(
            lsl_config.horizons_minutes
        )
        neural_values["forecast_mode"] = "ambient_no_cgm"
        neural_values["input_cgm"] = False
        neural_values["feature_registry"] = build_audit["feature_registry"]
        generated_config = workspace.canonical / "lsl_neural_glucose_config.json"
        _save_json(generated_config, neural_values)
        print(f"\nSaved combined external LSL cohort: {destination}")
        print(f"Generated matching neural config: {generated_config}")
        return

    if arguments.study == "prepare-mimic-neural":
        from src.neuroglycemic.mimic_neural import prepare_mimic_neural_file
        from src.neuroglycemic.workspace import ResearchWorkspace

        if arguments.data is None or arguments.workspace is None:
            parser.error("prepare-mimic-neural requires --data and --workspace.")
        _require_external_runtime_path(
            arguments.data,
            label="MIMIC neural source",
            parser=parser,
        )
        workspace = ResearchWorkspace.create(
            arguments.workspace, repository_root=PROJECT_ROOT
        )
        windows = prepare_mimic_neural_file(arguments.data)
        print_frame("CAUSAL MIMIC-IV DEMO NEURAL WINDOWS", windows)
        destination = workspace.aligned / "mimiciv_demo_neural_windows.csv.gz"
        if destination.exists() and not arguments.rebuild:
            parser.error(
                f"Refusing to overwrite existing aligned cohort: {destination}\n"
                "Re-run with --rebuild only after verifying the source cohort."
            )
        _atomic_write_csv(windows, destination, compression="gzip")
        print(f"\nSaved external MIMIC neural cohort: {destination}")
        return

    if arguments.study == "prepare-big-ideas":
        from src.neuroglycemic.big_ideas_data import (
            BigIdeasBuildConfig,
            big_ideas_build_manifest,
            build_big_ideas_dataset,
            discover_big_ideas_patients,
        )
        from src.neuroglycemic.workspace import ResearchWorkspace

        if arguments.workspace is None or arguments.source_dir is None:
            parser.error("prepare-big-ideas requires --workspace and --source-dir.")
        if not arguments.source_timezone:
            parser.error("prepare-big-ideas requires --source-timezone.")
        source_root = _require_external_runtime_path(
            arguments.source_dir,
            label="Big Ideas source data",
            parser=parser,
        )
        workspace = ResearchWorkspace.create(
            arguments.workspace, repository_root=PROJECT_ROOT
        )
        build_config = BigIdeasBuildConfig(
            source_timezone=arguments.source_timezone,
            clock_uncertainty_ms=arguments.clock_uncertainty_ms,
            horizons_minutes=(
                tuple(int(value) for value in arguments.horizons_minutes)
                if arguments.horizons_minutes
                else (30, 60)
            ),
        )
        windows, audit = build_big_ideas_dataset(
            discover_big_ideas_patients(source_root), config=build_config
        )
        print_frame("BIG IDEAS SOURCE AUDIT", audit)
        print_frame("BIG IDEAS CAUSAL WEARABLE/CGM WINDOWS", windows)
        destination = workspace.aligned / "big_ideas_wearable_cgm_windows.csv.gz"
        if destination.exists():
            parser.error(f"Refusing to overwrite existing aligned cohort: {destination}")
        windows.to_csv(destination, index=False, compression="gzip")
        audit.to_csv(workspace.canonical / "big_ideas_ingestion_audit.csv", index=False)
        _save_json(
            workspace.canonical / "big_ideas_build_manifest.json",
            big_ideas_build_manifest(
                windows, source_root=source_root, config=build_config
            ),
        )
        generated_config = workspace.canonical / "big_ideas_neural.json"
        _save_json(
            generated_config,
            json.loads(
                (PROJECT_ROOT / "config" / "big_ideas_neural.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        print(f"\nSaved aligned Big Ideas cohort outside Git: {destination}")
        print(f"Generated matching neural config: {generated_config}")
        return

    if arguments.study == "prepare-physiocgm":
        from src.neuroglycemic.physiocgm_data import (
            build_physiocgm_aligned_windows,
            write_physiocgm_build,
        )
        from src.neuroglycemic.workspace import ResearchWorkspace

        if arguments.workspace is None or arguments.source_dir is None:
            parser.error("prepare-physiocgm requires --workspace and --source-dir.")
        if not arguments.source_timezone:
            parser.error("prepare-physiocgm requires --source-timezone.")
        if not arguments.trust_pickle:
            parser.error(
                "prepare-physiocgm requires --trust-pickle after verifying the official files."
            )
        source_root = _require_external_runtime_path(
            arguments.source_dir,
            label="PhysioCGM processed source data",
            parser=parser,
        )
        workspace = ResearchWorkspace.create(
            arguments.workspace, repository_root=PROJECT_ROOT
        )
        horizons = (30, 60)
        result = build_physiocgm_aligned_windows(
            source_root,
            horizons_minutes=horizons,
            source_timezone=arguments.source_timezone,
            trust_pickle=True,
            clock_uncertainty_ms=arguments.clock_uncertainty_ms,
        )
        print_frame("PHYSIOCGM SOURCE AUDIT", result.audit)
        print_frame("PHYSIOCGM CAUSAL WEARABLE/CGM WINDOWS", result.frame)
        destination = workspace.aligned / "physiocgm_wearable_cgm_windows.csv.gz"
        if destination.exists():
            parser.error(f"Refusing to overwrite existing aligned cohort: {destination}")
        write_physiocgm_build(
            result,
            destination,
            input_dir=source_root,
            horizons_minutes=horizons,
            source_timezone=arguments.source_timezone,
        )
        generated_config = workspace.canonical / "physiocgm_neural.json"
        _save_json(
            generated_config,
            json.loads(
                (PROJECT_ROOT / "config" / "neural_glucose_physio.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        print(f"\nSaved aligned PhysioCGM cohort outside Git: {destination}")
        print(f"Generated matching neural config: {generated_config}")
        return

    if arguments.study == "prepare-diatrend":
        from src.neuroglycemic.diatrend import (
            DiaTrendBuildConfig,
            build_diatrend_dataset,
            discover_diatrend_workbooks,
        )
        from src.neuroglycemic.workspace import ResearchWorkspace

        if arguments.workspace is None or arguments.source_dir is None:
            parser.error("prepare-diatrend requires --workspace and --source-dir.")
        if not arguments.source_timezone:
            parser.error("prepare-diatrend requires --source-timezone.")
        _require_external_runtime_path(
            arguments.source_dir,
            label="DiaTrend source data",
            parser=parser,
        )
        workspace = ResearchWorkspace.create(
            arguments.workspace, repository_root=PROJECT_ROOT
        )
        workbooks = discover_diatrend_workbooks(arguments.source_dir)
        build_config = DiaTrendBuildConfig(
            source_timezone=arguments.source_timezone
        )
        windows, audit = build_diatrend_dataset(workbooks, config=build_config)
        print_frame("DIATREND SOURCE AUDIT", audit)
        print_frame("DIATREND CAUSAL PATIENT WINDOWS", windows)
        destination = workspace.aligned / "diatrend_patient_windows.csv.gz"
        audit_path = workspace.canonical / "diatrend_ingestion_audit.csv"
        windows.to_csv(destination, index=False, compression="gzip")
        audit.to_csv(audit_path, index=False)
        _save_json(
            workspace.canonical / "diatrend_build_config.json",
            {
                "source_timezone": build_config.source_timezone,
                "horizons_minutes": list(build_config.horizons_minutes),
                "grid_minutes": build_config.grid_minutes,
                "history_minutes": build_config.history_minutes,
                "anchor_stride_minutes": build_config.anchor_stride_minutes,
                "minimum_history_coverage": build_config.minimum_history_coverage,
                "workbook_count": len(workbooks),
            },
        )
        print(f"\nSaved aligned DiaTrend windows outside Git: {destination}")
        return

    if arguments.study == "lsl-replay":
        from src.neuroglycemic.lsl import replay_numeric_table

        if arguments.data is None or not arguments.channel:
            parser.error("lsl-replay requires --data and at least one --channel.")
        _require_external_runtime_path(
            arguments.data,
            label="LSL replay data",
            parser=parser,
        )
        suffix = arguments.data.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            replay_frame = pd.read_parquet(arguments.data)
        elif suffix in {".csv", ".gz"}:
            replay_frame = pd.read_csv(arguments.data)
        else:
            parser.error("lsl-replay data must be CSV, CSV.GZ, or Parquet.")
        replay_audit = replay_numeric_table(
            replay_frame,
            timestamp_column=arguments.timestamp_column,
            timestamp_format=arguments.timestamp_format,
            channel_columns=arguments.channel,
            stream_name=arguments.stream_name,
            stream_type=arguments.stream_type,
            source_id=arguments.source_id,
            speed=arguments.replay_speed,
            max_rows=arguments.max_rows,
        )
        print(json.dumps(replay_audit, indent=2))
        return

    if arguments.study == "lsl-session-replay":
        from src.neuroglycemic.lsl import (
            load_replay_session_manifest,
            replay_session_manifest,
        )

        if arguments.manifest is None:
            parser.error("lsl-session-replay requires --manifest.") 

        manifest_path = _require_external_runtime_path(
            arguments.manifest,
            label="LSL replay manifest",
            parser=parser,
        )

        if not manifest_path.is_file():
            parser.error(
                "LSL replay manifest does not exist or is not a file: "
                f"{manifest_path}"
            )

        session = load_replay_session_manifest(manifest_path)

        for stream in session.streams:
            data_path = _require_external_runtime_path(
                stream.data_path,
                label=f"LSL replay data for {stream.source_id}",
                parser=parser,
            )

            if not data_path.is_file():
                parser.error(
                    f"LSL replay data for {stream.source_id} does not exist "
                    f"or is not a file: {data_path}"
                )

        replay_audit = replay_session_manifest(
            manifest_path,
            speed=arguments.replay_speed,
            startup_delay_seconds=arguments.startup_delay_seconds,
            max_rows_per_stream=arguments.max_rows,
            dry_run=arguments.dry_run,
        )
        print(json.dumps(replay_audit, indent=2))
        return

    if arguments.study in {"train-neural", "evaluate-neural"}:
        if arguments.data is None:
            parser.error(f"{arguments.study} requires --data with a real aligned table.")
        from src.neuroglycemic.workspace import ResearchWorkspace

        if arguments.workspace is None:
            parser.error(
                f"{arguments.study} requires --workspace outside the software repository."
            )
        data_path = _require_external_runtime_path(
            arguments.data,
            label="Neural training/evaluation data",
            parser=parser,
        )
        if not data_path.is_file():
            parser.error(
                _missing_file_message(
                    data_path, label="Neural training/evaluation data"
                )
            )
        workspace = ResearchWorkspace.create(
            arguments.workspace, repository_root=PROJECT_ROOT
        )
        neural_config = arguments.config or PROJECT_ROOT / "config" / "neural_glucose.json"
        neural_config = neural_config.expanduser().resolve()
        if not neural_config.is_file():
            parser.error(
                _missing_file_message(neural_config, label="Neural configuration")
            )
        print(f"Software repository: {PROJECT_ROOT}")
        print(f"External workspace: {workspace.root}")
        neural_outputs = arguments.output_dir or workspace.run_directory(
            arguments.run_name
        )
        _require_external_runtime_path(
            neural_outputs,
            label="Neural outputs",
            parser=parser,
        )
        neural_checkpoint = arguments.checkpoint or (
            workspace.models / f"{arguments.run_name}.pt"
        )
        _require_external_runtime_path(
            neural_checkpoint,
            label="Neural checkpoint",
            parser=parser,
        )
        if arguments.study == "train-neural":
            run_neural_train(
                data_path=data_path,
                config_path=neural_config,
                checkpoint_path=neural_checkpoint,
                output_dir=neural_outputs,
                batch_size=arguments.batch_size,
                train_fraction=arguments.train_fraction,
                validation_fraction=arguments.validation_fraction,
                pretrain_epochs=arguments.pretrain_epochs,
                pretrain_checkpoint=arguments.pretrain_checkpoint,
                init_from_pretrain=arguments.init_from_pretrain,
            )
        else:
            run_neural_evaluate(
                data_path=data_path,
                config_path=neural_config,
                checkpoint_path=neural_checkpoint,
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
        from src.neuroglycemic.neural_training import load_neural_training_config
        from src.neuroglycemic.workspace import ResearchWorkspace

        neural_config = load_neural_training_config(neural_config_path)
        if arguments.workspace is None:
            parser.error("neural-case requires --workspace outside the repository.")
        _require_external_runtime_path(
            arguments.request,
            label="Neural case request",
            parser=parser,
        )
        workspace = ResearchWorkspace.create(
            arguments.workspace, repository_root=PROJECT_ROOT
        )
        checkpoint = arguments.checkpoint or (
            workspace.models / f"{arguments.run_name}.pt"
        )
        case_output = arguments.output_dir or workspace.run_directory(arguments.run_name)
        _require_external_runtime_path(
            checkpoint,
            label="Neural checkpoint",
            parser=parser,
        )
        _require_external_runtime_path(
            case_output,
            label="Neural case outputs",
            parser=parser,
        )
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
            workspace_root=workspace.root,
            health_agent=agent,
            include_raw_llm_response=arguments.print_llm_raw,
            output_path=case_output / "case_study.json",
        )
        if arguments.fhir_patient_reference:
            from src.neuroglycemic.fhir import neural_forecast_observation

            fhir_observation = neural_forecast_observation(
                case["neural_forecast"],
                patient_reference=arguments.fhir_patient_reference,
            )
            _save_json(case_output / "forecast_observation.fhir.json", fhir_observation)
            case["fhir_forecast_observation"] = fhir_observation
        print("\nCHECKPOINT-BACKED NEURAL ARCHITECTURE CASE STUDY")
        print(json.dumps(case, indent=2))
        return

    if arguments.study not in {"ehr-glucose", "architecture"}:
        parser.error(f"No command handler is registered for {arguments.study!r}.")

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