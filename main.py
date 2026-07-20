import argparse
import json
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

def _save_json(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


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
        session = patient_session_predictions(group, "combined_probability")
        metrics = classification_metrics(session[TARGET], session["probability"])
        rows.append({"model_or_scenario": f"missingness: {scenario}", **metrics})
    return pd.DataFrame(rows)


def main(config_path: Path | None = None, *, rebuild_features: bool = False) -> None:
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
    print(f"\nSaved derived study artifacts to: {config.output_dir}")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        nargs="?",
        choices=("eeg-wearable", "ehr-glucose"),
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
    arguments = parser.parse_args()
    if arguments.study == "eeg-wearable":
        config_path = arguments.config or PROJECT_ROOT / "config" / "study.json"
        main(config_path, rebuild_features=arguments.rebuild)
        return

    from src.neuroglycemic.config import load_ehr_config
    from src.neuroglycemic.pipeline import run_ehr_glucose_pipeline

    config_path = arguments.config or PROJECT_ROOT / "config" / "ehr_glucose.json"
    run_ehr_glucose_pipeline(load_ehr_config(config_path), rebuild=arguments.rebuild)


if __name__ == "__main__":
    cli()
