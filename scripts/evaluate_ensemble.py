#!/usr/bin/env python3
"""Evaluate a K-seed neural ensemble on the checkpoint-recorded test split.

Example:
    python scripts/evaluate_ensemble.py \
        --data /path/aligned.csv.gz \
        --config config/neural_glucose.json \
        --workspace /path/neuroglycemic-runtime \
        --run-name dvxr-bridge-v1 \
        --members 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.neuroglycemic.ensemble import (  # noqa: E402
    combine_member_predictions,
    load_ensemble_members,
)
from src.neuroglycemic.neural_dataset import (  # noqa: E402
    TrainOnlyFeatureStandardizer,
    attach_recorded_split,
    glucose_forecast_metrics,
    load_aligned_window_frame,
    make_neural_batches,
    predict_neural_batches,
)
from src.neuroglycemic.neural_training import load_neural_training_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    checkpoints = [
        args.workspace
        / "models"
        / f"{args.run_name}-seed-{seed}.pt"
        for seed in range(args.first_seed, args.first_seed + args.members)
    ]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing ensemble member checkpoints: {missing}")

    members = load_ensemble_members(checkpoints)
    print(f"Loaded {len(members)} ensemble members with a verified shared contract.")

    reference = members[0]
    metadata = reference["metadata"]
    config = load_neural_training_config(args.config)
    feature_standardizer = TrainOnlyFeatureStandardizer.from_dict(
        metadata["feature_schema"]
    )
    modalities = tuple(config.feature_registry) or tuple(
        reference["model"].modalities
    )
    frame, _ = load_aligned_window_frame(
        args.data,
        config.forecast_horizons_minutes,
        modalities=modalities,
        horizon_tolerance_minutes=config.horizon_tolerance_minutes,
        feature_registry=config.feature_registry or None,
        input_cgm=config.input_cgm,
    )
    frame = attach_recorded_split(frame, metadata["patient_split"])
    test = frame.loc[frame["split"] == "test"].copy()

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
    batches = make_neural_batches(
        test,
        feature_standardizer,
        config.forecast_horizons_minutes,
        batch_size=args.batch_size,
        auxiliary_tasks=config.auxiliary_tasks,
        patient_to_index=patient_to_index,
        event_basis_columns=event_basis_columns,
    )
    member_frames = [
        predict_neural_batches(
            member["model"],
            batches,
            member["target_standardizer"],
            config.forecast_horizons_minutes,
            hypoglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl["hypoglycemia"],
            hyperglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl[
                "hyperglycemia"
            ],
        )
        for member in members
    ]
    combined = combine_member_predictions(
        member_frames,
        hypoglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl["hypoglycemia"],
        hyperglycemia_threshold_mg_dl=config.risk_thresholds_mg_dl["hyperglycemia"],
    )
    metrics = glucose_forecast_metrics(combined)

    output_dir = args.workspace / "runs" / f"{args.run_name}-ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "ensemble_test_predictions.csv", index=False)
    from main import _json_safe

    safe_metrics = _json_safe(metrics)
    with (output_dir / "ensemble_test_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(safe_metrics, fh, indent=2, allow_nan=False)
    print(json.dumps(safe_metrics, indent=2, allow_nan=False))
    print(f"\nSaved ensemble artifacts to: {output_dir}")


if __name__ == "__main__":
    main()