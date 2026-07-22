"""K-member deep-ensemble combination for the neural glucose forecaster.

Training the same architecture with different seeds and averaging the
predictive distributions (a mixture of the per-member mixtures) improves both
point accuracy and uncertainty quality.  The between-member spread is the
epistemic component the abstention logic can use.

Members must share the same data, patient split, feature schema, horizons and
target standardization; only the training seed differs.  The combiner verifies
these contracts before merging so a mismatched member cannot silently enter
the ensemble.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .neural_model import NeuroGlycemicNet
from .neural_training import (
    GlucoseTargetStandardizer,
    load_neural_checkpoint,
)

ENSEMBLE_SCHEMA = "neuroglycemic-ensemble-v1"


def model_from_spec(spec: Mapping[str, Any]) -> NeuroGlycemicNet:
    """Reconstruct one member from its checkpoint ``model_spec`` metadata."""

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
        raise ValueError(f"model_spec is missing fields: {sorted(missing)}")
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


def load_ensemble_members(
    checkpoint_paths: Sequence[Path],
    *,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    """Load and contract-check ensemble members (same split/schema/target)."""

    if len(checkpoint_paths) < 2:
        raise ValueError("An ensemble needs at least two member checkpoints.")
    members: list[dict[str, Any]] = []
    reference: dict[str, Any] = {}
    for path in checkpoint_paths:
        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or "model_spec" not in metadata:
            raise ValueError(f"{path} is missing serving metadata.")
        model = model_from_spec(metadata["model_spec"])
        load_neural_checkpoint(path, model, device=device)
        standardizer = GlucoseTargetStandardizer.from_dict(
            payload["target_standardizer"]
        )
        contract = {
            "feature_schema": metadata.get("feature_schema"),
            "patient_split": metadata.get("patient_split"),
            "target_standardizer": payload.get("target_standardizer"),
            "data_sha256": metadata.get("data_sha256"),
        }
        if not reference:
            reference = contract
        elif contract != reference:
            raise ValueError(
                f"{path} does not share the first member's data/split/schema "
                "contract; ensemble members may differ only by training seed."
            )
        members.append(
            {
                "path": path,
                "model": model.eval(),
                "target_standardizer": standardizer,
                "metadata": metadata,
                "payload": payload,
            }
        )
    return members


def combine_member_predictions(
    member_frames: Sequence[pd.DataFrame],
    *,
    member_prefixes: Sequence[str] | None = None,
    hypoglycemia_threshold_mg_dl: float = 70.0,
    hyperglycemia_threshold_mg_dl: float = 180.0,
) -> pd.DataFrame:
    """Merge per-member prediction frames into mixture-of-mixtures rows.

    Every member contributes its experts with weight ``w_m / K``.  The output
    frame follows the same column contract as ``predict_neural_batches`` (with
    seed-prefixed expert names), so the standard metrics, bootstrap and
    ablation tooling consume it unchanged.
    """

    from .evaluation import gaussian_mixture_quantile

    if len(member_frames) < 2:
        raise ValueError("Combining requires at least two member frames.")
    prefixes = member_prefixes or [f"m{index}" for index in range(len(member_frames))]
    if len(prefixes) != len(member_frames) or len(set(prefixes)) != len(prefixes):
        raise ValueError("Member prefixes must be unique and match the frame count.")

    key = ["participant_key", "anchor_time", "horizon_minutes"]
    base_columns = [
        "patient_id",
        "cohort_id",
        "participant_key",
        "anchor_time",
        "horizon_minutes",
        "target_glucose_mg_dl",
        "target_hypoglycemia",
        "target_hyperglycemia",
        "persistence_glucose_mg_dl",
        "abstained",
    ]
    reference = member_frames[0].set_index(key).sort_index()
    aligned: list[pd.DataFrame] = []
    for frame in member_frames:
        indexed = frame.set_index(key).sort_index()
        if not indexed.index.equals(reference.index):
            raise ValueError(
                "Member predictions must cover identical patients, anchors and "
                "horizons for a paired ensemble combination."
            )
        aligned.append(indexed)

    member_modalities: list[tuple[str, ...]] = []
    for frame in aligned:
        names = tuple(
            column.removeprefix("weight_")
            for column in frame.columns
            if column.startswith("weight_")
        )
        if not names:
            raise ValueError("A member frame contains no learned modality weights.")
        member_modalities.append(names)

    # Uniform expert schema across rows: rows with a missing modality carry
    # that expert with weight exactly zero, never a NaN cell.
    all_expert_names = [
        f"{prefix}_{name}"
        for prefix, names in zip(prefixes, member_modalities)
        for name in names
    ]
    rows: list[dict[str, Any]] = []
    member_count = len(aligned)
    for index_values in reference.index:
        location = {name: value for name, value in zip(key, index_values)}
        member_rows = [frame.loc[index_values] for frame in aligned]
        combined_means: list[float] = []
        combined_scales: list[float] = []
        combined_weights: list[float] = []
        combined_names: list[str] = []
        for prefix, row, names in zip(prefixes, member_rows, member_modalities):
            for name in names:
                weight = float(row[f"weight_{name}"])
                if not math.isfinite(weight) or weight <= 0:
                    continue
                combined_means.append(float(row[f"expert_mean_{name}_mg_dl"]))
                combined_scales.append(float(row[f"expert_sd_{name}_mg_dl"]))
                combined_weights.append(weight / member_count)
                combined_names.append(f"{prefix}_{name}")
        total_weight = sum(combined_weights)
        if total_weight <= 0:
            abstained = True
            # Zero-total weight marks abstention for every metrics consumer.
            combined_weights = [0.0] * max(len(combined_weights), 1)
        else:
            abstained = bool(member_rows[0]["abstained"])
            combined_weights = [value / total_weight for value in combined_weights]
        means = np.asarray(combined_means, dtype=float)
        scales = np.asarray(combined_scales, dtype=float)
        weights = np.asarray(combined_weights, dtype=float)
        mean = float((weights * means).sum())
        second_moment = float((weights * (scales**2 + means**2)).sum())
        variance = max(second_moment - mean**2, 0.0)
        sqrt_two = math.sqrt(2.0)
        erf = np.vectorize(math.erf)
        hypo = float(
            (
                weights
                * 0.5
                * (
                    1.0
                    + erf(
                        (hypoglycemia_threshold_mg_dl - means) / (sqrt_two * scales)
                    )
                )
            ).sum()
        )
        hyper = float(
            (
                weights
                * 0.5
                * (
                    1.0
                    - erf(
                        (hyperglycemia_threshold_mg_dl - means) / (sqrt_two * scales)
                    )
                )
            ).sum()
        )
        row: dict[str, Any] = {
            **location,
            "patient_id": member_rows[0]["patient_id"],
            "cohort_id": member_rows[0]["cohort_id"],
            "target_glucose_mg_dl": float(member_rows[0]["target_glucose_mg_dl"]),
            "target_hypoglycemia": member_rows[0]["target_hypoglycemia"],
            "target_hyperglycemia": member_rows[0]["target_hyperglycemia"],
            "persistence_glucose_mg_dl": float(
                member_rows[0]["persistence_glucose_mg_dl"]
            ),
            "abstained": abstained,
            "predicted_glucose_mg_dl": (math.nan if abstained else mean),
            "predicted_standard_deviation_mg_dl": (
                math.nan if abstained else math.sqrt(variance)
            ),
            "prediction_lower_mg_dl": (
                math.nan
                if abstained
                else gaussian_mixture_quantile(
                    combined_means, combined_scales, combined_weights, 0.025
                )
            ),
            "prediction_upper_mg_dl": (
                math.nan
                if abstained
                else gaussian_mixture_quantile(
                    combined_means, combined_scales, combined_weights, 0.975
                )
            ),
            "hypoglycemia_probability": (math.nan if abstained else hypo),
            "hyperglycemia_probability": (math.nan if abstained else hyper),
            "ensemble_member_count": member_count,
            "ensemble_mean_spread_mg_dl": float(
                np.std(
                    [
                        float(member_row["predicted_glucose_mg_dl"])
                        for member_row in member_rows
                    ]
                )
            ),
        }
        included = dict(
            zip(
                combined_names,
                zip(combined_means, combined_scales, combined_weights),
            )
        )
        for name in all_expert_names:
            if name in included:
                mean_value, scale_value, weight_value = included[name]
            else:
                # Zero-weight placeholder: excluded by every mixture consumer
                # but keeps the frame schema uniform (no NaN cells).
                mean_value, scale_value, weight_value = mean, 1.0, 0.0
            row[f"weight_{name}"] = weight_value
            row[f"expert_mean_{name}_mg_dl"] = mean_value
            row[f"expert_sd_{name}_mg_dl"] = scale_value
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)