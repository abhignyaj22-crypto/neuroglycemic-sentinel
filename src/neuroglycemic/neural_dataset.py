"""Causal, patient-aligned data preparation for the neural glucose model.

This module consumes one *already aligned* wide table.  It never joins EEG,
wearable, EHR, meal, or CGM records and therefore cannot manufacture a paired
patient from disjoint cohorts.  Source adapters are responsible for producing
the explicit patient/cohort/time provenance checked here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.cogwear_study.split import PatientSplit, attach_split, split_patients


MODALITIES = ("eeg", "wearable", "ehr")
FEATURE_SCHEMA_VERSION = "neuroglycemic-aligned-window-v1"
SHARED_FEATURE_PREFIXES = ("meal_", "context_")


def target_column(horizon_minutes: int) -> str:
    return f"target_glucose_{int(horizon_minutes)}m_mg_dl"


def target_time_column(horizon_minutes: int) -> str:
    return f"target_glucose_{int(horizon_minutes)}m_time"


def data_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Aligned neural dataset does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError("Aligned neural data must be CSV, CSV.GZ, or Parquet.")


def _as_boolean(values: pd.Series, *, name: str) -> pd.Series:
    if values.dtype == bool:
        return values.copy()
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError(f"{name} must contain only boolean or 0/1 values.")
    return numeric.astype(bool)


def discover_feature_columns(
    frame: pd.DataFrame,
    modalities: Sequence[str] = MODALITIES,
    *,
    feature_registry: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Resolve ordered inputs, preferring a versioned explicit allowlist.

    Prefix discovery remains only for backwards compatibility with the original
    v3 aligned-window contract. Production experiments pass ``feature_registry``
    so a newly added target/provenance column cannot silently become a predictor.
    """

    if feature_registry:
        if set(feature_registry) != set(modalities):
            raise ValueError("The feature registry must cover exactly the modeled modalities.")
        result = {
            modality: tuple(str(name) for name in feature_registry[modality])
            for modality in modalities
        }
        for modality, names in result.items():
            if not names or len(names) != len(set(names)):
                raise ValueError(f"Feature registry for {modality!r} is empty or duplicated.")
            missing = set(names) - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Feature registry for {modality!r} references missing columns: {sorted(missing)}"
                )
            if any(not name.startswith(f"{modality}_") for name in names):
                raise ValueError(
                    f"Every registered {modality!r} feature must start with {modality}_."
                )
            unsafe = _unsafe_feature_names(names, modality=modality)
            if unsafe:
                raise ValueError(f"Unsafe model features are prohibited: {unsafe}")
        return result, ()

    shared = tuple(
        column
        for column in frame.columns
        if any(column.startswith(prefix) for prefix in SHARED_FEATURE_PREFIXES)
    )
    result: dict[str, tuple[str, ...]] = {}
    for modality in modalities:
        excluded = {
            f"{modality}_available",
            f"{modality}_quality",
            f"{modality}_staleness_minutes",
            f"{modality}_patient_id",
            f"{modality}_cohort_id",
            f"{modality}_anchor_time",
            f"{modality}_available_time",
            f"{modality}_clock_uncertainty_ms",
        }
        specific = tuple(
            column
            for column in frame.columns
            if column.startswith(f"{modality}_") and column not in excluded
        )
        if not specific:
            raise ValueError(f"No {modality!r} feature columns were found.")
        unsafe = _unsafe_feature_names(specific, modality=modality)
        if unsafe:
            raise ValueError(f"Unsafe model features are prohibited: {unsafe}")
        result[modality] = (*specific, *shared)
    return result, shared


def _unsafe_feature_names(
    names: Sequence[str], *, modality: str
) -> list[str]:
    """Reject fields whose names disclose a label or post-anchor computation.

    The check is applied after removing the required modality prefix; otherwise
    a field such as ``eeg_target_glucose_30m_mg_dl`` evades a naïve
    ``startswith('target_')`` test.
    """

    prefix = f"{modality}_"
    forbidden_tokens = {"target", "future", "label", "outcome", "postanchor"}
    unsafe: list[str] = []
    for name in names:
        local = name[len(prefix) :] if name.startswith(prefix) else name
        tokens = set(local.lower().replace("-", "_").split("_"))
        if (
            tokens & forbidden_tokens
            or local.endswith("_time")
            or local.endswith("_patient_id")
            or local.endswith("_cohort_id")
            or local in {
                "available",
                "quality",
                "staleness_minutes",
                "available_time",
                "anchor_time",
                "clock_uncertainty_ms",
            }
        ):
            unsafe.append(name)
    return unsafe


def load_aligned_window_frame(
    path: Path,
    horizons_minutes: Sequence[int],
    *,
    modalities: Sequence[str] = MODALITIES,
    horizon_tolerance_minutes: float = 5.0,
    feature_registry: Mapping[str, Sequence[str]] | None = None,
    input_cgm: bool = False,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Load and validate one source-built, same-patient multimodal window table.

    Every available modality must repeat the canonical patient, cohort, and
    anchor.  This redundancy is intentional: it turns accidental cross-cohort
    or cross-patient joins into a hard error before model fitting.
    """

    frame = _read_table(path)
    if frame.empty:
        raise ValueError("Aligned neural dataset is empty.")
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns are not allowed: {duplicated}")
    required = {"patient_id", "cohort_id", "anchor_time"}
    for horizon in horizons_minutes:
        required.update({target_column(horizon), target_time_column(horizon)})
    for modality in modalities:
        required.update(
            {
                f"{modality}_available",
                f"{modality}_quality",
                f"{modality}_staleness_minutes",
                f"{modality}_patient_id",
                f"{modality}_cohort_id",
                f"{modality}_anchor_time",
                f"{modality}_available_time",
                f"{modality}_clock_uncertainty_ms",
            }
        )
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Aligned neural dataset is missing columns: {sorted(missing)}")

    frame = frame.copy()
    for name in ("patient_id", "cohort_id"):
        frame[name] = frame[name].astype("string")
        if frame[name].isna().any() or frame[name].str.strip().eq("").any():
            raise ValueError(f"{name} must be populated on every row.")
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], utc=True, errors="coerce")
    if frame["anchor_time"].isna().any():
        raise ValueError("anchor_time contains invalid timestamps.")
    if frame.duplicated(["cohort_id", "patient_id", "anchor_time"]).any():
        raise ValueError(
            "cohort_id + patient_id + anchor_time must uniquely identify a window."
        )

    feature_columns, _ = discover_feature_columns(
        frame, modalities, feature_registry=feature_registry
    )
    all_features = sorted({item for values in feature_columns.values() for item in values})
    forbidden_cgm = [name for name in all_features if "cgm" in name.lower()]
    if forbidden_cgm and not input_cgm:
        raise ValueError(
            "input_cgm=false forbids CGM predictor columns; found: "
            f"{forbidden_cgm}"
        )
    for column in all_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for modality in modalities:
        available_name = f"{modality}_available"
        quality_name = f"{modality}_quality"
        stale_name = f"{modality}_staleness_minutes"
        available = _as_boolean(frame[available_name], name=available_name)
        frame[available_name] = available
        quality = pd.to_numeric(frame[quality_name], errors="coerce")
        staleness = pd.to_numeric(frame[stale_name], errors="coerce")
        if quality.isna().any() or (~quality.between(0.0, 1.0)).any():
            raise ValueError(f"{quality_name} must be finite and in [0, 1].")
        if staleness.isna().any() or (staleness < 0).any():
            raise ValueError(f"{stale_name} must be finite and non-negative.")
        frame[quality_name] = quality.astype(float)
        frame[stale_name] = staleness.astype(float)
        clock_name = f"{modality}_clock_uncertainty_ms"
        clock_uncertainty = pd.to_numeric(frame[clock_name], errors="coerce")
        if clock_uncertainty.isna().any() or (clock_uncertainty < 0).any():
            raise ValueError(f"{clock_name} must be finite and non-negative.")
        frame[clock_name] = clock_uncertainty.astype(float)

        patient = frame[f"{modality}_patient_id"].astype("string")
        cohort = frame[f"{modality}_cohort_id"].astype("string")
        anchor = pd.to_datetime(
            frame[f"{modality}_anchor_time"], utc=True, errors="coerce"
        )
        mismatch = available & (
            patient.ne(frame["patient_id"]).fillna(True)
            | cohort.ne(frame["cohort_id"]).fillna(True)
            | anchor.isna()
            | anchor.ne(frame["anchor_time"])
        )
        if mismatch.any():
            raise ValueError(
                f"{modality} provenance does not match the canonical patient/cohort/anchor. "
                "Cross-patient and cross-cohort fusion is prohibited."
            )
        available_time_name = f"{modality}_available_time"
        available_time = pd.to_datetime(
            frame[available_time_name], utc=True, errors="coerce"
        )
        future_information = available & (
            available_time.isna() | available_time.gt(frame["anchor_time"])
        )
        if future_information.any():
            raise ValueError(
                f"{available_time_name} must be populated and no later than anchor_time."
            )
        frame[available_time_name] = available_time
        specific = [name for name in feature_columns[modality] if name.startswith(f"{modality}_")]
        has_specific_value = frame[specific].notna().any(axis=1)
        if (available & ~has_specific_value).any():
            raise ValueError(f"Available {modality} rows need at least one observed feature.")
        if ((~available) & has_specific_value).any():
            raise ValueError(
                f"Unavailable {modality} rows must leave modality features missing, not zero-filled."
            )

    if (~frame[[f"{name}_available" for name in modalities]].any(axis=1)).any():
        raise ValueError("Training rows with all modalities missing are not allowed.")

    for horizon in horizons_minutes:
        value_name = target_column(horizon)
        time_name = target_time_column(horizon)
        values = pd.to_numeric(frame[value_name], errors="coerce")
        times = pd.to_datetime(frame[time_name], utc=True, errors="coerce")
        valid = values.notna()
        if valid.sum() < 2:
            raise ValueError(f"{value_name} needs at least two observed labels.")
        if (values[valid] <= 0).any():
            raise ValueError(f"{value_name} must contain positive mg/dL values.")
        delta_minutes = (times - frame["anchor_time"]).dt.total_seconds() / 60.0
        invalid_time = valid & (
            times.isna()
            | (delta_minutes <= 0)
            | ((delta_minutes - float(horizon)).abs() > horizon_tolerance_minutes)
        )
        if invalid_time.any():
            raise ValueError(
                f"{time_name} must be a future timestamp within "
                f"{horizon_tolerance_minutes:g} minutes of the {horizon}-minute horizon."
            )
        frame[value_name] = values.astype(float)
        frame[time_name] = times

    frame = frame.sort_values(
        ["cohort_id", "patient_id", "anchor_time"], ignore_index=True
    )
    return frame, feature_columns


def patient_grouped_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, PatientSplit]:
    participant_key = (
        frame["cohort_id"].astype(str) + "::" + frame["patient_id"].astype(str)
    )
    split = split_patients(
        participant_key,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    lookup = {
        **{value: "train" for value in split.train},
        **{value: "validation" for value in split.validation},
        **{value: "test" for value in split.test},
    }
    result = frame.copy()
    result["participant_key"] = participant_key
    result["split"] = result["participant_key"].map(lookup)
    if result["split"].isna().any():
        raise RuntimeError("At least one cohort-patient key has no split assignment.")
    sets = {
        name: set(part["participant_key"].astype(str))
        for name, part in result.groupby("split", sort=False)
    }
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise RuntimeError("Patient leakage was detected across partitions.")
    return result, split


@dataclass(frozen=True)
class TrainOnlyFeatureStandardizer:
    """Per-feature z-score statistics fitted on available training rows only."""

    feature_names: Mapping[str, tuple[str, ...]]
    means: Mapping[str, tuple[float, ...]]
    scales: Mapping[str, tuple[float, ...]]
    valid_counts: Mapping[str, tuple[int, ...]]
    fit_split: str = "train"
    schema_version: str = FEATURE_SCHEMA_VERSION

    @classmethod
    def fit(
        cls,
        training: pd.DataFrame,
        feature_names: Mapping[str, Sequence[str]],
    ) -> "TrainOnlyFeatureStandardizer":
        if "split" in training and not training["split"].eq("train").all():
            raise ValueError("Feature statistics may only be fit on training rows.")
        means: dict[str, tuple[float, ...]] = {}
        scales: dict[str, tuple[float, ...]] = {}
        counts: dict[str, tuple[int, ...]] = {}
        ordered: dict[str, tuple[str, ...]] = {}
        for modality, names_value in feature_names.items():
            names = tuple(names_value)
            available = training[f"{modality}_available"].astype(bool).to_numpy()
            modality_means: list[float] = []
            modality_scales: list[float] = []
            modality_counts: list[int] = []
            for name in names:
                values = pd.to_numeric(training[name], errors="coerce").to_numpy(float)
                observed = available & np.isfinite(values)
                if not observed.any():
                    raise ValueError(
                        f"Feature {name!r} has no observed values in available training {modality} rows."
                    )
                selected = values[observed]
                mean = float(selected.mean())
                scale = float(selected.std(ddof=0))
                if not math.isfinite(scale) or scale <= 1e-12:
                    scale = 1.0
                modality_means.append(mean)
                modality_scales.append(scale)
                modality_counts.append(int(selected.size))
            ordered[modality] = names
            means[modality] = tuple(modality_means)
            scales[modality] = tuple(modality_scales)
            counts[modality] = tuple(modality_counts)
        return cls(ordered, means, scales, counts)

    @property
    def input_dims(self) -> dict[str, int]:
        return {name: len(values) for name, values in self.feature_names.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.schema_version,
            "fit_split": self.fit_split,
            "modalities": list(self.feature_names),
            "ordered_feature_names": {
                name: list(values) for name, values in self.feature_names.items()
            },
            "feature_names": {
                name: list(values) for name, values in self.feature_names.items()
            },
            "means": {name: list(values) for name, values in self.means.items()},
            "scales": {name: list(values) for name, values in self.scales.items()},
            "valid_counts": {
                name: list(values) for name, values in self.valid_counts.items()
            },
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainOnlyFeatureStandardizer":
        names = values.get("ordered_feature_names", values.get("feature_names"))
        if not isinstance(names, Mapping):
            raise ValueError("Checkpoint feature schema is missing ordered feature names.")
        result = cls(
            feature_names={key: tuple(item) for key, item in names.items()},
            means={key: tuple(float(value) for value in item) for key, item in values["means"].items()},
            scales={key: tuple(float(value) for value in item) for key, item in values["scales"].items()},
            valid_counts={key: tuple(int(value) for value in item) for key, item in values["valid_counts"].items()},
            fit_split=str(values["fit_split"]),
            schema_version=str(values["version"]),
        )
        if result.fit_split != "train" or result.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("Unsupported or non-training feature standardization provenance.")
        for modality, names_value in result.feature_names.items():
            size = len(names_value)
            if any(len(mapping[modality]) != size for mapping in (result.means, result.scales, result.valid_counts)):
                raise ValueError("Checkpoint feature statistics do not match feature names.")
            if any(not math.isfinite(value) or value <= 0 for value in result.scales[modality]):
                raise ValueError("Checkpoint feature scales must be finite and positive.")
        return result

    def transform_modality(self, frame: pd.DataFrame, modality: str) -> tuple[Tensor, Tensor]:
        names = self.feature_names[modality]
        missing = set(names) - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset is missing checkpoint features: {sorted(missing)}")
        values = frame[list(names)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        observed = np.isfinite(values)
        available = frame[f"{modality}_available"].astype(bool).to_numpy()[:, None]
        observed &= available
        means = np.asarray(self.means[modality], dtype=float)
        scales = np.asarray(self.scales[modality], dtype=float)
        standardized = (values - means) / scales
        standardized[~observed] = np.nan
        return torch.tensor(standardized.tolist(), dtype=torch.float32), torch.tensor(
            observed.tolist(), dtype=torch.bool
        )


def make_neural_batches(
    frame: pd.DataFrame,
    standardizer: TrainOnlyFeatureStandardizer,
    horizons_minutes: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    auxiliary_tasks: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    ordered = frame.reset_index(drop=True)
    if shuffle:
        order = np.random.default_rng(seed).permutation(len(ordered))
        ordered = ordered.iloc[order].reset_index(drop=True)
    target_names = [target_column(value) for value in horizons_minutes]
    targets = ordered[target_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(targets).any():
        raise ValueError("A partition must contain at least one observed glucose target.")

    full_features: dict[str, Tensor] = {}
    full_masks: dict[str, Tensor] = {}
    for modality in standardizer.feature_names:
        full_features[modality], full_masks[modality] = standardizer.transform_modality(
            ordered, modality
        )
    availability = torch.tensor(
        ordered[
            [f"{name}_available" for name in standardizer.feature_names]
        ].to_numpy(bool).tolist(),
        dtype=torch.bool,
    )
    quality = torch.tensor(
        ordered[
            [f"{name}_quality" for name in standardizer.feature_names]
        ].to_numpy(float).tolist(),
        dtype=torch.float32,
    )
    staleness = torch.tensor(
        ordered[
            [f"{name}_staleness_minutes" for name in standardizer.feature_names]
        ].to_numpy(float).tolist(),
        dtype=torch.float32,
    )
    clock_uncertainty = torch.tensor(
        ordered[
            [f"{name}_clock_uncertainty_ms" for name in standardizer.feature_names]
        ].to_numpy(float).tolist(),
        dtype=torch.float32,
    )
    if bool((clock_uncertainty < 0).any()) or not torch.isfinite(clock_uncertainty).all():
        raise ValueError("Clock uncertainty must be finite and non-negative.")
    target_tensor = torch.tensor(targets.tolist(), dtype=torch.float32)
    participant_keys = ordered.get(
        "participant_key",
        ordered["cohort_id"].astype(str) + "::" + ordered["patient_id"].astype(str),
    ).astype(str)
    patient_counts = participant_keys.map(participant_keys.value_counts()).astype(float)
    # Every patient contributes equal total objective mass even when recording
    # durations differ. Normalize to mean one so optimizer scale remains stable.
    patient_weight = (1.0 / patient_counts)
    patient_weight = patient_weight / patient_weight.mean()
    patient_weight_tensor = torch.tensor(
        patient_weight.to_numpy(float).tolist(), dtype=torch.float32
    )
    persistence_name = next(
        (
            name
            for name in ("cgm_current_mg_dl", "ehr_current_glucose_mg_dl")
            if name in ordered
        ),
        None,
    )
    persistence = torch.tensor(
        pd.to_numeric(
            ordered[persistence_name]
            if persistence_name is not None
            else pd.Series(np.nan, index=ordered.index),
            errors="coerce",
        ).to_numpy(float).tolist(),
        dtype=torch.float32,
    )
    auxiliary_targets: dict[str, Tensor] = {}
    auxiliary_masks: dict[str, Tensor] = {}
    for task_name, specification in (auxiliary_tasks or {}).items():
        column = str(specification["target_column"])
        if column not in ordered:
            raise ValueError(
                f"Configured auxiliary task {task_name!r} requires column {column!r}."
            )
        values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(float)
        kind = str(specification["kind"])
        valid = np.isfinite(values)
        if kind == "binary" and not np.isin(values[valid], (0.0, 1.0)).all():
            raise ValueError(f"Auxiliary binary target {column!r} must contain 0/1 values.")
        auxiliary_targets[task_name] = torch.tensor(values.tolist(), dtype=torch.float32)
        auxiliary_masks[task_name] = torch.tensor(valid.tolist(), dtype=torch.bool)
    batches: list[dict[str, Any]] = []
    for start in range(0, len(ordered), batch_size):
        stop = min(start + batch_size, len(ordered))
        batches.append(
            {
                "features": {name: value[start:stop] for name, value in full_features.items()},
                "feature_masks": {name: value[start:stop] for name, value in full_masks.items()},
                "availability": availability[start:stop],
                "quality": quality[start:stop],
                "staleness": staleness[start:stop],
                "clock_uncertainty": clock_uncertainty[start:stop],
                "targets": target_tensor[start:stop],
                "target_mask": torch.isfinite(target_tensor[start:stop]),
                "sample_weight": patient_weight_tensor[start:stop],
                "persistence_glucose": persistence[start:stop],
                "auxiliary_targets": {
                    name: values[start:stop]
                    for name, values in auxiliary_targets.items()
                },
                "auxiliary_masks": {
                    name: values[start:stop]
                    for name, values in auxiliary_masks.items()
                },
                "patient_ids": ordered["patient_id"].astype(str).iloc[start:stop].tolist(),
                "cohort_ids": ordered["cohort_id"].astype(str).iloc[start:stop].tolist(),
                "participant_keys": participant_keys.iloc[start:stop].tolist(),
                "anchor_times": [value.isoformat() for value in ordered["anchor_time"].iloc[start:stop]],
            }
        )
    return batches


def attach_recorded_split(frame: pd.DataFrame, split_values: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    split = PatientSplit(
        train=tuple(str(value) for value in split_values["train"]),
        validation=tuple(str(value) for value in split_values["validation"]),
        test=tuple(str(value) for value in split_values["test"]),
    )
    expected = set(split.train) | set(split.validation) | set(split.test)
    raw_patient = frame["patient_id"].astype(str)
    composite = frame["cohort_id"].astype(str) + "::" + raw_patient
    identity = composite if set(composite) == expected else raw_patient
    if set(identity) != expected:
        raise ValueError("Evaluation patient IDs do not exactly match the recorded split.")
    lookup = {
        **{value: "train" for value in split.train},
        **{value: "validation" for value in split.validation},
        **{value: "test" for value in split.test},
    }
    result = frame.copy()
    result["participant_key"] = identity
    result["split"] = identity.map(lookup)
    return result


def predict_neural_batches(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, Any]],
    target_standardizer: Any,
    horizons_minutes: Sequence[int],
    *,
    hypoglycemia_threshold_mg_dl: float = 70.0,
    hyperglycemia_threshold_mg_dl: float = 180.0,
) -> pd.DataFrame:
    from .neural_training import inverse_transform_neuroglycemic_outputs
    from .evaluation import gaussian_mixture_quantile

    model.eval()
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    modality_names = tuple(model.modalities)
    with torch.no_grad():
        for batch in batches:
            outputs = model(
                {name: value.to(device) for name, value in batch["features"].items()},
                {
                    name: value.to(device)
                    for name, value in batch["feature_masks"].items()
                },
                batch["availability"].to(device),
                batch["quality"].to(device),
                batch["staleness"].to(device),
                batch.get("clock_uncertainty", torch.zeros_like(batch["staleness"])).to(
                    device
                ),
            )
            converted = inverse_transform_neuroglycemic_outputs(outputs, target_standardizer)
            # ``tolist`` avoids depending on PyTorch's optional NumPy ABI at
            # serving time; pandas can consume the resulting Python numbers.
            mean = converted["mixture_mean"].detach().cpu().tolist()
            standard_deviation = (
                converted["mixture_variance"].sqrt().detach().cpu().tolist()
            )
            expert_mean = converted["expert_mean"].detach().cpu().tolist()
            expert_scale = converted["expert_scale"].detach().cpu().tolist()
            sqrt_two = math.sqrt(2.0)
            hypoglycemia_cdf = 0.5 * (
                1.0
                + torch.erf(
                    (hypoglycemia_threshold_mg_dl - converted["expert_mean"])
                    / (sqrt_two * converted["expert_scale"])
                )
            )
            hyperglycemia_cdf = 0.5 * (
                1.0
                + torch.erf(
                    (hyperglycemia_threshold_mg_dl - converted["expert_mean"])
                    / (sqrt_two * converted["expert_scale"])
                )
            )
            expanded_weights = outputs.get(
                "fusion_weights_by_horizon",
                outputs["fusion_weights"].unsqueeze(-1).expand_as(converted["expert_mean"]),
            )
            hypoglycemia_probability = (
                expanded_weights * hypoglycemia_cdf
            ).sum(dim=1).masked_fill(
                outputs["abstained"].unsqueeze(-1), torch.nan
            ).detach().cpu().tolist()
            hyperglycemia_probability = (
                expanded_weights * (1.0 - hyperglycemia_cdf)
            ).sum(dim=1).masked_fill(
                outputs["abstained"].unsqueeze(-1), torch.nan
            ).detach().cpu().tolist()
            target = batch["targets"].detach().cpu().tolist()
            horizon_weights = outputs.get(
                "fusion_weights_by_horizon",
                outputs["fusion_weights"].unsqueeze(-1).expand_as(converted["expert_mean"]),
            )
            weights = horizon_weights.detach().cpu().tolist()
            abstained = outputs["abstained"].detach().cpu().tolist()
            auxiliary_values = {
                name: (
                    torch.sigmoid(values)
                    if getattr(model, "auxiliary_task_kinds", {}).get(name) == "binary"
                    else values
                ).detach().cpu().tolist()
                for name, values in outputs.get("auxiliary_outputs", {}).items()
            }
            auxiliary_targets = {
                name: values.detach().cpu().tolist()
                for name, values in batch.get("auxiliary_targets", {}).items()
            }
            auxiliary_masks = {
                name: values.detach().cpu().tolist()
                for name, values in batch.get("auxiliary_masks", {}).items()
            }
            for row_index, (patient_id, cohort_id, participant_key, anchor_time) in enumerate(
                zip(
                    batch["patient_ids"],
                    batch["cohort_ids"],
                    batch["participant_keys"],
                    batch["anchor_times"],
                    strict=True,
                )
            ):
                for horizon_index, horizon in enumerate(horizons_minutes):
                    actual = float(target[row_index][horizon_index])
                    row: dict[str, Any] = {
                        "patient_id": patient_id,
                        "cohort_id": cohort_id,
                        "participant_key": participant_key,
                        "anchor_time": anchor_time,
                        "horizon_minutes": int(horizon),
                        "target_glucose_mg_dl": actual,
                        "target_hypoglycemia": (
                            int(actual < hypoglycemia_threshold_mg_dl)
                            if math.isfinite(actual)
                            else None
                        ),
                        "target_hyperglycemia": (
                            int(actual > hyperglycemia_threshold_mg_dl)
                            if math.isfinite(actual)
                            else None
                        ),
                        "predicted_glucose_mg_dl": float(mean[row_index][horizon_index]),
                        "predicted_standard_deviation_mg_dl": float(
                            standard_deviation[row_index][horizon_index]
                        ),
                        "prediction_lower_mg_dl": gaussian_mixture_quantile(
                            [
                                expert_mean[row_index][index][horizon_index]
                                for index in range(len(modality_names))
                            ],
                            [
                                expert_scale[row_index][index][horizon_index]
                                for index in range(len(modality_names))
                            ],
                            [
                                weights[row_index][index][horizon_index]
                                for index in range(len(modality_names))
                            ],
                            0.025,
                        ),
                        "prediction_upper_mg_dl": gaussian_mixture_quantile(
                            [
                                expert_mean[row_index][index][horizon_index]
                                for index in range(len(modality_names))
                            ],
                            [
                                expert_scale[row_index][index][horizon_index]
                                for index in range(len(modality_names))
                            ],
                            [
                                weights[row_index][index][horizon_index]
                                for index in range(len(modality_names))
                            ],
                            0.975,
                        ),
                        "persistence_glucose_mg_dl": float(
                            batch["persistence_glucose"][row_index].item()
                        ),
                        "hypoglycemia_probability": float(
                            hypoglycemia_probability[row_index][horizon_index]
                        ),
                        "hyperglycemia_probability": float(
                            hyperglycemia_probability[row_index][horizon_index]
                        ),
                        "abstained": bool(abstained[row_index]),
                    }
                    row.update(
                        {
                            f"weight_{name}": float(
                                weights[row_index][index][horizon_index]
                            )
                            for index, name in enumerate(modality_names)
                        }
                    )
                    row.update(
                        {
                            f"expert_mean_{name}_mg_dl": float(
                                expert_mean[row_index][index][horizon_index]
                            )
                            for index, name in enumerate(modality_names)
                        }
                    )
                    row.update(
                        {
                            f"expert_sd_{name}_mg_dl": float(
                                expert_scale[row_index][index][horizon_index]
                            )
                            for index, name in enumerate(modality_names)
                        }
                    )
                    row.update(
                        {
                            f"auxiliary_{name}": float(values[row_index])
                            for name, values in auxiliary_values.items()
                        }
                    )
                    row.update(
                        {
                            f"target_auxiliary_{name}": (
                                float(values[row_index])
                                if auxiliary_masks[name][row_index]
                                else float("nan")
                            )
                            for name, values in auxiliary_targets.items()
                        }
                    )
                    row.update(
                        {
                            f"auxiliary_kind_{name}": str(kind)
                            for name, kind in getattr(
                                model, "auxiliary_task_kinds", {}
                            ).items()
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def modality_ablation_predictions(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, Any]],
    target_standardizer: Any,
    horizons_minutes: Sequence[int],
    *,
    hypoglycemia_threshold_mg_dl: float = 70.0,
    hyperglycemia_threshold_mg_dl: float = 180.0,
) -> dict[str, pd.DataFrame]:
    """Run paired test rows with one or all modalities removed.

    This changes only availability and observation masks at inference time; it
    never fabricates a replacement feature. Every scenario therefore retains
    identical patients, anchors, and targets for a paired comparison.
    """

    modalities = tuple(model.modalities)

    def predict(masked_modalities: set[str]) -> pd.DataFrame:
        scenario_batches: list[dict[str, Any]] = []
        for batch in batches:
            copied = dict(batch)
            availability = batch["availability"].clone()
            feature_masks = {
                name: values.clone()
                for name, values in batch["feature_masks"].items()
            }
            for name in masked_modalities:
                index = modalities.index(name)
                availability[:, index] = False
                feature_masks[name].fill_(False)
            copied["availability"] = availability
            copied["feature_masks"] = feature_masks
            scenario_batches.append(copied)
        return predict_neural_batches(
            model,
            scenario_batches,
            target_standardizer,
            horizons_minutes,
            hypoglycemia_threshold_mg_dl=hypoglycemia_threshold_mg_dl,
            hyperglycemia_threshold_mg_dl=hyperglycemia_threshold_mg_dl,
        )

    return {
        "observed_modalities": predict(set()),
        **{f"without_{name}": predict({name}) for name in modalities},
        "all_unavailable": predict(set(modalities)),
    }


def modality_ablation_table(
    scenarios: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    from .evaluation import missing_modality_ablation_summary

    reference = scenarios.get("observed_modalities")
    if reference is None:
        raise KeyError("Ablations require the observed_modalities reference.")
    rows: list[pd.DataFrame] = []
    for horizon in sorted(reference["horizon_minutes"].unique()):
        horizon_scenarios = {
            name: frame.loc[frame["horizon_minutes"].eq(horizon)].reset_index(
                drop=True
            )
            for name, frame in scenarios.items()
        }
        summary = missing_modality_ablation_summary(
            horizon_scenarios,
            "predicted_glucose_mg_dl",
            reference_scenario="observed_modalities",
        )
        summary.insert(0, "horizon_minutes", int(horizon))
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def glucose_forecast_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    from .evaluation import (
        binary_event_metrics,
        gaussian_mixture_nll,
        neural_regression_metrics,
        prediction_interval_metrics,
        paired_patient_bootstrap_delta,
    )

    metrics: dict[str, Any] = {"by_horizon": {}}
    for horizon, group in predictions.groupby("horizon_minutes", sort=True):
        group = group.reset_index(drop=True)
        modality_names = tuple(
            column.removeprefix("weight_")
            for column in group.columns
            if column.startswith("weight_")
        )
        if not modality_names:
            raise ValueError("Predictions do not contain learned modality weights.")
        expert_means = np.stack(
            [
                group[f"expert_mean_{name}_mg_dl"].to_numpy(float)
                for name in modality_names
            ],
            axis=1,
        )[:, :, None]
        expert_scales = np.stack(
            [
                group[f"expert_sd_{name}_mg_dl"].to_numpy(float)
                for name in modality_names
            ],
            axis=1,
        )[:, :, None]
        weights = group[[f"weight_{name}" for name in modality_names]].to_numpy(float)
        targets = group["target_glucose_mg_dl"].to_numpy(float)[:, None]
        patient_column = (
            "participant_key" if "participant_key" in group else "patient_id"
        )
        regression = neural_regression_metrics(
            group,
            "predicted_glucose_mg_dl",
            patient_column=patient_column,
        )
        intervals = prediction_interval_metrics(group)
        hypoglycemia = binary_event_metrics(
            group,
            target_column="target_hypoglycemia",
            probability_column="hypoglycemia_probability",
        )
        hyperglycemia = binary_event_metrics(
            group,
            target_column="target_hyperglycemia",
            probability_column="hyperglycemia_probability",
        )
        metrics["by_horizon"][str(int(horizon))] = {
            **regression,
            "gaussian_mixture_nll": gaussian_mixture_nll(
                targets, expert_means, expert_scales, weights
            ),
            "prediction_interval_95_coverage": intervals["interval_coverage"],
            "mean_prediction_interval_95_width_mg_dl": intervals[
                "mean_interval_width_mg_dl"
            ],
            "hypoglycemia_event": hypoglycemia,
            "hyperglycemia_event": hyperglycemia,
        }
        if np.isfinite(group["persistence_glucose_mg_dl"].to_numpy(float)).any():
            persistence = neural_regression_metrics(
                group,
                "persistence_glucose_mg_dl",
                patient_column=patient_column,
            )
            metrics["by_horizon"][str(int(horizon))]["persistence_baseline"] = persistence
            metrics["by_horizon"][str(int(horizon))][
                "paired_patient_bootstrap_model_minus_persistence"
            ] = paired_patient_bootstrap_delta(
                group,
                model_column="predicted_glucose_mg_dl",
                baseline_column="persistence_glucose_mg_dl",
                patient_column=patient_column,
                seed=42 + int(horizon),
                replicates=1000,
            )
    auxiliary_names = tuple(
        column.removeprefix("auxiliary_")
        for column in predictions.columns
        if column.startswith("auxiliary_")
        and not column.startswith("auxiliary_kind_")
        and f"target_{column}" in predictions
    )
    if auxiliary_names:
        patient_column = (
            "participant_key" if "participant_key" in predictions else "patient_id"
        )
        one_per_window = predictions.drop_duplicates(
            [patient_column, "anchor_time"], keep="first"
        )
        metrics["auxiliary_tasks"] = {}
        for name in auxiliary_names:
            prediction_column = f"auxiliary_{name}"
            target_name = f"target_auxiliary_{name}"
            kind_values = one_per_window[f"auxiliary_kind_{name}"].dropna().unique()
            if len(kind_values) != 1:
                raise ValueError(f"Auxiliary task {name!r} has inconsistent kind metadata.")
            if kind_values[0] == "binary":
                task_metrics = binary_event_metrics(
                    one_per_window,
                    target_column=target_name,
                    probability_column=prediction_column,
                )
            elif kind_values[0] == "continuous":
                target_values = one_per_window[target_name].to_numpy(float)
                predicted_values = one_per_window[prediction_column].to_numpy(float)
                valid = np.isfinite(target_values) & np.isfinite(predicted_values)
                if not valid.any():
                    raise ValueError(f"Auxiliary task {name!r} has no held-out labels.")
                errors = predicted_values[valid] - target_values[valid]
                per_patient = pd.DataFrame(
                    {
                        "patient": one_per_window.loc[valid, patient_column].astype(str),
                        "absolute_error": np.abs(errors),
                        "squared_error": errors**2,
                    }
                ).groupby("patient", sort=False).mean()
                task_metrics = {
                    "n_predictions": int(valid.sum()),
                    "n_patients": int(per_patient.shape[0]),
                    "mae": float(np.abs(errors).mean()),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "patient_macro_mae": float(per_patient["absolute_error"].mean()),
                    "patient_macro_rmse": float(
                        np.sqrt(per_patient["squared_error"]).mean()
                    ),
                }
            else:
                raise ValueError(f"Unsupported auxiliary task kind {kind_values[0]!r}.")
            metrics["auxiliary_tasks"][name] = task_metrics
    return metrics
