"""Turn a same-participant LabRecorder session into causal neural windows.

The adapter uses only observed LSL/XDF samples. EEG band powers and wearable
summaries are deterministic signal features; glucose labels are taken from a
recorded CGM reference stream at later timestamps and are never interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import welch

from .interoperability import convert_measurement_unit
from .lsl import audit_xdf
from .neural_dataset import target_column, target_time_column


EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

LSL_EEG_FEATURES = tuple(
    [f"eeg_{band}_mean" for band in EEG_BANDS]
    + [f"eeg_{band}_variability" for band in EEG_BANDS]
    + ["eeg_theta_alpha_ratio", "eeg_beta_alpha_ratio"]
)


@dataclass(frozen=True)
class EEGSource:
    source_id: str
    channels: tuple[str, ...]
    clock_uncertainty_ms: float = 0.0


@dataclass(frozen=True)
class WearableSummary:
    source_id: str
    column: str
    feature_name: str
    aggregation: str
    clock_uncertainty_ms: float = 0.0


@dataclass(frozen=True)
class CGMReference:
    source_id: str
    column: str
    unit: str
    clock_uncertainty_ms: float = 0.0


@dataclass(frozen=True)
class LSLWindowConfig:
    patient_id: str
    cohort_id: str
    session_id: str
    session_start_utc: str
    horizons_minutes: tuple[int, ...]
    lookback_seconds: int
    stride_seconds: int
    target_tolerance_minutes: float
    minimum_eeg_coverage: float
    eeg_sources: tuple[EEGSource, ...]
    wearable_summaries: tuple[WearableSummary, ...]
    cgm_reference: CGMReference


def _nonempty_text(value: Any, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    return result


def load_lsl_window_config(path: Path) -> LSLWindowConfig:
    """Load the strict device-to-feature map for one LabRecorder session."""

    values = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "patient_id",
        "cohort_id",
        "session_id",
        "session_start_utc",
        "horizons_minutes",
        "lookback_seconds",
        "stride_seconds",
        "target_tolerance_minutes",
        "minimum_eeg_coverage",
        "eeg_sources",
        "wearable_summaries",
        "cgm_reference",
    }
    if not isinstance(values, dict) or set(values) != required:
        missing = required - set(values) if isinstance(values, dict) else required
        extra = set(values) - required if isinstance(values, dict) else set()
        raise ValueError(
            f"LSL window config has missing={sorted(missing)} and extra={sorted(extra)}."
        )
    if values["schema_version"] != "neuroglycemic-lsl-window-v1":
        raise ValueError("Unsupported LSL window config schema_version.")
    session_start = pd.Timestamp(values["session_start_utc"])
    if session_start.tzinfo is None:
        raise ValueError("session_start_utc must include a timezone offset.")
    horizons = tuple(int(value) for value in values["horizons_minutes"])
    if not horizons or any(value <= 0 for value in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons_minutes must contain unique positive integers.")
    lookback = int(values["lookback_seconds"])
    stride = int(values["stride_seconds"])
    tolerance = float(values["target_tolerance_minutes"])
    minimum_coverage = float(values["minimum_eeg_coverage"])
    if lookback <= 0 or stride <= 0:
        raise ValueError("lookback_seconds and stride_seconds must be positive.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("target_tolerance_minutes must be finite and positive.")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_eeg_coverage must be in (0, 1].")

    eeg_sources: list[EEGSource] = []
    for index, item in enumerate(values["eeg_sources"]):
        if not isinstance(item, dict) or set(item) != {
            "source_id",
            "channels",
            "clock_uncertainty_ms",
        }:
            raise ValueError(f"eeg_sources[{index}] has an invalid schema.")
        channels = tuple(_nonempty_text(value, name="EEG channel") for value in item["channels"])
        uncertainty = float(item["clock_uncertainty_ms"])
        if not channels or len(channels) != len(set(channels)):
            raise ValueError("Every EEG source needs unique channel labels.")
        if not math.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError("EEG clock uncertainty must be finite and non-negative.")
        eeg_sources.append(
            EEGSource(
                source_id=_nonempty_text(item["source_id"], name="EEG source_id"),
                channels=channels,
                clock_uncertainty_ms=uncertainty,
            )
        )

    wearable: list[WearableSummary] = []
    allowed_aggregations = {"mean", "std", "last", "min", "max", "sum", "delta", "slope", "rmssd"}
    for index, item in enumerate(values["wearable_summaries"]):
        if not isinstance(item, dict) or set(item) != {
            "source_id",
            "column",
            "feature_name",
            "aggregation",
            "clock_uncertainty_ms",
        }:
            raise ValueError(f"wearable_summaries[{index}] has an invalid schema.")
        feature_name = _nonempty_text(item["feature_name"], name="wearable feature_name")
        aggregation = _nonempty_text(item["aggregation"], name="wearable aggregation")
        uncertainty = float(item["clock_uncertainty_ms"])
        if not feature_name.startswith("wearable_"):
            raise ValueError("Every wearable feature_name must start with wearable_.")
        if aggregation not in allowed_aggregations:
            raise ValueError(f"Unsupported wearable aggregation {aggregation!r}.")
        if not math.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError("Wearable clock uncertainty must be finite and non-negative.")
        wearable.append(
            WearableSummary(
                source_id=_nonempty_text(item["source_id"], name="wearable source_id"),
                column=_nonempty_text(item["column"], name="wearable column"),
                feature_name=feature_name,
                aggregation=aggregation,
                clock_uncertainty_ms=uncertainty,
            )
        )
    if len({value.feature_name for value in wearable}) != len(wearable):
        raise ValueError("Wearable feature names must be unique.")
    cgm = values["cgm_reference"]
    if not isinstance(cgm, dict) or set(cgm) != {
        "source_id",
        "column",
        "unit",
        "clock_uncertainty_ms",
    }:
        raise ValueError("cgm_reference has an invalid schema.")
    cgm_uncertainty = float(cgm["clock_uncertainty_ms"])
    if not math.isfinite(cgm_uncertainty) or cgm_uncertainty < 0:
        raise ValueError("CGM clock uncertainty must be finite and non-negative.")
    if not eeg_sources and not wearable:
        raise ValueError("At least one EEG or wearable input is required.")
    return LSLWindowConfig(
        patient_id=_nonempty_text(values["patient_id"], name="patient_id"),
        cohort_id=_nonempty_text(values["cohort_id"], name="cohort_id"),
        session_id=_nonempty_text(values["session_id"], name="session_id"),
        session_start_utc=session_start.tz_convert("UTC").isoformat(),
        horizons_minutes=horizons,
        lookback_seconds=lookback,
        stride_seconds=stride,
        target_tolerance_minutes=tolerance,
        minimum_eeg_coverage=minimum_coverage,
        eeg_sources=tuple(eeg_sources),
        wearable_summaries=tuple(wearable),
        cgm_reference=CGMReference(
            source_id=_nonempty_text(cgm["source_id"], name="CGM source_id"),
            column=_nonempty_text(cgm["column"], name="CGM column"),
            unit=_nonempty_text(cgm["unit"], name="CGM unit"),
            clock_uncertainty_ms=cgm_uncertainty,
        ),
    )


def _frames_by_source_id(
    audit: pd.DataFrame, frames: Mapping[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    if len(audit) != len(frames):
        raise ValueError("XDF audit and stream-frame counts do not match.")
    frame_values = list(frames.values())
    result: dict[str, pd.DataFrame] = {}
    metadata: dict[str, pd.Series] = {}
    for index, row in audit.reset_index(drop=True).iterrows():
        source_id = str(row["source_id"]).strip()
        if not source_id:
            raise ValueError("Every training XDF stream needs a non-empty source_id.")
        if source_id in result:
            raise ValueError(f"Duplicate XDF source_id {source_id!r}.")
        result[source_id] = frame_values[index]
        metadata[source_id] = row
    return result, metadata


def _slice(frame: pd.DataFrame, start: float, stop: float) -> pd.DataFrame:
    timestamp = pd.to_numeric(frame["lsl_timestamp"], errors="coerce")
    return frame.loc[timestamp.ge(start) & timestamp.lt(stop)]


def _sample_rate(frame: pd.DataFrame, nominal_rate: float) -> float:
    if math.isfinite(nominal_rate) and nominal_rate > 0:
        return float(nominal_rate)
    timestamp = pd.to_numeric(frame["lsl_timestamp"], errors="coerce").dropna().to_numpy(float)
    differences = np.diff(np.sort(timestamp))
    differences = differences[differences > 0]
    if not len(differences):
        return 0.0
    return float(1.0 / np.median(differences))


def _eeg_band_features(values: np.ndarray, sampling_rate: float) -> dict[str, float] | None:
    if sampling_rate < 2 * EEG_BANDS["gamma"][1] or values.ndim != 2:
        return None
    per_channel: list[dict[str, float]] = []
    for column in range(values.shape[1]):
        signal = values[:, column]
        signal = signal[np.isfinite(signal)]
        if len(signal) < max(32, int(round(2.0 * sampling_rate))):
            continue
        signal = signal - float(np.mean(signal))
        frequency, density = welch(
            signal,
            fs=sampling_rate,
            nperseg=min(len(signal), max(64, int(round(4.0 * sampling_rate)))),
        )
        total_mask = (frequency >= 1.0) & (frequency <= 45.0)
        total = float(trapezoid(density[total_mask], x=frequency[total_mask]))
        if not math.isfinite(total) or total <= 0:
            continue
        channel_values: dict[str, float] = {}
        for band, (low, high) in EEG_BANDS.items():
            mask = (frequency >= low) & (frequency < high)
            power = float(trapezoid(density[mask], x=frequency[mask]))
            channel_values[band] = max(power / total, np.finfo(float).eps)
        per_channel.append(channel_values)
    if not per_channel:
        return None
    result: dict[str, float] = {}
    for band in EEG_BANDS:
        values_by_band = np.asarray([value[band] for value in per_channel], dtype=float)
        result[f"eeg_{band}_mean"] = float(np.mean(values_by_band))
        result[f"eeg_{band}_variability"] = float(np.std(values_by_band))
    result["eeg_theta_alpha_ratio"] = float(
        result["eeg_theta_mean"] / max(result["eeg_alpha_mean"], np.finfo(float).eps)
    )
    result["eeg_beta_alpha_ratio"] = float(
        result["eeg_beta_mean"] / max(result["eeg_alpha_mean"], np.finfo(float).eps)
    )
    return result


def _aggregate(values: np.ndarray, times: np.ndarray, method: str) -> float:
    valid = np.isfinite(values) & np.isfinite(times)
    values, times = values[valid], times[valid]
    if not len(values):
        return float("nan")
    if method == "mean":
        return float(np.mean(values))
    if method == "std":
        return float(np.std(values))
    if method == "last":
        return float(values[-1])
    if method == "min":
        return float(np.min(values))
    if method == "max":
        return float(np.max(values))
    if method == "sum":
        return float(np.sum(values))
    if method == "delta":
        return float(values[-1] - values[0]) if len(values) >= 2 else float("nan")
    if method == "rmssd":
        return float(np.sqrt(np.mean(np.diff(values) ** 2))) if len(values) >= 2 else float("nan")
    if method == "slope":
        if len(values) < 2 or float(np.ptp(times)) <= 0:
            return float("nan")
        return float(np.polyfit(times - times[0], values, 1)[0])
    raise ValueError(f"Unsupported aggregation {method!r}.")


def _target_at(
    cgm_times: np.ndarray,
    cgm_values: np.ndarray,
    desired: float,
    tolerance_seconds: float,
) -> tuple[float, float] | None:
    position = int(np.searchsorted(cgm_times, desired))
    candidates = [value for value in (position - 1, position) if 0 <= value < len(cgm_times)]
    if not candidates:
        return None
    best = min(candidates, key=lambda value: abs(float(cgm_times[value]) - desired))
    if abs(float(cgm_times[best]) - desired) > tolerance_seconds:
        return None
    return float(cgm_values[best]), float(cgm_times[best])


def build_lsl_glucose_windows(
    audit: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    config: LSLWindowConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build same-session EEG/wearable windows with future CGM labels."""

    by_source, metadata = _frames_by_source_id(audit, frames)
    required_sources = {
        *[value.source_id for value in config.eeg_sources],
        *[value.source_id for value in config.wearable_summaries],
        config.cgm_reference.source_id,
    }
    missing = required_sources - set(by_source)
    if missing:
        raise ValueError(f"XDF is missing configured source_id values: {sorted(missing)}")
    for source_id, frame in by_source.items():
        if "lsl_timestamp" not in frame:
            raise ValueError(f"XDF stream {source_id!r} has no lsl_timestamp column.")

    cgm = by_source[config.cgm_reference.source_id]
    if config.cgm_reference.column not in cgm:
        raise ValueError("Configured CGM channel is absent from the XDF stream.")
    cgm_times = pd.to_numeric(cgm["lsl_timestamp"], errors="coerce").to_numpy(float)
    cgm_values = pd.to_numeric(cgm[config.cgm_reference.column], errors="coerce").to_numpy(float)
    cgm_values = np.asarray(
        [
            convert_measurement_unit(
                value,
                source_unit=config.cgm_reference.unit,
                target_unit="mg/dL",
            )
            if math.isfinite(value)
            else np.nan
            for value in cgm_values
        ],
        dtype=float,
    )
    keep_cgm = np.isfinite(cgm_times) & np.isfinite(cgm_values) & (cgm_values > 0)
    cgm_times, cgm_values = cgm_times[keep_cgm], cgm_values[keep_cgm]
    order = np.argsort(cgm_times, kind="stable")
    cgm_times, cgm_values = cgm_times[order], cgm_values[order]
    if len(cgm_times) < 3 or bool(np.any(np.diff(cgm_times) <= 0)):
        raise ValueError("CGM reference needs at least three strictly increasing samples.")

    input_sources = required_sources - {config.cgm_reference.source_id}
    input_start = min(
        float(pd.to_numeric(by_source[source]["lsl_timestamp"], errors="coerce").min())
        for source in input_sources
    )
    input_stop = max(
        float(pd.to_numeric(by_source[source]["lsl_timestamp"], errors="coerce").max())
        for source in input_sources
    )
    first_anchor = max(input_start + config.lookback_seconds, float(cgm_times[0]))
    last_anchor = min(
        input_stop,
        float(cgm_times[-1]) - max(config.horizons_minutes) * 60.0,
    )
    if last_anchor < first_anchor:
        raise ValueError("No causal input/CGM forecast interval overlaps in this XDF session.")
    anchors = np.arange(
        first_anchor,
        last_anchor + 1e-9,
        float(config.stride_seconds),
        dtype=float,
    )
    origin_lsl = min(
        float(pd.to_numeric(value["lsl_timestamp"], errors="coerce").min())
        for value in by_source.values()
    )
    session_start = pd.Timestamp(config.session_start_utc)

    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        start = anchor - float(config.lookback_seconds)
        row: dict[str, Any] = {
            "patient_id": config.patient_id,
            "cohort_id": config.cohort_id,
            "session_id": config.session_id,
            "anchor_time": session_start + pd.to_timedelta(anchor - origin_lsl, unit="s"),
        }

        eeg_feature_sets: list[dict[str, float]] = []
        eeg_quality: list[float] = []
        eeg_last_samples: list[float] = []
        for source in config.eeg_sources:
            window = _slice(by_source[source.source_id], start, anchor)
            absent_channels = set(source.channels) - set(window.columns)
            if absent_channels:
                raise ValueError(
                    f"EEG source {source.source_id!r} lacks channels: {sorted(absent_channels)}"
                )
            rate = _sample_rate(
                by_source[source.source_id],
                float(metadata[source.source_id]["nominal_rate_hz"]),
            )
            expected = max(1.0, config.lookback_seconds * rate)
            coverage = min(1.0, len(window) / expected)
            if coverage < config.minimum_eeg_coverage:
                continue
            features = _eeg_band_features(
                window[list(source.channels)].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                rate,
            )
            if features is not None:
                eeg_feature_sets.append(features)
                eeg_quality.append(coverage)
                eeg_last_samples.append(
                    float(pd.to_numeric(window["lsl_timestamp"], errors="coerce").max())
                )
        eeg_available = bool(eeg_feature_sets)
        for feature in LSL_EEG_FEATURES:
            row[feature] = (
                float(np.mean([value[feature] for value in eeg_feature_sets]))
                if eeg_available
                else np.nan
            )
        row.update(
            {
                "eeg_available": eeg_available,
                "eeg_quality": float(np.mean(eeg_quality)) if eeg_available else 0.0,
                "eeg_staleness_minutes": (
                    float(max(0.0, anchor - max(eeg_last_samples)) / 60.0)
                    if eeg_available
                    else 0.0
                ),
                "eeg_clock_uncertainty_ms": (
                    max(value.clock_uncertainty_ms for value in config.eeg_sources)
                    if eeg_available
                    else 0.0
                ),
                "eeg_patient_id": config.patient_id if eeg_available else None,
                "eeg_cohort_id": config.cohort_id if eeg_available else None,
                "eeg_anchor_time": row["anchor_time"] if eeg_available else pd.NaT,
                "eeg_available_time": row["anchor_time"] if eeg_available else pd.NaT,
            }
        )

        wearable_last_samples: list[float] = []
        wearable_observed = 0
        for specification in config.wearable_summaries:
            stream = by_source[specification.source_id]
            if specification.column not in stream:
                raise ValueError(
                    f"Wearable source {specification.source_id!r} lacks "
                    f"column {specification.column!r}."
                )
            window = _slice(stream, start, anchor)
            values = pd.to_numeric(window[specification.column], errors="coerce").to_numpy(float)
            times = pd.to_numeric(window["lsl_timestamp"], errors="coerce").to_numpy(float)
            summary = _aggregate(values, times, specification.aggregation)
            row[specification.feature_name] = summary
            if math.isfinite(summary):
                wearable_observed += 1
                wearable_last_samples.append(float(np.max(times[np.isfinite(times)])))
        wearable_available = wearable_observed > 0
        row.update(
            {
                "wearable_available": wearable_available,
                "wearable_quality": (
                    wearable_observed / len(config.wearable_summaries)
                    if config.wearable_summaries
                    else 0.0
                ),
                "wearable_staleness_minutes": (
                    float(max(0.0, anchor - max(wearable_last_samples)) / 60.0)
                    if wearable_available
                    else 0.0
                ),
                "wearable_clock_uncertainty_ms": (
                    max(value.clock_uncertainty_ms for value in config.wearable_summaries)
                    if wearable_available
                    else 0.0
                ),
                "wearable_patient_id": config.patient_id if wearable_available else None,
                "wearable_cohort_id": config.cohort_id if wearable_available else None,
                "wearable_anchor_time": row["anchor_time"] if wearable_available else pd.NaT,
                "wearable_available_time": row["anchor_time"] if wearable_available else pd.NaT,
            }
        )
        if not (eeg_available or wearable_available):
            continue
        for horizon in config.horizons_minutes:
            target = _target_at(
                cgm_times,
                cgm_values,
                anchor + horizon * 60.0,
                config.target_tolerance_minutes * 60.0,
            )
            row[target_column(horizon)] = target[0] if target else np.nan
            row[target_time_column(horizon)] = (
                session_start + pd.to_timedelta(target[1] - origin_lsl, unit="s")
                if target
                else pd.NaT
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No synchronized LSL windows survived the configured quality gates.")
    for horizon in config.horizons_minutes:
        if result[target_column(horizon)].notna().sum() < 2:
            raise ValueError(f"Fewer than two observed {horizon}-minute CGM labels survived.")
    audit_result = {
        "patient_id": config.patient_id,
        "cohort_id": config.cohort_id,
        "session_id": config.session_id,
        "source_ids": sorted(required_sources),
        "eeg_channel_map": {
            source.source_id: list(source.channels) for source in config.eeg_sources
        },
        "eeg_channel_counts": {
            source.source_id: len(source.channels) for source in config.eeg_sources
        },
        "windows": int(len(result)),
        "eeg_available_windows": int(result["eeg_available"].sum()),
        "wearable_available_windows": int(result["wearable_available"].sum()),
        "horizon_label_counts": {
            str(value): int(result[target_column(value)].notna().sum())
            for value in config.horizons_minutes
        },
        "feature_registry": {
            "eeg": list(LSL_EEG_FEATURES),
            "wearable": [value.feature_name for value in config.wearable_summaries],
        },
    }
    return result, audit_result


def build_lsl_glucose_windows_from_xdf(
    xdf_path: Path, config: LSLWindowConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    audit, frames = audit_xdf(xdf_path)
    windows, build_audit = build_lsl_glucose_windows(audit, frames, config)
    return windows, audit, build_audit
