"""Device-neutral sensor schemas used by offline files and LSL streams."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


DEVICE_SIGNAL_CONTRACTS: dict[str, dict[str, str]] = {
    "muse": {
        "delta": "log10_power",
        "theta": "log10_power",
        "alpha": "log10_power",
        "beta": "log10_power",
        "gamma": "log10_power",
        "eeg": "microvolt",
    },
    "emotiv": {
        "eeg": "microvolt",
        "delta": "power",
        "theta": "power",
        "alpha": "power",
        "beta": "power",
        "gamma": "power",
        "contact_quality": "score",
    },
    "galea": {
        "eeg": "microvolt",
        "eda": "microsiemens",
        "ppg": "arbitrary_unit",
        "emg": "microvolt",
        "eog": "microvolt",
    },
    "empatica": {
        "bvp": "arbitrary_unit",
        "eda": "microsiemens",
        "skin_temperature": "celsius",
        "accelerometer": "g",
    },
    "ihealth": {
        "heart_rate": "beats_per_minute",
        "systolic_blood_pressure": "mmHg",
        "diastolic_blood_pressure": "mmHg",
        "spo2": "percent",
        "step_count": "count",
        "weight": "kilogram",
    },
    "pulse": {
        "ppg": "arbitrary_unit",
        "heart_rate": "beats_per_minute",
        "spo2": "percent",
    },
    "cgm": {"glucose": "mg/dL"},
    "ehr": {
        "glucose": "mg/dL",
        "diagnosis": "code",
        "medication": "code",
        "laboratory": "source_unit",
    },
}


@dataclass(frozen=True)
class CanonicalSignalRecord:
    patient_id: str
    session_id: str
    device: str
    signal: str
    channel: str
    value: float
    unit: str
    event_time_utc: str
    sampling_rate_hz: float
    quality: float
    source: str
    lsl_timestamp: float | None = None
    available_time_utc: str | None = None
    clock_uncertainty_ms: float = 0.0
    sequence_number: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_iso(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def validate_device_signal(device: str, signal: str, unit: str) -> None:
    device_key = device.strip().lower()
    signal_key = signal.strip().lower()
    if device_key not in DEVICE_SIGNAL_CONTRACTS:
        raise ValueError(f"Unsupported device {device!r}.")
    expected = DEVICE_SIGNAL_CONTRACTS[device_key].get(signal_key)
    if expected is None:
        raise ValueError(f"Unsupported signal {signal!r} for {device!r}.")
    if expected != "source_unit" and unit != expected:
        raise ValueError(
            f"Unit mismatch for {device_key}.{signal_key}: expected {expected!r}, got {unit!r}."
        )


def convert_measurement_unit(
    value: float, *, source_unit: str, target_unit: str
) -> float:
    """Apply a small, explicit clinical unit conversion registry.

    Unknown conversions fail closed. Device adapters must never relabel a value
    with a target unit without numerically converting it.
    """

    source = source_unit.strip().lower()
    target = target_unit.strip().lower()
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Measurement values must be finite.")
    if source == target:
        return numeric
    conversions = {
        ("mmol/l", "mg/dl"): lambda item: item * 18.0182,
        ("mg/dl", "mmol/l"): lambda item: item / 18.0182,
        ("fahrenheit", "celsius"): lambda item: (item - 32.0) * 5.0 / 9.0,
        ("celsius", "fahrenheit"): lambda item: item * 9.0 / 5.0 + 32.0,
        ("kpa", "mmhg"): lambda item: item * 7.50062,
        ("fraction", "percent"): lambda item: item * 100.0,
    }
    conversion = conversions.get((source, target))
    if conversion is None:
        raise ValueError(f"Unsupported unit conversion: {source_unit!r} -> {target_unit!r}.")
    converted = float(conversion(numeric))
    if not math.isfinite(converted):
        raise ValueError("Unit conversion produced a non-finite value.")
    return converted


def canonicalize_wide_stream(
    frame: pd.DataFrame,
    *,
    patient_id: str,
    session_id: str,
    device: str,
    signal: str,
    unit: str,
    source_unit: str | None = None,
    timestamp_column: str,
    channel_columns: Mapping[str, str],
    sampling_rate_hz: float,
    source: str,
    quality_column: str | None = None,
    unix_timestamps: bool = False,
    lsl_timestamp_column: str | None = None,
    available_time_column: str | None = None,
    clock_uncertainty_ms: float = 0.0,
    sequence_column: str | None = None,
) -> pd.DataFrame:
    """Convert an adapter-specific wide table to one canonical long schema."""
    validate_device_signal(device, signal, unit)
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive.")
    required = {timestamp_column, *channel_columns.keys()}
    if quality_column:
        required.add(quality_column)
    if lsl_timestamp_column:
        required.add(lsl_timestamp_column)
    if available_time_column:
        required.add(available_time_column)
    if sequence_column:
        required.add(sequence_column)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing stream columns: {sorted(missing)}")

    if not math.isfinite(clock_uncertainty_ms) or clock_uncertainty_ms < 0:
        raise ValueError("clock_uncertainty_ms must be finite and non-negative.")
    rows: list[dict[str, Any]] = []
    # ``itertuples`` rewrites vendor column names containing spaces, hyphens,
    # parentheses, or duplicate-looking identifiers. Record dictionaries retain
    # the exact adapter contract (for example ``SpO2 (%)`` or ``BP-systolic``).
    for row_index, values in enumerate(frame.to_dict(orient="records")):
        raw_time = values[timestamp_column]
        event_time = (
            datetime.fromtimestamp(float(raw_time), tz=timezone.utc).isoformat()
            if unix_timestamps
            else _utc_iso(raw_time)
        )
        quality = float(values[quality_column]) if quality_column else 1.0
        if not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("Every quality value must be in [0, 1].")
        for input_column, canonical_channel in channel_columns.items():
            value = float(values[input_column])
            if not math.isfinite(value):
                continue
            converted = convert_measurement_unit(
                value,
                source_unit=source_unit or unit,
                target_unit=unit,
            )
            rows.append(
                CanonicalSignalRecord(
                    patient_id=patient_id,
                    session_id=session_id,
                    device=device.lower(),
                    signal=signal.lower(),
                    channel=canonical_channel,
                    value=converted,
                    unit=unit,
                    event_time_utc=event_time,
                    sampling_rate_hz=float(sampling_rate_hz),
                    quality=quality,
                    source=source,
                    lsl_timestamp=(
                        float(values[lsl_timestamp_column])
                        if lsl_timestamp_column
                        else None
                    ),
                    available_time_utc=(
                        _utc_iso(values[available_time_column])
                        if available_time_column
                        else event_time
                    ),
                    clock_uncertainty_ms=float(clock_uncertainty_ms),
                    sequence_number=(
                        int(values[sequence_column])
                        if sequence_column
                        else row_index
                    ),
                ).as_dict()
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No finite sensor values were produced.")
    return result.sort_values(["event_time_utc", "channel"], ignore_index=True)


def audit_canonical_streams(frame: pd.DataFrame) -> pd.DataFrame:
    """Report sampling, gaps, quality, and duration before feature extraction."""
    required = set(CanonicalSignalRecord.__dataclass_fields__)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical stream is missing columns: {sorted(missing)}")
    values = frame.copy()
    values["event_time_utc"] = pd.to_datetime(values["event_time_utc"], utc=True)
    rows: list[dict[str, Any]] = []
    group_columns = ["patient_id", "session_id", "device", "signal", "channel"]
    for keys, group in values.groupby(group_columns, sort=False):
        ordered = group.sort_values("event_time_utc")
        seconds = ordered["event_time_utc"].astype("int64").to_numpy() / 1e9
        differences = np.diff(seconds)
        expected_rate = float(ordered["sampling_rate_hz"].median())
        duration = float(max(0.0, seconds[-1] - seconds[0]))
        expected_samples = duration * expected_rate + 1.0
        coverage = min(1.0, len(ordered) / max(expected_samples, 1.0))
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "rows": len(ordered),
                "start_utc": ordered["event_time_utc"].iloc[0].isoformat(),
                "stop_utc": ordered["event_time_utc"].iloc[-1].isoformat(),
                "duration_seconds": duration,
                "sampling_rate_hz": expected_rate,
                "median_observed_rate_hz": (
                    float(1.0 / np.median(differences[differences > 0]))
                    if np.any(differences > 0)
                    else float("nan")
                ),
                "largest_gap_seconds": float(np.max(differences)) if len(differences) else 0.0,
                "duplicate_timestamps": int(np.sum(differences == 0)),
                "coverage_fraction": coverage,
                "mean_quality": float(ordered["quality"].mean()),
            }
        )
    return pd.DataFrame(rows)


def overlap_audit(streams: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Find the same-patient/session UTC intersection across modalities."""
    if not streams:
        raise ValueError("At least one stream is required.")
    identities: set[tuple[str, str]] = set()
    starts: dict[str, pd.Timestamp] = {}
    stops: dict[str, pd.Timestamp] = {}
    for name, stream in streams.items():
        if stream.empty:
            raise ValueError(f"Stream {name!r} is empty.")
        pairs = set(zip(stream["patient_id"], stream["session_id"], strict=False))
        if len(pairs) != 1:
            raise ValueError(f"Stream {name!r} must contain exactly one patient/session.")
        identities |= pairs
        timestamps = pd.to_datetime(stream["event_time_utc"], utc=True)
        starts[name] = timestamps.min()
        stops[name] = timestamps.max()
    if len(identities) != 1:
        raise ValueError("Streams do not belong to the same patient and session.")
    start = max(starts.values())
    stop = min(stops.values())
    overlap_seconds = max(0.0, float((stop - start).total_seconds()))
    patient_id, session_id = next(iter(identities))
    return {
        "patient_id": patient_id,
        "session_id": session_id,
        "stream_count": len(streams),
        "overlap_start_utc": start.isoformat(),
        "overlap_stop_utc": stop.isoformat(),
        "overlap_seconds": overlap_seconds,
        "has_overlap": overlap_seconds > 0,
        "stream_starts": {name: value.isoformat() for name, value in starts.items()},
        "stream_stops": {name: value.isoformat() for name, value in stops.items()},
    }
