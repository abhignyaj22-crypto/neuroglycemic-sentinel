from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .neural_dataset import target_column, target_time_column


PHYSIOCGM_COHORT_ID = "PhysioCGM-2025"


@dataclass(frozen=True)
class PhysioCGMBuildResult:
    frame: pd.DataFrame
    audit: pd.DataFrame
    source_files: int


def _utc_timestamp(value: Any, *, source_timezone: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"Invalid PhysioCGM timestamp: {value!r}")
    if result.tzinfo is None:
        result = result.tz_localize(
            source_timezone, ambiguous="raise", nonexistent="raise"
        )
    return result.tz_convert("UTC")


def _trusted_pickle(path: Path, *, trust_pickle: bool) -> Mapping[str, Any]:
    if not trust_pickle:
        raise ValueError(
            "PhysioCGM processed clips are pickle files. Re-run with "
            "--trust-pickle only after verifying they came from the official dataset."
        )
    with path.open("rb") as stream:
        value = pickle.load(stream)  # noqa: S301 - guarded explicit trust boundary
    if not isinstance(value, Mapping):
        raise ValueError(f"PhysioCGM clip must contain a mapping: {path}")
    return value


def discover_physiocgm_clips(input_dir: Path) -> dict[str, tuple[Path, ...]]:
    """Discover ``processed/<subject>/**/*.pkl`` without inventing identities."""

    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PhysioCGM processed directory does not exist: {root}")
    grouped: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.pkl")):
        relative = path.relative_to(root)
        subject = relative.parts[0] if len(relative.parts) > 1 else path.parent.name
        if not subject.strip():
            continue
        grouped.setdefault(subject, []).append(path)
    if not grouped:
        raise FileNotFoundError(
            f"No PhysioCGM processed *.pkl clips were found below {root}."
        )
    return {key: tuple(values) for key, values in grouped.items()}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _causal_series(
    stream: Mapping[str, Any],
    value_name: str,
    *,
    anchor: pd.Timestamp,
    source_timezone: str,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    raw_values = stream.get(value_name)
    raw_times = stream.get("Time")
    if raw_values is None or raw_times is None:
        return np.empty(0, dtype=float), pd.DatetimeIndex([])
    values = pd.to_numeric(pd.Series(list(raw_values)), errors="coerce").to_numpy(float)
    times = pd.DatetimeIndex(pd.to_datetime(list(raw_times), errors="coerce"))
    size = min(len(values), len(times))
    values, times = values[:size], times[:size]
    if times.tz is None:
        times = times.tz_localize(
            source_timezone, ambiguous="NaT", nonexistent="NaT"
        )
    times = times.tz_convert("UTC")
    valid = np.isfinite(values) & ~times.isna() & (times <= anchor)
    return values[valid], times[valid]


def _stats(
    prefix: str, values: np.ndarray, times: pd.DatetimeIndex
) -> dict[str, float]:
    if values.size == 0:
        return {}
    result = {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
    }
    if values.size >= 2:
        elapsed = (times - times[0]).total_seconds().to_numpy(float) / 60.0
        if np.ptp(elapsed) > 0:
            result[f"{prefix}_slope_per_minute"] = float(
                np.polyfit(elapsed, values, 1)[0]
            )
        result[f"{prefix}_successive_difference_rmssd"] = float(
            np.sqrt(np.mean(np.diff(values) ** 2))
        )
    return result


def _acceleration_stats(
    stream: Mapping[str, Any],
    names: Sequence[str],
    *,
    prefix: str,
    anchor: pd.Timestamp,
    source_timezone: str,
) -> tuple[dict[str, float], list[pd.Timestamp]]:
    series = [
        _causal_series(
            stream, name, anchor=anchor, source_timezone=source_timezone
        )
        for name in names
    ]
    if not series or any(values.size == 0 for values, _ in series):
        return {}, []
    size = min(values.size for values, _ in series)
    magnitude = np.sqrt(sum(values[:size] ** 2 for values, _ in series))
    times = series[0][1][:size]
    return _stats(prefix, magnitude, times), list(times)


def extract_causal_wearable_features(
    clip: Mapping[str, Any],
    *,
    anchor: pd.Timestamp,
    source_timezone: str,
) -> tuple[dict[str, float], float, float]:
    """Summarize only samples whose timestamps are no later than the anchor."""

    specifications = (
        ("wearable_e4_hr", _nested(clip, "e4", "HR"), "HR"),
        ("wearable_bvp", _nested(clip, "e4", "BVP"), "BVP"),
        ("wearable_eda", _nested(clip, "e4", "EDA"), "EDA"),
        ("wearable_temperature", _nested(clip, "e4", "TEMP"), "TEMP"),
        ("wearable_ecg", _nested(clip, "zephyr", "ECG"), "EcgWaveform"),
        ("wearable_zephyr_hr", _nested(clip, "zephyr", "Summary"), "HR"),
        ("wearable_breathing_rate", _nested(clip, "zephyr", "Summary"), "BR"),
        ("wearable_activity", _nested(clip, "zephyr", "Summary"), "Activity"),
        ("wearable_hr_confidence", _nested(clip, "zephyr", "Summary"), "HRConfidence"),
        ("wearable_ecg_noise", _nested(clip, "zephyr", "Summary"), "ECGNoise"),
    )
    features: dict[str, float] = {}
    latest_times: list[pd.Timestamp] = []
    populated_streams = 0
    for prefix, stream, value_name in specifications:
        values, times = _causal_series(
            stream,
            value_name,
            anchor=anchor,
            source_timezone=source_timezone,
        )
        if values.size:
            populated_streams += 1
            latest_times.append(times.max())
            features.update(_stats(prefix, values, times))
    for stream, names, prefix in (
        (_nested(clip, "e4", "ACC"), ("x", "y", "z"), "wearable_e4_acceleration"),
        (
            _nested(clip, "zephyr", "Accel"),
            ("Vertical", "Lateral", "Sagittal"),
            "wearable_zephyr_acceleration",
        ),
    ):
        values, times = _acceleration_stats(
            stream,
            names,
            prefix=prefix,
            anchor=anchor,
            source_timezone=source_timezone,
        )
        if values:
            populated_streams += 1
            latest_times.extend(times)
            features.update(values)
    quality = float(populated_streams / (len(specifications) + 2))
    staleness = (
        max(0.0, (anchor - max(latest_times)).total_seconds() / 60.0)
        if latest_times
        else 0.0
    )
    return features, quality, float(staleness)


def _nearest_future_target(
    anchors: Sequence[pd.Timestamp],
    glucose: Sequence[float],
    *,
    anchor: pd.Timestamp,
    horizon_minutes: int,
    tolerance_minutes: float,
) -> tuple[float, pd.Timestamp] | None:
    expected = anchor + pd.Timedelta(minutes=int(horizon_minutes))
    deltas = np.asarray(
        [abs((value - expected).total_seconds()) / 60.0 for value in anchors]
    )
    if deltas.size == 0:
        return None
    index = int(np.argmin(deltas))
    if deltas[index] > tolerance_minutes or anchors[index] <= anchor:
        return None
    value = float(glucose[index])
    if not math.isfinite(value) or value <= 0:
        return None
    return value, anchors[index]


def build_physiocgm_aligned_windows(
    input_dir: Path,
    *,
    horizons_minutes: Sequence[int] = (30, 60),
    horizon_tolerance_minutes: float = 5.0,
    source_timezone: str = "UTC",
    trust_pickle: bool = False,
) -> PhysioCGMBuildResult:
    if not trust_pickle:
        raise ValueError(
            "Refusing to load pickle input without trust_pickle=True. Verify the "
            "files came from the official PhysioCGM release, then use --trust-pickle."
        )
    grouped = discover_physiocgm_clips(input_dir)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    source_files = 0
    for patient_id, paths in grouped.items():
        clips: list[tuple[pd.Timestamp, float, Mapping[str, Any], Path]] = []
        for path in paths:
            source_files += 1
            try:
                clip = _trusted_pickle(path, trust_pickle=trust_pickle)
                anchor = _utc_timestamp(
                    clip.get("Timestamp"), source_timezone=source_timezone
                )
                glucose = float(clip.get("glucose"))
                if not math.isfinite(glucose) or glucose <= 0:
                    raise ValueError("reference glucose must be finite and positive")
                clips.append((anchor, glucose, clip, path))
            except Exception as exc:
                audits.append(
                    {
                        "patient_id": patient_id,
                        "source_file": str(path),
                        "status": "rejected",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        clips.sort(key=lambda value: value[0])
        anchors = [value[0] for value in clips]
        glucose_values = [value[1] for value in clips]
        for anchor, _current_glucose, clip, path in clips:
            features, quality, staleness = extract_causal_wearable_features(
                clip, anchor=anchor, source_timezone=source_timezone
            )
            if not features:
                audits.append(
                    {
                        "patient_id": patient_id,
                        "source_file": str(path),
                        "status": "rejected",
                        "reason": "no causal wearable samples",
                    }
                )
                continue
            row: dict[str, Any] = {
                "patient_id": patient_id,
                "cohort_id": PHYSIOCGM_COHORT_ID,
                "anchor_time": anchor,
                "wearable_available": True,
                "wearable_quality": quality,
                "wearable_staleness_minutes": staleness,
                "wearable_patient_id": patient_id,
                "wearable_cohort_id": PHYSIOCGM_COHORT_ID,
                "wearable_anchor_time": anchor,
                **features,
            }
            complete = True
            for horizon in horizons_minutes:
                target = _nearest_future_target(
                    anchors,
                    glucose_values,
                    anchor=anchor,
                    horizon_minutes=int(horizon),
                    tolerance_minutes=float(horizon_tolerance_minutes),
                )
                if target is None:
                    complete = False
                    break
                row[target_column(int(horizon))] = target[0]
                row[target_time_column(int(horizon))] = target[1]
            if complete:
                rows.append(row)
                audits.append(
                    {
                        "patient_id": patient_id,
                        "source_file": str(path),
                        "status": "kept",
                        "reason": "",
                    }
                )
            else:
                audits.append(
                    {
                        "patient_id": patient_id,
                        "source_file": str(path),
                        "status": "rejected",
                        "reason": "missing future CGM target within horizon tolerance",
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(
            "PhysioCGM preparation produced no complete causal forecast windows. "
            "Check the processed folder, timezone, and horizon tolerance."
        )
    frame = frame.sort_values(["patient_id", "anchor_time"], ignore_index=True)
    if frame.duplicated(["patient_id", "anchor_time"]).any():
        raise ValueError("PhysioCGM contains duplicate patient/anchor windows.")
    return PhysioCGMBuildResult(
        frame=frame,
        audit=pd.DataFrame(audits),
        source_files=source_files,
    )


def write_physiocgm_build(
    result: PhysioCGMBuildResult,
    output_path: Path,
    *,
    input_dir: Path,
    horizons_minutes: Sequence[int],
    source_timezone: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".parquet", ".pq"}:
        result.frame.to_parquet(output, index=False)
    elif output.suffix.lower() in {".csv", ".gz"}:
        result.frame.to_csv(output, index=False)
    else:
        raise ValueError("Prepared data output must be CSV, CSV.GZ, or Parquet.")
    audit_path = output.with_name(f"{output.stem}_alignment_audit.csv")
    result.audit.to_csv(audit_path, index=False)
    manifest = {
        "schema": "neuroglycemic-physiocgm-builder-v1",
        "cohort_id": PHYSIOCGM_COHORT_ID,
        "input_dir": str(Path(input_dir).resolve()),
        "output_file": str(output.resolve()),
        "source_files": result.source_files,
        "rows": int(len(result.frame)),
        "patients": int(result.frame["patient_id"].nunique()),
        "horizons_minutes": [int(value) for value in horizons_minutes],
        "source_timezone_for_naive_timestamps": source_timezone,
        "input_cgm_used_as_feature": False,
    }
    output.with_name(f"{output.stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
