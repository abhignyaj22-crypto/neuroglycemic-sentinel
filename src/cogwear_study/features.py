from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import StudyConfig


EEG_BANDS: dict[str, tuple[str, ...]] = {
    "delta": ("Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10"),
    "theta": ("Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10"),
    "alpha": ("Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10"),
    "beta": ("Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10"),
    "gamma": ("Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP10"),
}

EEG_FEATURES = tuple(
    [f"eeg_{band}_mean" for band in EEG_BANDS]
    + [f"eeg_{band}_variability" for band in EEG_BANDS]
    + ["eeg_theta_alpha_ratio", "eeg_beta_alpha_ratio"]
)

WEARABLE_FEATURES = (
    "wearable_pulse_rate_bpm",
    "wearable_pulse_interval_rmssd_ms",
    "wearable_bvp_variability",
    "wearable_eda_mean",
    "wearable_eda_variability",
    "wearable_eda_slope",
    "wearable_temperature_mean",
    "wearable_temperature_variability",
    "wearable_temperature_slope",
)


@dataclass(frozen=True)
class SessionStreams:
    eeg: pd.DataFrame
    bvp: pd.DataFrame
    eda: pd.DataFrame
    temperature: pd.DataFrame


def _numeric_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=columns)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time", ignore_index=True)
    return frame


def load_session_streams(session: pd.Series) -> SessionStreams:
    eeg_columns = ["time", "HeadBandOn", *[column for values in EEG_BANDS.values() for column in values]]
    eeg = _numeric_frame(Path(session["eeg_path"]), eeg_columns)
    eeg = eeg.loc[eeg["HeadBandOn"].fillna(0.0) > 0.0].reset_index(drop=True)
    return SessionStreams(
        eeg=eeg,
        bvp=_numeric_frame(Path(session["bvp_path"]), ["time", "bvp"]),
        eda=_numeric_frame(Path(session["eda_path"]), ["time", "eda"]),
        temperature=_numeric_frame(Path(session["temp_path"]), ["time", "temp"]),
    )


def _slice(frame: pd.DataFrame, start: float, stop: float) -> pd.DataFrame:
    return frame.loc[(frame["time"] >= start) & (frame["time"] < stop)]


def _safe_slope(values: np.ndarray, times: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(times)
    if valid.sum() < 2 or np.ptp(times[valid]) == 0:
        return float("nan")
    return float(np.polyfit(times[valid] - times[valid][0], values[valid], 1)[0])


def _eeg_features(window: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    band_means: dict[str, float] = {}
    for band, columns in EEG_BANDS.items():
        values = window[list(columns)].to_numpy(dtype=float)
        band_means[band] = float(np.nanmean(values))
        result[f"eeg_{band}_mean"] = band_means[band]
        result[f"eeg_{band}_variability"] = float(np.nanstd(values))

    # Muse reports log-band-power values. Ratios are therefore expressed as
    # differences in log space, which is stable and device-agnostic.
    result["eeg_theta_alpha_ratio"] = band_means["theta"] - band_means["alpha"]
    result["eeg_beta_alpha_ratio"] = band_means["beta"] - band_means["alpha"]
    return result


def _pulse_features(window: pd.DataFrame) -> dict[str, float]:
    values = window["bvp"].to_numpy(dtype=float)
    times = window["time"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(times)
    values, times = values[valid], times[valid]
    if len(values) < 3:
        return {
            "wearable_pulse_rate_bpm": float("nan"),
            "wearable_pulse_interval_rmssd_ms": float("nan"),
            "wearable_bvp_variability": float("nan"),
        }

    sample_rate = 1.0 / float(np.median(np.diff(times)))
    centered = values - np.median(values)
    prominence = max(float(np.std(centered)) * 0.25, 1e-6)
    peaks, _ = find_peaks(
        centered,
        distance=max(1, int(round(sample_rate * 0.35))),
        prominence=prominence,
    )
    intervals = np.diff(times[peaks])
    intervals = intervals[(intervals >= 0.33) & (intervals <= 1.50)]
    pulse_rate = 60.0 / float(np.mean(intervals)) if len(intervals) else float("nan")
    rmssd = (
        float(np.sqrt(np.mean(np.diff(intervals) ** 2)) * 1000.0)
        if len(intervals) >= 2
        else float("nan")
    )
    return {
        "wearable_pulse_rate_bpm": pulse_rate,
        "wearable_pulse_interval_rmssd_ms": rmssd,
        "wearable_bvp_variability": float(np.std(values)),
    }


def _wearable_features(
    bvp: pd.DataFrame, eda: pd.DataFrame, temperature: pd.DataFrame
) -> dict[str, float]:
    result = _pulse_features(bvp)
    eda_values = eda["eda"].to_numpy(dtype=float)
    eda_times = eda["time"].to_numpy(dtype=float)
    temp_values = temperature["temp"].to_numpy(dtype=float)
    temp_times = temperature["time"].to_numpy(dtype=float)
    result.update(
        {
            "wearable_eda_mean": float(np.nanmean(eda_values)),
            "wearable_eda_variability": float(np.nanstd(eda_values)),
            "wearable_eda_slope": _safe_slope(eda_values, eda_times),
            "wearable_temperature_mean": float(np.nanmean(temp_values)),
            "wearable_temperature_variability": float(np.nanstd(temp_values)),
            "wearable_temperature_slope": _safe_slope(temp_values, temp_times),
        }
    )
    return result


def _overlap(streams: SessionStreams) -> tuple[float, float]:
    frames = (streams.eeg, streams.bvp, streams.eda, streams.temperature)
    start = max(float(frame["time"].min()) for frame in frames)
    stop = min(float(frame["time"].max()) for frame in frames)
    return start, stop


def build_paired_feature_table(
    sessions: pd.DataFrame, config: StudyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract synchronized features without learning from (or) excluding test patients."""
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []

    for session in sessions.itertuples(index=False):
        session_series = pd.Series(session._asdict())
        streams = load_session_streams(session_series)
        overlap_start, overlap_stop = _overlap(streams)
        analysis_start = overlap_start + config.warmup_seconds
        allowed_stop = min(
            overlap_stop,
            analysis_start + config.window_seconds * config.max_windows_per_condition,
        )
        possible_windows = int(
            np.floor((allowed_stop - analysis_start + 1e-6) / config.window_seconds)
        )

        audits.append(
            {
                "patient_id": session.patient_id,
                "condition": session.condition,
                "eeg_rows": len(streams.eeg),
                "bvp_rows": len(streams.bvp),
                "eda_rows": len(streams.eda),
                "temperature_rows": len(streams.temperature),
                "overlap_seconds": overlap_stop - overlap_start,
                "windows_kept": possible_windows,
            }
        )

        for window_index in range(possible_windows):
            start = analysis_start + window_index * config.window_seconds
            stop = start + config.window_seconds
            eeg_window = _slice(streams.eeg, start, stop)
            bvp_window = _slice(streams.bvp, start, stop)
            eda_window = _slice(streams.eda, start, stop)
            temp_window = _slice(streams.temperature, start, stop)

            # These thresholds require at least ~1 second of each stream. A bad
            # window is omitted, never replaced with made-up measurements.
            if min(len(eeg_window), len(bvp_window)) < 32 or min(len(eda_window), len(temp_window)) < 4:
                continue

            row: dict[str, object] = {
                "patient_id": session.patient_id,
                "condition": session.condition,
                "window_index": window_index,
                "window_start_unix": start,
                "target_cognitive_load": int(session.target_cognitive_load),
                "eeg_available": 1,
                "wearable_available": 1,
            }
            row.update(_eeg_features(eeg_window))
            row.update(_wearable_features(bvp_window, eda_window, temp_window))
            rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        raise ValueError("No synchronized EEG/wearable windows survived quality checks.")

    # A few resting sessions are shorter than their matching Stroop sessions.
    # Keep the same number of windows from both conditions for each participant
    # so session duration cannot silently reweight the target.
    balanced_groups: list[pd.DataFrame] = []
    for patient_id, patient_rows in features.groupby("patient_id", sort=False):
        counts = patient_rows.groupby("condition").size()
        if len(counts) != 2:
            continue
        windows_per_condition = int(counts.min())
        for _, condition_rows in patient_rows.groupby("condition", sort=False):
            balanced_groups.append(
                condition_rows.sort_values("window_index").head(windows_per_condition)
            )
    features = pd.concat(balanced_groups, ignore_index=True)
    features = features.sort_values(
        ["patient_id", "target_cognitive_load", "window_index"], ignore_index=True
    )
    audit = pd.DataFrame(audits)
    used = (
        features.groupby(["patient_id", "condition"])
        .size()
        .rename("windows_used_after_balancing")
        .reset_index()
    )
    audit = audit.merge(used, on=["patient_id", "condition"], how="left")
    return features, audit
