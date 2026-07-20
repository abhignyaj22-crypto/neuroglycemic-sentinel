"""Lab Streaming Layer and XDF acquisition adapters.

LSL supplies synchronized transport metadata. 
LabRecorder/XDF is the preferred durable recording path. Imports are
kept optional so offline MIMIC and CogWear experiments do not require liblsl.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LSLStreamAudit:
    name: str
    stream_type: str
    source_id: str
    channel_count: int
    nominal_rate_hz: float
    sample_count: int
    first_timestamp: float | None
    last_timestamp: float | None
    duration_seconds: float
    duplicate_timestamps: int
    backward_timestamps: int
    largest_gap_seconds: float | None
    clock_offset_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _import_pylsl() -> Any:
    try:
        import pylsl
    except ImportError as exc:  # pragma: no cover - depends on acquisition host
        raise RuntimeError(
            "Live LSL requires the optional acquisition dependency: pip install pylsl."
        ) from exc
    return pylsl


def _import_pyxdf() -> Any:
    try:
        import pyxdf
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "XDF import requires the optional acquisition dependency: pip install pyxdf."
        ) from exc
    return pyxdf


def discover_streams(*, timeout_seconds: float = 2.0) -> pd.DataFrame:
    """Discover current LSL outlets and print-ready metadata."""
    pylsl = _import_pylsl()
    streams = pylsl.resolve_streams(wait_time=float(timeout_seconds))
    rows = [
        {
            "name": info.name(),
            "type": info.type(),
            "source_id": info.source_id(),
            "channel_count": int(info.channel_count()),
            "nominal_rate_hz": float(info.nominal_srate()),
            "hostname": info.hostname(),
        }
        for info in streams
    ]
    return pd.DataFrame(rows)


def capture_stream(
    *,
    stream_type: str,
    duration_seconds: float,
    timeout_seconds: float = 5.0,
) -> tuple[pd.DataFrame, LSLStreamAudit]:
    """Capture one live stream with LSL clock sync/dejitter post-processing."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    pylsl = _import_pylsl()
    resolved = pylsl.resolve_byprop("type", stream_type, timeout=float(timeout_seconds))
    if not resolved:
        raise RuntimeError(f"No LSL stream with type {stream_type!r} was discovered.")
    info = resolved[0]
    flags = pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_monotonize
    inlet = pylsl.StreamInlet(info, processing_flags=flags)
    correction = float(inlet.time_correction(timeout=float(timeout_seconds)))
    started = time.monotonic()
    samples: list[list[float]] = []
    timestamps: list[float] = []
    while time.monotonic() - started < duration_seconds:
        chunk, chunk_times = inlet.pull_chunk(timeout=0.2)
        samples.extend(chunk)
        timestamps.extend(chunk_times)
    if not samples:
        raise RuntimeError(f"LSL stream {stream_type!r} produced no samples.")
    matrix = np.asarray(samples, dtype=float)
    frame = pd.DataFrame(matrix, columns=[f"channel_{i}" for i in range(matrix.shape[1])])
    frame.insert(0, "lsl_timestamp", np.asarray(timestamps, dtype=float))
    audit = audit_timestamp_array(
        timestamps=np.asarray(timestamps, dtype=float),
        name=info.name(),
        stream_type=info.type(),
        source_id=info.source_id(),
        channel_count=int(info.channel_count()),
        nominal_rate_hz=float(info.nominal_srate()),
        clock_offset_seconds=correction,
    )
    return frame, audit


def audit_timestamp_array(
    *,
    timestamps: np.ndarray,
    name: str,
    stream_type: str,
    source_id: str,
    channel_count: int,
    nominal_rate_hz: float,
    clock_offset_seconds: float | None = None,
) -> LSLStreamAudit:
    timestamps = np.asarray(timestamps, dtype=float)
    timestamps = timestamps[np.isfinite(timestamps)]
    differences = np.diff(timestamps)
    return LSLStreamAudit(
        name=name,
        stream_type=stream_type,
        source_id=source_id,
        channel_count=channel_count,
        nominal_rate_hz=nominal_rate_hz,
        sample_count=len(timestamps),
        first_timestamp=float(timestamps[0]) if len(timestamps) else None,
        last_timestamp=float(timestamps[-1]) if len(timestamps) else None,
        duration_seconds=(
            float(max(0.0, timestamps[-1] - timestamps[0])) if len(timestamps) else 0.0
        ),
        duplicate_timestamps=int(np.sum(differences == 0)),
        backward_timestamps=int(np.sum(differences < 0)),
        largest_gap_seconds=float(np.max(differences)) if len(differences) else None,
        clock_offset_seconds=clock_offset_seconds,
    )


def audit_xdf(path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load a LabRecorder XDF file and audit every recorded stream."""
    if not path.exists():
        raise FileNotFoundError(path)
    pyxdf = _import_pyxdf()
    streams, header = pyxdf.load_xdf(str(path), synchronize_clocks=True, dejitter_timestamps=True)
    audits: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for index, stream in enumerate(streams):
        info = stream["info"]
        name = str(info.get("name", [f"stream_{index}"])[0])
        stream_type = str(info.get("type", [""])[0])
        source_id = str(info.get("source_id", [""])[0])
        rate = float(info.get("nominal_srate", [0.0])[0])
        channel_count = int(info.get("channel_count", [0])[0])
        timestamps = np.asarray(stream.get("time_stamps", []), dtype=float)
        audit = audit_timestamp_array(
            timestamps=timestamps,
            name=name,
            stream_type=stream_type,
            source_id=source_id,
            channel_count=channel_count,
            nominal_rate_hz=rate,
        )
        audits.append(audit.as_dict())
        series = np.asarray(stream.get("time_series", []))
        if series.ndim == 1:
            series = series.reshape(-1, 1)
        frame = pd.DataFrame(series, columns=[f"channel_{i}" for i in range(series.shape[1])])
        frame.insert(0, "lsl_timestamp", timestamps)
        frames[f"{index}:{name}"] = frame
    audit_frame = pd.DataFrame(audits)
    audit_frame.attrs["xdf_header"] = header
    return audit_frame, frames
