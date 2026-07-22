"""Lab Streaming Layer and XDF acquisition adapters.

LSL supplies synchronized transport metadata. 
LabRecorder/XDF is the preferred durable recording path. Imports are
kept optional so offline MIMIC and CogWear experiments do not require liblsl.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
from pathlib import Path
import time
from typing import Any
from xml.etree import ElementTree

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
    channel_labels: tuple[str, ...] = ()
    channel_units: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayChannel:
    """One numeric source column and its LSL metadata."""

    column: str
    label: str
    unit: str


@dataclass(frozen=True)
class ReplayStream:
    """A real recorded table exposed as one replay-only LSL outlet."""

    name: str
    stream_type: str
    source_id: str
    data_path: Path
    timestamp_column: str
    timestamp_format: str
    channels: tuple[ReplayChannel, ...]
    nominal_rate_hz: float
    manufacturer: str
    model: str
    participant_key: str
    session_id: str


@dataclass(frozen=True)
class ReplaySession:
    schema_version: str
    streams: tuple[ReplayStream, ...]


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_replay_session_manifest(path: Path) -> ReplaySession:
    """Load a strict multi-device replay manifest.

    The manifest contains metadata and paths only; it never manufactures sensor
    values. Relative data paths are resolved beside the manifest so the entire
    protected runtime can be moved without editing source code.
    """

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) != {"schema_version", "streams"}:
        raise ValueError("Replay manifest requires exactly schema_version and streams.")
    if values["schema_version"] != "neuroglycemic-lsl-replay-v1":
        raise ValueError("Unsupported LSL replay manifest schema_version.")
    raw_streams = values["streams"]
    if not isinstance(raw_streams, list) or not raw_streams:
        raise ValueError("Replay manifest streams must be a non-empty array.")

    required_stream = {
        "name",
        "type",
        "source_id",
        "data_path",
        "timestamp_column",
        "timestamp_format",
        "channels",
        "nominal_rate_hz",
        "manufacturer",
        "model",
        "participant_key",
        "session_id",
    }
    streams: list[ReplayStream] = []
    for stream_index, raw in enumerate(raw_streams):
        if not isinstance(raw, dict) or set(raw) != required_stream:
            missing = required_stream - set(raw) if isinstance(raw, dict) else required_stream
            extra = set(raw) - required_stream if isinstance(raw, dict) else set()
            raise ValueError(
                f"Replay stream {stream_index} has missing={sorted(missing)} "
                f"and extra={sorted(extra)} fields."
            )
        text_fields = {
            name: str(raw[name]).strip()
            for name in (
                "name",
                "type",
                "source_id",
                "timestamp_column",
                "manufacturer",
                "model",
                "participant_key",
                "session_id",
            )
        }
        if any(not value for value in text_fields.values()):
            raise ValueError(f"Replay stream {stream_index} has an empty identity field.")
        timestamp_format = str(raw["timestamp_format"]).strip()
        if timestamp_format not in {"iso8601", "unix_seconds", "lsl_seconds"}:
            raise ValueError(
                "timestamp_format must be iso8601, unix_seconds, or lsl_seconds."
            )
        data_path = Path(str(raw["data_path"])).expanduser()
        if not data_path.is_absolute():
            data_path = path.parent / data_path
        data_path = data_path.resolve()
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        channels_raw = raw["channels"]
        if not isinstance(channels_raw, list) or not channels_raw:
            raise ValueError(f"Replay stream {stream_index} needs at least one channel.")
        channels: list[ReplayChannel] = []
        for channel_index, channel in enumerate(channels_raw):
            if not isinstance(channel, dict) or set(channel) != {"column", "label", "unit"}:
                raise ValueError(
                    f"Replay stream {stream_index} channel {channel_index} requires "
                    "exactly column, label, and unit."
                )
            values_by_name = {
                name: str(channel[name]).strip() for name in ("column", "label", "unit")
            }
            if any(not value for value in values_by_name.values()):
                raise ValueError("Replay channel fields cannot be empty.")
            if values_by_name["column"].startswith("target_"):
                raise ValueError("Future target columns cannot be replayed as model inputs.")
            channels.append(ReplayChannel(**values_by_name))
        if len({value.column for value in channels}) != len(channels):
            raise ValueError("Replay channel source columns must be unique within a stream.")
        if len({value.label for value in channels}) != len(channels):
            raise ValueError("Replay channel labels must be unique within a stream.")
        nominal_rate = float(raw["nominal_rate_hz"])
        if not np.isfinite(nominal_rate) or nominal_rate < 0:
            raise ValueError("nominal_rate_hz must be finite and non-negative.")
        streams.append(
            ReplayStream(
                name=text_fields["name"],
                stream_type=text_fields["type"],
                source_id=text_fields["source_id"],
                data_path=data_path,
                timestamp_column=text_fields["timestamp_column"],
                timestamp_format=timestamp_format,
                channels=tuple(channels),
                nominal_rate_hz=nominal_rate,
                manufacturer=text_fields["manufacturer"],
                model=text_fields["model"],
                participant_key=text_fields["participant_key"],
                session_id=text_fields["session_id"],
            )
        )
    identities = [(value.name, value.stream_type, value.source_id) for value in streams]
    if len(set(identities)) != len(identities):
        raise ValueError("Every replay outlet name/type/source_id identity must be unique.")
    if len({value.source_id for value in streams}) != len(streams):
        raise ValueError("Every replay outlet source_id must be globally unique.")
    return ReplaySession(
        schema_version=str(values["schema_version"]), streams=tuple(streams)
    )


def _read_replay_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Replay input must be CSV, CSV.GZ, or Parquet: {path}")


def _source_seconds(values: pd.Series, timestamp_format: str) -> np.ndarray:
    if timestamp_format == "iso8601":
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
        seconds = parsed.astype("int64").to_numpy(dtype=float) / 1e9
        seconds[pd.isna(parsed).to_numpy()] = np.nan
        return seconds
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _prepare_replay_stream(
    stream: ReplayStream, *, max_rows: int | None
) -> tuple[np.ndarray, np.ndarray]:
    frame = _read_replay_table(stream.data_path)
    columns = [value.column for value in stream.channels]
    required = {stream.timestamp_column, *columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Replay stream {stream.source_id!r} is missing columns: {sorted(missing)}"
        )
    seconds = _source_seconds(frame[stream.timestamp_column], stream.timestamp_format)
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(seconds) & np.isfinite(matrix).all(axis=1)
    seconds, matrix = seconds[keep], matrix[keep]
    if max_rows is not None:
        seconds, matrix = seconds[:max_rows], matrix[:max_rows]
    if not len(seconds):
        raise ValueError(f"Replay stream {stream.source_id!r} has no complete numeric rows.")
    differences = np.diff(seconds)
    if bool(np.any(differences <= 0)):
        duplicate_steps = int(np.sum(differences == 0))
        backward_steps = int(np.sum(differences < 0))
        raise ValueError(
            f"Replay stream {stream.source_id!r} timestamps must be strictly increasing; "
            f"found duplicate_steps={duplicate_steps}, backward_steps={backward_steps}. "
            "Use the raw acquisition clock/sample index or a documented repair; "
            "the replayer will not invent an order."
        )
    return seconds, matrix.astype(np.float32)


def _lsl_stream_info(pylsl: Any, stream: ReplayStream) -> Any:
    info = pylsl.StreamInfo(
        stream.name,
        stream.stream_type,
        len(stream.channels),
        stream.nominal_rate_hz,
        "float32",
        stream.source_id,
    )
    channels = info.desc().append_child("channels")
    for specification in stream.channels:
        channel = channels.append_child("channel")
        channel.append_child_value("label", specification.label)
        channel.append_child_value("unit", specification.unit)
        channel.append_child_value("type", stream.stream_type)
    acquisition = info.desc().append_child("acquisition")
    acquisition.append_child_value("manufacturer", stream.manufacturer)
    acquisition.append_child_value("model", stream.model)
    acquisition.append_child_value("participant_key", stream.participant_key)
    acquisition.append_child_value("session_id", stream.session_id)
    acquisition.append_child_value("replay", "true")
    acquisition.append_child_value("source_file_sha256", _sha256(stream.data_path))
    return info


def replay_session_manifest(
    manifest_path: Path,
    *,
    speed: float = 60.0,
    startup_delay_seconds: float = 5.0,
    max_rows_per_stream: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Replay multiple recorded devices on one shared LSL time base.

    Emotiv/Galea/iHealth/pulse/CGM tables retain independent outlets and source
    rates. A priority queue interleaves their source samples, while explicit LSL
    timestamps preserve cross-stream timing after acceleration. LabRecorder can
    discover the outlets during ``startup_delay_seconds`` and record them to XDF.
    """

    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Replay speed must be finite and positive.")
    if not np.isfinite(startup_delay_seconds) or startup_delay_seconds < 0:
        raise ValueError("startup_delay_seconds must be finite and non-negative.")
    if max_rows_per_stream is not None and max_rows_per_stream <= 0:
        raise ValueError("max_rows_per_stream must be positive when supplied.")
    session = load_replay_session_manifest(manifest_path)
    prepared = [
        _prepare_replay_stream(stream, max_rows=max_rows_per_stream)
        for stream in session.streams
    ]
    global_start = min(float(times[0]) for times, _ in prepared)
    global_stop = max(float(times[-1]) for times, _ in prepared)
    stream_audits = [
        {
            "name": stream.name,
            "type": stream.stream_type,
            "source_id": stream.source_id,
            "participant_key": stream.participant_key,
            "session_id": stream.session_id,
            "channels": [value.label for value in stream.channels],
            "samples": int(len(prepared[index][0])),
            "source_start_seconds": float(prepared[index][0][0]),
            "source_stop_seconds": float(prepared[index][0][-1]),
            "source_sha256": _sha256(stream.data_path),
        }
        for index, stream in enumerate(session.streams)
    ]
    result: dict[str, Any] = {
        "schema_version": session.schema_version,
        "stream_count": len(session.streams),
        "source_duration_seconds": float(global_stop - global_start),
        "speed": float(speed),
        "dry_run": bool(dry_run),
        "streams": stream_audits,
    }
    if dry_run:
        result["samples_replayed"] = 0
        return result

    pylsl = _import_pylsl()
    outlets = [pylsl.StreamOutlet(_lsl_stream_info(pylsl, stream)) for stream in session.streams]
    if startup_delay_seconds:
        time.sleep(float(startup_delay_seconds))
    lsl_start = float(pylsl.local_clock())
    wall_start = time.monotonic()
    pending: list[tuple[float, int, int]] = [
        (float(times[0]), stream_index, 0)
        for stream_index, (times, _) in enumerate(prepared)
    ]
    heapq.heapify(pending)
    sent = [0 for _ in session.streams]
    while pending:
        source_time, stream_index, row_index = heapq.heappop(pending)
        scheduled = lsl_start + (source_time - global_start) / float(speed)
        remaining = scheduled - float(pylsl.local_clock())
        if remaining > 0:
            time.sleep(remaining)
        sample = prepared[stream_index][1][row_index].tolist()
        outlets[stream_index].push_sample(sample, timestamp=scheduled)
        sent[stream_index] += 1
        next_index = row_index + 1
        if next_index < len(prepared[stream_index][0]):
            heapq.heappush(
                pending,
                (
                    float(prepared[stream_index][0][next_index]),
                    stream_index,
                    next_index,
                ),
            )
    result["samples_replayed"] = int(sum(sent))
    result["wall_duration_seconds"] = float(time.monotonic() - wall_start)
    for index, count in enumerate(sent):
        result["streams"][index]["samples_replayed"] = int(count)
    return result


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
    stream_name: str | None = None,
    source_id: str | None = None,
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
    candidates = [
        info
        for info in resolved
        if (stream_name is None or info.name() == stream_name)
        and (source_id is None or info.source_id() == source_id)
    ]
    if len(candidates) != 1:
        identities = [
            {"name": value.name(), "type": value.type(), "source_id": value.source_id()}
            for value in candidates
        ]
        raise RuntimeError(
            "LSL capture requires exactly one matching name/type/source_id; "
            f"found {len(candidates)}: {identities}"
        )
    info = candidates[0]
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
    channel_labels, channel_units = _channel_metadata_from_xml(
        info.as_xml(), matrix.shape[1]
    )
    frame = pd.DataFrame(matrix, columns=_unique_labels(channel_labels))
    frame.insert(0, "lsl_timestamp", np.asarray(timestamps, dtype=float))
    audit = audit_timestamp_array(
        timestamps=np.asarray(timestamps, dtype=float),
        name=info.name(),
        stream_type=info.type(),
        source_id=info.source_id(),
        channel_count=int(info.channel_count()),
        nominal_rate_hz=float(info.nominal_srate()),
        clock_offset_seconds=correction,
        channel_labels=channel_labels,
        channel_units=channel_units,
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
    channel_labels: tuple[str, ...] = (),
    channel_units: tuple[str, ...] = (),
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
        channel_labels=channel_labels,
        channel_units=channel_units,
    )


def _unique_labels(labels: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    counts: dict[str, int] = {}
    for index, raw in enumerate(labels):
        label = raw.strip() or f"channel_{index}"
        count = counts.get(label, 0)
        counts[label] = count + 1
        result.append(label if count == 0 else f"{label}_{count + 1}")
    return result


def _channel_metadata_from_xml(
    xml: str, channel_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        root = None
    labels: list[str] = []
    units: list[str] = []
    if root is not None:
        for channel in root.findall(".//channels/channel"):
            labels.append(channel.findtext("label", default=""))
            units.append(channel.findtext("unit", default="source_unit"))
    while len(labels) < channel_count:
        labels.append(f"channel_{len(labels)}")
        units.append("source_unit")
    return tuple(labels[:channel_count]), tuple(units[:channel_count])


def _xdf_channel_metadata(
    info: dict[str, Any], channel_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    labels: list[str] = []
    units: list[str] = []
    try:
        channels = info["desc"][0]["channels"][0]["channel"]
    except (KeyError, IndexError, TypeError):
        channels = []
    for channel in channels:
        label = channel.get("label", [""])
        unit = channel.get("unit", ["source_unit"])
        labels.append(str(label[0] if isinstance(label, list) else label))
        units.append(str(unit[0] if isinstance(unit, list) else unit))
    while len(labels) < channel_count:
        labels.append(f"channel_{len(labels)}")
        units.append("source_unit")
    return tuple(labels[:channel_count]), tuple(units[:channel_count])


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
        channel_labels, channel_units = _xdf_channel_metadata(info, channel_count)
        audit = audit_timestamp_array(
            timestamps=timestamps,
            name=name,
            stream_type=stream_type,
            source_id=source_id,
            channel_count=channel_count,
            nominal_rate_hz=rate,
            channel_labels=channel_labels,
            channel_units=channel_units,
        )
        audits.append(audit.as_dict())
        series = np.asarray(stream.get("time_series", []))
        if series.ndim == 1:
            series = series.reshape(-1, 1)
        if series.ndim != 2 or series.shape[1] != channel_count:
            raise ValueError(
                f"XDF stream {name!r} declares {channel_count} channels but "
                f"contains shape {series.shape}."
            )
        frame = pd.DataFrame(series, columns=_unique_labels(channel_labels))
        frame.insert(0, "lsl_timestamp", timestamps)
        frame.attrs["channel_units"] = dict(
            zip(_unique_labels(channel_labels), channel_units, strict=True)
        )
        frames[f"{index}:{name}"] = frame
    audit_frame = pd.DataFrame(audits)
    audit_frame.attrs["xdf_header"] = header
    return audit_frame, frames


def replay_numeric_table(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    timestamp_format: str = "iso8601",
    channel_columns: list[str] | tuple[str, ...],
    stream_name: str,
    stream_type: str,
    source_id: str,
    speed: float = 60.0,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Replay recorded inputs through an LSL outlet at accelerated wall time.

    This is an interoperability test, not a new dataset. Source timestamps set
    relative pacing; the outlet uses the acquisition computer's LSL clock so it
    can be synchronized with live EEG/wearable streams. Target columns are
    prohibited to prevent future glucose leakage into an acquisition stream.
    """

    if not channel_columns:
        raise ValueError("At least one replay channel is required.")
    if any(name.startswith("target_") for name in channel_columns):
        raise ValueError("Future target columns cannot be replayed as model inputs.")
    required = {timestamp_column, *channel_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Replay table is missing columns: {sorted(missing)}")
    if not stream_name.strip() or not stream_type.strip() or not source_id.strip():
        raise ValueError("LSL replay name, type, and source_id are required.")
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Replay speed must be finite and positive.")
    if timestamp_format not in {"iso8601", "unix_seconds", "lsl_seconds"}:
        raise ValueError(
            "timestamp_format must be iso8601, unix_seconds, or lsl_seconds."
        )
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when supplied.")

    selected = frame[[timestamp_column, *channel_columns]].copy()
    selected["_source_seconds"] = _source_seconds(
        selected[timestamp_column], timestamp_format
    )
    for column in channel_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna().sort_values("_source_seconds")
    if max_rows is not None:
        selected = selected.head(max_rows)
    if selected.empty:
        raise ValueError("Replay table contains no complete numeric rows.")
    differences = np.diff(selected["_source_seconds"].to_numpy(dtype=float))
    if bool(np.any(differences <= 0)):
        duplicate_steps = int(np.sum(differences == 0))
        backward_steps = int(np.sum(differences < 0))
        raise ValueError(
            "Replay timestamps must be strictly increasing after channel "
            f"filtering; duplicate_steps={duplicate_steps}, "
            f"backward_steps={backward_steps}."
        )

    pylsl = _import_pylsl()
    info = pylsl.StreamInfo(
        stream_name,
        stream_type,
        len(channel_columns),
        0.0,
        "float32",
        source_id,
    )
    channels = info.desc().append_child("channels")
    for label in channel_columns:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", "source_unit")
        channel.append_child_value("type", stream_type)
    info.desc().append_child_value("replay", "true")
    info.desc().append_child_value("source_timestamp_column", timestamp_column)
    outlet = pylsl.StreamOutlet(info)

    source_seconds = selected["_source_seconds"].to_numpy(dtype=float)
    matrix = selected[list(channel_columns)].to_numpy(dtype=np.float32)
    started = time.monotonic()
    previous = float(source_seconds[0])
    for index, (timestamp, sample) in enumerate(zip(source_seconds, matrix, strict=True)):
        if index:
            delay = max(0.0, float(timestamp - previous) / float(speed))
            if delay:
                time.sleep(delay)
        outlet.push_sample(sample.tolist(), timestamp=pylsl.local_clock())
        previous = float(timestamp)
    return {
        "stream_name": stream_name,
        "stream_type": stream_type,
        "source_id": source_id,
        "channels": list(channel_columns),
        "samples_replayed": int(len(selected)),
        "source_duration_seconds": float(source_seconds[-1] - source_seconds[0]),
        "wall_duration_seconds": float(time.monotonic() - started),
        "speed": float(speed),
    }
