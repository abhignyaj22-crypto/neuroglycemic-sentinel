import json
from pathlib import Path

import pandas as pd
import pytest

from src.neuroglycemic.lsl import (
    load_replay_session_manifest,
    replay_session_manifest,
)


def _manifest(tmp_path: Path, *, second_participant: str = "P001") -> Path:
    first = tmp_path / "eeg.csv"
    second = tmp_path / "wearable.csv"
    pd.DataFrame(
        {"time": [1.0, 2.0, 3.0], "AF3": [1.0, None, 3.0], "AF4": [1.0, 2.0, 3.0]}
    ).to_csv(first, index=False)
    pd.DataFrame(
        {"time": [1.0, 2.0, 3.0], "heart_rate": [70.0, 71.0, 72.0]}
    ).to_csv(second, index=False)
    values = {
        "schema_version": "neuroglycemic-lsl-replay-v1",
        "streams": [
            {
                "name": "Emotiv",
                "type": "EEG",
                "source_id": "emotiv-p001",
                "data_path": str(first),
                "timestamp_column": "time",
                "timestamp_format": "unix_seconds",
                "channels": [
                    {"column": "AF3", "label": "AF3", "unit": "microvolt"},
                    {"column": "AF4", "label": "AF4", "unit": "microvolt"},
                ],
                "nominal_rate_hz": 1.0,
                "manufacturer": "Emotiv",
                "model": "fixture",
                "participant_key": "P001",
                "session_id": "S001",
            },
            {
                "name": "Pulse",
                "type": "Vitals",
                "source_id": "pulse-p001",
                "data_path": str(second),
                "timestamp_column": "time",
                "timestamp_format": "unix_seconds",
                "channels": [
                    {
                        "column": "heart_rate",
                        "label": "heart_rate",
                        "unit": "beats_per_minute",
                    }
                ],
                "nominal_rate_hz": 1.0,
                "manufacturer": "Pulse",
                "model": "fixture",
                "participant_key": second_participant,
                "session_id": "S001",
            },
        ],
    }
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_replay_dry_run_preserves_partially_missing_packets(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    result = replay_session_manifest(path, dry_run=True)
    assert result["stream_count"] == 2
    assert result["streams"][0]["samples"] == 3


def test_replay_rejects_mixed_participants(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="participant_key"):
        load_replay_session_manifest(_manifest(tmp_path, second_participant="P999"))
