#!/usr/bin/env python3
"""Build one checkpoint-schema request from an aligned, target-free row view."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.neuroglycemic.service import NeuralGlucoseService  # noqa: E402


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError("Aligned data must be CSV, CSV.GZ, or Parquet.")


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def build_request(
    checkpoint: Path,
    data: Path,
    *,
    row_index: int,
    horizon_minutes: int | None,
    split: str = "all",
    allow_research_only: bool = False,
) -> dict[str, object]:
    service = NeuralGlucoseService.from_checkpoint(
        checkpoint, allow_research_only=allow_research_only
    )
    frame = _read_table(data)
    if split != "all":
        import torch

        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch.
            payload = torch.load(checkpoint, map_location="cpu")
        patient_split = payload.get("metadata", {}).get("patient_split", {})
        selected = patient_split.get(split)
        if not isinstance(selected, (list, tuple)) or not selected:
            raise ValueError(
                f"Checkpoint does not contain a non-empty {split!r} patient split."
            )
        if "cohort_id" in frame:
            participant_key = (
                frame["cohort_id"].astype(str) + "::" + frame["patient_id"].astype(str)
            )
        else:
            participant_key = frame["patient_id"].astype(str)
        frame = frame.loc[
            participant_key.isin({str(value) for value in selected})
        ].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"Aligned data has no patients from the checkpoint {split!r} split."
            )
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(
            f"row_index must be in [0, {len(frame) - 1}] after {split!r} filtering."
        )
    row = frame.iloc[row_index]
    horizon = horizon_minutes or service.supported_horizons_minutes[0]
    if horizon not in service.supported_horizons_minutes:
        raise ValueError(
            f"horizon_minutes must be one of {service.supported_horizons_minutes}."
        )
    modalities = tuple(service.feature_schema.feature_names)
    required = {"patient_id", "anchor_time"}
    for modality in modalities:
        required.update(
            {
                f"{modality}_available",
                f"{modality}_quality",
                f"{modality}_staleness_minutes",
                f"{modality}_clock_uncertainty_ms",
                *service.feature_schema.feature_names[modality],
            }
        )
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Aligned row is missing checkpoint fields: {sorted(missing)}")
    anchor = pd.Timestamp(row["anchor_time"])
    if anchor.tzinfo is None:
        raise ValueError("anchor_time must include a timezone offset.")
    return {
        "patient_id": str(row["patient_id"]),
        "anchor_time": anchor.isoformat(),
        "horizon_minutes": int(horizon),
        "feature_schema_version": service.feature_schema.version,
        "features": {
            modality: {
                name: _finite_or_none(row[name])
                for name in service.feature_schema.feature_names[modality]
            }
            for modality in modalities
        },
        "availability": {
            modality: bool(row[f"{modality}_available"])
            for modality in modalities
        },
        "quality": {
            modality: float(row[f"{modality}_quality"])
            for modality in modalities
        },
        "staleness_minutes": {
            modality: float(row[f"{modality}_staleness_minutes"])
            for modality in modalities
        },
        "clock_uncertainty_ms": {
            modality: float(row[f"{modality}_clock_uncertainty_ms"])
            for modality in modalities
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--horizon-minutes", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation", "test"),
        default="all",
        help=(
            "Select a checkpoint-recorded patient split before --row-index. "
            "Use --split test for a held-out case study."
        ),
    )
    parser.add_argument(
        "--research-only",
        action="store_true",
        help="Explicitly allow a research_only checkpoint release manifest.",
    )
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error(f"Refusing to overwrite {arguments.output}.")
    request = build_request(
        arguments.checkpoint,
        arguments.data,
        row_index=arguments.row_index,
        horizon_minutes=arguments.horizon_minutes,
        split=arguments.split,
        allow_research_only=arguments.research_only,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(request, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved target-free checkpoint request: {arguments.output}")
    print(json.dumps(request, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
