import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudyConfig:
    cohort: str
    participants: tuple[int, ...]
    conditions: dict[str, int]
    window_seconds: float
    warmup_seconds: float
    max_windows_per_condition: int
    seed: int
    train_fraction: float
    validation_fraction: float
    head_learning_rate: float
    head_epochs: int
    head_l2: float
    fusion_learning_rate: float
    fusion_epochs: int
    project_root: Path

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw" / "cogwear" / "pilot"

    @property
    def processed_path(self) -> Path:
        return self.project_root / "data" / "processed" / "cogwear_paired_windows.csv"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs"


def load_config(path: Path) -> StudyConfig:
    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)

    project_root = path.parent.parent
    return StudyConfig(
        cohort=str(values["cohort"]),
        participants=tuple(int(value) for value in values["participants"]),
        conditions={str(key): int(value) for key, value in values["conditions"].items()},
        window_seconds=float(values["window_seconds"]),
        warmup_seconds=float(values["warmup_seconds"]),
        max_windows_per_condition=int(values["max_windows_per_condition"]),
        seed=int(values["seed"]),
        train_fraction=float(values["train_fraction"]),
        validation_fraction=float(values["validation_fraction"]),
        head_learning_rate=float(values["head_learning_rate"]),
        head_epochs=int(values["head_epochs"]),
        head_l2=float(values["head_l2"]),
        fusion_learning_rate=float(values["fusion_learning_rate"]),
        fusion_epochs=int(values["fusion_epochs"]),
        project_root=project_root,
    )
