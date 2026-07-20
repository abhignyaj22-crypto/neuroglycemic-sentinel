import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EHRGlucoseConfig:
    cohort: str
    mimic_version: str
    raw_relative_dir: str
    processed_relative_path: str
    output_relative_dir: str
    forecast_horizon_hours: float
    forecast_tolerance_hours: float
    lookback_hours: float
    min_glucose_history: int
    minimum_glucose_mg_dl: float
    maximum_glucose_mg_dl: float
    hyperglycemia_threshold_mg_dl: float
    seed: int
    train_fraction: float
    validation_fraction: float
    learning_rate: float
    epochs: int
    l2: float
    classification_loss_weight: float
    prediction_interval: float
    bootstrap_replicates: int
    project_root: Path

    @property
    def raw_dir(self) -> Path:
        return self.project_root / self.raw_relative_dir

    @property
    def processed_path(self) -> Path:
        return self.project_root / self.processed_relative_path

    @property
    def output_dir(self) -> Path:
        return self.project_root / self.output_relative_dir


def load_ehr_config(path: Path) -> EHRGlucoseConfig:
    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    root = path.parent.parent
    return EHRGlucoseConfig(
        cohort=str(values["cohort"]),
        mimic_version=str(values["mimic_version"]),
        raw_relative_dir=str(values["raw_relative_dir"]),
        processed_relative_path=str(values["processed_relative_path"]),
        output_relative_dir=str(values["output_relative_dir"]),
        forecast_horizon_hours=float(values["forecast_horizon_hours"]),
        forecast_tolerance_hours=float(values["forecast_tolerance_hours"]),
        lookback_hours=float(values["lookback_hours"]),
        min_glucose_history=int(values["min_glucose_history"]),
        minimum_glucose_mg_dl=float(values["minimum_glucose_mg_dl"]),
        maximum_glucose_mg_dl=float(values["maximum_glucose_mg_dl"]),
        hyperglycemia_threshold_mg_dl=float(values["hyperglycemia_threshold_mg_dl"]),
        seed=int(values["seed"]),
        train_fraction=float(values["train_fraction"]),
        validation_fraction=float(values["validation_fraction"]),
        learning_rate=float(values["learning_rate"]),
        epochs=int(values["epochs"]),
        l2=float(values["l2"]),
        classification_loss_weight=float(values["classification_loss_weight"]),
        prediction_interval=float(values["prediction_interval"]),
        bootstrap_replicates=int(values["bootstrap_replicates"]),
        project_root=root,
    )

