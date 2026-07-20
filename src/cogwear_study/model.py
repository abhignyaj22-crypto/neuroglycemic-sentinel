from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def binary_cross_entropy(target: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    target = np.asarray(target, dtype=float)
    return float(-np.mean(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability)))


@dataclass
class TrainOnlyStandardizer:
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "TrainOnlyStandardizer":
        values = np.asarray(values, dtype=float)
        self.medians = np.nanmedian(values, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        filled = np.where(np.isnan(values), self.medians, values)
        self.means = np.mean(filled, axis=0)
        self.scales = np.std(filled, axis=0)
        self.scales[self.scales < 1e-8] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("The standardizer must be fit on training data first.")
        values = np.asarray(values, dtype=float)
        filled = np.where(np.isnan(values), self.medians, values)
        return (filled - self.means) / self.scales


@dataclass
class LogisticHead:
    feature_names: tuple[str, ...]
    standardizer: TrainOnlyStandardizer
    weights: np.ndarray
    bias: float
    best_epoch: int
    best_validation_loss: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        standardized = self.standardizer.transform(frame[list(self.feature_names)].to_numpy(float))
        return sigmoid(standardized @ self.weights + self.bias)

    def feature_contributions(self, row: pd.Series) -> pd.Series:
        values = row[list(self.feature_names)].to_numpy(dtype=float).reshape(1, -1)
        standardized = self.standardizer.transform(values)[0]
        return pd.Series(standardized * self.weights, index=self.feature_names).sort_values(
            key=np.abs, ascending=False
        )

    def coefficient_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "standardized_coefficient": self.weights,
                "absolute_coefficient": np.abs(self.weights),
            }
        ).sort_values("absolute_coefficient", ascending=False, ignore_index=True)

    def save(self, path: Path) -> None:
        if (
            self.standardizer.medians is None
            or self.standardizer.means is None
            or self.standardizer.scales is None
        ):
            raise RuntimeError("Cannot save an unfitted standardizer.")
        payload = {
            "schema_version": "cogwear-logistic-head-v1",
            "feature_names": list(self.feature_names),
            "standardizer": {
                "medians": self.standardizer.medians.tolist(),
                "means": self.standardizer.means.tolist(),
                "scales": self.standardizer.scales.tolist(),
            },
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LogisticHead":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "cogwear-logistic-head-v1":
            raise ValueError("Unsupported CogWear head schema.")
        values = payload["standardizer"]
        return cls(
            feature_names=tuple(payload["feature_names"]),
            standardizer=TrainOnlyStandardizer(
                medians=np.asarray(values["medians"], dtype=float),
                means=np.asarray(values["means"], dtype=float),
                scales=np.asarray(values["scales"], dtype=float),
            ),
            weights=np.asarray(payload["weights"], dtype=float),
            bias=float(payload["bias"]),
            best_epoch=int(payload["best_epoch"]),
            best_validation_loss=float(payload["best_validation_loss"]),
        )


def fit_logistic_head(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: tuple[str, ...],
    *,
    target_column: str,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> tuple[LogisticHead, pd.DataFrame]:
    """A visible forward -> loss -> gradient -> update training loop."""
    standardizer = TrainOnlyStandardizer().fit(train[list(feature_names)].to_numpy(float))
    x_train = standardizer.transform(train[list(feature_names)].to_numpy(float))
    x_validation = standardizer.transform(validation[list(feature_names)].to_numpy(float))
    y_train = train[target_column].to_numpy(float)
    y_validation = validation[target_column].to_numpy(float)

    weights = np.zeros(x_train.shape[1], dtype=float)
    bias = 0.0
    best_weights = weights.copy()
    best_bias = bias
    best_epoch = 0
    best_validation_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs + 1):
        train_probability = sigmoid(x_train @ weights + bias)
        validation_probability = sigmoid(x_validation @ weights + bias)
        train_loss = binary_cross_entropy(y_train, train_probability) + 0.5 * l2 * float(weights @ weights)
        validation_loss = binary_cross_entropy(y_validation, validation_probability)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = weights.copy()
            best_bias = bias
            best_epoch = epoch

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )

        # Backpropagation for a linear sigmoid classifier.
        error = train_probability - y_train
        gradient_weights = (x_train.T @ error) / len(x_train) + l2 * weights
        gradient_bias = float(np.mean(error))
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    head = LogisticHead(
        feature_names=feature_names,
        standardizer=standardizer,
        weights=best_weights,
        bias=best_bias,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
    return head, pd.DataFrame(history)


def classification_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
    }


def patient_session_predictions(
    frame: pd.DataFrame, probability_column: str
) -> pd.DataFrame:
    return (
        frame.groupby(["patient_id", "condition", "target_cognitive_load"], as_index=False)
        .agg(probability=(probability_column, "mean"), windows=(probability_column, "size"))
        .sort_values(["patient_id", "condition"], ignore_index=True)
    )
