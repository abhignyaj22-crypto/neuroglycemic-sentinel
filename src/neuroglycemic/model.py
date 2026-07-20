import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class Standardizer:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=float)
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isnan(values), medians, values)
        means = np.mean(filled, axis=0)
        scales = np.std(filled, axis=0)
        scales = np.where(scales < 1e-8, 1.0, scales)
        return cls(medians=medians, means=means, scales=scales)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        filled = np.where(np.isnan(values), self.medians, values)
        return (filled - self.means) / self.scales


@dataclass
class MultitaskParameters:
    regression_weights: np.ndarray
    regression_bias: float
    log_variance: float
    classification_weights: np.ndarray
    classification_bias: float

    def copy(self) -> "MultitaskParameters":
        return MultitaskParameters(
            regression_weights=self.regression_weights.copy(),
            regression_bias=float(self.regression_bias),
            log_variance=float(self.log_variance),
            classification_weights=self.classification_weights.copy(),
            classification_bias=float(self.classification_bias),
        )


def multitask_loss_and_gradients(
    x: np.ndarray,
    target_regression: np.ndarray,
    target_classification: np.ndarray,
    parameters: MultitaskParameters,
    *,
    positive_weight: float,
    classification_loss_weight: float,
    l2: float,
) -> tuple[dict[str, float], MultitaskParameters]:
    """Gaussian NLL + class-balanced BCE, with exact analytic gradients."""
    x = np.asarray(x, dtype=float)
    y_reg = np.asarray(target_regression, dtype=float)
    y_cls = np.asarray(target_classification, dtype=float)
    sample_count = len(x)

    mean = x @ parameters.regression_weights + parameters.regression_bias
    log_variance = float(np.clip(parameters.log_variance, -6.0, 4.0))
    precision = float(np.exp(-log_variance))
    residual = mean - y_reg
    gaussian_nll = float(
        0.5 * np.mean(precision * residual**2 + log_variance + np.log(2.0 * np.pi))
    )

    probability = sigmoid(x @ parameters.classification_weights + parameters.classification_bias)
    probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    class_weights = np.where(y_cls == 1.0, positive_weight, 1.0)
    weight_sum = float(np.sum(class_weights))
    balanced_bce = float(
        -np.sum(
            class_weights
            * (y_cls * np.log(probability) + (1.0 - y_cls) * np.log(1.0 - probability))
        )
        / weight_sum
    )

    regularization = 0.5 * l2 * float(
        parameters.regression_weights @ parameters.regression_weights
        + parameters.classification_weights @ parameters.classification_weights
    )
    total = gaussian_nll + classification_loss_weight * balanced_bce + regularization

    regression_output_gradient = precision * residual / sample_count
    regression_weights_gradient = x.T @ regression_output_gradient + l2 * parameters.regression_weights
    regression_bias_gradient = float(np.sum(regression_output_gradient))
    log_variance_gradient = float(0.5 * np.mean(1.0 - precision * residual**2))

    classification_output_gradient = class_weights * (probability - y_cls) / weight_sum
    classification_weights_gradient = (
        classification_loss_weight * (x.T @ classification_output_gradient)
        + l2 * parameters.classification_weights
    )
    classification_bias_gradient = float(
        classification_loss_weight * np.sum(classification_output_gradient)
    )

    gradients = MultitaskParameters(
        regression_weights=regression_weights_gradient,
        regression_bias=regression_bias_gradient,
        log_variance=log_variance_gradient,
        classification_weights=classification_weights_gradient,
        classification_bias=classification_bias_gradient,
    )
    losses = {
        "total_loss": total,
        "gaussian_nll": gaussian_nll,
        "balanced_bce": balanced_bce,
        "l2_penalty": regularization,
    }
    return losses, gradients


@dataclass
class ProbabilisticGlucoseModel:
    feature_names: tuple[str, ...]
    standardizer: Standardizer
    target_center: float
    target_scale: float
    regression_target: str
    reference_feature: str | None
    parameters: MultitaskParameters
    hyperglycemia_threshold_mg_dl: float
    prediction_interval: float
    best_epoch: int
    best_validation_loss: float
    training_metadata: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = self.standardizer.transform(frame[list(self.feature_names)].to_numpy(dtype=float))
        mean_z = x @ self.parameters.regression_weights + self.parameters.regression_bias
        sigma_z = float(np.sqrt(np.exp(np.clip(self.parameters.log_variance, -6.0, 4.0))))
        alpha = 1.0 - self.prediction_interval
        critical_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)

        center = self.target_center + self.target_scale * mean_z
        lower_center = self.target_center + self.target_scale * (
            mean_z - critical_value * sigma_z
        )
        upper_center = self.target_center + self.target_scale * (
            mean_z + critical_value * sigma_z
        )
        if self.regression_target == "delta_from_reference":
            if self.reference_feature is None:
                raise ValueError("A reference feature is required for residual forecasting.")
            reference = frame[self.reference_feature].to_numpy(dtype=float)
            predicted = reference + center
            lower = reference + lower_center
            upper = reference + upper_center
        elif self.regression_target == "log1p_absolute":
            predicted = np.expm1(center)
            lower = np.expm1(lower_center)
            upper = np.expm1(upper_center)
        else:
            raise ValueError(f"Unknown regression target: {self.regression_target}")
        probability = sigmoid(
            x @ self.parameters.classification_weights + self.parameters.classification_bias
        )
        return pd.DataFrame(
            {
                "predicted_glucose_mg_dl": np.maximum(0.0, predicted),
                "prediction_lower_mg_dl": np.maximum(0.0, lower),
                "prediction_upper_mg_dl": np.maximum(0.0, upper),
                "hyperglycemia_probability": probability,
            },
            index=frame.index,
        )

    def feature_coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "future_minus_current_glucose_z_coefficient": self.parameters.regression_weights,
                "hyperglycemia_logit_coefficient": self.parameters.classification_weights,
            }
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "ehr-glucose-model-v2",
            "feature_names": list(self.feature_names),
            "standardizer": {
                "medians": self.standardizer.medians.tolist(),
                "means": self.standardizer.means.tolist(),
                "scales": self.standardizer.scales.tolist(),
            },
            "target_center": self.target_center,
            "target_scale": self.target_scale,
            "regression_target": self.regression_target,
            "reference_feature": self.reference_feature,
            "parameters": {
                "regression_weights": self.parameters.regression_weights.tolist(),
                "regression_bias": self.parameters.regression_bias,
                "log_variance": self.parameters.log_variance,
                "classification_weights": self.parameters.classification_weights.tolist(),
                "classification_bias": self.parameters.classification_bias,
            },
            "hyperglycemia_threshold_mg_dl": self.hyperglycemia_threshold_mg_dl,
            "prediction_interval": self.prediction_interval,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "training_metadata": self.training_metadata,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProbabilisticGlucoseModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "ehr-glucose-model-v2":
            raise ValueError("Unsupported glucose model schema.")
        scaler = payload["standardizer"]
        parameters = payload["parameters"]
        return cls(
            feature_names=tuple(payload["feature_names"]),
            standardizer=Standardizer(
                medians=np.asarray(scaler["medians"], dtype=float),
                means=np.asarray(scaler["means"], dtype=float),
                scales=np.asarray(scaler["scales"], dtype=float),
            ),
            target_center=float(payload["target_center"]),
            target_scale=float(payload["target_scale"]),
            regression_target=str(payload["regression_target"]),
            reference_feature=payload["reference_feature"],
            parameters=MultitaskParameters(
                regression_weights=np.asarray(parameters["regression_weights"], dtype=float),
                regression_bias=float(parameters["regression_bias"]),
                log_variance=float(parameters["log_variance"]),
                classification_weights=np.asarray(
                    parameters["classification_weights"], dtype=float
                ),
                classification_bias=float(parameters["classification_bias"]),
            ),
            hyperglycemia_threshold_mg_dl=float(payload["hyperglycemia_threshold_mg_dl"]),
            prediction_interval=float(payload["prediction_interval"]),
            best_epoch=int(payload["best_epoch"]),
            best_validation_loss=float(payload["best_validation_loss"]),
            training_metadata=dict(payload["training_metadata"]),
        )


def _zeros_like(parameters: MultitaskParameters) -> MultitaskParameters:
    return MultitaskParameters(
        regression_weights=np.zeros_like(parameters.regression_weights),
        regression_bias=0.0,
        log_variance=0.0,
        classification_weights=np.zeros_like(parameters.classification_weights),
        classification_bias=0.0,
    )


def _adam_update(
    parameters: MultitaskParameters,
    gradients: MultitaskParameters,
    first_moment: MultitaskParameters,
    second_moment: MultitaskParameters,
    *,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> None:
    for name in (
        "regression_weights",
        "regression_bias",
        "log_variance",
        "classification_weights",
        "classification_bias",
    ):
        gradient = getattr(gradients, name)
        first = beta1 * getattr(first_moment, name) + (1.0 - beta1) * gradient
        second = beta2 * getattr(second_moment, name) + (1.0 - beta2) * gradient**2
        setattr(first_moment, name, first)
        setattr(second_moment, name, second)
        corrected_first = first / (1.0 - beta1**step)
        corrected_second = second / (1.0 - beta2**step)
        updated = getattr(parameters, name) - learning_rate * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )
        setattr(parameters, name, updated)
    parameters.log_variance = float(np.clip(parameters.log_variance, -6.0, 4.0))


def fit_probabilistic_glucose_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: tuple[str, ...],
    *,
    target_regression_column: str,
    target_classification_column: str,
    hyperglycemia_threshold_mg_dl: float,
    prediction_interval: float,
    learning_rate: float,
    epochs: int,
    l2: float,
    classification_loss_weight: float,
    training_metadata: dict[str, Any],
    regression_target: str = "delta_from_reference",
    reference_feature: str = "current_glucose_mg_dl",
) -> tuple[ProbabilisticGlucoseModel, pd.DataFrame]:
    standardizer = Standardizer.fit(train[list(feature_names)].to_numpy(dtype=float))
    x_train = standardizer.transform(train[list(feature_names)].to_numpy(dtype=float))
    x_validation = standardizer.transform(validation[list(feature_names)].to_numpy(dtype=float))

    if regression_target == "delta_from_reference":
        train_raw_target = (
            train[target_regression_column].to_numpy(dtype=float)
            - train[reference_feature].to_numpy(dtype=float)
        )
        validation_raw_target = (
            validation[target_regression_column].to_numpy(dtype=float)
            - validation[reference_feature].to_numpy(dtype=float)
        )
    elif regression_target == "log1p_absolute":
        train_raw_target = np.log1p(train[target_regression_column].to_numpy(dtype=float))
        validation_raw_target = np.log1p(
            validation[target_regression_column].to_numpy(dtype=float)
        )
    else:
        raise ValueError(f"Unknown regression target: {regression_target}")
    target_center = float(np.mean(train_raw_target))
    target_scale = float(np.std(train_raw_target))
    if target_scale < 1e-8:
        raise ValueError("Training glucose targets have no variation.")
    y_train = (train_raw_target - target_center) / target_scale
    y_validation = (validation_raw_target - target_center) / target_scale
    c_train = train[target_classification_column].to_numpy(dtype=float)
    c_validation = validation[target_classification_column].to_numpy(dtype=float)
    positives = float(np.sum(c_train == 1.0))
    negatives = float(np.sum(c_train == 0.0))
    positive_weight = negatives / max(positives, 1.0)

    parameters = MultitaskParameters(
        regression_weights=np.zeros(x_train.shape[1], dtype=float),
        regression_bias=0.0,
        log_variance=0.0,
        classification_weights=np.zeros(x_train.shape[1], dtype=float),
        classification_bias=0.0,
    )
    first_moment = _zeros_like(parameters)
    second_moment = _zeros_like(parameters)
    best_parameters = parameters.copy()
    best_validation_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs + 1):
        train_losses, gradients = multitask_loss_and_gradients(
            x_train,
            y_train,
            c_train,
            parameters,
            positive_weight=positive_weight,
            classification_loss_weight=classification_loss_weight,
            l2=l2,
        )
        validation_losses, _ = multitask_loss_and_gradients(
            x_validation,
            y_validation,
            c_validation,
            parameters,
            positive_weight=positive_weight,
            classification_loss_weight=classification_loss_weight,
            l2=l2,
        )
        if validation_losses["total_loss"] < best_validation_loss:
            best_validation_loss = validation_losses["total_loss"]
            best_epoch = epoch
            best_parameters = parameters.copy()

        if epoch % max(1, epochs // 20) == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "train_total_loss": train_losses["total_loss"],
                    "train_gaussian_nll": train_losses["gaussian_nll"],
                    "train_balanced_bce": train_losses["balanced_bce"],
                    "validation_total_loss": validation_losses["total_loss"],
                    "validation_gaussian_nll": validation_losses["gaussian_nll"],
                    "validation_balanced_bce": validation_losses["balanced_bce"],
                    "log_variance": parameters.log_variance,
                }
            )
        if epoch < epochs:
            _adam_update(
                parameters,
                gradients,
                first_moment,
                second_moment,
                step=epoch + 1,
                learning_rate=learning_rate,
            )

    metadata = {
        **training_metadata,
        "positive_class_weight": positive_weight,
        "loss": (
            "Gaussian NLL on standardized future-minus-current glucose + weighted BCE"
            if regression_target == "delta_from_reference"
            else "Gaussian NLL on standardized log1p glucose + weighted BCE"
        ),
    }
    model = ProbabilisticGlucoseModel(
        feature_names=feature_names,
        standardizer=standardizer,
        target_center=target_center,
        target_scale=target_scale,
        regression_target=regression_target,
        reference_feature=reference_feature if regression_target == "delta_from_reference" else None,
        parameters=best_parameters,
        hyperglycemia_threshold_mg_dl=hyperglycemia_threshold_mg_dl,
        prediction_interval=prediction_interval,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        training_metadata=metadata,
    )
    return model, pd.DataFrame(history)
