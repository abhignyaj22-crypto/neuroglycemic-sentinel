from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import binary_cross_entropy


def _softmax(values: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if eligible is None:
        eligible = np.ones(len(values), dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != values.shape or not eligible.any():
        raise ValueError("At least one correctly shaped eligible modality is required.")
    weights = np.zeros_like(values)
    shifted = values[eligible] - np.max(values[eligible])
    exponent = np.exp(shifted)
    weights[eligible] = exponent / exponent.sum()
    return weights


@dataclass
class LateFusion:
    modality_names: tuple[str, ...]
    weights: np.ndarray
    fallback_probability: float

    def predict(
        self, probabilities: np.ndarray, availability: np.ndarray | None = None
    ) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        if probabilities.shape[1] != len(self.weights):
            raise ValueError("Probability columns do not match fusion weights.")
        if availability is None:
            availability = np.isfinite(probabilities)
        availability = np.asarray(availability, dtype=bool)

        weighted = np.where(availability, probabilities, 0.0) * self.weights
        denominator = availability @ self.weights
        result = np.full(len(probabilities), self.fallback_probability, dtype=float)
        present = denominator > 0
        result[present] = weighted[present].sum(axis=1) / denominator[present]
        return result


def fit_late_fusion(
    validation_probabilities: np.ndarray,
    validation_target: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    modality_names: tuple[str, ...] = ("eeg", "wearable"),
    fallback_probability: float = 0.5,
    eligible_modalities: np.ndarray | None = None,
) -> tuple[LateFusion, pd.DataFrame]:
    """Learn weighted-average fusion on validation patients only."""
    probabilities = np.asarray(validation_probabilities, dtype=float)
    target = np.asarray(validation_target, dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Fusion fitting requires complete paired validation predictions.")
    if probabilities.shape[1] != len(modality_names):
        raise ValueError("One validation probability column is required per modality.")
    if eligible_modalities is None:
        eligible_modalities = np.ones(probabilities.shape[1], dtype=bool)
    eligible_modalities = np.asarray(eligible_modalities, dtype=bool)
    if eligible_modalities.shape != (probabilities.shape[1],) or not eligible_modalities.any():
        raise ValueError("At least one modality head must be eligible for fusion.")

    raw_weights = np.zeros(probabilities.shape[1], dtype=float)
    best_raw = raw_weights.copy()
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs + 1):
        weights = _softmax(raw_weights, eligible_modalities)
        fused = probabilities @ weights
        loss = binary_cross_entropy(target, fused)
        if loss < best_loss:
            best_loss = loss
            best_raw = raw_weights.copy()

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "validation_loss": loss,
                    **{f"weight_{name}": weights[index] for index, name in enumerate(modality_names)},
                }
            )

        clipped = np.clip(fused, 1e-7, 1.0 - 1e-7)
        loss_gradient = (clipped - target) / (clipped * (1.0 - clipped) * len(target))
        # d(sum_j softmax(z)_j * p_j)/dz_k = w_k * (p_k - fused)
        gradient = np.array(
            [
                np.sum(loss_gradient * weights[k] * (probabilities[:, k] - fused))
                for k in range(probabilities.shape[1])
            ]
        )
        raw_weights[eligible_modalities] -= learning_rate * gradient[eligible_modalities]

    fusion = LateFusion(
        modality_names=modality_names,
        weights=_softmax(best_raw, eligible_modalities),
        fallback_probability=float(fallback_probability),
    )
    return fusion, pd.DataFrame(history)


def missing_modality_scenarios(
    frame: pd.DataFrame, fusion: LateFusion
) -> pd.DataFrame:
    probabilities = frame[["alpha_eeg", "beta_wearable"]].to_numpy(float)
    scenarios = {
        "both_available": np.ones_like(probabilities, dtype=bool),
        "eeg_missing": np.column_stack(
            [np.zeros(len(frame), dtype=bool), np.ones(len(frame), dtype=bool)]
        ),
        "wearable_missing": np.column_stack(
            [np.ones(len(frame), dtype=bool), np.zeros(len(frame), dtype=bool)]
        ),
        "both_missing": np.zeros_like(probabilities, dtype=bool),
    }
    rows: list[pd.DataFrame] = []
    for name, availability in scenarios.items():
        scenario = frame[["patient_id", "condition", "target_cognitive_load"]].copy()
        scenario["scenario"] = name
        scenario["combined_probability"] = fusion.predict(probabilities, availability)
        rows.append(scenario)
    return pd.concat(rows, ignore_index=True)
