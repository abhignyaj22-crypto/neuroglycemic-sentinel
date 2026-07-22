"""Publication-ready figures generated only from recorded experiment artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _pyplot(cache_directory: Path):
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_training_figure(history: pd.DataFrame, destination: Path) -> Path:
    required = {"epoch", "train_loss", "validation_loss"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Training history is missing columns: {sorted(missing)}")
    destination = Path(destination)
    plt = _pyplot(destination.parent / ".matplotlib-cache")
    fig, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    axis.plot(history["epoch"], history["train_loss"], label="Train", linewidth=2)
    axis.plot(
        history["epoch"], history["validation_loss"], label="Validation", linewidth=2
    )
    best = history.loc[history["validation_loss"].idxmin()]
    axis.scatter([best["epoch"]], [best["validation_loss"]], color="#b42318", zorder=3)
    axis.set(title="Neural optimization", xlabel="Epoch", ylabel="Negative log-likelihood")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def save_forecast_figure(predictions: pd.DataFrame, destination: Path) -> Path:
    required = {
        "target_glucose_mg_dl",
        "predicted_glucose_mg_dl",
        "prediction_lower_mg_dl",
        "prediction_upper_mg_dl",
        "horizon_minutes",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    destination = Path(destination)
    plt = _pyplot(destination.parent / ".matplotlib-cache")
    horizons = sorted(predictions["horizon_minutes"].dropna().astype(int).unique())
    fig, axes = plt.subplots(
        1, len(horizons), figsize=(5.0 * len(horizons), 4.5), squeeze=False,
        constrained_layout=True,
    )
    for axis, horizon in zip(axes[0], horizons, strict=True):
        group = predictions.loc[predictions["horizon_minutes"].eq(horizon)]
        actual = group["target_glucose_mg_dl"].to_numpy(float)
        predicted = group["predicted_glucose_mg_dl"].to_numpy(float)
        valid = np.isfinite(actual) & np.isfinite(predicted)
        axis.scatter(actual[valid], predicted[valid], s=14, alpha=0.5)
        if valid.any():
            low = float(min(actual[valid].min(), predicted[valid].min()))
            high = float(max(actual[valid].max(), predicted[valid].max()))
            axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        axis.set(
            title=f"{horizon}-minute forecast",
            xlabel="Observed reference glucose (mg/dL)",
            ylabel="Predicted glucose (mg/dL)",
        )
        axis.grid(alpha=0.2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def save_fusion_weight_figure(predictions: pd.DataFrame, destination: Path) -> Path:
    weight_columns = [name for name in predictions if name.startswith("weight_")]
    if not weight_columns:
        raise ValueError("Prediction table contains no learned modality weights.")
    summary = predictions.groupby("horizon_minutes", sort=True)[weight_columns].mean()
    destination = Path(destination)
    plt = _pyplot(destination.parent / ".matplotlib-cache")
    fig, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    summary.rename(columns=lambda value: value.removeprefix("weight_")).plot.bar(
        ax=axis, width=0.75
    )
    axis.set(
        title="Mean learned fusion weight by forecast horizon",
        xlabel="Forecast horizon (minutes)",
        ylabel="Mean gate weight",
        ylim=(0.0, 1.0),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Modality", frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination
