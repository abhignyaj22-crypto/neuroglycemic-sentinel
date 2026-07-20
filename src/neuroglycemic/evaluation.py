import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def forecast_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    actual = frame["target_glucose_mg_dl"].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    error = predicted - actual
    result = {
        "rmse_mg_dl": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_mg_dl": float(mean_absolute_error(actual, predicted)),
        "median_ae_mg_dl": float(np.median(np.abs(error))),
        "mard_percent": float(np.mean(np.abs(error) / np.maximum(actual, 1.0)) * 100.0),
        "r2": float(r2_score(actual, predicted)),
    }
    if "patient_id" in frame:
        errors = frame[["patient_id"]].copy()
        errors["absolute_error"] = np.abs(error)
        errors["squared_error"] = error**2
        by_patient = errors.groupby("patient_id", sort=False).agg(
            mae=("absolute_error", "mean"), mse=("squared_error", "mean")
        )
        result["patient_macro_mae_mg_dl"] = float(by_patient["mae"].mean())
        result["patient_macro_rmse_mg_dl"] = float(np.sqrt(by_patient["mse"]).mean())
    return result


def probabilistic_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["target_glucose_mg_dl"].to_numpy(dtype=float)
    lower = frame["prediction_lower_mg_dl"].to_numpy(dtype=float)
    upper = frame["prediction_upper_mg_dl"].to_numpy(dtype=float)
    return {
        "interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_interval_width_mg_dl": float(np.mean(upper - lower)),
    }


def hyperglycemia_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame["target_hyperglycemia"].to_numpy(dtype=int)
    probability = frame["hyperglycemia_probability"].to_numpy(dtype=float)
    result = {
        "brier_score": float(brier_score_loss(target, probability)),
        "prevalence": float(np.mean(target)),
    }
    if len(np.unique(target)) == 2:
        result["auroc"] = float(roc_auc_score(target, probability))
        result["average_precision"] = float(average_precision_score(target, probability))
    else:
        result["auroc"] = float("nan")
        result["average_precision"] = float("nan")
    return result


def patient_cluster_bootstrap(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    seed: int,
    replicates: int,
) -> dict[str, dict[str, float]]:
    patients = frame["patient_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {"rmse_mg_dl": [], "mae_mg_dl": []}
    for _ in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        pieces = [frame.loc[frame["patient_id"] == patient] for patient in sampled]
        bootstrap = pd.concat(pieces, ignore_index=True)
        metrics = forecast_metrics(bootstrap, prediction_column)
        for name in estimates:
            estimates[name].append(metrics[name])
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for name, values in estimates.items()
    }


def comparison_table(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, column in (
        ("persistence_current_glucose", "current_glucose_mg_dl"),
        ("validation_shrunk_multitask_ehr", "predicted_glucose_mg_dl"),
    ):
        rows.append({"model": name, **forecast_metrics(test, column)})
    return pd.DataFrame(rows)


def subgroup_table(test: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "sex_female=0": test.loc[test["sex_female"] == 0],
        "sex_female=1": test.loc[test["sex_female"] == 1],
        "age<65": test.loc[test["age_years"] < 65],
        "age>=65": test.loc[test["age_years"] >= 65],
    }
    rows = []
    for name, group in groups.items():
        if len(group) < 5:
            continue
        rows.append(
            {
                "subgroup": name,
                "n_anchors": len(group),
                "n_patients": group["patient_id"].nunique(),
                **forecast_metrics(group, "predicted_glucose_mg_dl"),
            }
        )
    return pd.DataFrame(rows)


def _finite_prediction_rows(
    frame: pd.DataFrame, prediction_column: str
) -> pd.DataFrame:
    """Return evaluable rows without treating abstentions as predictions."""

    required = {"target_glucose_mg_dl", prediction_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation frame is missing columns: {sorted(missing)}")
    target = pd.to_numeric(frame["target_glucose_mg_dl"], errors="coerce")
    prediction = pd.to_numeric(frame[prediction_column], errors="coerce")
    valid = target.notna() & prediction.notna()
    if "abstained" in frame:
        valid &= ~frame["abstained"].astype(bool)
    return frame.loc[valid].copy()


def neural_regression_metrics(
    frame: pd.DataFrame, prediction_column: str
) -> dict[str, float | int]:
    """Regression metrics with explicit coverage for neural forecasts."""

    valid = _finite_prediction_rows(frame, prediction_column)
    result: dict[str, float | int] = {
        "evaluated_rows": int(len(valid)),
        "prediction_coverage": float(len(valid) / len(frame)) if len(frame) else 0.0,
        "abstention_rate": float(1.0 - len(valid) / len(frame)) if len(frame) else 1.0,
    }
    if valid.empty:
        result.update(
            {
                "rmse_mg_dl": float("nan"),
                "mae_mg_dl": float("nan"),
                "median_ae_mg_dl": float("nan"),
                "mard_percent": float("nan"),
                "r2": float("nan"),
            }
        )
        return result
    return {**result, **forecast_metrics(valid, prediction_column)}


def prediction_interval_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    required = {
        "target_glucose_mg_dl",
        "prediction_lower_mg_dl",
        "prediction_upper_mg_dl",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Interval frame is missing columns: {sorted(missing)}")
    values = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1)
    if "abstained" in frame:
        valid &= ~frame["abstained"].astype(bool)
    selected = values.loc[valid]
    if selected.empty:
        return {
            "evaluated_rows": 0,
            "interval_coverage": float("nan"),
            "mean_interval_width_mg_dl": float("nan"),
        }
    lower = selected["prediction_lower_mg_dl"].to_numpy(float)
    upper = selected["prediction_upper_mg_dl"].to_numpy(float)
    if np.any(upper < lower):
        raise ValueError("Prediction interval upper bounds must not be below lower bounds.")
    target = selected["target_glucose_mg_dl"].to_numpy(float)
    return {
        "evaluated_rows": int(len(selected)),
        "interval_coverage": float(np.mean((target >= lower) & (target <= upper))),
        "mean_interval_width_mg_dl": float(np.mean(upper - lower)),
    }


def binary_event_metrics(
    frame: pd.DataFrame,
    *,
    target_column: str,
    probability_column: str,
) -> dict[str, float | int]:
    missing = {target_column, probability_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Event frame is missing columns: {sorted(missing)}")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    probability = pd.to_numeric(frame[probability_column], errors="coerce")
    valid = target.notna() & probability.notna()
    if "abstained" in frame:
        valid &= ~frame["abstained"].astype(bool)
    target_values = target.loc[valid].to_numpy(int)
    probability_values = probability.loc[valid].to_numpy(float)
    if np.any((probability_values < 0.0) | (probability_values > 1.0)):
        raise ValueError("Event probabilities must be in [0, 1].")
    result: dict[str, float | int] = {
        "evaluated_rows": int(valid.sum()),
        "prediction_coverage": float(valid.mean()) if len(valid) else 0.0,
        "prevalence": float(np.mean(target_values)) if len(target_values) else float("nan"),
        "brier_score": (
            float(brier_score_loss(target_values, probability_values))
            if len(target_values)
            else float("nan")
        ),
        "auroc": float("nan"),
        "average_precision": float("nan"),
    }
    if len(np.unique(target_values)) == 2:
        result["auroc"] = float(roc_auc_score(target_values, probability_values))
        result["average_precision"] = float(
            average_precision_score(target_values, probability_values)
        )
    return result


def gaussian_mixture_nll(
    targets: np.ndarray,
    expert_means: np.ndarray,
    expert_scales: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Stable mean Gaussian-mixture NLL for reporting, not training."""

    targets = np.asarray(targets, dtype=float)
    means = np.asarray(expert_means, dtype=float)
    scales = np.asarray(expert_scales, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if means.ndim != 3 or scales.shape != means.shape:
        raise ValueError("Expert means/scales must be [row, modality, horizon].")
    if targets.shape != (means.shape[0], means.shape[2]):
        raise ValueError("Targets must be [row, horizon].")
    if weights.shape != means.shape[:2]:
        raise ValueError("Weights must be [row, modality].")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Expert scales must be finite and positive.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Mixture weights must be finite and non-negative.")
    weight_sum = weights.sum(axis=1)
    valid_rows = np.isfinite(targets).all(axis=1) & (weight_sum > 0)
    if not valid_rows.any():
        return float("nan")
    selected_targets = targets[valid_rows, None, :]
    selected_means = means[valid_rows]
    selected_scales = scales[valid_rows]
    selected_weights = weights[valid_rows] / weight_sum[valid_rows, None]
    log_density = -0.5 * (
        np.log(2.0 * np.pi)
        + 2.0 * np.log(selected_scales)
        + ((selected_targets - selected_means) / selected_scales) ** 2
    )
    log_weights = np.where(
        selected_weights > 0, np.log(np.maximum(selected_weights, 1e-300)), -np.inf
    )
    terms = log_density + log_weights[:, :, None]
    maximum = np.max(terms, axis=1, keepdims=True)
    log_probability = (
        maximum.squeeze(1)
        + np.log(np.exp(terms - maximum).sum(axis=1))
    )
    return float(-np.mean(log_probability))


def missing_modality_ablation_summary(
    scenarios: dict[str, pd.DataFrame],
    prediction_column: str,
    *,
    reference_scenario: str,
) -> pd.DataFrame:
    """Summarize strictly paired missing-modality evaluations."""

    if reference_scenario not in scenarios:
        raise KeyError(f"Missing reference scenario {reference_scenario!r}.")
    keys = ["patient_id", "anchor_time", "horizon_minutes", "target_glucose_mg_dl"]
    reference = scenarios[reference_scenario].reset_index(drop=True)
    missing = set(keys) - set(reference.columns)
    if missing:
        raise ValueError(f"Ablation reference is missing keys: {sorted(missing)}")
    reference_keys = reference[keys].astype(str)
    reference_metrics = neural_regression_metrics(reference, prediction_column)
    rows: list[dict[str, float | int | str]] = []
    for name, frame in scenarios.items():
        candidate = frame.reset_index(drop=True)
        if len(candidate) != len(reference) or not candidate[keys].astype(str).equals(
            reference_keys
        ):
            raise ValueError("Missing-modality ablations must use identical ordered rows.")
        metrics = neural_regression_metrics(candidate, prediction_column)
        rows.append(
            {
                "scenario": name,
                **metrics,
                "mae_delta_vs_observed_mg_dl": (
                    float(metrics["mae_mg_dl"] - reference_metrics["mae_mg_dl"])
                    if np.isfinite(metrics["mae_mg_dl"])
                    and np.isfinite(reference_metrics["mae_mg_dl"])
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)
