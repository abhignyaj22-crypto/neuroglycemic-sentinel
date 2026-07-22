from collections.abc import Mapping, Sequence
import math

import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def neural_regression_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    target_column: str = "target_glucose_mg_dl",
    patient_column: str = "patient_id",
) -> dict[str, float | int]:
    """Evaluate neural forecasts without hiding abstentions or patient imbalance.

    Metrics use finite target/prediction pairs.  ``prediction_coverage`` reports
    the fraction of finite labels that received a prediction, so abstaining rows
    cannot silently disappear from the report.  Patient-macro metrics give every
    patient equal weight, independently of their number of windows.
    """

    required = {target_column, prediction_column, patient_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Evaluation frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Evaluation frame cannot be empty.")

    actual = frame[target_column].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    labelled = np.isfinite(actual)
    valid = labelled & np.isfinite(predicted)
    labelled_count = int(labelled.sum())
    if labelled_count == 0:
        raise ValueError("Evaluation requires at least one finite glucose target.")
    if not valid.any():
        return {
            "n_labelled": labelled_count,
            "n_predicted": 0,
            "n_patients": 0,
            "prediction_coverage": 0.0,
            "abstention_rate": 1.0,
            "mae_mg_dl": float("nan"),
            "rmse_mg_dl": float("nan"),
            "mard_percent": float("nan"),
            "patient_macro_mae_mg_dl": float("nan"),
            "patient_macro_rmse_mg_dl": float("nan"),
            "patient_macro_mard_percent": float("nan"),
        }
    if np.any(actual[valid] <= 0):
        raise ValueError("MARD requires positive reference glucose values.")

    errors = predicted[valid] - actual[valid]
    absolute = np.abs(errors)
    relative = absolute / actual[valid]
    result: dict[str, float | int] = {
        "n_labelled": labelled_count,
        "n_predicted": int(valid.sum()),
        "n_patients": int(frame.loc[valid, patient_column].nunique()),
        "prediction_coverage": float(valid.sum() / labelled_count),
        "abstention_rate": float(1.0 - valid.sum() / labelled_count),
        "mae_mg_dl": float(absolute.mean()),
        "rmse_mg_dl": float(np.sqrt(np.mean(errors**2))),
        "mard_percent": float(100.0 * relative.mean()),
    }

    errors_frame = pd.DataFrame(
        {
            "patient": frame.loc[valid, patient_column].astype(str).to_numpy(),
            "absolute_error": absolute,
            "squared_error": errors**2,
            "absolute_relative_error": relative,
        }
    )
    by_patient = errors_frame.groupby("patient", sort=False).agg(
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        mard=("absolute_relative_error", "mean"),
    )
    result.update(
        {
            "patient_macro_mae_mg_dl": float(by_patient["mae"].mean()),
            "patient_macro_rmse_mg_dl": float(np.sqrt(by_patient["mse"]).mean()),
            "patient_macro_mard_percent": float(100.0 * by_patient["mard"].mean()),
        }
    )
    return result


def gaussian_mixture_nll(
    target: np.ndarray | Sequence[float],
    expert_means: np.ndarray,
    expert_scales: np.ndarray,
    fusion_weights: np.ndarray,
) -> float:
    """Mean negative log likelihood of neural Gaussian-mixture predictions.

    Expert arrays are ``[example, modality, horizon]`` and targets are
    ``[example, horizon]``.  A one-horizon target may be passed as a vector.
    Rows with missing labels or zero total modality weight are excluded; callers
    should report abstention separately with :func:`neural_regression_metrics`.
    """

    means = np.asarray(expert_means, dtype=float)
    scales = np.asarray(expert_scales, dtype=float)
    weights = np.asarray(fusion_weights, dtype=float)
    values = np.asarray(target, dtype=float)
    if means.ndim != 3 or scales.shape != means.shape:
        raise ValueError("Expert means and scales must be [example, modality, horizon].")
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (means.shape[0], means.shape[2]):
        raise ValueError("target must be [example, horizon].")
    if weights.shape != means.shape[:2]:
        raise ValueError("fusion_weights must be [example, modality].")
    if not np.isfinite(means).all():
        raise ValueError("Expert means must be finite.")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Expert scales must be finite and positive.")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Fusion weights must be finite and non-negative.")

    weight_sums = weights.sum(axis=1)
    active = weight_sums > 0
    if np.any(active) and not np.allclose(weight_sums[active], 1.0, atol=1e-5):
        raise ValueError("Fusion weights must sum to one for non-abstained examples.")
    valid = np.isfinite(values) & active[:, None]
    if not valid.any():
        raise ValueError("No labelled, non-abstained predictions are available for NLL.")

    safe_target = np.where(np.isfinite(values), values, 0.0)[:, None, :]
    log_density = -0.5 * (
        math.log(2.0 * math.pi)
        + 2.0 * np.log(scales)
        + ((safe_target - means) / scales) ** 2
    )
    log_weights = np.full_like(weights, -np.inf)
    positive = weights > 0
    log_weights[positive] = np.log(weights[positive])
    nll = -logsumexp(log_weights[:, :, None] + log_density, axis=1)
    return float(nll[valid].mean())


def gaussian_mixture_quantile(
    expert_means: Sequence[float] | np.ndarray,
    expert_scales: Sequence[float] | np.ndarray,
    fusion_weights: Sequence[float] | np.ndarray,
    probability: float,
    *,
    iterations: int = 64,
) -> float:
    """Numerically invert a one-dimensional Gaussian-mixture CDF.

    A Gaussian mixture is generally not Gaussian, so ``mean ± 1.96 * SD`` is
    not its 95% interval. Bisection is deterministic and sufficiently fast for
    evaluation and single-request serving.
    """

    means = np.asarray(expert_means, dtype=float)
    scales = np.asarray(expert_scales, dtype=float)
    weights = np.asarray(fusion_weights, dtype=float)
    if means.ndim != 1 or scales.shape != means.shape or weights.shape != means.shape:
        raise ValueError("Mixture means, scales, and weights must be equal-length vectors.")
    if not 0 < probability < 1:
        raise ValueError("probability must be strictly between zero and one.")
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("Mixture parameters must be finite.")
    if np.any(scales <= 0) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Mixture scales and weights are invalid.")
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    weights = weights / total
    lower = float(np.min(means - 10.0 * scales))
    upper = float(np.max(means + 10.0 * scales))
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        cdf = float(np.sum(weights * ndtr((midpoint - means) / scales)))
        if cdf < probability:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def paired_patient_bootstrap_delta(
    frame: pd.DataFrame,
    *,
    model_column: str,
    baseline_column: str,
    seed: int = 42,
    replicates: int = 1000,
    patient_column: str = "patient_id",
) -> dict[str, float | int | str]:
    """Patient-macro bootstrap CI for model MAE minus baseline MAE.

    Each patient contributes one mean absolute-error difference before
    resampling.  This matches the patient-level estimand used by the study and
    prevents a patient with many laboratory measurements from dominating a
    patient with fewer observations.
    """

    required = {patient_column, "target_glucose_mg_dl", model_column, baseline_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Bootstrap frame is missing columns: {sorted(missing)}")
    if replicates <= 0:
        raise ValueError("replicates must be positive.")
    valid = frame[list(required)].replace([np.inf, -np.inf], np.nan).dropna()
    patients = valid[patient_column].astype(str).drop_duplicates().to_numpy()
    if patients.size < 2:
        return {
            "patients": int(patients.size),
            "replicates": 0,
            "delta_mae_mg_dl": float("nan"),
            "lower_95": float("nan"),
            "upper_95": float("nan"),
        }

    patient_deltas: list[float] = []
    for _, values in valid.groupby(patient_column, sort=False):
        actual = values["target_glucose_mg_dl"].to_numpy(float)
        model_error = np.abs(values[model_column].to_numpy(float) - actual).mean()
        baseline_error = np.abs(
            values[baseline_column].to_numpy(float) - actual
        ).mean()
        patient_deltas.append(float(model_error - baseline_error))
    deltas = np.asarray(patient_deltas, dtype=float)

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        deltas, size=(int(replicates), len(deltas)), replace=True
    ).mean(axis=1)
    actual = valid["target_glucose_mg_dl"].to_numpy(float)
    row_weighted_delta = float(
        np.abs(valid[model_column].to_numpy(float) - actual).mean()
        - np.abs(valid[baseline_column].to_numpy(float) - actual).mean()
    )
    return {
        "patients": int(patients.size),
        "replicates": int(replicates),
        "estimand": "patient_macro_mae_model_minus_baseline",
        "delta_mae_mg_dl": float(deltas.mean()),
        "row_weighted_delta_mae_mg_dl": row_weighted_delta,
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def prediction_interval_metrics(
    frame: pd.DataFrame,
    *,
    target_column: str = "target_glucose_mg_dl",
    lower_column: str = "prediction_lower_mg_dl",
    upper_column: str = "prediction_upper_mg_dl",
    nominal_coverage: float = 0.95,
) -> dict[str, float | int]:
    """Return coverage, sharpness, and interval score on finite intervals."""

    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must be strictly between zero and one.")

    required = {target_column, lower_column, upper_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Interval frame is missing columns: {sorted(missing)}")
    actual = frame[target_column].to_numpy(dtype=float)
    lower = frame[lower_column].to_numpy(dtype=float)
    upper = frame[upper_column].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        raise ValueError("No finite target intervals are available.")
    if np.any(lower[valid] > upper[valid]):
        raise ValueError("Prediction interval lower bounds cannot exceed upper bounds.")
    valid_actual = actual[valid]
    valid_lower = lower[valid]
    valid_upper = upper[valid]
    covered = (valid_actual >= valid_lower) & (valid_actual <= valid_upper)
    width = valid_upper - valid_lower
    alpha = 1.0 - nominal_coverage
    interval_score = width.copy()
    below = valid_actual < valid_lower
    above = valid_actual > valid_upper
    interval_score[below] += (2.0 / alpha) * (
        valid_lower[below] - valid_actual[below]
    )
    interval_score[above] += (2.0 / alpha) * (
        valid_actual[above] - valid_upper[above]
    )
    coverage = float(covered.mean())
    return {
        "n_intervals": int(valid.sum()),
        "nominal_coverage": float(nominal_coverage),
        "interval_coverage": coverage,
        "coverage_error": float(coverage - nominal_coverage),
        "mean_interval_width_mg_dl": float(width.mean()),
        "median_interval_width_mg_dl": float(np.median(width)),
        "mean_interval_score_mg_dl": float(interval_score.mean()),
    }


def binary_event_metrics(
    frame: pd.DataFrame,
    *,
    target_column: str,
    probability_column: str,
    minimum_positive_events: int = 10,
    minimum_negative_events: int = 10,
) -> dict[str, float | int | bool]:
    """Evaluate a learned event probability; rank metrics need both classes."""

    required = {target_column, probability_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Event frame is missing columns: {sorted(missing)}")
    if minimum_positive_events <= 0 or minimum_negative_events <= 0:
        raise ValueError("Minimum event-support counts must be positive.")
    target = frame[target_column].to_numpy(dtype=float)
    probability = frame[probability_column].to_numpy(dtype=float)
    valid = np.isfinite(target) & np.isfinite(probability)
    if not valid.any():
        raise ValueError("No finite event targets and probabilities are available.")
    y = target[valid]
    p = probability[valid]
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Event targets must be binary.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Event probabilities must be in [0, 1].")
    positive_events = int(y.sum())
    negative_events = int(len(y) - positive_events)
    minimum_support_met = bool(
        positive_events >= minimum_positive_events
        and negative_events >= minimum_negative_events
    )
    result: dict[str, float | int] = {
        "n_event_predictions": int(valid.sum()),
        "positive_events": positive_events,
        "negative_events": negative_events,
        "event_prevalence": float(y.mean()),
        "brier_score": float(brier_score_loss(y.astype(int), p)),
        "rank_metrics_estimable": bool(np.unique(y).size == 2),
        "minimum_support_met": minimum_support_met,
        "minimum_positive_events": int(minimum_positive_events),
        "minimum_negative_events": int(minimum_negative_events),
        "auroc": float("nan"),
        "average_precision": float("nan"),
    }
    if np.unique(y).size == 2:
        result["auroc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(average_precision_score(y, p))
    return result


def missing_modality_ablation_summary(
    scenarios: Mapping[str, pd.DataFrame],
    prediction_column: str,
    *,
    reference_scenario: str = "all_available",
    target_column: str = "target_glucose_mg_dl",
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """Summarize missing-modality scenarios against the same reference rows.

    Each frame must contain the same ordered targets and patients.  Standalone
    metrics expose abstention.  Deltas are recomputed on rows where both the
    scenario and reference produced a prediction, which makes the comparison
    paired instead of rewarding a model for abstaining on difficult windows.
    Positive deltas mean the ablation is worse than the all-modality reference.
    """

    if not scenarios:
        raise ValueError("At least one ablation scenario is required.")
    if reference_scenario not in scenarios:
        raise KeyError(f"Missing reference scenario {reference_scenario!r}.")
    reference = scenarios[reference_scenario].reset_index(drop=True)
    required = {target_column, prediction_column, patient_column}
    missing = required - set(reference.columns)
    if missing:
        raise KeyError(f"Reference frame is missing columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    for name, supplied in scenarios.items():
        frame = supplied.reset_index(drop=True)
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"Scenario {name!r} is missing columns: {sorted(missing)}")
        if len(frame) != len(reference):
            raise ValueError("Every ablation scenario must contain the same rows.")
        if not frame[patient_column].astype(str).equals(
            reference[patient_column].astype(str)
        ):
            raise ValueError("Ablation scenarios must use the same ordered patients.")
        if not np.allclose(
            frame[target_column].to_numpy(dtype=float),
            reference[target_column].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise ValueError("Ablation scenarios must use identical targets.")

        metrics = neural_regression_metrics(
            frame,
            prediction_column,
            target_column=target_column,
            patient_column=patient_column,
        )
        actual = frame[target_column].to_numpy(dtype=float)
        predicted = frame[prediction_column].to_numpy(dtype=float)
        reference_prediction = reference[prediction_column].to_numpy(dtype=float)
        paired = (
            np.isfinite(actual)
            & np.isfinite(predicted)
            & np.isfinite(reference_prediction)
        )
        if paired.any():
            scenario_error = predicted[paired] - actual[paired]
            reference_error = reference_prediction[paired] - actual[paired]
            paired_delta_mae = np.abs(scenario_error).mean() - np.abs(
                reference_error
            ).mean()
            paired_delta_rmse = np.sqrt(np.mean(scenario_error**2)) - np.sqrt(
                np.mean(reference_error**2)
            )
        else:
            paired_delta_mae = float("nan")
            paired_delta_rmse = float("nan")
        rows.append(
            {
                "scenario": str(name),
                **metrics,
                "n_paired_with_reference": int(paired.sum()),
                "paired_delta_mae_mg_dl": float(paired_delta_mae),
                "paired_delta_rmse_mg_dl": float(paired_delta_rmse),
            }
        )
    return pd.DataFrame(rows)


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
