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
