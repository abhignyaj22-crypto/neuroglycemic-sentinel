"""DiaTrend ingestion and causal glucose-window construction.

DiaTrend stores one Excel workbook per participant with CGM, Bolus, and
occasionally Basal sheets.  This adapter follows that published schema and
produces the same-patient aligned-window contract consumed by the neural model.
No protected records are bundled with the software and no interpolation is used
to create labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .neural_dataset import target_column, target_time_column


DIATREND_CGM_FEATURES = (
    "cgm_current_mg_dl",
    "cgm_lag_5m_mg_dl",
    "cgm_lag_15m_mg_dl",
    "cgm_lag_30m_mg_dl",
    "cgm_lag_60m_mg_dl",
    "cgm_lag_120m_mg_dl",
    "cgm_delta_5m_mg_dl",
    "cgm_delta_15m_mg_dl",
    "cgm_delta_30m_mg_dl",
    "cgm_mean_30m_mg_dl",
    "cgm_sd_30m_mg_dl",
    "cgm_mean_60m_mg_dl",
    "cgm_sd_60m_mg_dl",
    "cgm_mean_120m_mg_dl",
    "cgm_sd_120m_mg_dl",
    "cgm_slope_30m_mg_dl_per_min",
    "cgm_slope_60m_mg_dl_per_min",
)

DIATREND_EVENT_FEATURES = (
    "events_bolus_units_30m",
    "events_bolus_units_60m",
    "events_bolus_units_120m",
    "events_carbohydrate_g_30m",
    "events_carbohydrate_g_60m",
    "events_carbohydrate_g_120m",
    "events_minutes_since_bolus",
    "events_current_basal_units_per_hour",
    "events_insulin_on_board_units",
)


@dataclass(frozen=True)
class DiaTrendBuildConfig:
    source_timezone: str
    horizons_minutes: tuple[int, ...] = (30, 60, 90, 120)
    grid_minutes: int = 5
    history_minutes: int = 120
    anchor_stride_minutes: int = 15
    minimum_history_coverage: float = 0.75
    glucose_min_mg_dl: float = 20.0
    glucose_max_mg_dl: float = 600.0
    cohort_id: str = "diatrend"

    def __post_init__(self) -> None:
        if not self.source_timezone.strip():
            raise ValueError("source_timezone is required; DiaTrend timestamps are naive.")
        if not self.horizons_minutes or any(value <= 0 for value in self.horizons_minutes):
            raise ValueError("At least one positive forecast horizon is required.")
        if self.grid_minutes <= 0 or self.history_minutes < self.grid_minutes:
            raise ValueError("The grid and history durations are invalid.")
        if self.anchor_stride_minutes <= 0 or self.anchor_stride_minutes % self.grid_minutes:
            raise ValueError("anchor_stride_minutes must be a positive grid multiple.")
        if not 0 < self.minimum_history_coverage <= 1:
            raise ValueError("minimum_history_coverage must be in (0, 1].")
        if self.glucose_min_mg_dl <= 0 or self.glucose_min_mg_dl >= self.glucose_max_mg_dl:
            raise ValueError("Glucose validity bounds must be positive and ordered.")


@dataclass(frozen=True)
class DiaTrendPatientAudit:
    patient_id: str
    source_file: str
    raw_cgm_rows: int
    valid_cgm_rows: int
    duplicate_cgm_timestamps: int
    bolus_rows: int
    basal_rows: int
    aligned_windows: int
    cgm_start_utc: str | None
    cgm_stop_utc: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _localize_timestamp(values: pd.Series, timezone_name: str, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{name} contains invalid timestamps.")
    if parsed.dt.tz is None:
        try:
            parsed = parsed.dt.tz_localize(
                timezone_name, ambiguous="NaT", nonexistent="NaT"
            )
        except Exception as exc:  # zoneinfo/pytz raises backend-specific errors.
            raise ValueError(f"Invalid source timezone {timezone_name!r}.") from exc
        if parsed.isna().any():
            raise ValueError(
                f"{name} contains ambiguous/nonexistent local times; resolve them before training."
            )
    return parsed.dt.tz_convert("UTC")


def _read_sheet(path: Path, sheet_name: str, *, required: bool) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ImportError as exc:
        raise RuntimeError(
            "DiaTrend Excel ingestion requires openpyxl; install the project requirements."
        ) from exc
    except ValueError:
        if required:
            raise ValueError(f"{path.name} is missing required sheet {sheet_name!r}.")
        return pd.DataFrame()


def read_diatrend_workbook(
    path: Path, *, source_timezone: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and validate one workbook using the dataset's published columns."""

    path = Path(path)
    cgm = _read_sheet(path, "CGM", required=True)
    bolus = _read_sheet(path, "Bolus", required=False)
    basal = _read_sheet(path, "Basal", required=False)
    if not {"date", "mg/dL"}.issubset(cgm.columns):
        raise ValueError(f"{path.name} CGM sheet requires date and mg/dL columns.")
    cgm = cgm[["date", "mg/dL"]].copy()
    cgm["date"] = _localize_timestamp(cgm["date"], source_timezone, name="CGM.date")
    cgm["mg/dL"] = pd.to_numeric(cgm["mg/dL"], errors="coerce")

    if not bolus.empty:
        if "date" not in bolus:
            raise ValueError(f"{path.name} Bolus sheet requires a date column.")
        bolus = bolus.copy()
        bolus["date"] = _localize_timestamp(
            bolus["date"], source_timezone, name="Bolus.date"
        )
        for column in (
            "normal",
            "carbInput",
            "insulinOnBoard",
            "bgInput",
            "insulinCarbRatio",
        ):
            if column not in bolus:
                bolus[column] = np.nan
            bolus[column] = pd.to_numeric(bolus[column], errors="coerce")

    if not basal.empty:
        if not {"date", "duration", "rate"}.issubset(basal.columns):
            raise ValueError(
                f"{path.name} Basal sheet requires date, duration, and rate columns."
            )
        basal = basal.copy()
        basal["date"] = _localize_timestamp(
            basal["date"], source_timezone, name="Basal.date"
        )
        for column in ("duration", "rate"):
            basal[column] = pd.to_numeric(basal[column], errors="coerce")
    return cgm, bolus, basal


def _rolling_slope(values: pd.Series, periods: int, grid_minutes: int) -> pd.Series:
    # Least-squares slope with a fixed equally spaced time basis. Missing values
    # remain missing; the network receives the observation mask separately.
    x = np.arange(periods, dtype=float) * float(grid_minutes)
    x = x - x.mean()
    denominator = float(np.sum(x * x))

    def slope(window: np.ndarray) -> float:
        if not np.isfinite(window).all():
            return np.nan
        return float(np.sum(x * (window - window.mean())) / denominator)

    return values.rolling(periods, min_periods=periods).apply(slope, raw=True)


def _sum_events_on_grid(
    frame: pd.DataFrame, column: str, grid_index: pd.DatetimeIndex, grid_minutes: int
) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(np.nan, index=grid_index, dtype=float)
    values = frame[["date", column]].dropna(subset=[column]).copy()
    if values.empty:
        return pd.Series(np.nan, index=grid_index, dtype=float)
    # Right-label the interval so an event at 12:04 first appears in the 12:05
    # forecast, never in the already-issued 12:00 forecast.
    values["grid_time"] = values["date"].dt.ceil(f"{grid_minutes}min")
    grouped = values.groupby("grid_time")[column].sum()
    return grouped.reindex(grid_index).astype(float)


def _nearest_future_cgm(
    anchors: pd.DatetimeIndex,
    raw: pd.DataFrame,
    *,
    horizon_minutes: int,
    tolerance_minutes: float,
) -> tuple[pd.Series, pd.Series]:
    """Match nominal horizons to observed CGM without interpolation."""

    query = pd.DataFrame(
        {
            "anchor_time": anchors,
            "nominal_target_time": anchors + pd.Timedelta(minutes=horizon_minutes),
        }
    ).sort_values("nominal_target_time")
    reference = raw[["date", "mg/dL"]].rename(
        columns={"date": "observed_target_time", "mg/dL": "observed_target"}
    ).sort_values("observed_target_time")
    matched = pd.merge_asof(
        query,
        reference,
        left_on="nominal_target_time",
        right_on="observed_target_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    ).sort_values("anchor_time")
    valid = matched["observed_target_time"].gt(matched["anchor_time"])
    return (
        matched["observed_target"].where(valid).reset_index(drop=True),
        matched["observed_target_time"].where(valid).reset_index(drop=True),
    )


def build_diatrend_patient_windows(
    patient_id: str,
    cgm: pd.DataFrame,
    bolus: pd.DataFrame,
    basal: pd.DataFrame,
    *,
    config: DiaTrendBuildConfig,
) -> pd.DataFrame:
    """Build causal, multi-horizon windows for one DiaTrend participant."""

    raw = cgm.dropna(subset=["date", "mg/dL"]).copy()
    raw = raw.loc[
        raw["mg/dL"].between(config.glucose_min_mg_dl, config.glucose_max_mg_dl)
    ]
    if raw.empty:
        return pd.DataFrame()
    raw = raw.sort_values("date")
    raw["grid_time"] = raw["date"].dt.ceil(f"{config.grid_minutes}min")
    glucose = raw.groupby("grid_time")["mg/dL"].median().sort_index()
    glucose_available_time = raw.groupby("grid_time")["date"].max().sort_index()
    grid_index = pd.date_range(
        glucose.index.min(), glucose.index.max(), freq=f"{config.grid_minutes}min"
    )
    glucose = glucose.reindex(grid_index)
    observed = glucose.notna().astype(float)
    history_periods = config.history_minutes // config.grid_minutes + 1
    history_coverage = observed.rolling(
        history_periods, min_periods=history_periods
    ).mean()

    feature = pd.DataFrame(index=grid_index)
    feature["cgm_current_mg_dl"] = glucose
    for minutes in (5, 15, 30, 60, 120):
        periods = minutes // config.grid_minutes
        feature[f"cgm_lag_{minutes}m_mg_dl"] = glucose.shift(periods)
    for minutes in (5, 15, 30):
        feature[f"cgm_delta_{minutes}m_mg_dl"] = glucose - glucose.shift(
            minutes // config.grid_minutes
        )
    for minutes in (30, 60, 120):
        periods = minutes // config.grid_minutes + 1
        feature[f"cgm_mean_{minutes}m_mg_dl"] = glucose.rolling(
            periods, min_periods=periods
        ).mean()
        feature[f"cgm_sd_{minutes}m_mg_dl"] = glucose.rolling(
            periods, min_periods=periods
        ).std(ddof=0)
    for minutes in (30, 60):
        periods = minutes // config.grid_minutes + 1
        feature[f"cgm_slope_{minutes}m_mg_dl_per_min"] = _rolling_slope(
            glucose, periods, config.grid_minutes
        )
    bolus_units = _sum_events_on_grid(bolus, "normal", grid_index, config.grid_minutes)
    carbohydrate = _sum_events_on_grid(
        bolus, "carbInput", grid_index, config.grid_minutes
    )
    for minutes in (30, 60, 120):
        periods = minutes // config.grid_minutes + 1
        feature[f"events_bolus_units_{minutes}m"] = bolus_units.rolling(
            periods, min_periods=1
        ).sum()
        feature[f"events_carbohydrate_g_{minutes}m"] = carbohydrate.rolling(
            periods, min_periods=1
        ).sum()

    bolus_times = pd.Series(pd.NaT, index=grid_index, dtype="datetime64[ns, UTC]")
    if not bolus.empty and bolus["normal"].notna().any():
        positive_bolus = bolus.loc[bolus["normal"].gt(0), ["date"]].copy()
        positive_bolus["grid_time"] = positive_bolus["date"].dt.ceil(
            f"{config.grid_minutes}min"
        )
        bolus_times.update(
            positive_bolus.groupby("grid_time")["date"].max().reindex(grid_index)
        )
    last_bolus = bolus_times.ffill()
    feature["events_minutes_since_bolus"] = (
        pd.Series(grid_index, index=grid_index) - last_bolus
    ).dt.total_seconds() / 60.0
    feature["events_minutes_since_bolus"] = feature[
        "events_minutes_since_bolus"
    ].where(feature["events_minutes_since_bolus"].le(120.0))

    feature["events_current_basal_units_per_hour"] = np.nan
    active_basal = pd.Series(False, index=grid_index)
    if not basal.empty:
        valid_basal = basal.dropna(
            subset=["date", "duration", "rate"]
        ).loc[lambda frame: frame["duration"].gt(0)].sort_values("date")
        if not valid_basal.empty:
            valid_basal = valid_basal.copy()
            # DiaTrend publishes basal duration in milliseconds. A basal rate is
            # valid only inside that recorded infusion interval; indefinite
            # last-observation carry-forward would invent exposure.
            valid_basal["stop_time"] = valid_basal["date"] + pd.to_timedelta(
                valid_basal["duration"], unit="ms"
            )
            aligned = pd.merge_asof(
                pd.DataFrame({"grid_time": grid_index}),
                valid_basal[["date", "stop_time", "rate"]].rename(
                    columns={"date": "basal_available_time"}
                ),
                left_on="grid_time",
                right_on="basal_available_time",
                direction="backward",
            )
            active = aligned["grid_time"].lt(aligned["stop_time"])
            active_basal = pd.Series(active.to_numpy(), index=grid_index)
            feature["events_current_basal_units_per_hour"] = aligned["rate"].where(
                active
            ).to_numpy()
    feature["events_insulin_on_board_units"] = np.nan
    if not bolus.empty and bolus["insulinOnBoard"].notna().any():
        valid_iob = bolus.dropna(subset=["date", "insulinOnBoard"]).sort_values("date")
        aligned_iob = pd.merge_asof(
            pd.DataFrame({"grid_time": grid_index}),
            valid_iob[["date", "insulinOnBoard"]].rename(
                columns={"date": "iob_available_time"}
            ),
            left_on="grid_time",
            right_on="iob_available_time",
            direction="backward",
            tolerance=pd.Timedelta(minutes=30),
        )
        feature["events_insulin_on_board_units"] = aligned_iob[
            "insulinOnBoard"
        ].to_numpy()

    # A gap between logged boluses is unknown, not proof that no bolus/meal
    # occurred. The event expert is available only while at least one recorded
    # feature contributes to the causal lookback or an explicit basal interval.
    event_feature_frame = feature[list(DIATREND_EVENT_FEATURES)]
    event_available = event_feature_frame.notna().any(axis=1)
    feature.loc[~event_available, list(DIATREND_EVENT_FEATURES)] = np.nan

    event_available_time = pd.Series(
        pd.NaT, index=grid_index, dtype="datetime64[ns, UTC]"
    )
    if not bolus.empty:
        logged = bolus.loc[
            bolus[["normal", "carbInput", "insulinOnBoard"]].notna().any(axis=1),
            ["date"],
        ].sort_values("date")
        if not logged.empty:
            latest = pd.merge_asof(
                pd.DataFrame({"grid_time": grid_index}),
                logged.rename(columns={"date": "bolus_available_time"}),
                left_on="grid_time",
                right_on="bolus_available_time",
                direction="backward",
                tolerance=pd.Timedelta(minutes=120),
            )["bolus_available_time"]
            event_available_time.update(pd.Series(latest.to_numpy(), index=grid_index))
    if not basal.empty and "aligned" in locals():
        basal_times = pd.Series(
            aligned["basal_available_time"].where(active).to_numpy(), index=grid_index
        )
        event_available_time = pd.concat(
            [event_available_time, basal_times], axis=1
        ).max(axis=1)
    event_available_time = event_available_time.where(event_available)

    result = feature.copy()
    result.insert(0, "anchor_time", grid_index)
    result.insert(0, "cohort_id", config.cohort_id)
    result.insert(0, "patient_id", str(patient_id))
    result["session_id"] = f"{config.cohort_id}:{patient_id}"
    result["cgm_available"] = glucose.notna().to_numpy()
    result["cgm_quality"] = history_coverage.fillna(0.0).to_numpy()
    cgm_times = glucose_available_time.reindex(grid_index)
    result["cgm_staleness_minutes"] = (
        pd.Series(grid_index, index=grid_index) - cgm_times
    ).dt.total_seconds().div(60.0).to_numpy()
    result["cgm_clock_uncertainty_ms"] = 0.0
    result["cgm_patient_id"] = str(patient_id)
    result["cgm_cohort_id"] = config.cohort_id
    result["cgm_anchor_time"] = result["anchor_time"]
    result["cgm_available_time"] = cgm_times.to_numpy()
    result["events_available"] = event_available.to_numpy()
    result["events_quality"] = (
        event_feature_frame.notna().mean(axis=1).where(event_available, 0.0).to_numpy()
    )
    result["events_staleness_minutes"] = (
        (pd.Series(grid_index, index=grid_index) - event_available_time)
        .dt.total_seconds()
        .div(60.0)
        .where(event_available, 0.0)
        .to_numpy()
    )
    result["events_clock_uncertainty_ms"] = 0.0
    result["events_patient_id"] = np.where(event_available, str(patient_id), None)
    result["events_cohort_id"] = np.where(event_available, config.cohort_id, None)
    result["events_anchor_time"] = result["anchor_time"].where(event_available)
    result["events_available_time"] = event_available_time.to_numpy()

    for horizon in config.horizons_minutes:
        target, actual_time = _nearest_future_cgm(
            grid_index,
            raw,
            horizon_minutes=horizon,
            tolerance_minutes=float(config.grid_minutes),
        )
        result[target_column(horizon)] = target.to_numpy()
        result[target_time_column(horizon)] = actual_time.to_numpy()

    stride = config.anchor_stride_minutes // config.grid_minutes
    eligible = (
        result["cgm_available"]
        & result["cgm_quality"].ge(config.minimum_history_coverage)
        & (np.arange(len(result)) % stride == 0)
    )
    target_names = [target_column(value) for value in config.horizons_minutes]
    eligible &= result[target_names].notna().any(axis=1)
    result = result.loc[eligible].reset_index(drop=True)
    ordered = [
        "patient_id",
        "cohort_id",
        "session_id",
        "anchor_time",
        *DIATREND_CGM_FEATURES,
        *DIATREND_EVENT_FEATURES,
        *[
            value
            for modality in ("cgm", "events")
            for value in (
                f"{modality}_available",
                f"{modality}_quality",
                f"{modality}_staleness_minutes",
                f"{modality}_clock_uncertainty_ms",
                f"{modality}_patient_id",
                f"{modality}_cohort_id",
                f"{modality}_anchor_time",
                f"{modality}_available_time",
            )
        ],
        *[
            value
            for horizon in config.horizons_minutes
            for value in (target_column(horizon), target_time_column(horizon))
        ],
    ]
    return result[ordered]


def discover_diatrend_workbooks(source_directory: Path) -> list[Path]:
    source_directory = Path(source_directory)
    if not source_directory.is_dir():
        raise NotADirectoryError(source_directory)
    workbooks = sorted(
        path
        for path in source_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".xlsx", ".xlsm"}
        and not path.name.startswith("~$")
    )
    if not workbooks:
        raise FileNotFoundError(f"No DiaTrend Excel workbooks found in {source_directory}.")
    return workbooks


def build_diatrend_dataset(
    workbooks: Iterable[Path], *, config: DiaTrendBuildConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for path_value in workbooks:
        path = Path(path_value)
        patient_id = path.stem
        cgm, bolus, basal = read_diatrend_workbook(
            path, source_timezone=config.source_timezone
        )
        duplicate_count = int(cgm.duplicated("date").sum())
        valid = cgm["mg/dL"].between(
            config.glucose_min_mg_dl, config.glucose_max_mg_dl
        ) & cgm["mg/dL"].notna()
        patient_windows = build_diatrend_patient_windows(
            patient_id, cgm, bolus, basal, config=config
        )
        windows.append(patient_windows)
        valid_times = cgm.loc[valid, "date"]
        audits.append(
            DiaTrendPatientAudit(
                patient_id=patient_id,
                source_file=path.name,
                raw_cgm_rows=len(cgm),
                valid_cgm_rows=int(valid.sum()),
                duplicate_cgm_timestamps=duplicate_count,
                bolus_rows=len(bolus),
                basal_rows=len(basal),
                aligned_windows=len(patient_windows),
                cgm_start_utc=(valid_times.min().isoformat() if not valid_times.empty else None),
                cgm_stop_utc=(valid_times.max().isoformat() if not valid_times.empty else None),
            ).as_dict()
        )
    combined = pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
    if combined.empty:
        raise ValueError("DiaTrend ingestion produced no eligible causal windows.")
    return combined, pd.DataFrame(audits)


def diatrend_feature_registry() -> dict[str, tuple[str, ...]]:
    return {
        "cgm": DIATREND_CGM_FEATURES,
        "events": DIATREND_EVENT_FEATURES,
    }
