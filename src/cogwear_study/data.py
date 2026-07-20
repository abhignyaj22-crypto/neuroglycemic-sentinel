from pathlib import Path

import pandas as pd

from .config import StudyConfig


REQUIRED_FILES = (
    "muse_eeg.csv",
    "empatica_bvp.csv",
    "empatica_eda.csv",
    "empatica_temp.csv",
)


def build_session_index(config: StudyConfig) -> pd.DataFrame:
    """Create one row per patient, condition multimodal session."""
    rows: list[dict[str, object]] = []
    missing: list[Path] = []

    for participant in config.participants:
        for condition, target in config.conditions.items():
            session_dir = config.raw_dir / str(participant) / condition
            paths = {name: session_dir / name for name in REQUIRED_FILES}
            missing.extend(path for path in paths.values() if not path.exists())
            rows.append(
                {
                    "patient_id": f"cogwear_{participant:02d}",
                    "participant_number": participant,
                    "condition": condition,
                    "target_cognitive_load": target,
                    "eeg_path": paths["muse_eeg.csv"],
                    "bvp_path": paths["empatica_bvp.csv"],
                    "eda_path": paths["empatica_eda.csv"],
                    "temp_path": paths["empatica_temp.csv"],
                }
            )

    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:8])
        raise FileNotFoundError(
            "CogWear raw files are missing."
            "Run 'scripts/download_cogwear.py` first. Example missing files:\n" + examples
        )

    return pd.DataFrame(rows).sort_values(
        ["participant_number", "target_cognitive_load"], ignore_index=True
    )


def inspect_first_session(session: pd.Series) -> dict[str, pd.DataFrame]:
    """Read five rows so the raw schemas are visible before modeling."""
    return {
        "Muse EEG": pd.read_csv(session["eeg_path"], nrows=5),
        "Empatica BVP": pd.read_csv(session["bvp_path"], nrows=5),
        "Empatica EDA": pd.read_csv(session["eda_path"], nrows=5),
        "Empatica temperature": pd.read_csv(session["temp_path"], nrows=5),
    }


def raw_file_sizes(session: pd.Series) -> pd.DataFrame:
    rows = []
    for label, column in (
        ("Muse EEG", "eeg_path"),
        ("Empatica BVP", "bvp_path"),
        ("Empatica EDA", "eda_path"),
        ("Empatica temperature", "temp_path"),
    ):
        path = Path(session[column])
        rows.append({"stream": label, "bytes": path.stat().st_size, "path": str(path)})
    return pd.DataFrame(rows)
