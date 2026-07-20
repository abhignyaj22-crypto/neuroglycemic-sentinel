from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatientSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def as_frame(self) -> pd.DataFrame:
        rows = [
            *({"patient_id": patient, "split": "train"} for patient in self.train),
            *({"patient_id": patient, "split": "validation"} for patient in self.validation),
            *({"patient_id": patient, "split": "test"} for patient in self.test),
        ]
        return pd.DataFrame(rows).sort_values(["split", "patient_id"], ignore_index=True)


def split_patients(
    patient_ids: pd.Series | list[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> PatientSplit:
    """Split participant IDs once; all their sessions/windows follow them."""
    patients = np.array(sorted(set(patient_ids)), dtype=object)
    if len(patients) < 5:
        raise ValueError("At least five patients are required for train/validation/test splitting.")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fraction must be less than one.")

    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    train_count = max(1, int(round(len(patients) * train_fraction)))
    validation_count = max(1, int(round(len(patients) * validation_fraction)))
    if train_count + validation_count >= len(patients):
        train_count = len(patients) - validation_count - 1

    split = PatientSplit(
        train=tuple(str(value) for value in patients[:train_count]),
        validation=tuple(
            str(value) for value in patients[train_count : train_count + validation_count]
        ),
        test=tuple(str(value) for value in patients[train_count + validation_count :]),
    )
    assert not (set(split.train) & set(split.validation))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.validation) & set(split.test))
    return split


def attach_split(frame: pd.DataFrame, split: PatientSplit) -> pd.DataFrame:
    lookup = {
        **{patient: "train" for patient in split.train},
        **{patient: "validation" for patient in split.validation},
        **{patient: "test" for patient in split.test},
    }
    result = frame.copy()
    result["split"] = result["patient_id"].map(lookup)
    if result["split"].isna().any():
        raise ValueError("At least one feature row has no patient split.")
    return result
