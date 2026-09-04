"""EMOTIV EPOC X — validation/testing only. Never used to fit a reportable model."""
from __future__ import annotations

from data_loaders.pairing import refuse_unmatched_eeg_glucose

SPLIT = "val_only"
_ALLOWED = frozenset({"val", "test", "val_only"})


class EmotivTrainForbidden(RuntimeError):
    """EMOTIV recordings are n=1 schema/validation, not a training cohort."""


def load(*, split: str = SPLIT, eeg_source: str = "emotiv", wearable_source: str = "",
         data_dir=None, **_kwargs):
    if split not in _ALLOWED:
        raise EmotivTrainForbidden(
            f"EMOTIV split={split!r} is forbidden; allowed={sorted(_ALLOWED)} (never train)"
        )
    refuse_unmatched_eeg_glucose(eeg_source or "emotiv", wearable_source)
    return {
        "device": "EMOTIV EPOC X",
        "split": split,
        "trainable": False,
        "data_dir": data_dir,
        "note": "schema-validation / n=1 disclosure only; not a scoreboard cohort",
    }
