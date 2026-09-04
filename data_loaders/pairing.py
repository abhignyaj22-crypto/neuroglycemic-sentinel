"""Fail-closed pairing: never train or fuse unmatched EEG onto CGMacros glucose."""
from __future__ import annotations


class CrossCohortPairingForbidden(RuntimeError):
    """Raised when a caller tries to pair recordings that are not the same people."""


_BANNED_EEG = (
    "wesad",
    "mumtaz",
    "eegmat",
    "deap",
    "case",
    "clamp",
    "hypoglycemia",
    "galea",
    "emotiv",
    "zenodo",
    "mimic",
)


def refuse_unmatched_eeg_glucose(eeg_source: str, wearable_source: str) -> None:
    eeg = (eeg_source or "").lower()
    wear = (wearable_source or "").lower()
    if any(token in eeg for token in _BANNED_EEG) and "cgmacro" in wear:
        raise CrossCohortPairingForbidden(
            f"refuse {eeg_source!r} × {wearable_source!r}: not the same people"
        )
