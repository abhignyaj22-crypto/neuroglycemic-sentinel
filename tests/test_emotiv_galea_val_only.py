"""EMOTIV / Galea are val/test only and cannot pair with CGMacros."""
from __future__ import annotations

import unittest

from data_loaders.emotiv import EmotivTrainForbidden, load as load_emotiv
from data_loaders.galea import GaleaTrainForbidden, load as load_galea
from data_loaders.pairing import CrossCohortPairingForbidden, refuse_unmatched_eeg_glucose


class EmotivGaleaValOnly(unittest.TestCase):
    def test_emotiv_val_only_ok(self):
        rec = load_emotiv(split="val_only")
        self.assertFalse(rec["trainable"])

    def test_emotiv_train_forbidden(self):
        with self.assertRaises(EmotivTrainForbidden):
            load_emotiv(split="train")

    def test_galea_train_forbidden(self):
        with self.assertRaises(GaleaTrainForbidden):
            load_galea(split="fit")

    def test_emotiv_times_cgmacros_refused(self):
        with self.assertRaises(CrossCohortPairingForbidden):
            load_emotiv(split="val", wearable_source="cgmacros")

    def test_galea_times_cgmacros_refused(self):
        with self.assertRaises(CrossCohortPairingForbidden):
            refuse_unmatched_eeg_glucose("galea", "cgmacros")


if __name__ == "__main__":
    unittest.main()
