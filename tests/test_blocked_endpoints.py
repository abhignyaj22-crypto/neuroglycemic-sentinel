"""Complication / progression cannot carry a score."""
from __future__ import annotations

import unittest

from pipeline.outcomes import (
    EvidenceStatus,
    OutcomeFitError,
    may_fit_reportable,
    serve_outcome,
)


class BlockedEndpoints(unittest.TestCase):
    def test_complication_blocked_no_value(self):
        pkt = serve_outcome("diabetes_complication", value=None)
        self.assertEqual(pkt.evidence_status, EvidenceStatus.BLOCKED)
        self.assertIsNone(pkt.value)

    def test_progression_blocked_no_value(self):
        pkt = serve_outcome("diabetes_progression", value=0.9)
        self.assertEqual(pkt.evidence_status, EvidenceStatus.BLOCKED)
        self.assertIsNone(pkt.value)

    def test_cannot_fit_reportable(self):
        self.assertFalse(may_fit_reportable("diabetes_complication"))
        with self.assertRaises(OutcomeFitError):
            may_fit_reportable("diabetes_progression", raise_on_deny=True)

    def test_blocked_packet_rejects_injected_score(self):
        from pipeline.outcomes import OutcomeResult, ValueType
        with self.assertRaises(ValueError):
            OutcomeResult(
                endpoint="diabetes_complication",
                value=0.42,
                value_type=ValueType.PROBABILITY,
                evidence_status=EvidenceStatus.BLOCKED,
                abstain_reason="no",
            )


if __name__ == "__main__":
    unittest.main()
