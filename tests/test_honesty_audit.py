"""Honesty audit for the public extract — fusion is not a winner, numbers trace to boards."""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_WINNER_PHRASES = (
    "fusion wins",
    "learned fusion is the best",
    "CACMF outperforms the specialist",
    "LLM as a predictor headline",
)


class ExtractHonestyAudit(unittest.TestCase):
    def test_readme_is_research_stage_not_diagnosis(self):
        text = (ROOT / "README.md").read_text().lower()
        self.assertIn("not a diagnosis", text)
        self.assertIn("research-stage", text)
        self.assertIn("--profile pow", text)
        self.assertIn("loses", text)

    def test_readme_does_not_sell_fusion_or_llm_as_winner(self):
        text = (ROOT / "README.md").read_text()
        for phrase in FORBIDDEN_WINNER_PHRASES:
            self.assertNotIn(phrase.lower(), text.lower())

    def test_synthesis_is_shipped(self):
        p = ROOT / "docs" / "RESEARCH_SYNTHESIS.md"
        self.assertTrue(p.is_file())
        text = p.read_text()
        self.assertIn("learned multimodal fusion does not reliably beat", text.lower())

    def test_config_stays_cpu_and_fm_probes_off(self):
        import json
        cfg_path = ROOT / "config.json"
        if not cfg_path.is_file():
            cfg_path = ROOT / "config.example.json"
        cfg = json.loads(cfg_path.read_text())
        self.assertEqual(cfg["device"], "cpu")
        self.assertFalse(cfg["fm_probes"])

    def test_paper_mh_and_cgmacros_boards_are_committed(self):
        mh = ROOT / "outputs" / "scoreboards" / "paper_mh_5x5" / "benchmark_scoreboard.csv"
        glu = ROOT / "outputs" / "scoreboards" / "paper_cgmacros_5x5" / "benchmark_scoreboard.csv"
        self.assertTrue(mh.is_file())
        self.assertTrue(glu.is_file())

    def test_paper_validate_profile_passes(self):
        import subprocess
        import sys
        proc = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--profile", "paper-validate"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout[-2000:] + proc.stderr[-1000:])

    def test_phase1_boards_are_committed_negatives(self):
        eeg = ROOT / "outputs" / "scoreboards" / "phase1_eegmat_workload" / "benchmark_scoreboard.csv"
        glu = ROOT / "outputs" / "scoreboards" / "phase1_cgmacros_glucose" / "benchmark_scoreboard.csv"
        self.assertTrue(eeg.is_file())
        self.assertTrue(glu.is_file())
        with eeg.open() as fh:
            eeg_row = next(csv.DictReader(fh))
        with glu.open() as fh:
            glu_row = next(csv.DictReader(fh))
        self.assertEqual(eeg_row["meets_>=50%"], "False")
        self.assertEqual(glu_row["meets_>=50%"], "False")
        self.assertEqual(glu_row["best_baseline"], "ridge_raw_sequence")


if __name__ == "__main__":
    unittest.main()
