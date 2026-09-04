"""Dry-run does not train and still refuse unmatched pairing in the demo infer."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DryRun(unittest.TestCase):
    def test_main_dry_run_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--dry-run", "--profile", "smoke"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**dict(**__import__("os").environ), "PYTHONPATH": str(ROOT)},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        combined = proc.stdout + proc.stderr
        self.assertIn("dry-run", combined)
        self.assertTrue("abstain" in combined.lower() or "refuse" in combined.lower(), combined)
