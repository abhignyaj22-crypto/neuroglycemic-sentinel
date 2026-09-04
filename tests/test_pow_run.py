"""POW walk is the honest subset: traces + blocked endpoints, not a fused-LLM product."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PowWalk(unittest.TestCase):
    def test_dry_run_pow_exits_zero_and_blocks_complication(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--dry-run", "--profile", "pow"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        combined = proc.stdout + proc.stderr
        self.assertIn("[trace]", combined)
        self.assertIn("diabetes_complication", combined)
        self.assertIn("blocked", combined.lower())
        self.assertIn("trainable=False", combined)
        self.assertNotIn("fusion wins", combined.lower())
        walk = json.loads((ROOT / "outputs" / "_scratch" / "pow" / "walk.json").read_text())
        self.assertFalse(walk["pow_llm_predictor"])
        self.assertFalse(walk["pow_emotiv_galea_trainable"])
        self.assertIsNone(walk["blocked"]["diabetes_complication"]["value"])
        self.assertIsNone(walk.get("lab_harness"))

    def test_trace_array_prints_shape(self):
        from utils.trace import trace_array
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            trace_array("demo", type("A", (), {"shape": (4, 8), "dtype": "float64"})())
        self.assertIn("shape=(4, 8)", buf.getvalue())
        self.assertIn("dtype=float64", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
