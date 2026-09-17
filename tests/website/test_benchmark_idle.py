"""A competing runner must never enter the benchmark's measurement window."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_idle


class IsolationTest(unittest.TestCase):
    def test_other_process_cannot_acquire_machine_lock(self):
        code = "import sys; sys.path.insert(0, sys.argv[1]); import benchmark_idle; "
        code += "\nwith benchmark_idle.exclusive(): print('acquired')"
        with tempfile.TemporaryDirectory() as temp, benchmark_idle.exclusive():
            result = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts")],
                                    env=dict(os.environ, TMPDIR=temp), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("acquired", result.stdout)
        self.assertIn("machine-wide lock", result.stderr)

    def test_busy_machine_times_out_without_admitting_a_run(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(benchmark_idle, "busy_percent", return_value=99):
            path = Path(temp) / "idle.jsonl"
            with self.assertRaisesRegex(RuntimeError, "no benchmark started"):
                benchmark_idle.wait_idle(path, timeout=.002)
            self.assertIn('"cpu_busy_percent": 99', path.read_text())
