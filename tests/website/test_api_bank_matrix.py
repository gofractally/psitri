"""A failing or unrun shim must not be promoted into a performance series."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import api_bank_matrix


class MatrixTest(unittest.TestCase):
    def test_failed_smoke_skips_measurement_and_keeps_other_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def run(args):
                calls.append((args.engines[0], args.profile))
                p = args.output / "run"; p.mkdir(parents=True)
                failed = args.engines[0] == "mdbx"
                (p / "manifest.json").write_text(json.dumps(dict(records=[dict(status="failed" if failed else "passed")], status="failed" if failed else "passed")))
                return int(failed)
            args = SimpleNamespace(output=Path(temp), machine="test", build_mdbx=Path(temp), build_integrations=Path(temp), repeats=3, timeout=10, idle_timeout=10)
            with patch.object(api_bank_matrix.fair_bank, "WORKLOAD", "api-bank-v1"), patch.object(api_bank_matrix.fair_bank, "run", side_effect=run), patch.object(api_bank_matrix.benchmark_idle, "exclusive", return_value=contextlib.nullcontext()):
                self.assertEqual(api_bank_matrix.collect(args), 1)
            self.assertNotIn(("mdbx", "performance"), calls)
            self.assertEqual(calls, [("mdbx", "smoke"), ("rocksdb", "smoke"), ("rocksdb", "performance"), ("sqlite", "smoke"), ("sqlite", "performance")])
            manifest = json.loads(next(Path(temp).glob("*/matrix.json")).read_text())
            self.assertEqual([p["status"] for p in manifest["pairs"]], ["failed", "passed", "passed"])
