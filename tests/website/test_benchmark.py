"""Evidence must survive failure, and comparisons must not manufacture wins."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class MetricsTest(unittest.TestCase):
    def test_real_archived_bank_and_missing_instrumentation(self):
        text = (ROOT / "logs/2026-06-17_gcart_spec-bank-vs-mdbx/bank_compare.log").read_text()
        metrics = benchmark.parse_bank(text)
        values = {(m["engine"], m["name"]): m["value"] for m in metrics}
        self.assertEqual(values["sal_art_ref", "write"], 551010)
        self.assertEqual(values["gcart_spec", "live"], 39)
        self.assertNotIn(("sal_art_ref", "live"), values)

    def test_correctness_failure_and_absent_data_cannot_pass(self):
        row = "big 100 200 300 400 500 0 64 FAIL"
        for output in (row, "benchmark crashed before printing metrics", ""):
            with self.subTest(output=output), self.assertRaises(ValueError):
                benchmark.parse_bank(output)

    def test_duplicate_repetitions_are_not_silently_averaged(self):
        row = "big 100 200 300 400 500 0 64 PASS\n"
        with self.assertRaises(ValueError):
            benchmark.parse_bank(row * 2)

    def test_nonfinite_negative_and_duplicate_metrics_rejected(self):
        metric = dict(engine="big", name="rate", value=10, unit="ops/s", direction="higher")
        for value in (-1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.validate_metrics([dict(metric, value=value)])
        with self.assertRaises(ValueError):
            benchmark.validate_metrics([metric, metric])


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.before = {"comparison": {"machine_id": "runner-a", "durability": "fsync"}, "metrics": [
            dict(engine="big", name="throughput", value=100, unit="ops/s", direction="higher"),
            dict(engine="big", name="latency", value=100, unit="ns/op", direction="lower")]}
        self.after = copy.deepcopy(self.before)

    def test_throughput_loss_and_latency_growth_are_regressions(self):
        self.after["metrics"][0]["value"] = 90
        self.after["metrics"][1]["value"] = 120
        rows, regressed = benchmark.compare_runs(self.before, self.after, 5)
        self.assertTrue(regressed)
        self.assertAlmostEqual(rows[0]["regression_percent"], 10)
        self.assertAlmostEqual(rows[1]["regression_percent"], 20)

    def test_improvements_and_threshold_boundary(self):
        self.after["metrics"][0]["value"] = 95
        self.after["metrics"][1]["value"] = 80
        self.assertFalse(benchmark.compare_runs(self.before, self.after, 5)[1])

    def test_durability_and_machine_mismatch_rejected(self):
        for key in ("machine_id", "durability"):
            after = copy.deepcopy(self.after)
            after["comparison"][key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                benchmark.compare_runs(self.before, after, 5)

    def test_missing_metrics_and_changed_units_rejected(self):
        self.after["metrics"].pop()
        with self.assertRaises(ValueError):
            benchmark.compare_runs(self.before, self.after, 5)
        self.after = copy.deepcopy(self.before)
        self.after["metrics"][0]["unit"] = "million ops/s"
        with self.assertRaises(ValueError):
            benchmark.compare_runs(self.before, self.after, 5)

    def test_zero_baseline_is_explicit(self):
        self.before["metrics"][0]["value"] = 0
        with self.assertRaises(ValueError):
            benchmark.compare_runs(self.before, self.after, 5)


class CaptureTest(unittest.TestCase):
    def run_capture(self, directory, command, parser="none"):
        return subprocess.run([sys.executable, str(ROOT / "scripts/benchmark.py"), "record",
            "--label", "recorder-test", "--workload", "test-fixture", "--machine", "test-runner",
            "--build", "test", "--compiler", "test", "--durability", "none",
            "--output", directory, "--parser", parser, "--", *command], capture_output=True, text=True)

    def test_failed_process_retains_both_streams_and_exit_code(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_capture(temp, [sys.executable, "-c", "import sys; print('partial data'); print('failure detail', file=sys.stderr); sys.exit(7)"])
            self.assertEqual(result.returncode, 7)
            run = next(Path(temp).iterdir())
            metadata = json.loads((run / "run.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["exit_code"], 7)
            self.assertIn("partial data", (run / "stdout.log").read_text())
            self.assertIn("failure detail", (run / "stderr.log").read_text())
            with self.assertRaises(ValueError):
                benchmark.read_run(run)

    def test_unique_archives_and_json_metrics(self):
        metrics = [dict(engine="big", name="rate", value=123, unit="ops/s", direction="higher")]
        with tempfile.TemporaryDirectory() as temp:
            for _ in range(2):
                result = self.run_capture(temp, [sys.executable, "-c", f"print({json.dumps(metrics)!r})"], parser="json")
                self.assertEqual(result.returncode, 0, result.stderr)
            records = [benchmark.read_run(path) for path in Path(temp).iterdir()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["metrics"], metrics)
            self.assertFalse(benchmark.compare_runs(*records, 5)[1])

    def test_success_exit_with_bad_metrics_is_failed_record(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_capture(temp, [sys.executable, "-c", "print('missing metrics')"], parser="bank")
            self.assertEqual(result.returncode, 2)
            run = json.loads(next(Path(temp).glob("*/run.json")).read_text())
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
