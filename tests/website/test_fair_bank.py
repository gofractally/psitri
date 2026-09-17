"""Bad correctness, durability, or cohort evidence must never produce a comparison."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import fair_bank


class FairBankTest(unittest.TestCase):
    def setUp(self):
        self.result = json.loads((Path(__file__).parent / "fixtures/fair-bank.json").read_text())
        self.config = copy.deepcopy(self.result["config"])

    def validate(self):
        fair_bank.validate_result(self.result, "small", self.config)

    def test_actual_smoke_result_is_accepted(self):
        self.validate()

    def test_failed_or_missing_checks_rejected(self):
        for key in fair_bank.CHECKS:
            with self.subTest(check=key):
                saved = self.result["validation"].pop(key)
                with self.assertRaises(ValueError):
                    self.validate()
                self.result["validation"][key] = False
                with self.assertRaises(ValueError):
                    self.validate()
                self.result["validation"][key] = saved

    def test_durability_and_snapshot_batch_mismatch_rejected(self):
        self.config["durability"] = "fsync"
        with self.assertRaisesRegex(ValueError, "settings"):
            self.validate()
        self.config = copy.deepcopy(self.result["config"])
        self.config["reads_per_snapshot"] += 1
        with self.assertRaisesRegex(ValueError, "settings"):
            self.validate()

    def test_skipped_transfers_and_commits_rejected(self):
        for key in ("write_transfers", "mixed_transfers", "write_commits", "mixed_commits", "read_only_reads"):
            with self.subTest(counter=key):
                self.result["counts"][key] -= 1
                with self.assertRaises(ValueError):
                    self.validate()
                self.result["counts"][key] += 1

    def test_missing_metrics_cannot_hide_costs(self):
        self.result["metrics"] = [m for m in self.result["metrics"] if m["name"] != "peak_rss"]
        with self.assertRaisesRegex(ValueError, "metric definitions"):
            self.validate()

    def records(self):
        return [dict(engine="small", repetition=i + 1, status="passed", result=copy.deepcopy(self.result)) for i in range(3)]

    def test_incomplete_failed_or_duplicate_cohort_has_no_metrics(self):
        for records in (self.records()[:2], self.records() + self.records()[:1]):
            result = fair_bank.summarize(records, ["small"], 3, "performance")
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["metrics"])
        records = self.records()
        records[2]["repetition"] = 1
        self.assertFalse(fair_bank.summarize(records, ["small"], 3, "performance")["metrics"])
        records = self.records()
        records[1]["status"] = "failed"
        self.assertFalse(fair_bank.summarize(records, ["small"], 3, "performance")["metrics"])

    def test_different_final_state_or_config_rejects_series(self):
        records = self.records()
        records[1]["result"]["validation"]["final_hash"] = "different"
        self.assertEqual(fair_bank.summarize(records, ["small"], 3, "performance")["status"], "failed")
        records = self.records()
        records[1]["result"]["config"]["seed"] += 1
        self.assertEqual(fair_bank.summarize(records, ["small"], 3, "performance")["status"], "failed")

    def test_smoke_never_becomes_performance_series(self):
        result = fair_bank.summarize(self.records(), ["small"], 3, "smoke")
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["performance_series_complete"])

    def test_diagnostic_is_one_run_and_never_a_performance_series(self):
        self.assertEqual(fair_bank.repetition_count("diagnostic", None), 1)
        self.assertEqual(fair_bank.repetition_count("diagnostic", 1), 1)
        with self.assertRaisesRegex(ValueError, "exactly one repetition"):
            fair_bank.repetition_count("diagnostic", 5)
        for records in (self.records()[:1], self.records()):
            result = fair_bank.summarize(records, ["small"], len(records), "diagnostic")
            self.assertEqual(result["status"], "passed")
            self.assertFalse(result["performance_series_complete"])

    def test_performance_still_requires_repetitions(self):
        self.assertEqual(fair_bank.repetition_count("performance", None), 3)
        with self.assertRaisesRegex(ValueError, "at least three repetitions"):
            fair_bank.repetition_count("performance", 1)

    def test_library_or_adapter_changes_reject_series(self):
        for key in ("version", "settings"):
            records = self.records()
            records[1]["result"][key] += " changed"
            self.assertFalse(fair_bank.summarize(records, ["small"], 3, "performance")["metrics"])

    def test_full_flush_cannot_substitute_for_fsync_or_none(self):
        desc = dict(engine="rocksdb", workload=fair_bank.WORKLOAD, durability=["none", "full"])
        fair_bank.validate_description(desc, "rocksdb", "full")
        fair_bank.validate_description(desc, "rocksdb", "none")
        with self.assertRaisesRegex(ValueError, "does not support fsync"):
            fair_bank.validate_description(desc, "rocksdb", "fsync")
        desc["durability"] = ["fsync"]
        with self.assertRaisesRegex(ValueError, "does not support full"):
            fair_bank.validate_description(desc, "rocksdb", "full")
        with self.assertRaisesRegex(ValueError, "does not support none"):
            fair_bank.validate_description(desc, "rocksdb", "none")

    def test_logged_native_storage_errors_are_fatal(self):
        for engine in ("small", "large"):
            for marker in ("\x1b[1;31m", "\x1b[1;35m"):
                with self.assertRaisesRegex(ValueError, "storage reported"):
                    fair_bank.validate_stderr(engine, "mapping.cpp:1 " + marker + "Flush failed\x1b[0m")
        fair_bank.validate_stderr("small", "load\nwarmup\nwrite\n")

    def test_capture_retains_failures_and_timeouts(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            code, _ = fair_bank.capture_run([sys.executable, "-c", "import sys; print('partial'); print('failure', file=sys.stderr); sys.exit(7)"], directory, 10)
            self.assertEqual(code, 7)
            self.assertIn("partial", (directory / "stdout.log").read_text())
            self.assertIn("failure", (directory / "stderr.log").read_text())
            code, _ = fair_bank.capture_run([sys.executable, "-c", "import time; time.sleep(10)"], directory, .05)
            self.assertEqual(code, 124)
            self.assertIn("timeout", (directory / "stderr.log").read_text())


if __name__ == "__main__":
    unittest.main()
