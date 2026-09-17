"""Keep public numbers faithful to the retained source evidence."""
import importlib.util
import json
import re
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_charts


class PublicEvidenceTest(unittest.TestCase):
    def test_api_pairs_only_publish_validated_matching_runs(self):
        data = json.loads((ROOT / "website/data/benchmarks.json").read_text())
        api = data["api_bank"]
        for platform in api["platforms"]:
            for pair in platform["pairs"]:
                if pair["status"] != "passed":
                    self.assertNotIn("native", pair)
                    self.assertNotIn("shim", pair)
                    continue
                logs = []
                for role in ("native", "shim"):
                    engine = pair[role]
                    log = (ROOT / platform["source_directory"] / engine["source"]).read_text()
                    logs.append(log)
                    self.assertIn("PASS  Total wall clock", log)
                    patterns = dict(write=r"Write-only:\s+([\d,]+) tx/sec",
                                    mixed_write=r"Write\+Read:\s+([\d,]+) tx/sec",
                                    mixed_read=r"Reader:\s+([\d,]+) reads/sec",
                                    load=r"Bulk load\s+[\d.]+ sec\s+([\d,]+) ops/sec")
                    for metric, pattern in patterns.items():
                        self.assertEqual(engine["values"][metric], int(re.search(pattern, log)[1].replace(",", "")))
                    self.assertEqual(engine["values"]["disk"], float(re.findall(r"Size: file=([\d.]+) MB", log)[-1]))
                for field in ("Accounts", "Transactions", "Batch size", "Sync every", "Sync mode", "Seed", "Read mode"):
                    self.assertEqual(re.search(rf"{field}:\s+(.+)", logs[0])[1], re.search(rf"{field}:\s+(.+)", logs[1])[1])
                self.assertEqual(re.findall(r"Successful:.*", logs[0]), re.findall(r"Successful:.*", logs[1]))
        html = benchmark_charts.api_chart(data, "test-api", "../evidence/")
        self.assertEqual(html.count('class="api-bars"'), 5)
        self.assertEqual(html.count('data-api-arch='), 2)

    def test_bank_chart_matches_archived_rows(self):
        data = json.loads((ROOT / "website/data/benchmarks.json").read_text())["bank"]
        archived = []
        for name in ("bank_compare.log", "mdbx.log"):
            archived.extend(benchmark.parse_bank((ROOT / data["source_directory"] / name).read_text()))
        actual = {(m["engine"], m["name"]): m["value"] for m in archived}
        published = {(e["id"], name): value for e in data["engines"] for name, value in e["values"].items() if value is not None}
        self.assertEqual(published, actual)
        self.assertEqual({e["id"] for e in data["engines"] if e["status"] == "maintained"}, {"sal_art_ref", "start_ref"})

    def test_scale_series_preserves_every_probe_and_ram_context(self):
        data = json.loads((ROOT / "website/data/benchmarks.json").read_text())["scale"]
        self.assertFalse(data["beyond_ram"])
        self.assertEqual(len(data["points"]), 100)
        self.assertEqual(data["points"][-1]["keys"], 1000000000)
        self.assertEqual(data["points"][-1]["rss_gb"], 58.08)
        self.assertTrue(all(p["hit_percent"] == 100 for p in data["points"]))
        self.assertIn("128 GB", data["machine"])


if __name__ == "__main__":
    unittest.main()
