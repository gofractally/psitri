"""Keep public numbers faithful to the retained source evidence."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class PublicEvidenceTest(unittest.TestCase):
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
