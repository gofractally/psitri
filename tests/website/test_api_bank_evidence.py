"""Public API-pair charts require complete, matching, repeated measurements."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import api_bank_evidence
import benchmark_charts


class ApiEvidenceTest(unittest.TestCase):
    def series(self):
        result = json.loads((Path(__file__).parent / 'fixtures/fair-bank.json').read_text())
        records = []
        for engine, factor in [('mdbx', 1), ('psitrimdbx', 2)]:
            for i in range(3):
                r = copy.deepcopy(result)
                r.update(engine=engine, workload='api-bank-v1')
                for metric in r['metrics']:
                    metric['engine'] = engine
                    if metric['name'] == 'write': metric['value'] = (i + 1) * 100 * factor
                records.append(dict(engine=engine, repetition=i+1, status='passed', exit_code=0, result=r))
        return dict(workload='api-bank-v1', profile='performance', status='passed', dirty=False,
            revision='a' * 40, source_status='', engines=['mdbx', 'psitrimdbx'], repeats=3,
            config=result['config'], records=records,
            isolation=dict(machine_lock=True, idle_seconds=10, max_cpu_busy_percent=3))

    def test_all_repetitions_contribute_to_median_and_range(self):
        s = api_bank_evidence.validated_series(self.series(), 'mdbx')
        self.assertEqual(s['metrics']['mdbx']['write']['median'], 200)
        self.assertEqual(s['metrics']['psitrimdbx']['write']['samples'], [200,400,600])
        self.assertEqual(s['metrics']['psitrimdbx']['write']['minimum'], 200)
        self.assertEqual(s['metrics']['psitrimdbx']['write']['maximum'], 600)

    def test_smoke_dirty_failed_and_unisolated_runs_cannot_be_published(self):
        for key, value in [('profile','smoke'),('profile','diagnostic'),('dirty',True),('status','failed'),('repeats',1)]:
            with self.subTest(key=key):
                manifest = self.series(); manifest[key] = value
                with self.assertRaises(ValueError): api_bank_evidence.validated_series(manifest, 'mdbx')
        manifest = self.series(); manifest['isolation']['max_cpu_busy_percent'] = 20
        with self.assertRaises(ValueError): api_bank_evidence.validated_series(manifest, 'mdbx')

    def test_missing_repetition_or_different_final_state_rejected(self):
        manifest = self.series(); manifest['records'].pop()
        with self.assertRaises(ValueError): api_bank_evidence.validated_series(manifest, 'mdbx')
        manifest = self.series(); manifest['records'][0]['result']['validation']['final_hash'] = 'different'
        with self.assertRaises(ValueError): api_bank_evidence.validated_series(manifest, 'mdbx')

    def test_paired_chart_labels_repeated_cohort_and_ranges(self):
        data = json.loads((ROOT/'website/data/benchmarks.json').read_text())
        p = data['api_bank']['platforms'][0]['pairs'][1]
        p['cohort'] = dict(date='2026-09-17', revision='b'*40, repeats=5, workload='100,000 accounts · four readers')
        for role in ['native','shim']:
            p[role]['ranges'] = {k:dict(minimum=v*0.9, maximum=v*1.1) for k,v in p[role]['values'].items()}
            p[role].update(evidence_kind='data',evidence='api-bank/test.json')
        html = benchmark_charts.api_chart(data, 'test', '../evidence/')
        self.assertIn('median of 5 runs', html)
        self.assertIn('class="api-range"', html)
        self.assertIn('../data/api-bank/test.json', html)
        self.assertIn('All repetitions', html)
        self.assertIn('100,000 accounts', html)

    def test_small_positive_ratio_is_not_displayed_as_zero(self):
        data = json.loads((ROOT/'website/data/benchmarks.json').read_text())
        pair = data['api_bank']['platforms'][0]['pairs'][1]
        pair['native']['values']['mixed_read'] = 3_000_000
        pair['shim']['values']['mixed_read'] = 270
        html = benchmark_charts.api_chart(data, 'test', '../evidence/')
        self.assertIn('0.00009× throughput', html)
