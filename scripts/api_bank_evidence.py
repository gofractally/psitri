#!/usr/bin/env python3
"""Curate a complete same-API performance series without dropping repetitions."""
import argparse
import json
from pathlib import Path

import fair_bank

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {'mdbx': ('mdbx', 'psitrimdbx'), 'rocksdb': ('rocksdb', 'psitrirocks'),
         'sqlite': ('sqlite', 'psitri_sqlite')}
NAMES = {'mdbx': 'libmdbx', 'psitrimdbx': 'psitri MDBX shim', 'rocksdb': 'RocksDB',
         'psitrirocks': 'psitri RocksDB shim', 'sqlite': 'SQLite', 'psitri_sqlite': 'psitri SQLite shim'}
METRICS = {'write': ('write', 1), 'mixed_write': ('mixed_write', 1),
           'mixed_read': ('mixed_read', 1), 'load': ('load', 1),
           'disk': ('allocated_after_close', 1024 * 1024)}


def validated_series(manifest, api):
    engines = PAIRS[api]
    if (manifest.get('workload'), manifest.get('profile'), manifest.get('status')) != ('api-bank-v1', 'performance', 'passed'):
        raise ValueError('Require a passed api-bank-v1 performance series')
    if manifest.get('dirty') is not False or manifest.get('source_status') or not manifest.get('revision'):
        raise ValueError('Require identified, clean source')
    if any(not d.get('revision') or d.get('status') for d in manifest.get('native_dependencies', {}).values()):
        raise ValueError('Require identified, clean dependency sources')
    if set(manifest.get('engines', [])) != set(engines):
        raise ValueError('Require both original library and matching shim')
    if manifest.get('repeats', 0) < 3:
        raise ValueError('Require at least three repetitions')
    isolation = manifest.get('isolation', {})
    if not isolation.get('machine_lock') or isolation.get('idle_seconds', 0) < 10 or isolation.get('max_cpu_busy_percent', 100) > 3:
        raise ValueError('Require exclusive execution and an idle host')
    records = manifest['records']
    previous = fair_bank.WORKLOAD
    fair_bank.WORKLOAD = 'api-bank-v1'
    try:
        for record in records:
            if record.get('status') != 'passed' or record.get('exit_code') != 0:
                raise ValueError('A repetition failed')
            fair_bank.validate_result(record['result'], record['engine'], manifest['config'])
        summary = fair_bank.summarize(records, list(engines), manifest['repeats'], 'performance')
    finally:
        fair_bank.WORKLOAD = previous
    if not summary['performance_series_complete']:
        raise ValueError('Incomplete or inconsistent paired series')
    return summary


def curate(archive, api, architecture, data_path):
    archive = archive.resolve()
    manifest = json.loads((archive / 'manifest.json').read_text())
    summary = validated_series(manifest, api)
    actual_arch = manifest["hardware"]["architecture"]
    if actual_arch not in ({"arm64", "aarch64"} if architecture == "arm64" else {"x86_64", "AMD64"}):
        raise ValueError("Requested architecture differs from the recorded machine")
    for record in manifest['records']:
        actual = json.loads((archive / record['directory'] / 'result.json').read_text())
        if actual != record['result']:
            raise ValueError('Manifest result differs from raw repetition')
        samples = [json.loads(line) for line in (archive / record['directory'] / 'idle.jsonl').read_text().splitlines()]
        quiet = []
        for sample in reversed(samples):
            if sample['cpu_busy_percent'] > 3: break
            quiet.append(sample['time'])
        if len(quiet) < 2 or quiet[0] - quiet[-1] < 10:
            raise ValueError('Missing evidence of ten continuous idle seconds before a run')
    source_directory = str(archive.relative_to(ROOT))
    data = json.loads(data_path.read_text())
    platform = next(p for p in data['api_bank']['platforms'] if p['id'] == architecture)
    pair = next(p for p in platform['pairs'] if p['id'] == api)
    c = manifest['config']
    cohort = dict(date=manifest['started_at'][:10], revision=manifest['revision'],
        source_directory=source_directory, repeats=manifest['repeats'],
        workload=f"{c['accounts']:,} accounts · {c['transfers']:,} transfers per phase · batch {c['batch']:,} · {c['readers']} readers · seed {c['seed']} · durability {c['durability']}",
        metric_labels={'disk': 'Allocated storage after close (MiB)'},
        limits='Medians of every retained repetition; ranges show minimum–maximum. File size means allocated storage after close.')
    for other in platform['pairs']:
        if other['status'] == 'passed' and 'cohort' not in other:
            other['cohort'] = {k: platform[k] for k in ('date','revision','source_directory','workload','limits')}
    pair.clear()
    pair.update(id=api, status='passed', cohort=cohort,
        note='Both implementations passed rollback, held snapshots, read-your-writes, exact account state, balance conservation and reopen on every repetition.')
    evidence_dir = ROOT / 'website/data/api-bank'
    evidence_dir.mkdir(exist_ok=True)
    for role, engine in zip(('native', 'shim'), PAIRS[api]):
        values, ranges = {}, {}
        for key, (metric, divisor) in METRICS.items():
            stats = summary['metrics'][engine][metric]
            values[key] = stats['median'] / divisor
            ranges[key] = {k: stats[k] / divisor for k in ('minimum', 'maximum')}
        filename = f'{architecture}-{engine}.json'
        records = [dict(repetition=r['repetition'], result=r['result']) for r in manifest['records'] if r['engine'] == engine]
        evidence = dict(workload='api-bank-v1', revision=manifest['revision'], machine_id=manifest['machine_id'],
            architecture=architecture, config=c, isolation=manifest['isolation'],
            source_manifest=f'https://github.com/gofractally/psitri/blob/sal2/{source_directory}/manifest.json',
            summary=summary['metrics'][engine], repetitions=records)
        (evidence_dir / filename).write_text(json.dumps(evidence, indent=2) + '\n')
        pair[role] = dict(name=NAMES[engine], engine=engine, values=values, ranges=ranges,
            source='manifest.json', evidence=f'api-bank/{filename}', evidence_kind='data')
    data['curated_on'] = cohort['date']
    data_path.write_text(json.dumps(data, indent=2) + '\n')
    return pair


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--api', choices=PAIRS, required=True)
    p.add_argument('--architecture', choices=('arm64', 'x86_64'), required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'website/data/benchmarks.json')
    a = p.parse_args()
    curate(a.archive, a.api, a.architecture, a.data)
