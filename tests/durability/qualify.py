#!/usr/bin/env python3
"""Retain commands, sources, dependencies and outputs for native durability qualification."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--cc', required=True)
p.add_argument('--cxx', required=True)
p.add_argument('--ucc-source', required=True, type=Path)
p.add_argument('--ucc-revision', required=True)
p.add_argument('--catch2-dir', type=Path)
p.add_argument('--catch2-source', type=Path)
p.add_argument('--build-dir', default='build-durability-native')
p.add_argument('--jobs', default='4')
p.add_argument('--label', default='native')
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
build = root / a.build_dir
evidence = root / 'logs' / ('durability-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + a.label)
evidence.mkdir(parents=True, exist_ok=False)
report = {'platform': platform.platform(), 'machine': platform.machine(), 'commands': [],
          'passed': False, 'source_sha256': {}, 'ucc_sha256': {}, 'ucc_revision': a.ucc_revision,
          'free_bytes_before': shutil.disk_usage(root).free, 'host_load_before': list(os.getloadavg()),
          'scope': 'Ordinary syscall failures and EINTR; public API regression; no power-cut or arbitrary storage-corruption guarantee.'}
for name in ['libraries/adb', 'libraries/sal', 'libraries/start_ref', 'libraries/sal_art_ref',
             'libraries/psitri_api', 'tests/api', 'tests/durability']:
    for source in sorted((root / name).rglob('*')):
        if source.is_file():
            report['source_sha256'][str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
for source in sorted(a.ucc_source.rglob('*')):
    if source.is_file() and '.git' not in source.parts:
        report['ucc_sha256'][str(source.relative_to(a.ucc_source))] = hashlib.sha256(source.read_bytes()).hexdigest()

def save():
    (evidence / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

def run(name, argv):
    print(name, flush=True)
    start = time.monotonic()
    result = subprocess.run(list(map(str, argv)), cwd=root, capture_output=True, text=True, timeout=900)
    (evidence / (name + '.stdout')).write_text(result.stdout)
    (evidence / (name + '.stderr')).write_text(result.stderr)
    report['commands'].append({'name': name, 'argv': list(map(str, argv)), 'exit_code': result.returncode,
                               'seconds': time.monotonic() - start})
    save()
    if result.returncode:
        raise RuntimeError(name + ' failed')

try:
    if report['free_bytes_before'] < 15 * 1024**3:
        raise RuntimeError('less than 15GiB free; no build started')
    overrides = [name for name in ['HEART_PAGE_BYTES', 'HEART_PINNED_MB', 'HEART_READ_CACHE',
                                   'HEART_READ_CACHE_SEC'] if os.getenv(name) is not None]
    if overrides:
        raise RuntimeError('unsupported qualification environment overrides: ' + ', '.join(overrides))
    run('compiler', [a.cxx, '--version'])
    if platform.system() == 'Linux':
        run('hardware', ['lscpu'])
    else:
        run('hardware', ['sysctl', 'machdep.cpu.brand_string', 'hw.memsize', 'hw.ncpu'])
    options = ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=' + a.cc,
               '-DCMAKE_CXX_COMPILER=' + a.cxx, '-DPSITRI_DURABILITY_UCC_SOURCE=' + str(a.ucc_source)]
    if a.catch2_dir:
        options += ['-DPSITRI_DURABILITY_API_TESTS=ON', '-DCatch2_DIR=' + str(a.catch2_dir)]
    if a.catch2_source:
        options += ['-DPSITRI_DURABILITY_CATCH2_SOURCE=' + str(a.catch2_source)]
    run('configure', ['cmake', '-S', root / 'tests/durability', '-B', build, '-G', 'Ninja', *options])
    run('build', ['cmake', '--build', build, '--parallel', a.jobs])
    run('tests', ['ctest', '--test-dir', build, '--verbose', '--output-on-failure'])
    report['binary_sha256'] = {name: hashlib.sha256((build / name).read_bytes()).hexdigest()
                              for name in ['durability_fault_driver', 'durability_api_regression']
                              if (build / name).is_file()}
    report['passed'] = True
except Exception as error:
    report['error'] = str(error)
finally:
    report['free_bytes_after'] = shutil.disk_usage(root).free
    report['host_load_after'] = list(os.getloadavg())
    save()
print(json.dumps({'evidence': str(evidence), 'passed': report['passed'], 'error': report.get('error')}))
raise SystemExit(0 if report['passed'] else 1)
