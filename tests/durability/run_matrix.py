#!/usr/bin/env python3
"""Ordinary interrupted/failed flushes must not acknowledge unknown durability."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--binary', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
evidence = a.output / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
evidence.mkdir(parents=True, exist_ok=False)
binary = a.binary.resolve()
report = {'platform': platform.platform(), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
          'passed': False, 'cases': [], 'ack_boundary': 'ACK is written only after commit/sync returns.',
          'terminal_contract': 'EIO exits the entire process with 74 before ACK; no later writes execute.',
          'reopen_contract': 'A fail-stopped store was not closed: a read-only open refuses it without modifying files, and the next read-write open recovers it. Without a power cut or corruption injection, the recovered fixture is complete old or complete new. Unacknowledged does not promise rollback. Read-only reopen never modifies files.'}

def save():
    (evidence / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

def digest(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob('*')) if path.is_file()}

def run(label, kind, mode, operation, number, durability):
    database = evidence / 'database' / label
    argv = [str(binary), kind, str(database), mode, operation, str(number), durability]
    start = time.monotonic()
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    (evidence / (label + '.stdout')).write_text(result.stdout)
    (evidence / (label + '.stderr')).write_text(result.stderr)
    record = {'label': label, 'argv': argv, 'exit_code': result.returncode,
              'seconds': time.monotonic() - start, 'acknowledged': 'ACK ' in result.stdout}
    report['cases'].append(record)
    save()
    if mode.startswith('error'):
        if result.returncode != 74 or record['acknowledged'] or 'SAL durability failure:' not in result.stderr:
            raise RuntimeError(label + ': terminal I/O failure did not fail-stop before acknowledgement')
        if kind.startswith('engine'):
            def reopen(step):
                before = digest(database)
                attempt = subprocess.run([str(binary), 'recover' if step == 'recover' else 'reopen', str(database)],
                                         capture_output=True, text=True, timeout=60)
                (evidence / (label + '-' + step + '.stdout')).write_text(attempt.stdout)
                (evidence / (label + '-' + step + '.stderr')).write_text(attempt.stderr)
                record[step + '_exit_code'] = attempt.returncode
                record[step + '_stdout'] = attempt.stdout.strip()
                if step != 'recover':
                    record[step + '_files_unchanged'] = before == digest(database)
                save()
                return attempt
            refused = reopen('unclean-readonly')
            if (refused.returncode == 0 or 'not shut down cleanly' not in refused.stderr
                    or not record['unclean-readonly_files_unchanged']):
                raise RuntimeError(label + ': read-only open accepted or modified a store that was not closed')
            if reopen('recover').returncode:
                raise RuntimeError(label + ': failed commit did not recover to a consistent root')
            if reopen('readonly').returncode or not record['readonly_files_unchanged']:
                raise RuntimeError(label + ': recovered store reopened inconsistently or read-only modified files')
        return None
    if result.returncode or not record['acknowledged']:
        raise RuntimeError(label + ': healthy/interrupted flush did not acknowledge success')
    count = re.search(r'ACK calls=(\d+)', result.stdout)
    if not count:
        raise RuntimeError(label + ': missing actual syscall count')
    return int(count.group(1))

try:
    for durability in ['fsync', 'full']:
        for kind, operations in [('mapping', ['msync', 'fsync']), ('block', ['fsync']),
                                 ('engine', ['msync', 'fsync']), ('engine-thread', ['fsync'])]:
            if durability == 'full' and platform.system() == 'Darwin':
                operations = [operation for operation in operations if operation != 'fsync'] + ['full']
                if kind.startswith('engine'):
                    operations += ['fsync']  # data file fsync precedes full root-map flush
            for operation in operations:
                prefix = durability + '-' + kind + '-' + operation
                count = run(prefix + '-healthy', kind, 'count', operation, 0, durability)
                if count == 0:
                    raise RuntimeError(prefix + ': injection did not intercept the intended syscall')
                for number in range(1, count + 1):
                    retried = run(prefix + '-eintr-' + str(number), kind, 'eintr', operation, number, durability)
                    if retried != count + 1:
                        raise RuntimeError(prefix + ': EINTR was not retried exactly once')
                    run(prefix + '-eio-' + str(number), kind, 'error', operation, number, durability)
    run('valid-fd-zero', 'fd0', 'count', 'fsync', 0, 'fsync')
    run('truncate-grow-shrink-prefix', 'block-grow', 'count', 'fsync', 0, 'fsync')
    run('readonly-truncate-rejected', 'readonly-truncate', 'count', 'fsync', 0, 'fsync')
    for mode in ['error-enospc', 'error-erofs', 'error-ebadf']:
        run(mode + '-engine-fsync', 'engine', mode, 'fsync', 1, 'fsync')
    report['passed'] = True
except Exception as error:
    report['error'] = str(error)
finally:
    save()
print(json.dumps({'evidence': str(evidence), 'passed': report['passed'], 'cases': len(report['cases']), 'error': report.get('error')}))
raise SystemExit(0 if report['passed'] else 1)
