#!/usr/bin/env python3
"""Audit Small Engine cohorts, idle evidence, hashes, and retained baseline."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
sweep=json.loads((root/'x86/sweep.json').read_text())
assert sweep['status']=='passed'
pins, hashes, binaries=set(),set(),{}
measured=smoke=0
for run in sweep['runs']:
    assert run['status']=='passed'
    path=root/'x86'/run['manifest']; m=json.loads(path.read_text())
    assert m['status']=='passed' and not m['dirty'] and not m['source_status']
    assert m['config']['durability']=='none' and m['repeats']==1
    assert set(m['engines'])=={'small','dwal_small'}
    pins.add(m['revision'])
    for engine,b in m['binaries'].items():
        assert engine not in binaries or binaries[engine]==b['sha256']
        binaries[engine]=b['sha256']
    for rec in m['records']:
        assert rec['status']=='passed' and rec['exit_code']==0
        idle=[json.loads(line) for line in (path.parent/rec['directory']/'idle.jsonl').read_text().splitlines()]
        assert len(idle)>=10 and all(x['cpu_busy_percent']<=3 for x in idle[-10:])
        result=rec['result']
        assert all(v is True for k,v in result['validation'].items() if k!='final_hash')
        if run['profile']=='diagnostic':
            measured+=1; hashes.add(str(result['validation']['final_hash']))
        else: smoke+=1
prior=json.loads((root.parent/'2026-09-17_dwal-bank-batching/comparison.json').read_text())['machines']['x86']
assert measured==12 and smoke==2 and len(pins)==1 and hashes==set(prior['hashes'])
base='/home/dlarimer/psitri-api-bench-20260917'
prov=dict(revision=next(iter(pins)),measured_runs=measured,smoke_runs=smoke,final_hash=next(iter(hashes)),
          binary_sha256=binaries,
          validation='All checks passed; clean source and stable binaries; ten final idle samples <=3%; one repeat per configuration.',
          baseline='../2026-09-17_dwal-bank-batching/comparison.json',
          baseline_revisions=dict(dwal=prior['revisions'],direct=prior['direct_revision']),
          functional_checks=dict(dwal_cases=159,dwal_assertions=602375,byte_cases=5,byte_assertions=72,session_cases=1,session_assertions=20,small_suite_cases=39,small_suite_assertions=1098734,python_checks=38),
          arm_status='Not measured; busy Mac, see local/arm-availability.json. Functional checks only.',
          failed_attempt='x86-attempt-01; stopped at batch 1 on exact-account mismatch, before the writer-session fix',
          scope='Native Small Engine direct and DWAL over Small Engine; MDBX shim bars remain Large Engine based.',
          build_commands=[
              f'cmake -S {base}/small-fixed-source/benchmarks/api_bank -B {base}/build-small-fixed -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 -DFETCHCONTENT_SOURCE_DIR_PSIO={base}/psio -DFETCHCONTENT_SOURCE_DIR_UCC={base}/ucc -DFAIR_MDBX_INCLUDE_DIR=/home/dlarimer/arbtrie/build/release/_deps/libmdbx-src -DFAIR_MDBX_LIBRARY=/home/dlarimer/arbtrie/build/release/_deps/libmdbx-build/libmdbx.a -DPSITRIMDBX_BACKEND=dwal -DAPI_BANK_BUILD_CONTRACT_TESTS=OFF -DCMAKE_EXE_LINKER_FLAGS=-flto=thin',
              f'cmake --build {base}/build-small-fixed --target fair-bank-small fair-bank-dwal_small -j4'])
(root/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
print(f'PASS: {measured} measurements and {smoke} smokes; all match baseline final hash')
