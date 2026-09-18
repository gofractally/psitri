#!/usr/bin/env python3
"""Verify tiered bank evidence, migration, idle samples, and baseline compatibility."""
import json, re
from pathlib import Path
root=Path(__file__).resolve().parent
sweep=json.loads((root/'x86/sweep.json').read_text())
assert sweep['status']=='passed'
pins,hashes,binaries=set(),set(),set()
measured=smoke=0
migration={}
for run in sweep['runs']:
    assert run['status']=='passed'
    p=root/'x86'/run['manifest']; m=json.loads(p.read_text())
    assert m['status']=='passed' and not m['dirty'] and not m['source_status']
    assert m['config']['durability']=='none' and m['repeats']==1 and m['engines']==['tiered']
    pins.add(m['revision']); binaries.add(m['binaries']['tiered']['sha256'])
    for rec in m['records']:
        assert rec['status']=='passed' and rec['exit_code']==0
        d=p.parent/rec['directory']
        idle=[json.loads(line) for line in (d/'idle.jsonl').read_text().splitlines()]
        assert len(idle)>=10 and all(x['cpu_busy_percent']<=3 for x in idle[-10:])
        result=rec['result']
        assert all(v is True for k,v in result['validation'].items() if k!='final_hash')
        if run['profile']=='diagnostic':
            measured+=1; hashes.add(str(result['validation']['final_hash']))
            counts=[int(v) for v in re.findall(r'tiered base commits: (\d+)',(d/'stderr.log').read_text())]
            assert len(counts)==5 and max(counts)>0, 'No completed migration'
            migration[str(run['batch'])]=dict(zip(('after_load','after_warmup','after_solo_writes','after_mixed_writes','after_reopen'),counts))
        else: smoke+=1
prior=json.loads((root.parent/'2026-09-17_small-bank-batching/comparison.json').read_text())['machines']['x86']
assert measured==6 and smoke==1 and len(pins)==len(binaries)==1 and hashes==set(prior['hashes'])
base='/home/dlarimer/psitri-api-bench-20260917'
prov=dict(revision=next(iter(pins)),measured_runs=measured,smoke_runs=smoke,
          final_hash=next(iter(hashes)),binary_sha256=next(iter(binaries)),base_commits_by_batch=migration,
          validation='All bank checks passed; migration verified; source clean, binary stable; ten final idle samples <=3%; one repetition.',
          baseline='../2026-09-17_small-bank-batching/comparison.json',
          functional_checks=dict(pair_cases=2,pair_assertions=26,python_checks=38),
          arm_status='Not measured; busy Mac. Local results are correctness evidence only.',
          configuration='Native tiered Small Engine + Large Engine; 16 MiB bytes-written flush trigger, latest merged reads, both tiers sync=none.',
          lifecycle='Close joins an active flush and retains unflushed committed hot data in Small Engine; does not force a full drain.',
          build_commands=[
              f'cmake -S {base}/tiered-source/benchmarks/api_bank -B {base}/build-tiered -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 -DFETCHCONTENT_SOURCE_DIR_PSIO={base}/psio -DFETCHCONTENT_SOURCE_DIR_UCC={base}/ucc -DFAIR_MDBX_INCLUDE_DIR=/home/dlarimer/arbtrie/build/release/_deps/libmdbx-src -DFAIR_MDBX_LIBRARY=/home/dlarimer/arbtrie/build/release/_deps/libmdbx-build/libmdbx.a -DPSITRIMDBX_BACKEND=dwal -DAPI_BANK_BUILD_CONTRACT_TESTS=OFF -DCMAKE_EXE_LINKER_FLAGS=-flto=thin',
              f'cmake --build {base}/build-tiered --target fair-bank-tiered -j4'])
(root/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
print(f'PASS: {measured} measurements, {smoke} smoke, migration in every measured run, matching baseline final hash')
