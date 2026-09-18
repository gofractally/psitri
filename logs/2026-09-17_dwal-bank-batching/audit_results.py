#!/usr/bin/env python3
"""Validate retained cohorts and record their shared provenance."""
import json
from pathlib import Path
root = Path(__file__).resolve().parent
sweep = json.loads((root/'x86/sweep.json').read_text())
assert sweep['status'] == 'passed'
pins, hashes, binary_hashes = set(), set(), {}
n = 0
for run in sweep['runs']:
    assert run['status'] == 'passed'
    mpath = root/'x86'/run['manifest']
    m = json.loads(mpath.read_text())
    assert m['status'] == 'passed' and not m['dirty'] and not m['source_status']
    pins.add(m['revision'])
    assert m['config']['durability'] == 'none' and m['repeats'] == 1
    for engine, b in m['binaries'].items():
        if engine in binary_hashes: assert binary_hashes[engine] == b['sha256']
        binary_hashes[engine] = b['sha256']
    for rec in m['records']:
        assert rec['status'] == 'passed' and rec['exit_code'] == 0
        idle = [json.loads(x) for x in (mpath.parent/rec['directory']/'idle.jsonl').read_text().splitlines()]
        assert len(idle) >= 10 and all(x['cpu_busy_percent'] <= 3 for x in idle[-10:])
        assert all(v is True for k, v in rec['result']['validation'].items() if k != 'final_hash')
        if run['profile'] == 'diagnostic':
            n += 1
            hashes.add(rec['result']['validation']['final_hash'])
assert n == 18 and len(pins) == len(hashes) == 1
prior = json.loads((root.parent/'2026-09-17_direct-bank-batching/provenance.json').read_text())
prior_m = json.loads(next((root.parent/'2026-09-17_direct-bank-batching/x86/batch-1').glob('*/manifest.json')).read_text())
current_m = json.loads(next((root/'x86/batch-1').glob('*/manifest.json')).read_text())
assert current_m['dependencies'] == prior_m['dependencies']
prov = {
    'revision': next(iter(pins)), 'measured_runs': n, 'smoke_runs': 3,
    'final_hash': next(iter(hashes)),
    'validation': 'All configurations passed; source clean and binary hashes stable; last ten idle samples <=3%; exactly one repetition per configuration.',
    'binary_sha256': binary_hashes,
    'native_mdbx_binary_identical_to_direct_sweep': binary_hashes['mdbx'] == prior_m['binaries']['mdbx']['sha256'],
    'x86_mdbx_build': prior['x86_mdbx_build'], 'mdbx_version': prior['mdbx_version'],
    'arm_status': 'Not measured; Mac did not satisfy idle availability check. Functional tests only.',
    'functional_checks': {'dwal_cases': 159, 'dwal_assertions': 602375, 'byte_adapter_cases': 1, 'byte_adapter_assertions': 20, 'python_checks': 38},
    'notes': 'DWAL and direct measurements are separate sequential sweeps on the same machine; they are single observations, not repeated performance distributions.',
    'x86_build_commands': [
        'cmake -S /home/dlarimer/psitri-api-bench-20260917/dwal-source/benchmarks/api_bank -B /home/dlarimer/psitri-api-bench-20260917/build-dwal -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 -DFETCHCONTENT_SOURCE_DIR_PSIO=/home/dlarimer/psitri-api-bench-20260917/psio -DFETCHCONTENT_SOURCE_DIR_UCC=/home/dlarimer/psitri-api-bench-20260917/ucc -DFAIR_MDBX_INCLUDE_DIR=/home/dlarimer/arbtrie/build/release/_deps/libmdbx-src -DFAIR_MDBX_LIBRARY=/home/dlarimer/arbtrie/build/release/_deps/libmdbx-build/libmdbx.a -DPSITRIMDBX_BACKEND=dwal -DAPI_BANK_BUILD_CONTRACT_TESTS=OFF -DCMAKE_EXE_LINKER_FLAGS=-flto=thin',
        'cmake --build /home/dlarimer/psitri-api-bench-20260917/build-dwal --target fair-bank-dwal fair-bank-psitrimdbx fair-bank-mdbx -j4'
    ]
}
(root/'provenance.json').write_text(json.dumps(prov, indent=2)+'\n')
print(f"PASS: {n} measurements; revision {prov['revision']}; identical native MDBX binary: {prov['native_mdbx_binary_identical_to_direct_sweep']}")
