#!/usr/bin/env python3
"""Functional checks only; this busy Mac is not a timing evidence source."""
import hashlib, json, os, subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
records=[]
for engine in ('small','dwal_small'):
    binary = root/'build/api-dwal/bin'/('fair-bank-'+engine)
    command = [str(binary), '--path', str(root/'build'/('small-functional-'+engine)),
               '--result', str(out/(engine+'.json')), '--accounts','128','--transfers','256',
               '--warmup','64','--batch','32','--readers','2','--reads-per-snapshot','16',
               '--read-batches','2','--seed','12345','--value-bytes','8','--cache-mib','64',
               '--map-gib','64','--durability','none']
    record=dict(command=command, engine=engine, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    with (out/(engine+'.stdout.log')).open('w') as stdout, (out/(engine+'.stderr.log')).open('w') as stderr:
        run=subprocess.run(command, stdout=stdout, stderr=stderr, env=dict(os.environ,SAL_LOG_LEVEL='ERROR'))
    record['exit_code']=run.returncode; records.append(record)
    (out/'functional.json').write_text(json.dumps(dict(purpose='correctness only; no idle timing claim',records=records),indent=2)+'\n')
    if run.returncode: raise SystemExit(run.returncode)
    result=json.loads((out/(engine+'.json')).read_text())
    assert result['status']=='passed'
    print(engine, result['validation'])
