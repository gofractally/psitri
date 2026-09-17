import datetime, json, os, subprocess, sys, uuid
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
import benchmark_idle
from fair_bank import SMOKE
root=Path('logs/2026-09-17_api-contract-fixes')/(datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')+'-'+uuid.uuid4().hex[:8])
root.mkdir(parents=True)
(root/'source.patch').write_bytes(subprocess.check_output(['git','diff','HEAD','--binary']))
integ=Path('/Users/dlarimer/psitri-merge/build/bench-sources/integrations')
(root/'integration.patch').write_bytes(subprocess.check_output(['git','diff','HEAD','--binary'],cwd=integ))
config=dict(SMOKE,durability='none')
records=[]
with benchmark_idle.exclusive():
 for engine in sys.argv[1:]:
  build='api-mdbx' if engine in ('mdbx','psitrimdbx') else 'api-integrations'
  out=root/engine;out.mkdir()
  command=[str(Path('build',build,'bin','api-contract-'+engine).resolve()),'--path',str((out/'database').resolve()),'--result',str((out/'result.json').resolve())]
  for key,value in config.items(): command += ['--'+key.replace('_','-'),str(value)]
  record=dict(engine=engine,command=command,kind='correctness-only',performance_evidence=False)
  (out/'command.json').write_text(json.dumps(record,indent=2))
  with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
   try: record['exit_code']=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=60,env=dict(os.environ,SAL_LOG_LEVEL='ERROR')).returncode
   except subprocess.TimeoutExpired: record['exit_code']=124
  (out/'run.json').write_text(json.dumps(record,indent=2));records.append(record)
  print(engine,record['exit_code'],out,flush=True)
(root/'manifest.json').write_text(json.dumps(dict(kind='correctness-only',performance_evidence=False,records=records),indent=2))
