import datetime, json, os, resource, subprocess, sys
from pathlib import Path
root=Path('/home/dlarimer/psitri-api-bench-20260917')
sys.path.insert(0,str(root/'psitri/scripts'))
import benchmark_idle
from fair_bank import SMOKE
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
out=root/'results'/('contract-fixes-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S'))
out.mkdir()
records=[]
with benchmark_idle.exclusive():
 for build in ['build-mdbx','build-integrations']:
  record={'kind':'storage-regressions','build':build}
  command=[str(root/build/'bin/api-storage-regressions')]
  with (out/(build+'.log')).open('w') as f:
   record['exit_code']=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=240,env=os.environ|{'SAL_LOG_LEVEL':'ERROR'}).returncode
  records.append(record)
  print(build,record['exit_code'],flush=True)
 for engine in ['psitrimdbx','psitrirocks','psitri_sqlite']:
  target=out/engine;target.mkdir()
  build='build-mdbx' if engine=='psitrimdbx' else 'build-integrations'
  command=[str(root/build/'bin'/('api-contract-'+engine)),'--path',str(target/'database'),'--result',str(target/'result.json')]
  for key,value in dict(SMOKE,durability='none').items(): command += ['--'+key.replace('_','-'),str(value)]
  record={'engine':engine,'command':command}
  with (target/'stdout.log').open('w') as stdout,(target/'stderr.log').open('w') as stderr:
   record['exit_code']=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=120,env=os.environ|{'SAL_LOG_LEVEL':'ERROR'}).returncode
  records.append(record)
  print(engine,record['exit_code'],flush=True)
(out/'manifest.json').write_text(json.dumps(dict(kind='correctness-only',performance_evidence=False,records=records),indent=2))
print(out,flush=True)
sys.exit(int(any(r['exit_code'] for r in records)))
