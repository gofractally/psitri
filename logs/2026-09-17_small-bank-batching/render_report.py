#!/usr/bin/env python3
"""Add validated Small Engine cohorts to retained Large Engine/MDBX results."""
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
prior_path = root.parent/'2026-09-17_dwal-bank-batching/comparison.json'
data = json.loads(prior_path.read_text())
data['engines'] = ['mdbx','shim_direct','shim_dwal','native_direct','native_dwal','small_direct','small_dwal']
data['names'].update(native_direct='Large Engine · direct',native_dwal='Large Engine · DWAL',
                     small_direct='Small Engine · direct',small_dwal='Small Engine · DWAL')
data['baseline'] = '../2026-09-17_dwal-bank-batching/comparison.json'
for machine, entry in data['machines'].items():
    folder = root/machine
    sweep = json.loads((folder/'sweep.json').read_text())
    assert sweep['status'] == 'passed'
    rows = {r['batch']:r for r in entry['rows']}
    revisions, hashes, batches = set(), set(), set()
    for run in sweep['runs']:
        if run['profile'] != 'diagnostic': continue
        assert run['status'] == 'passed'
        path = folder/run['manifest']
        m = json.loads(path.read_text())
        assert m['status'] == 'passed' and m['repeats'] == 1 and m['profile'] == 'diagnostic'
        assert not m['source_status'] and not m['dirty']
        assert set(m['engines']) == {'small','dwal_small'}
        assert m['hardware'] == entry['hardware']
        assert all(v == entry['config'][k] for k,v in m['config'].items() if k != 'batch')
        revisions.add(m['revision']); batches.add(run['batch'])
        row = rows[run['batch']]
        row['previous_manifest'] = '../2026-09-17_dwal-bank-batching/' + row.pop('manifest')
        row['small_manifest'] = str(path.relative_to(root))
        for rec in m['records']:
            assert rec['status'] == 'passed'
            result=rec['result']; hashes.add(str(result['validation']['final_hash']))
            name={'small':'small_direct','dwal_small':'small_dwal'}[rec['engine']]
            row['records'][name]={'metrics':{v['name']:v['value'] for v in result['metrics']},
                                  'counts':result['counts'],'settings':result['settings'],'version':result['version']}
    assert batches == set(rows) and len(revisions) == 1 and hashes == set(entry['hashes'])
    entry['small_revision'] = next(iter(revisions))
(root/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')

html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>psitri · Transfers per commit</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#152a3b;background:#f2f5f7;font-size:15px}*{box-sizing:border-box}
body{margin:0}main{max-width:1120px;margin:auto;padding:42px 30px}header{margin-bottom:28px}.eyebrow{font-size:12px;font-weight:750;letter-spacing:2px;color:#536b7e}h1{font-size:42px;letter-spacing:-1.7px;margin:12px 0}p{line-height:1.6;color:#536574}.badge{display:inline-block;background:#e7eef5;color:#31516c;padding:7px 12px;border-radius:6px;font-weight:650;font-size:12px}.panel{background:white;border:1px solid #dce4eb;border-radius:14px;padding:24px;margin:18px 0}.controls{display:flex;align-items:end;gap:24px;flex-wrap:wrap}label{font-size:13px}select{display:block;min-width:240px;border:1px solid #bdcbd8;padding:10px 12px;border-radius:7px;background:white;font:inherit;margin-top:6px}.engines{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0 0}.engines label{display:flex;align-items:center;gap:6px}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}.group{display:grid;grid-template-columns:115px 1fr;gap:20px;padding:17px 0;border-bottom:1px solid #e8edf2}.group:last-child{border:0}.batch{font-size:21px;font-weight:700;padding-top:5px}.batch small{display:block;color:#758593;font-size:11px;font-weight:500}.bars{display:grid;gap:9px}.barrow{display:grid;grid-template-columns:180px 1fr 100px;align-items:center;gap:10px;font-size:12px}.track{height:21px;background:#f3f6f8;border-radius:3px}.bar{height:100%;border-radius:3px;min-width:2px}.val{text-align:right;font-variant-numeric:tabular-nums;font-weight:650}.meta{font-size:13px}.charthead{display:flex;justify-content:space-between;gap:10px;align-items:center}.charthead h2{font-size:19px;margin:0}.charthead span{font-size:12px;color:#617586}.details{display:grid;grid-template-columns:1fr 1fr;gap:22px}.details h3{font-size:15px;margin:0}.details p{font-size:13px;margin-bottom:0}a{color:#245db7}footer{font-size:12px;color:#72818e;padding:10px 0}@media(max-width:680px){main{padding:25px 15px}h1{font-size:32px}.group{grid-template-columns:65px 1fr;gap:10px}.batch{font-size:16px}.barrow{grid-template-columns:1fr 80px}.barname{grid-column:1/-1;font-size:11px}.details{grid-template-columns:1fr}.panel{padding:17px}}
</style><main><header><div class="eyebrow">PSITRI / SMALL + LARGE ENGINE DIAGNOSTIC</div><h1>Two engines. Two write paths.</h1><p>The same bank transfers through native MDBX and psitri: Small Engine and Large Engine, each with direct writes or DWAL. Retained MDBX shim results use Large Engine and show the compatibility-layer cost.</p><span class="badge">One run per configuration · no per-commit sync</span><p class="meta">ARM not measured: the Mac did not satisfy the idle gate.</p></header>
<section class="panel"><div class="controls"><label>Machine<select id="machine"></select></label><label>Measurement<select id="metric"><option value="write">Writes without concurrent readers</option><option value="mixed_write_active">Writes with four concurrent readers</option><option value="mixed_read">Reader throughput during writes</option><option value="write_batch_p99">Write-batch p99 latency</option><option value="peak_rss">Peak process memory</option><option value="allocated_after_close">Allocated storage after close</option><option value="close">Close and drain time</option><option value="read">Read-only throughput</option></select></label></div><div class="engines" id="engines"></div><p id="hardware" class="meta"></p></section>
<section class="panel"><p class="meta">For mixed writes, also inspect reader throughput. DWAL fresh snapshots acquire the root’s writer lock, so a larger commit can raise writer throughput while delaying readers.</p><div class="charthead"><h2 id="charttitle"></h2><span id="units"></span></div><div id="chart"></div></section>
<section class="panel details"><div><h3>Matched work</h3><p>100,000 accounts. One million transfers per timed write phase. Each transfer reads two balances and updates both. Eight-byte keys and values; identical deterministic traces. Batch size also governs loading and warmup.</p></div><div><h3>Validation before ranking</h3><p>Every plotted cohort passed rollback, snapshot stability, read-your-writes, exact account state, balance conservation, cardinality, and clean-close/reopen checks. This does not test power-loss recovery.</p></div><div><h3>Same workload, different storage</h3><p>DWAL buffers into ART and merges into its selected Small Engine or Large Engine base. Each commit submits WAL bytes without fsync; rotation and close retain full barriers. Direct storage has no WAL. Small Engine uses its default 32 GiB fixed main arena limit. Large Engine read promotion is disabled, with a 256 MiB pinned-cache budget. This dataset fits in RAM.</p></div><div><h3>Diagnostic, not a release claim</h3><p>One isolated process at a time after a sustained idle check. Small Engine bars were measured in this sweep; MDBX, Large Engine, and shim bars are retained from the preceding sweeps on the same host. DWAL configurations are experimental. One run per configuration, without uncertainty intervals. Include close/drain time and memory when evaluating the buffering tradeoff.</p></div></section>
<footer><a href="README.md">Methodology and results</a> · <a href="comparison.json">Download comparison data</a> · <span id="source"></span></footer></main>
<script>const DATA=__DATA__;
const colors={mdbx:'#5379ab',shim_direct:'#b7a4c7',shim_dwal:'#8a51b8',native_direct:'#84baac',native_dwal:'#007f6a',small_direct:'#ce9c33',small_dwal:'#b75524'}, short={mdbx:'Native MDBX',shim_direct:'MDBX shim · direct',shim_dwal:'MDBX shim · DWAL',native_direct:'Large Engine · direct',native_dwal:'Large Engine · DWAL',small_direct:'Small Engine · direct',small_dwal:'Small Engine · DWAL'};
const machine=document.getElementById('machine'), metric=document.getElementById('metric'), chosen=new Set(DATA.engines);
for(const [id,m] of Object.entries(DATA.machines)){const o=document.createElement('option');o.value=id;o.textContent=(id==='x86'?'x86 · ':'ARM · ')+m.hardware.cpu;machine.append(o)}
for(const id of DATA.engines){const label=document.createElement('label'), box=document.createElement('input'),dot=document.createElement('span');box.type='checkbox';box.checked=true;box.onchange=()=>{box.checked?chosen.add(id):chosen.delete(id);render()};dot.className='dot';dot.style.background=colors[id];label.append(box,dot,document.createTextNode(DATA.names[id]));document.getElementById('engines').append(label)}
function render(){const m=DATA.machines[machine.value],key=metric.value, bytes=['peak_rss','allocated_after_close'].includes(key),latency=key==='write_batch_p99',closing=key==='close', scale=bytes?1048576:latency?1000:1;const unit=bytes?'MiB':latency?'milliseconds':closing?'seconds':['mixed_read','read'].includes(key)?'reads / second':'transfers / second';let max=0;for(const r of m.rows)for(const id of chosen)max=Math.max(max,r.records[id].metrics[key]/scale);document.getElementById('hardware').textContent=m.hardware.cpu+' · '+m.hardware.system+' · '+m.hardware.logical_cpus+' logical CPUs · '+m.rows.length+' batch sizes · all validation passed';document.getElementById('source').textContent='Small Engine '+m.small_revision.slice(0,12)+' · Large DWAL '+m.revisions[0].slice(0,12)+' · Large direct '+m.direct_revision.slice(0,12);document.getElementById('charttitle').textContent=metric.selectedOptions[0].textContent;document.getElementById('units').textContent=unit+(bytes||latency||closing?' · lower is better':' · higher is better');const chart=document.getElementById('chart');chart.replaceChildren();if(!chosen.size){chart.textContent='Select an engine to display.';return}for(const r of m.rows){const group=document.createElement('div');group.className='group';const batch=document.createElement('div');batch.className='batch';batch.textContent=r.batch.toLocaleString();const small=document.createElement('small');small.textContent='transfers / commit';batch.append(small);const bars=document.createElement('div');bars.className='bars';for(const id of DATA.engines.filter(e=>chosen.has(e))){const v=r.records[id].metrics[key]/scale,row=document.createElement('div');row.className='barrow';const name=document.createElement('span');name.className='barname';name.textContent=short[id];const track=document.createElement('div');track.className='track';const bar=document.createElement('div');bar.className='bar';bar.style.background=colors[id];bar.style.width=(100*v/max)+'%';track.title=DATA.names[id]+': '+v.toLocaleString()+' '+unit;track.append(bar);const val=document.createElement('span');val.className='val';val.textContent=v.toLocaleString(undefined,{maximumFractionDigits:closing?3:bytes||latency?1:0});row.append(name,track,val);bars.append(row)}group.append(batch,bars);chart.append(group)}}machine.onchange=render;metric.onchange=render;render();</script></html>'''
(root / 'comparison.html').write_text(html.replace('__DATA__', json.dumps(data).replace('</', '<\\/')))
print(root / 'comparison.html')
