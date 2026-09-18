#!/usr/bin/env python3
"""Render the retained direct-bank diagnostic evidence without external assets."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
machines = {}
engines = ['mdbx', 'psitrimdbx', 'large']
names = {'mdbx': 'Native MDBX', 'psitrimdbx': 'MDBX shim · Large Engine', 'large': 'Native psitri · Large Engine'}
for machine in ('x86', 'arm'):
    folder = root / machine
    if not (folder / 'sweep.json').exists():
        continue
    sweep = json.loads((folder / 'sweep.json').read_text())
    entry = {'status': sweep['status'], 'rows': [], 'hardware': {}, 'revisions': set(), 'hashes': set()}
    for run in sweep['runs']:
        if run['profile'] != 'diagnostic' or run['status'] != 'passed':
            continue
        p = folder / run['manifest']
        m = json.loads(p.read_text())
        assert m['status'] == 'passed' and m['profile'] == 'diagnostic' and m['repeats'] == 1
        assert set(m['engines']) == set(engines)
        assert m['config']['durability'] == 'none'
        entry['hardware'] = m['hardware']
        entry['revisions'].add(m['revision'])
        entry['config'] = m['config']
        records = {}
        for rec in m['records']:
            assert rec['status'] == 'passed'
            result = rec['result']
            entry['hashes'].add(str(result['validation']['final_hash']))
            records[rec['engine']] = {
                'metrics': {v['name']: v['value'] for v in result['metrics']},
                'counts': result['counts'], 'settings': result['settings'], 'version': result['version']}
        entry['rows'].append({'batch': run['batch'], 'records': records,
                              'manifest': str(p.relative_to(root))})
    if entry['rows']:
        assert len(entry['hashes']) == 1 and len(entry['revisions']) == 1
        entry['hashes'] = list(entry['hashes'])
        entry['revisions'] = list(entry['revisions'])
        entry['rows'].sort(key=lambda v: v['batch'])
        machines[machine] = entry
assert machines, 'No validated cohorts'
data = {'machines': machines, 'engines': engines, 'names': names}
(root / 'comparison.json').write_text(json.dumps(data, indent=2) + '\n')

md = ['# Direct engine batching diagnostic', '',
      'One run per configuration; these are diagnostic measurements, not repeated release benchmarks.', '',
      'On x86, native psitri and the direct MDBX shim lead at one transfer per commit. '
      'Native MDBX leads both at every measured batch from 10 through 100,000. '
      'At 100,000, native psitri reaches 1.202M transfers/s versus MDBX at 1.376M '
      '(87% of MDBX), while the shim reaches 662K. The gap narrows at the largest '
      'batches, but this sweep does not demonstrate a large-batch crossover.', '',
      'The ARM sweep was interrupted before its first benchmark because the Mac '
      'never satisfied the idle gate. Its waiting log and interrupted manifest are '
      'retained under `arm/`; there are no ARM timings in this report.', '',
      '100,000 accounts; 1,000,000 successful transfers in each timed write phase; '
      '10,000 warmup transfers; eight-byte keys and balances; seed 12345. '
      'Each transfer reads two balances and writes both, without an appended transfer log. '
      'The selected batch size also governs loading and warmup. '
      'The mixed phase adds four readers, each refreshing its snapshot every 100 reads.', '',
      'All paths use no per-commit sync. MDBX and its shim receive WRITEMAP | UTTERLY_NOSYNC. '
      'Both psitri paths use direct sal_art_ref, read promotion disabled, and a 256 MiB pinned-cache budget. '
      'Native psitri sets a 64 GiB capacity limit; the shim retains the allocator default. '
      'This resident dataset does not test capacity, persistent subtree creation, or crash durability.', '',
      'Each process runs alone after ten consecutive seconds below 3% aggregate CPU activity. '
      'The runner retains idle samples, commands, binary hashes, source revisions, compiler flags, '
      'all raw output, and validation. Errors and interrupted attempts are retained separately. '
      'ARM/x86 comparisons also change CPU, OS, compiler and storage; they do not isolate instruction set.', '',
      '[Interactive bar charts](comparison.html) · [Machine-readable comparison](comparison.json)', '']
for machine, entry in machines.items():
    md += [f'## {machine.upper()}: {entry["hardware"]["cpu"]}', '',
           f'Sweep status: {entry["status"]}. Source: `{entry["revisions"][0]}`. '
           f'All completed configurations produced final hash `{entry["hashes"][0]}`.', '',
           'Write-only throughput, transfers/second:', '',
           '| Transfers/commit | Native MDBX | MDBX shim | Native psitri | Shim/MDBX | Native/MDBX |',
           '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in entry['rows']:
        vals = [row['records'][e]['metrics']['write'] for e in engines]
        md.append(f'| {row["batch"]:,} | ' + ' | '.join(f'{v:,.0f}' for v in vals) +
                  f' | {vals[1]/vals[0]:.2f}× | {vals[2]/vals[0]:.2f}× |')
    md += ['', 'With four concurrent readers, writer throughput (active writer interval), transfers/second:', '',
           '| Transfers/commit | Native MDBX | MDBX shim | Native psitri |', '| ---: | ---: | ---: | ---: |']
    for row in entry['rows']:
        md.append(f'| {row["batch"]:,} | ' + ' | '.join(f'{row["records"][e]["metrics"]["mixed_write_active"]:,.0f}' for e in engines) + ' |')
    md += ['', 'Each run checks read-your-writes, rollback, held snapshot stability, exact account '
           'state, conservation of balances, account count, and clean-close/reopen state. '
           'Reopen validation does not establish power-failure recovery.', '']
(root / 'README.md').write_text('\n'.join(md).rstrip() + '\n')

html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>psitri · Transfers per commit</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#152a3b;background:#f2f5f7;font-size:15px}*{box-sizing:border-box}
body{margin:0}main{max-width:1120px;margin:auto;padding:42px 30px}header{margin-bottom:28px}.eyebrow{font-size:12px;font-weight:750;letter-spacing:2px;color:#536b7e}h1{font-size:42px;letter-spacing:-1.7px;margin:12px 0}p{line-height:1.6;color:#536574}.badge{display:inline-block;background:#e7eef5;color:#31516c;padding:7px 12px;border-radius:6px;font-weight:650;font-size:12px}.panel{background:white;border:1px solid #dce4eb;border-radius:14px;padding:24px;margin:18px 0}.controls{display:flex;align-items:end;gap:24px;flex-wrap:wrap}label{font-size:13px}select{display:block;min-width:240px;border:1px solid #bdcbd8;padding:10px 12px;border-radius:7px;background:white;font:inherit;margin-top:6px}.engines{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0 0}.engines label{display:flex;align-items:center;gap:6px}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}.group{display:grid;grid-template-columns:115px 1fr;gap:20px;padding:17px 0;border-bottom:1px solid #e8edf2}.group:last-child{border:0}.batch{font-size:21px;font-weight:700;padding-top:5px}.batch small{display:block;color:#758593;font-size:11px;font-weight:500}.bars{display:grid;gap:9px}.barrow{display:grid;grid-template-columns:110px 1fr 100px;align-items:center;gap:10px;font-size:12px}.track{height:21px;background:#f3f6f8;border-radius:3px}.bar{height:100%;border-radius:3px;min-width:2px}.val{text-align:right;font-variant-numeric:tabular-nums;font-weight:650}.meta{font-size:13px}.charthead{display:flex;justify-content:space-between;gap:10px;align-items:center}.charthead h2{font-size:19px;margin:0}.charthead span{font-size:12px;color:#617586}.details{display:grid;grid-template-columns:1fr 1fr;gap:22px}.details h3{font-size:15px;margin:0}.details p{font-size:13px;margin-bottom:0}a{color:#245db7}footer{font-size:12px;color:#72818e;padding:10px 0}@media(max-width:680px){main{padding:25px 15px}h1{font-size:32px}.group{grid-template-columns:65px 1fr;gap:10px}.barrow{grid-template-columns:1fr 80px}.barname{display:none}.details{grid-template-columns:1fr}.panel{padding:17px}}
</style><main><header><div class="eyebrow">PSITRI / DIRECT ENGINE DIAGNOSTIC</div><h1>How much does batching buy?</h1><p>The same bank transfers through native MDBX, the MDBX compatibility API, and the native Large Engine API.</p><span class="badge">One run per configuration · no per-commit sync</span><p class="meta">ARM not measured: the Mac did not satisfy the idle gate.</p></header>
<section class="panel"><div class="controls"><label>Machine<select id="machine"></select></label><label>Measurement<select id="metric"><option value="write">Writes without concurrent readers</option><option value="mixed_write_active">Writes with four concurrent readers</option><option value="mixed_read">Reader throughput during writes</option><option value="write_batch_p99">Write-batch p99 latency</option><option value="peak_rss">Peak process memory</option><option value="allocated_after_close">Allocated storage after close</option></select></label></div><div class="engines" id="engines"></div><p id="hardware" class="meta"></p></section>
<section class="panel"><div class="charthead"><h2 id="charttitle"></h2><span id="units"></span></div><div id="chart"></div></section>
<section class="panel details"><div><h3>Matched work</h3><p>100,000 accounts. One million transfers per timed write phase. Each transfer reads two balances and updates both. Eight-byte keys and values; identical deterministic traces. Batch size also governs loading and warmup.</p></div><div><h3>Validation before ranking</h3><p>Every plotted cohort passed rollback, snapshot stability, read-your-writes, exact account state, balance conservation, cardinality, and clean-close/reopen checks. This does not test power-loss recovery.</p></div><div><h3>Direct storage on both psitri paths</h3><p>No DWAL or tiered engine. Read promotion disabled; 256 MiB pinned-cache budget. Native psitri sets a 64 GiB capacity limit; the shim retains the allocator default. The dataset fits in RAM.</p></div><div><h3>Diagnostic, not a release claim</h3><p>One isolated process at a time after a sustained idle check. No repeated trials or uncertainty intervals. Machine selection changes CPU, OS, compiler and storage; it does not isolate ARM versus x86.</p></div></section>
<footer><a href="README.md">Methodology and tables</a> · <a href="comparison.json">Download comparison data</a> · <span id="source"></span></footer></main>
<script>const DATA=__DATA__;
const colors={mdbx:'#5379ab',psitrimdbx:'#a775ce',large:'#139783'}, short={mdbx:'Native MDBX',psitrimdbx:'MDBX shim',large:'Native psitri'};
const machine=document.getElementById('machine'), metric=document.getElementById('metric'), chosen=new Set(DATA.engines);
for(const [id,m] of Object.entries(DATA.machines)){const o=document.createElement('option');o.value=id;o.textContent=(id==='x86'?'x86 · ':'ARM · ')+m.hardware.cpu;machine.append(o)}
for(const id of DATA.engines){const label=document.createElement('label'), box=document.createElement('input'),dot=document.createElement('span');box.type='checkbox';box.checked=true;box.onchange=()=>{box.checked?chosen.add(id):chosen.delete(id);render()};dot.className='dot';dot.style.background=colors[id];label.append(box,dot,document.createTextNode(DATA.names[id]));document.getElementById('engines').append(label)}
function render(){const m=DATA.machines[machine.value],key=metric.value, bytes=['peak_rss','allocated_after_close'].includes(key),latency=key==='write_batch_p99', scale=bytes?1048576:latency?1000:1;const unit=bytes?'MiB':latency?'milliseconds':key==='mixed_read'?'reads / second':'transfers / second';let max=0;for(const r of m.rows)for(const id of chosen)max=Math.max(max,r.records[id].metrics[key]/scale);document.getElementById('hardware').textContent=m.hardware.cpu+' · '+m.hardware.system+' · '+m.hardware.logical_cpus+' logical CPUs · '+m.rows.length+' batch sizes · all validation passed';document.getElementById('source').textContent='Source '+m.revisions[0].slice(0,12);document.getElementById('charttitle').textContent=metric.selectedOptions[0].textContent;document.getElementById('units').textContent=unit+(bytes||latency?' · lower is better':' · higher is better');const chart=document.getElementById('chart');chart.replaceChildren();if(!chosen.size){chart.textContent='Select an engine to display.';return}for(const r of m.rows){const group=document.createElement('div');group.className='group';const batch=document.createElement('div');batch.className='batch';batch.textContent=r.batch.toLocaleString();const small=document.createElement('small');small.textContent='transfers / commit';batch.append(small);const bars=document.createElement('div');bars.className='bars';for(const id of DATA.engines.filter(e=>chosen.has(e))){const v=r.records[id].metrics[key]/scale,row=document.createElement('div');row.className='barrow';const name=document.createElement('span');name.className='barname';name.textContent=short[id];const track=document.createElement('div');track.className='track';const bar=document.createElement('div');bar.className='bar';bar.style.background=colors[id];bar.style.width=(100*v/max)+'%';track.title=DATA.names[id]+': '+v.toLocaleString()+' '+unit;track.append(bar);const val=document.createElement('span');val.className='val';val.textContent=v.toLocaleString(undefined,{maximumFractionDigits:bytes||latency?1:0});row.append(name,track,val);bars.append(row)}group.append(batch,bars);chart.append(group)}}machine.onchange=render;metric.onchange=render;render();</script></html>'''
(root / 'comparison.html').write_text(html.replace('__DATA__', json.dumps(data).replace('</', '<\\/')))
print(root / 'comparison.html')
