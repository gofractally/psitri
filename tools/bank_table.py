#!/usr/bin/env python3
"""bank_table.py <log...> — aligned, color-coded bank comparison table.
Higher is better for rates, lower for sizes; green* marks each winner."""
import re, sys, os

rows = []
for f in sys.argv[1:]:
    name = re.sub(r'^bm_|\.log$', '', os.path.basename(f))
    t = open(f).read()
    def num(pat, idx=1):
        m = re.findall(pat, t)
        return float(m[-1].replace(',', '')) if m else 0.0
    rows.append({
        'name': name,
        'w':  num(r'Write-only:\s+([\d,]+) tx/sec'),
        'wr': num(r'Write\+Read:\s+([\d,]+) tx/sec'),
        'rw': num(r'([\d,]+) reads/sec \(with writer\)'),
        'rs': num(r'([\d,]+) reads/sec \(solo\)'),
        'fs': num(r'Size: file=([\d.]+) MB'),
        'lv': num(r'live=([\d.]+) MB'),
    })

cols = [('write tx/s','w',max), ('w+r tx/s','wr',max), ('reads w/wr','rw',max),
        ('reads solo','rs',max), ('file MB','fs',min), ('live MB','lv',min)]
G, R = '\033[1;32m', '\033[0m'
print(f"{'engine':<8}" + ''.join(f"{h:>13}" for h, _, _ in cols))
best = {k: agg(r[k] for r in rows) for _, k, agg in cols}
for r in rows:
    line = f"{r['name']:<8}"
    for _, k, _ in cols:
        v, win = r[k], r[k] == best[k]
        s = f"{v:,.0f}" if k in 'w wr rw rs'.split() else f"{v:,.1f}"
        line += (G if win else '') + f"{s:>12}" + ('*' if win else ' ') + (R if win else '')
    print(line)
