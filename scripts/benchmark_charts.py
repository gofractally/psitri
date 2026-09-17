"""Render curated benchmark bars once, with optional client-side filtering."""
from html import escape


METRIC_TITLES = {
    "write": "Write throughput", "read": "Read throughput",
    "mixed_write": "Writes + readers", "mixed_read": "Reads + writer",
    "load": "Bulk load", "disk": "Disk high-water",
}
ART_CASES = {
    "u64_hit": "Integer hit", "u64_miss": "Integer miss",
    "dictionary_hit": "Dictionary hit", "dictionary_miss": "Dictionary miss",
    "u64_insert": "Integer insert", "dictionary_insert": "Dictionary insert",
}


def controls(engines):
    choices = []
    for engine in engines:
        selected = engine["status"] != "research"
        badge = '<small>experimental</small>' if not selected else ''
        choices.append(
            f'<label class="engine-choice"{" hidden" if not selected else ""}><input type="checkbox" data-chart-choice '
            f'value="{escape(engine["id"])}" aria-label="{escape(engine["name"])}" '
            f'data-default="{str(selected).lower()}" data-status="{engine["status"]}" '
            f'{"checked" if selected else ""} disabled><span>{escape(engine["name"])}{badge}</span></label>'
        )
    experimental = ''
    if any(e["status"] == "research" for e in engines):
        experimental = '<label class="research-toggle"><input type="checkbox" data-chart-research disabled> Include experimental engines</label>'
    return f'''<fieldset class="engine-filter"><legend>Choose engines to compare</legend>
<div class="engine-choices">{''.join(choices)}</div>
<div class="engine-actions">{experimental}<button type="button" data-chart-reset disabled>Reset selection</button>
<button type="button" data-chart-clear disabled>Clear selection</button></div></fieldset>'''


def row(engine, value, largest, baseline, direction, ratio_only=False):
    ratio = value / baseline
    worse = (ratio > 1) if direction == "lower" else (ratio < 1)
    numbers = (f'<span>{value:.2f}×</span>' if ratio_only else
               f'<span>{value:,.0f}</span><b class="{"worse" if worse else ""}">{ratio:.2f}×</b>')
    return f'''<div class="bench-row {engine["status"]}" data-chart-engine="{escape(engine["id"])}" data-value="{value}">
<div class="bench-label">{escape(engine["name"])}<small>{escape(engine["detail"])}</small></div>
<div class="bar-track" aria-hidden="true"><div class="bar-fill" style="--bar-width:{100 * value / largest:.4f}%"></div></div>
<div class="bench-value{' worse' if ratio_only and worse else ''}">{numbers}</div></div>'''


def bank_chart(data, chart_id):
    bank = data["bank"]
    baseline = next(e for e in bank["engines"] if e["id"] == "mdbx")
    tabs, panels = [], []
    for metric, title in METRIC_TITLES.items():
        spec = bank["metrics"][metric]
        tabs.append(f'<button type="button" data-chart-metric="{metric}" aria-pressed="{str(metric == "write").lower()}" aria-controls="{chart_id}-{metric}" disabled>{title}</button>')
        largest = max(e["values"][metric] for e in bank["engines"])
        bars = ''.join(row(e, e["values"][metric], largest, baseline["values"][metric], spec["direction"]) for e in bank["engines"])
        panels.append(f'''<section data-chart-panel="{metric}" id="{chart_id}-{metric}" aria-label="{title}">
<h4>{title}</h4><p class="chart-caption">{escape(spec["label"])} ({spec["unit"]}) · {spec["direction"]} is better</p>
<div class="benchmark-bars">{bars}</div></section>''')
    return f'''<div class="benchmark-panel comparison-chart" data-comparison-chart="bank" data-baseline="mdbx" data-baseline-name="libmdbx" aria-label="Native bank engine comparison">
<div class="benchmark-title"><div><p class="eyebrow">Native engines · June 17, 2026</p><h3>Compare the engines</h3></div></div>
{controls(bank["engines"])}
<div class="metric-tabs" role="group" aria-label="Select benchmark metric">{''.join(tabs)}</div>
<p class="chart-selection" role="status" data-chart-status>All archived results shown. Enable JavaScript to filter engines and metrics.</p>
<p class="chart-empty" data-chart-empty hidden>Select at least one engine to show the comparison.</p>
{''.join(panels)}
<div class="benchmark-footnote"><p>Bars start at zero and rescale to the selected engines. Ratios always use libmdbx as the reference, even when its bar is hidden. Higher throughput is better; lower disk high-water is better.</p>
<p>One recorded run per command. Disk high-water includes allocation and retained versions; it is not settled live payload.</p></div></div>'''


def art_chart(data):
    engines = [
        dict(id="libart", name="libart", detail="Reference latency", status="baseline"),
        dict(id="psitri32", name="psitri (32-bit handles)", detail="Measured direct-address configuration", status="maintained"),
    ]
    ratios = data["art"]["ratios"]
    largest = max(1, *ratios.values())
    panels = []
    for case, title in ART_CASES.items():
        bars = row(engines[0], 1, largest, 1, "lower", True) + row(engines[1], ratios[case], largest, 1, "lower", True)
        panels.append(f'<section data-chart-panel="{case}" aria-label="{title}"><h4>{title}</h4><div class="benchmark-bars">{bars}</div></section>')
    return f'''<div class="benchmark-panel comparison-chart art-comparison" data-comparison-chart="art" data-baseline="libart" data-baseline-name="libart" aria-label="ART latency comparison">
<div class="benchmark-title"><div><p class="eyebrow">32-bit handles · June 11, 2026</p><h3>Latency by operation</h3></div></div>
{controls(engines)}
<p class="chart-caption">Latency relative to libart · lower is better · libart = 1.00×</p>
<p class="chart-selection" role="status" data-chart-status>All recorded ratios shown. Enable JavaScript to filter engines.</p>
<p class="chart-empty" data-chart-empty hidden>Select at least one engine to show the comparison.</p>
{''.join(panels)}
<div class="benchmark-footnote"><p>All operations share one scale starting at zero. Ratios keep libart at 1.00× even when its bar is hidden. Absolute timings were not retained.</p>
<p>This measures the four-byte configuration, not the supplied five-byte Small Engine profile. Insert latency was higher in both measured cases.</p></div></div>'''
