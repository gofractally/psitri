#!/usr/bin/env python3
"""Build a dependency-light GitHub Pages artifact from an explicit public allowlist."""
from __future__ import annotations

import argparse
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "website"
API_NAMES = ("README", "quickstart", "reference", "operations", "engines", "conformance", "chunked-values", "range-reads")
CONTENT_NAMES = ("compatibility", "benchmarks", "benchmarking")
EVIDENCE = {
    "bank_compare.log": "logs/2026-06-17_gcart_spec-bank-vs-mdbx/bank_compare.log",
    "mdbx.log": "logs/2026-06-17_gcart_spec-bank-vs-mdbx/mdbx.log",
    "scale-1b.log": "logs/2026-06-13_delta_art-1B/run.log",
    "integration-bank-rocksdb.log": "logs/2026-04-06_integration-bank/rocksdb.log",
    "integration-bank-psitrirocks.log": "logs/2026-04-06_integration-bank/psitrirocks.log",
    "integration-bank-psitri-sqlite.log": "logs/2026-04-06_integration-bank/psitri-sqlite.log",
    "integration-bank-system-sqlite.log": "logs/2026-04-06_integration-bank/system-sqlite.log",
    "integration-bank-machine.json": "logs/2026-04-06_integration-bank/machine.json",
    "integration-bank-progress.txt": "logs/2026-04-06_integration-bank/progress.txt",
    "integration-bank-provenance.json": "logs/2026-04-06_integration-bank/provenance.json",
}
NAV = (("index", "API overview"), ("quickstart", "Quickstart"), ("reference", "API reference"),
       ("operations", "Lifetimes & durability"), ("engines", "Choose an engine"),
       ("range-reads", "Range reads"), ("chunked-values", "Large values"),
       ("conformance", "Engine conformance"), ("compatibility", "Compatibility layers"),
       ("benchmarks", "Benchmark evidence"), ("benchmarking", "Record & compare runs"))


def html_name(name: str) -> str:
    return "index" if name == "README" else name


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.errors.append(f"duplicate id: {attrs['id']}")
            self.ids.add(attrs["id"])
        for attribute in ("href", "src"):
            if attribute in attrs:
                self.links.append(attrs[attribute])
        if tag == "img" and "alt" not in attrs:
            self.errors.append("image missing alt")


def validate(output: Path) -> int:
    documents = {}
    errors = []
    for path in output.rglob("*.html"):
        document = Document()
        document.feed(path.read_text())
        documents[path.resolve()] = document
        errors.extend(f"{path.relative_to(output)}: {error}" for error in document.errors)
    for path, document in documents.items():
        for link in document.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            if url.path.startswith("/"):
                errors.append(f"{path.name}: root-relative URL breaks project Pages: {link}")
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(output.resolve()) or not target.exists():
                errors.append(f"{path.relative_to(output)}: missing local target {link}")
            elif url.fragment and target in documents and unquote(url.fragment) not in documents[target].ids:
                errors.append(f"{path.relative_to(output)}: missing anchor {link}")
    if errors:
        raise ValueError("Site validation failed:\n" + "\n".join(errors))
    return len(documents)


def rewrite_api_links(text: str, revision: str) -> str:
    # Rewrite only Markdown link destinations, never example code or prose.
    def replace(match):
        destination = match.group(1)
        url = urlsplit(destination)
        if url.scheme or url.netloc:
            return match.group(0)
        suffix = ("#" + url.fragment) if url.fragment else ""
        if url.path.startswith("../../examples/api/"):
            return "](" + "../examples/" + url.path.rsplit("/", 1)[1] + suffix + ")"
        if url.path.startswith("../../"):
            return f"](https://github.com/gofractally/psitri/blob/{revision}/{url.path[6:]}{suffix})"
        if url.path.endswith(".md"):
            return "](" + html_name(url.path[:-3]) + ".html" + suffix + ")"
        return match.group(0)
    return re.sub(r"\]\(([^\s)]+)\)", replace, text)


def template(title: str, content: str, page: str, revision: str) -> str:
    navigation = "".join(f'<a href="{name}.html"' + (' aria-current="page"' if name == page else '') + f'>{label}</a>' for name, label in NAV)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} — psitri</title><meta name="description" content="{escape(title)}. psitri developer documentation and benchmark evidence.">
<meta name="theme-color" content="#163f35"><link rel="icon" href="../assets/mark.svg" type="image/svg+xml">
<link rel="stylesheet" href="../assets/site.css"><script src="../assets/site.js" defer></script></head>
<body><a class="skip-link" href="#main">Skip to content</a>
<header class="site-header"><a class="brand" href="../"><img src="../assets/mark.svg" alt="" width="30" height="30">psitri<span class="brand-dot">.</span></a>
<button class="menu-toggle" aria-expanded="false" aria-controls="navigation">Menu <span aria-hidden="true">＋</span></button>
<nav id="navigation" aria-label="Main navigation"><a href="../#architecture">Why psitri</a><a href="../#engines">Engines</a><a href="../#benchmarks">Benchmarks</a><a href="quickstart.html">Documentation ↗</a></nav></header>
<div class="docs-layout section-wrap"><aside class="docs-nav" aria-label="Documentation"><p class="eyebrow">Developer guide</p>{navigation}</aside>
<main id="main" class="prose">{content}<div class="doc-meta">Built from psitri revision {escape(revision[:12])}. Source links may require repository access. C++ source API version 1.</div></main></div>
<footer class="site-footer section-wrap"><a class="brand" href="../"><img src="../assets/mark.svg" alt="" width="26" height="26">psitri<span class="brand-dot">.</span></a><nav aria-label="Footer"><a href="index.html">API guide</a><a href="benchmarks.html">Evidence</a><a href="https://github.com/gofractally/psitri">Repository ↗</a></nav></footer></body></html>'''


def build(output: Path) -> None:
    # Only replace a prior generated artifact. Never remove an arbitrary directory.
    output = output.resolve()
    if output == ROOT or ROOT.is_relative_to(output) or output.is_relative_to(SOURCE):
        raise ValueError("Output must not overlap repository sources")
    marker = output / ".psitri-site-build"
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise ValueError(f"Refusing to replace non-generated directory: {output}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    marker.write_text("Generated by scripts/build_site.py\n")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    index = (SOURCE / "index.html").read_text()
    data = json.loads((SOURCE / "data/benchmarks.json").read_text())
    points = [p for p in data["scale"]["points"] if p["keys"] == 10000000 or p["keys"] % 100000000 == 0]
    min_keys, max_keys = points[0]["keys"], points[-1]["keys"]
    max_rss = max(p["rss_gb"] for p in points)
    coordinates = [(400 * (p["keys"] - min_keys) / (max_keys - min_keys), 83 - 80 * p["rss_gb"] / max_rss) for p in points]
    curve = "M" + "L".join(f"{x:.2f} {y:.2f}" for x, y in coordinates)
    index, count = re.subn(r'(<path\s+id="scale-curve"\s+d=")[^"]*', lambda m: m[1] + curve, index)
    if count != 1:
        raise ValueError("Scale plot must have exactly one data-driven curve")
    (output / "index.html").write_text(index)
    for name in ("assets", "data"):
        shutil.copytree(SOURCE / name, output / name)
    (output / ".nojekyll").touch()
    (output / "docs").mkdir()
    for name in API_NAMES:
        text = rewrite_api_links((ROOT / "docs/api" / f"{name}.md").read_text(), revision)
        title = text.splitlines()[0].lstrip("# ")
        body = markdown.markdown(text, extensions=["fenced_code", "tables", "toc", "sane_lists"])
        page = html_name(name)
        (output / "docs" / f"{page}.html").write_text(template(title, body, page, revision))
    for name in CONTENT_NAMES:
        text = (SOURCE / "content" / f"{name}.md").read_text()
        title = text.splitlines()[0].lstrip("# ")
        body = markdown.markdown(text, extensions=["fenced_code", "tables", "toc", "sane_lists"])
        (output / "docs" / f"{name}.html").write_text(template(title, body, name, revision))
    (output / "examples").mkdir()
    for name in ("basic.cpp", "revisions.cpp"):
        shutil.copy2(ROOT / "examples/api" / name, output / "examples" / name)
    (output / "evidence").mkdir()
    for name, path in EVIDENCE.items():
        shutil.copy2(ROOT / path, output / "evidence" / name)
    # Publish only the benchmark paragraphs, not the evolving internal work log.
    art_source = (ROOT / "libraries/start_ref/STATUS.md").read_text()
    art_excerpt = art_source.split("## What exists")[0] + "## Gate results\n\n" + art_source.split("## Gate results\n\n", 1)[1].split("**Bank vs heart", 1)[0]
    (output / "evidence/start_ref-art-excerpt.txt").write_text(
        "Verbatim benchmark excerpts from libraries/start_ref/STATUS.md.\n"
        "Unrelated implementation notes are omitted.\n\n" + art_excerpt)
    (output / "build-info.json").write_text(json.dumps({"revision": revision, "benchmark_data": "data/benchmarks.json"}, indent=2) + "\n")
    count = validate(output)
    print(f"Built {count} validated HTML pages in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/site")
    args = parser.parse_args()
    build(args.output)
