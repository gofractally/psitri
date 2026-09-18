#!/usr/bin/env python3
"""Capture benchmark evidence and compare runs with matched conditions (stdlib only)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
BANK_METRICS = (
    ("load", "accounts/s", "higher"),
    ("write", "transfers/s", "higher"),
    ("mixed_read", "reads/s", "higher"),
    ("mixed_write", "transfers/s", "higher"),
    ("read", "reads/s", "higher"),
    ("live", "MB", "lower"),
    ("disk", "MB", "lower"),
)
RECORDED_ENV = ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "DELTA_HI_WATER_MB")


def capture(*args: str, cwd: Path = ROOT) -> str | None:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_metrics(metrics: object) -> list[dict]:
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Expected a nonempty JSON list of metrics")
    seen = set()
    for metric in metrics:
        if not isinstance(metric, dict):
            raise ValueError("Each metric must be an object")
        for key in ("engine", "name", "unit"):
            if not isinstance(metric.get(key), str) or not metric[key]:
                raise ValueError(f"Metric requires a nonempty {key}")
        value = metric.get("value")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("Metric values must be finite nonnegative numbers")
        if metric.get("direction") not in ("higher", "lower"):
            raise ValueError("Metric direction must be higher or lower")
        identity = (metric["engine"], metric["name"])
        if identity in seen:
            raise ValueError(f"Duplicate metric: {identity}")
        seen.add(identity)
    return metrics


def parse_bank(output: str) -> list[dict]:
    metrics = []
    engines = set()
    for line in output.splitlines():
        cols = line.split()
        if len(cols) != 9 or not all(re.fullmatch(r"\d+", v) for v in cols[1:8]):
            continue
        engine = cols[0]
        if cols[8] != "PASS":
            raise ValueError(f"Correctness check failed for {engine}: {cols[8]}")
        if engine in engines:
            raise ValueError(f"Duplicate engine {engine}; record each repetition separately")
        engines.add(engine)
        for (name, unit, direction), value in zip(BANK_METRICS, map(int, cols[1:8])):
            if name == "live" and value == 0:
                continue  # Historical adapters use zero for missing instrumentation.
            metrics.append(dict(engine=engine, name=name, value=value, unit=unit, direction=direction))
    return validate_metrics(metrics)


def machine_info() -> dict:
    result = dict(system=platform.system(), release=platform.release(), architecture=platform.machine())
    if sys.platform == "darwin":
        for key, query in (("model", "hw.model"), ("cpu", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")):
            result[key] = capture("sysctl", "-n", query)
    elif sys.platform.startswith("linux"):
        cpu = Path("/proc/cpuinfo").read_text()
        result["cpu"] = next((line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("model name")), platform.processor())
        result["memory"] = next((line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")), None)
    result["logical_cpus"] = os.cpu_count()
    return result


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Supply the benchmark command after --")
    params = {}
    for entry in args.param:
        key, separator, value = entry.partition("=")
        if not separator or not key or key in params:
            raise ValueError("Each --param must be a unique KEY=VALUE")
        params[key] = value
    label = re.sub(r"[^a-zA-Z0-9_-]+", "-", args.label).strip("-")
    if not label:
        raise ValueError("Label must contain letters or numbers")
    now = dt.datetime.now(dt.timezone.utc)
    directory = args.output.resolve() / f"{now:%Y-%m-%dT%H%M%S}-{label}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    status = capture("git", "status", "--porcelain", "--untracked-files=all") or ""
    diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, capture_output=True).stdout
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "id": directory.name,
        "label": args.label,
        "started_at": now.isoformat(),
        "status": "running",
        "command": command,
        "cwd": str(Path.cwd()),
        "revision": capture("git", "rev-parse", "HEAD"),
        "branch": capture("git", "branch", "--show-current"),
        "dirty": bool(status),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": [line[3:] for line in status.splitlines() if line.startswith("?? ")],
        "comparison": {
            "workload": args.workload,
            "machine_id": args.machine,
            "hardware": machine_info(),
            "build": args.build,
            "compiler": args.compiler,
            "durability": args.durability,
            "parameters": params,
            "environment": {k: os.environ[k] for k in RECORDED_ENV if k in os.environ},
            "parser": args.parser,
            "command": [Path(command[0]).name, *command[1:]],
        },
        "notes": args.notes,
        "exit_code": None,
        "metrics": [],
    }
    write_json(directory / "run.json", metadata)  # Retain interrupted runs, too.
    (directory / "README.md").write_text(
        f"# {args.label}\n\nCaptured {now.isoformat()}. See `run.json` for complete metadata.\n\n"
        f"```sh\n{shlex.join(command)}\n```\n\n{args.notes}\n\n"
        "Raw output is in `stdout.log` and `stderr.log`. Review paths and notes before publishing.\n"
    )
    print(f"Recording: {directory}", file=sys.stderr, flush=True)
    start = time.monotonic()
    try:
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            def pump(source, destination, terminal):
                for block in iter(lambda: source.read1(8192), b""):
                    destination.write(block)
                    destination.flush()
                    try:
                        terminal.buffer.write(block)
                        terminal.buffer.flush()
                    except (BrokenPipeError, OSError):
                        pass  # A closed terminal must not lose the evidence.

            threads = [threading.Thread(target=pump, args=(process.stdout, stdout, sys.stdout)),
                       threading.Thread(target=pump, args=(process.stderr, stderr, sys.stderr))]
            for thread in threads:
                thread.start()
            try:
                metadata["exit_code"] = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                metadata["exit_code"] = 130
                metadata["error"] = "Interrupted by user"
            finally:
                for thread in threads:
                    thread.join()
        metadata["status"] = "passed" if metadata["exit_code"] == 0 else "failed"
        if metadata["status"] == "passed" and args.parser != "none":
            raw = (directory / "stdout.log").read_text(errors="replace")
            metadata["metrics"] = parse_bank(raw) if args.parser == "bank" else validate_metrics(json.loads(raw))
    except (OSError, ValueError) as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        print(f"Capture failed: {error}", file=sys.stderr)
    finally:
        metadata["elapsed_seconds"] = round(time.monotonic() - start, 6)
        metadata["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(directory / "run.json", metadata)
    print(f"Saved {metadata['status']} run: {directory / 'run.json'}", file=sys.stderr)
    return 0 if metadata["status"] == "passed" else (metadata["exit_code"] or 2) if (metadata["exit_code"] or 0) > 0 else 2


def read_run(path: Path) -> dict:
    data = json.loads((path / "run.json" if path.is_dir() else path).read_text())
    if data.get("schema_version") != SCHEMA_VERSION or data.get("status") != "passed":
        raise ValueError(f"{path}: comparison requires a passed schema-{SCHEMA_VERSION} run")
    validate_metrics(data.get("metrics"))
    if not isinstance(data.get("comparison"), dict) or not data["comparison"]:
        raise ValueError(f"{path}: missing comparison metadata")
    return data


def compare_runs(baseline: dict, candidate: dict, threshold: float) -> tuple[list[dict], bool]:
    if baseline["comparison"] != candidate["comparison"]:
        differing = sorted(key for key in baseline["comparison"].keys() | candidate["comparison"].keys()
                           if baseline["comparison"].get(key) != candidate["comparison"].get(key))
        raise ValueError("Incompatible conditions: " + ", ".join(differing))
    identity = lambda m: (m["engine"], m["name"])
    old = {identity(m): m for m in validate_metrics(baseline["metrics"])}
    new = {identity(m): m for m in validate_metrics(candidate["metrics"])}
    if old.keys() != new.keys():
        raise ValueError("Metric sets differ; missing metrics cannot pass a regression check")
    rows = []
    for key, before in old.items():
        after = new[key]
        if (before["unit"], before["direction"]) != (after["unit"], after["direction"]):
            raise ValueError(f"Metric definition changed: {key}")
        a, b = before["value"], after["value"]
        if a == 0 and b != 0:
            raise ValueError(f"Zero baseline cannot produce a percentage comparison: {key}")
        change = 0 if a == 0 else (b - a) / a * 100
        regression = -change if before["direction"] == "higher" else change
        rows.append({"engine": key[0], "metric": key[1], "baseline": a, "candidate": b,
                     "unit": before["unit"], "change_percent": change,
                     "regression_percent": regression, "regressed": regression > threshold})
    return rows, any(row["regressed"] for row in rows)


def compare(args: argparse.Namespace) -> int:
    if not math.isfinite(args.threshold) or args.threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    old, new = read_run(args.baseline), read_run(args.candidate)
    rows, failed = compare_runs(old, new, args.threshold)
    print(f"{'Engine / metric':38} {'Baseline':>13} {'Candidate':>13} {'Change':>10}  Status")
    for row in rows:
        print(f"{row['engine'] + ' / ' + row['metric']:38} {row['baseline']:13,.2f} {row['candidate']:13,.2f} "
              f"{row['change_percent']:+9.2f}%  {'REGRESSION' if row['regressed'] else 'ok'}")
    if args.report:
        write_json(args.report, {"schema_version": SCHEMA_VERSION, "baseline": old["id"],
                                "candidate": new["id"], "threshold_percent": args.threshold,
                                "regressed": failed, "metrics": rows})
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    rec = commands.add_parser("record", help="Capture a command and retain its output even if it fails")
    for name in ("label", "workload", "machine", "build", "compiler", "durability"):
        rec.add_argument("--" + name, required=True)
    rec.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    rec.add_argument("--notes", default="")
    rec.add_argument("--output", type=Path, default=ROOT / "logs")
    rec.add_argument("--parser", choices=("none", "bank", "json"), default="none")
    rec.add_argument("command", nargs=argparse.REMAINDER)
    cmp = commands.add_parser("compare", help="Reject mismatched runs and flag performance regressions")
    cmp.add_argument("baseline", type=Path)
    cmp.add_argument("candidate", type=Path)
    cmp.add_argument("--threshold", type=float, default=5.0, help="Allowed regression percentage (default 5)")
    cmp.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        return record(args) if args.action == "record" else compare(args)
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
