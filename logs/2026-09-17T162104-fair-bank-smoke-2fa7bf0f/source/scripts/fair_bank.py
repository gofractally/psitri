#!/usr/bin/env python3
"""Run one common bank workload in isolated processes; retain every result and failure."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import uuid

import benchmark

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("small", "large", "mdbx", "rocksdb", "sqlite", "duckdb")
CHECKS = ("read_your_writes", "rollback", "snapshot", "exact_accounts", "balance", "reopen")
WORKLOAD = "fair-bank-v1"
SMOKE = dict(accounts=128, transfers=256, warmup=64, batch=32, readers=2,
             reads_per_snapshot=16, read_batches=2, seed=12345, value_bytes=8,
             cache_mib=64, map_gib=64, durability="sync")
PERFORMANCE = dict(SMOKE, accounts=100000, transfers=100000, warmup=10000, batch=1000,
                   readers=4, reads_per_snapshot=100, read_batches=100, cache_mib=256)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_result(result: dict, engine: str, config: dict) -> None:
    if (result.get("schema_version"), result.get("workload"), result.get("engine"), result.get("status")) != (1, WORKLOAD, engine, "passed"):
        raise ValueError("Wrong result schema, workload, engine, or status")
    if result.get("config") != config:
        raise ValueError("Result does not match the requested workload/durability settings")
    validation = result.get("validation", {})
    if any(validation.get(key) is not True for key in CHECKS) or not validation.get("final_hash"):
        raise ValueError("Missing or failed correctness check")
    counts = result.get("counts", {})
    commits = (config["transfers"] + config["batch"] - 1) // config["batch"]
    for key in ("write_transfers", "mixed_transfers"):
        if counts.get(key) != config["transfers"]:
            raise ValueError("Missing or skipped transfers")
    for key in ("write_commits", "mixed_commits"):
        if counts.get(key) != commits:
            raise ValueError("Mismatched commit grouping")
    if counts.get("read_only_reads") != config["readers"] * config["read_batches"] * config["reads_per_snapshot"]:
        raise ValueError("Incomplete read-only workload")
    if not isinstance(counts.get("mixed_reads"), int) or counts["mixed_reads"] <= 0:
        raise ValueError("No concurrent reader work")
    if not result.get("version") or not result.get("settings"):
        raise ValueError("Missing adapter/library configuration")
    metrics = benchmark.validate_metrics(result.get("metrics"))
    if any(m["engine"] != engine for m in metrics):
        raise ValueError("Metric engine mismatch")
    expected = {
        "load": ("accounts/s", "higher"),
        "write": ("transfers/s", "higher"),
        "mixed_write": ("transfers/s", "higher"),
        "mixed_write_active": ("transfers/s", "higher"),
        "mixed_read": ("reads/s", "higher"), "read": ("reads/s", "higher"),
        "write_batch_p50": ("us", "lower"), "write_batch_p99": ("us", "lower"),
        "mixed_write_active_batch_p50": ("us", "lower"),
        "mixed_write_active_batch_p99": ("us", "lower"),
        "disk_before_close": ("bytes", "lower"), "disk_after_close": ("bytes", "lower"),
        "allocated_before_close": ("bytes", "lower"), "allocated_after_close": ("bytes", "lower"),
        "peak_rss": ("bytes", "lower"), "close": ("seconds", "lower"), "reopen": ("seconds", "lower"),
    }
    actual = {m["name"]: (m["unit"], m["direction"]) for m in metrics}
    if actual != expected:
        raise ValueError("Incomplete or changed metric definitions")


def summarize(records: list[dict], engines: list[str], repeats: int, profile: str) -> dict:
    """No speed ranking unless the entire requested cohort passed."""
    expected = {(engine, repetition + 1) for engine in engines for repetition in range(repeats)}
    actual = {(r["engine"], r["repetition"]) for r in records}
    complete = len(records) == len(expected) and actual == expected and all(r["status"] == "passed" for r in records)
    configs = {json.dumps(r["result"]["config"], sort_keys=True) for r in records if r["status"] == "passed"}
    complete &= len(configs) == 1
    hashes = {r["result"]["validation"]["final_hash"] for r in records if r["status"] == "passed"}
    complete &= len(hashes) == 1
    result = dict(status="passed" if complete else "failed", profile=profile,
                  performance_series_complete=bool(complete and profile == "performance" and repeats >= 3),
                  all_engines_same_final_hash=len(hashes) == 1, metrics={})
    if not complete:
        return result
    for engine in engines:
        runs = [r["result"] for r in records if r["engine"] == engine]
        output = {}
        for metric in runs[0]["metrics"]:
            values = [next(m["value"] for m in r["metrics"] if m["name"] == metric["name"]) for r in runs]
            output[metric["name"]] = dict(unit=metric["unit"], direction=metric["direction"],
                                          median=statistics.median(values), minimum=min(values), maximum=max(values),
                                          samples=values)
        result["metrics"][engine] = output
    return result


def capture_run(command: list[str], directory: Path, timeout: float) -> tuple[int, float]:
    start = time.monotonic()
    with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
        try:
            process = subprocess.Popen(command, stdout=out, stderr=err)
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
                err.write((f"\nRunner: {'timeout' if code == 124 else 'interrupted'}\n").encode())
        except OSError as error:
            err.write(str(error).encode()); code = 127
    return code, time.monotonic() - start


def run(args) -> int:
    config = dict(SMOKE if args.profile == "smoke" else PERFORMANCE)
    for key in config:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    engines = args.engines
    if not engines or len(set(engines)) != len(engines):
        raise ValueError("Choose unique engines")
    repeats = args.repeats if args.repeats is not None else (1 if args.profile == "smoke" else 3)
    if repeats < 1 or (args.profile == "performance" and repeats < 3):
        raise ValueError("Performance comparisons require at least three repetitions")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")

    now = dt.datetime.now(dt.timezone.utc)
    archive = args.output.resolve() / f"{now:%Y-%m-%dT%H%M%S}-fair-bank-{args.profile}-{uuid.uuid4().hex[:8]}"
    source_status = benchmark.capture("git", "status", "--porcelain", "--untracked-files=all") or ""
    archive.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, workload=WORKLOAD, profile=args.profile, status="running",
                    started_at=now.isoformat(), machine_id=args.machine, hardware=benchmark.machine_info(),
                    config=config, engines=engines, repeats=repeats, order_seed=args.order_seed,
                    revision=benchmark.capture("git", "rev-parse", "HEAD"), source_status=source_status,
                    dirty=bool(source_status), command=sys.argv, records=[], dependencies={}, binaries={})
    manifest["environment"] = {k: os.environ[k] for k in (
        "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "HEART_PINNED_MB", "HEART_READ_CACHE", "HEART_READ_CACHE_SEC") if k in os.environ}
    manifest["notes"] = args.notes
    benchmark.write_json(archive / "manifest.json", manifest)
    print(f"Archive: {archive}", flush=True)
    try:
        if any(k.startswith("HEART_") for k in manifest["environment"]):
            raise ValueError("Unset HEART_* overrides so the recorded adapter configuration is authoritative")
        if args.profile == "performance" and source_status:
            raise ValueError("Performance profile requires a clean source checkout; commit code/evidence first")
        build = args.build.resolve()
        build_info = json.loads((build / "fair-bank-build.json").read_text())
        if build_info["build_type"] != "Release":
            raise ValueError("Use a Release build for this comparison suite")
        manifest["build"] = build_info
        manifest["compiler_version_output"] = benchmark.capture(build_info["compiler"], "--version")
        manifest["native_dependencies"] = {}
        for name, source in build_info.get("native_dependencies", {}).items():
            path = Path(source)
            revision = benchmark.capture("git", "rev-parse", "HEAD", cwd=path)
            status = benchmark.capture("git", "status", "--porcelain", cwd=path)
            manifest["native_dependencies"][name] = dict(path=source, revision=revision, status=status)
            if args.profile == "performance" and (not revision or status):
                raise ValueError(f"Performance profile requires a clean, identified {name} dependency")
        for name in ("CMakeCache.txt", "compile_commands.json", "fair-bank-build.json"):
            shutil.copyfile(build / name, archive / name)
        diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
        (archive / "source.patch").write_bytes(diff)
        source_files = [ROOT / "scripts/fair_bank.py", ROOT / "scripts/benchmark.py", *sorted((ROOT / "benchmarks/fair_bank").glob("*"))]
        manifest["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in source_files if p.is_file()}
        # Preserve uncommitted benchmark source too, not merely its hash.
        for path in source_files:
            if path.is_file():
                dest = archive / "source" / path.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
        for library in build_info["libraries"]:
            path = Path(library).resolve()
            manifest["dependencies"][library] = dict(resolved_path=str(path), sha256=digest(path),
                compiler_and_build_flags="prebuilt dependency; inspect its separate build provenance")
        for engine in engines:
            binary = build / "bin" / ("fair-bank-" + engine)
            description = subprocess.run([str(binary), "--describe"], text=True, capture_output=True, check=True, timeout=30)
            desc = json.loads(description.stdout)
            if desc.get("engine") != engine or desc.get("workload") != WORKLOAD:
                raise ValueError("Incompatible benchmark binary")
            if config["durability"] == "relaxed" and not desc["relaxed"]:
                raise ValueError(f"{engine} does not support matched relaxed commits; select sync or explicitly choose a smaller cohort")
            manifest["binaries"][engine] = dict(path=str(binary), sha256=digest(binary), description=desc)
        benchmark.write_json(archive / "manifest.json", manifest)
        order = list(engines)
        random.Random(args.order_seed).shuffle(order)
        # Rotate the initial randomized order so each repetition changes thermal/cache position.
        for repetition in range(repeats):
            schedule = order[repetition % len(order):] + order[:repetition % len(order)]
            for engine in schedule:
                directory = archive / f"r{repetition + 1}-{engine}"
                directory.mkdir()
                command = [manifest["binaries"][engine]["path"], "--path", str(directory / "database"),
                           "--result", str(directory / "result.json")]
                for key, value in config.items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                record = dict(engine=engine, repetition=repetition + 1, directory=directory.name,
                              command=command, status="running", exit_code=None)
                manifest["records"].append(record); benchmark.write_json(archive / "manifest.json", manifest)
                print(f"[{repetition + 1}/{repeats}] {engine}", flush=True)
                if digest(Path(command[0])) != manifest["binaries"][engine]["sha256"]:
                    raise ValueError("Benchmark binary changed during the series")
                code, elapsed = capture_run(command, directory, args.timeout)
                record.update(exit_code=code, elapsed_seconds=elapsed, status="failed")
                try:
                    if code:
                        raise ValueError(f"Process exited {code}; see stderr.log")
                    result = json.loads((directory / "result.json").read_text())
                    validate_result(result, engine, config)
                    record.update(status="passed", result=result)
                except (ValueError, OSError) as error:
                    record["error"] = str(error)
                benchmark.write_json(directory / "run.json", record)
                benchmark.write_json(archive / "manifest.json", manifest)
                print(f"  {record['status']} ({elapsed:.2f}s)", flush=True)
                if record["status"] == "passed" and not args.keep_databases:
                    shutil.rmtree(directory / "database")  # Only this run's generated, validated database.
                if code == 130:
                    raise KeyboardInterrupt
        summary = summarize(manifest["records"], engines, repeats, args.profile)
        benchmark.write_json(archive / "summary.json", summary)
        manifest["status"] = summary["status"]
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=str(error) or "Interrupted")
        print(f"Comparison failed: {manifest['error']}", file=sys.stderr)
    finally:
        manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        benchmark.write_json(archive / "manifest.json", manifest)
    print(f"{manifest['status'].upper()}: {archive / 'manifest.json'}", flush=True)
    return 0 if manifest["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fair")
    parser.add_argument("--output", type=Path, default=ROOT / "logs")
    parser.add_argument("--machine", required=True, help="Stable physical runner identity")
    parser.add_argument("--profile", choices=("smoke", "performance"), default="smoke")
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--order-seed", type=int, default=20260917)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--notes", default="")
    parser.add_argument("--keep-databases", action="store_true")
    for key in SMOKE:
        if key == "durability":
            parser.add_argument("--durability", choices=("sync", "relaxed"))
        else:
            parser.add_argument("--" + key.replace("_", "-"), type=int)
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
