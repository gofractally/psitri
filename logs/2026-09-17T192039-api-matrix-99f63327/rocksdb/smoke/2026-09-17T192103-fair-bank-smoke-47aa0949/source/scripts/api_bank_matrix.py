#!/usr/bin/env python3
"""Collect the three API pairs sequentially, measuring only passing smoke pairs."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import benchmark
import benchmark_idle
import fair_bank


def collect(args):
    fair_bank.WORKLOAD = "api-bank-v1"
    root = args.output.resolve() / (dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%S") + "-api-matrix-" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    manifest = dict(workload=fair_bank.WORKLOAD, status="running", machine=args.machine,
                    revision=benchmark.capture("git", "rev-parse", "HEAD"), pairs=[])
    def save():
        benchmark.write_json(root / "matrix.json", manifest)
    save()
    print(f"Matrix: {root}", flush=True)
    try:
        with benchmark_idle.exclusive():
            for name, engines, build in (
                ("mdbx", ["mdbx", "psitrimdbx"], args.build_mdbx),
                ("rocksdb", ["rocksdb", "psitrirocks"], args.build_integrations),
                ("sqlite", ["sqlite", "psitri_sqlite"], args.build_integrations),
            ):
                pair = dict(api=name, status="running", runs=[])
                manifest["pairs"].append(pair)
                save()
                for profile in ("smoke", "performance"):
                    destination = root / name / profile
                    run = SimpleNamespace(build=build, output=destination, machine=args.machine,
                        profile=profile, engines=engines, repeats=1 if profile == "smoke" else args.repeats,
                        order_seed=20260917, timeout=args.timeout, idle_timeout=args.idle_timeout,
                        notes="Same-API pair; no concurrent benchmarks; wait for idle before every process.",
                        keep_databases=False, durability="none")
                    code = fair_bank.run(run)
                    records = list(destination.glob("*/manifest.json"))
                    if len(records) != 1:
                        raise RuntimeError("Expected exactly one new run archive")
                    result = json.loads(records[0].read_text())
                    pair["runs"].append(str(records[0].relative_to(root)))
                    save()
                    if code:
                        pair["status"] = "failed"
                        if any(r["status"] == "not_run" for r in result["records"]):
                            raise RuntimeError(result.get("error", "Host was not idle"))
                        break  # Preserve failure; do not measure this pair.
                else:
                    pair["status"] = "passed"
                save()
            manifest["status"] = "passed" if all(p["status"] == "passed" for p in manifest["pairs"]) else "failed"
    except (OSError, RuntimeError, KeyboardInterrupt) as error:
        manifest.update(status="blocked", error=str(error) or "Interrupted")
    finally:
        save()
    print(f"{manifest['status'].upper()}: {root / 'matrix.json'}", flush=True)
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-mdbx", type=Path, required=True)
    parser.add_argument("--build-integrations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--idle-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("At least three performance repetitions are required")
    sys.exit(collect(args))
