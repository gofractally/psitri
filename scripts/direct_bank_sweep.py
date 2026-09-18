#!/usr/bin/env python3
"""One sequential diagnostic per batch size for the three direct bank adapters."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

import benchmark

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--transfers", type=int, default=1_000_000)
    parser.add_argument("--idle-timeout", type=int, default=1800)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    steps = [("smoke", 32)] + [("diagnostic", b) for b in (1, 10, 100, 1000, 10000, 100000)]
    evidence = dict(status="running", started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    command=sys.argv, machine=args.machine, runs=[])
    benchmark.write_json(output / "sweep.json", evidence)
    for index, (profile, batch) in enumerate(steps):
        directory = output / ("smoke" if profile == "smoke" else f"batch-{batch}")
        command = [sys.executable, str(ROOT / "scripts/fair_bank.py"), "--suite", "direct",
                   "--build", str(args.build.resolve()), "--output", str(directory),
                   "--machine", args.machine, "--profile", profile, "--durability", "none",
                   "--batch", str(batch), "--order-seed", str(20260917 + index),
                   "--idle-timeout", str(args.idle_timeout), "--notes",
                   "Direct-engine batch sweep; one run per configuration; read cache disabled "
                   "for both psitri paths; no per-commit sync; no persistent subtree creation."]
        if profile != "smoke":
            command += ["--accounts", "100000", "--transfers", str(args.transfers),
                        "--warmup", "10000", "--readers", "4", "--reads-per-snapshot", "100",
                        "--read-batches", "100", "--value-bytes", "8", "--seed", "12345",
                        "--cache-mib", "256", "--map-gib", "64"]
        run = dict(profile=profile, batch=batch, command=command, status="running")
        evidence["runs"].append(run)
        benchmark.write_json(output / "sweep.json", evidence)
        print(f"Starting {profile}, batch {batch}", flush=True)
        with (output / f"{profile}-{batch}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.send_signal(2)
                process.wait()
                code = 130
        run.update(exit_code=code, status="passed" if code == 0 else "failed")
        manifests = list(directory.glob("*/manifest.json"))
        if len(manifests) == 1:
            run["manifest"] = str(manifests[0].relative_to(output))
        benchmark.write_json(output / "sweep.json", evidence)
        if code:
            evidence["status"] = "failed"
            break
    else:
        evidence["status"] = "passed"
    evidence["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    benchmark.write_json(output / "sweep.json", evidence)
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
