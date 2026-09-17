"""Serialize benchmark runners and wait for sustained system CPU idle time."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def linux_ticks():
    values = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    # iowait is not idle for a storage benchmark.
    return sum(values), values[3]


def busy_percent():
    if sys.platform.startswith("linux"):
        total, idle = linux_ticks()
        time.sleep(1)
        new_total, new_idle = linux_ticks()
        return 100 * (1 - (new_idle - idle) / max(1, new_total - total))
    if sys.platform == "darwin":
        output = subprocess.check_output(["top", "-l", "2", "-n", "0", "-s", "1"], text=True)
        idle = re.findall(r"CPU usage:.*?([\d.]+)% idle", output)
        if not idle:
            raise RuntimeError("Could not inspect CPU idle state")
        return 100 - float(idle[-1])
    raise RuntimeError("Idle inspection supports Linux and macOS")


@contextlib.contextmanager
def exclusive():
    path = Path(tempfile.gettempdir()) / "psitri-benchmark.lock"
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another psitri benchmark runner holds the machine-wide lock")
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def wait_idle(log, quiet_seconds=10, threshold=3.0, timeout=1800):
    start = time.monotonic()
    quiet_since = None
    last_notice = start - 30
    with Path(log).open("a") as out:
        while time.monotonic() - start < timeout:
            busy = busy_percent()
            now = time.monotonic()
            quiet_since = (quiet_since or now) if busy <= threshold else None
            out.write(json.dumps(dict(time=time.time(), cpu_busy_percent=round(busy, 3))) + "\n")
            out.flush()
            if quiet_since is not None and now - quiet_since >= quiet_seconds:
                return
            if now - last_notice >= 30:
                print(f"Waiting for {quiet_seconds}s idle (CPU busy {busy:.1f}%, limit {threshold:.1f}%)", flush=True)
                last_notice = now
    raise RuntimeError("Machine did not become idle; no benchmark started")
