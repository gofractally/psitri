"""Serialize benchmark runners and wait for sustained system CPU idle time."""
import contextlib
import ctypes
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def linux_ticks():
    values = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    # iowait is not idle for a storage benchmark.
    return sum(values), values[3]


def mac_ticks():
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    ticks = (ctypes.c_uint * 4)()
    count = ctypes.c_uint(4)
    port = lib.mach_host_self()
    try:
        if lib.host_statistics(port, 3, ticks, ctypes.byref(count)):
            raise RuntimeError("Could not inspect CPU idle state")
        return sum(ticks), ticks[2]
    finally:
        task = ctypes.c_uint.in_dll(lib, "mach_task_self_").value
        lib.mach_port_deallocate(task, port)


def busy_percent():
    ticks = linux_ticks if sys.platform.startswith("linux") else mac_ticks if sys.platform == "darwin" else None
    if not ticks:
        raise RuntimeError("Idle inspection supports Linux and macOS")
    total, idle = ticks()
    time.sleep(1)
    new_total, new_idle = ticks()
    return 100 * (1 - (new_idle - idle) / max(1, new_total - total))


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
