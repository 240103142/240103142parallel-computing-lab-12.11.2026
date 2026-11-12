"""Shared helpers for all lab phases (stdlib only).

Set STUDENT_ID to your own ID before running.
Set environment variable LAB_QUICK=1 only for a fast smoke test (small sizes);
the real lab runs must be done WITHOUT it.
"""
import json
import multiprocessing as mp
import os
import sys
from multiprocessing import shared_memory

STUDENT_ID = 240103142          # <-- your Student ID (integer)

QUICK = os.environ.get("LAB_QUICK") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def scale(full, quick):
    return quick if QUICK else full


def save_results(phase: str, data: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{phase}.json"), "w") as f:
        json.dump(data, f, indent=2)


def load_results(phase: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{phase}.json")
    if not os.path.exists(path):
        sys.exit(f"Missing {path}: run the earlier phase first.")
    with open(path) as f:
        return json.load(f)


def hw() -> dict:
    """Hardware baseline recorded by Phase 0."""
    return load_results("phase0")["hardware"]


def p_phys() -> int:
    return int(hw()["P_phys"])


def p_log() -> int:
    return mp.cpu_count()


def attach_shm(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing block without registering it with the
    resource tracker (avoids bogus 'leaked shared_memory' warnings on macOS)."""
    try:
        return shared_memory.SharedMemory(name=name, track=False)   # Python 3.13+
    except TypeError:
        shm = shared_memory.SharedMemory(name=name)
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        return shm


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def log(lines: list, s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)
