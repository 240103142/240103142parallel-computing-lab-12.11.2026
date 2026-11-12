"""Phase 0 - hardware topology audit + timer noise-floor calibration."""
import multiprocessing as mp
import platform
import re
import subprocess
import sys
import time

from harness import emit_verify_tag
from lab_common import STUDENT_ID, save_results, scale, log


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def fmt_bytes(n):
    if not n:
        return "n/a"
    n = int(n)
    if n >= 1 << 20:
        return f"{n / (1 << 20):g} MB"
    return f"{n / 1024:g} KB"


def detect_mac():
    def s(key):
        return sh(f"sysctl -n {key} 2>/dev/null")
    h = {
        "cpu_model": s("machdep.cpu.brand_string") or "unknown",
        "P_phys": int(s("hw.physicalcpu") or mp.cpu_count()),
        "P_log": int(s("hw.logicalcpu") or mp.cpu_count()),
        "line_bytes": int(s("hw.cachelinesize") or 64),
        "l1d_bytes": int(s("hw.l1dcachesize") or 0),
        "l2_bytes": int(s("hw.l2cachesize") or 0),
        "l3_bytes": int(s("hw.l3cachesize") or 0),
    }
    # Apple Silicon: perflevel0 = Performance cores, perflevel1 = Efficiency cores
    p0 = s("hw.perflevel0.physicalcpu")
    p1 = s("hw.perflevel1.physicalcpu")
    if p0:
        h["p_cores"] = int(p0)
        h["e_cores"] = int(p1 or 0)
        l1 = s("hw.perflevel0.l1dcachesize")
        l2 = s("hw.perflevel0.l2cachesize")
        if l1:
            h["l1d_bytes"] = int(l1)
        if l2:
            h["l2_bytes"] = int(l2)
        h["e_l1d_bytes"] = int(s("hw.perflevel1.l1dcachesize") or 0)
        h["e_l2_bytes"] = int(s("hw.perflevel1.l2cachesize") or 0)
    h["raw"] = sh("sysctl -n machdep.cpu.brand_string && sysctl hw.physicalcpu hw.logicalcpu "
                  "hw.cachelinesize hw.l1dcachesize hw.l2cachesize hw.l3cachesize "
                  "hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu 2>/dev/null")
    return h


def parse_size(txt):
    m = re.search(r"([\d.]+)\s*(K|M|G)i?B?", txt or "", re.I)
    if not m:
        return 0
    mult = {"K": 1024, "M": 1 << 20, "G": 1 << 30}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


def detect_linux():
    raw = sh("lscpu")
    def f(key):
        m = re.search(rf"^{key}:\s*(.+)$", raw, re.M)
        return m.group(1).strip() if m else ""
    sockets = int(f("Socket\\(s\\)") or 1)
    cores = int(f("Core\\(s\\) per socket") or mp.cpu_count())
    l1 = parse_size(f("L1d cache"))
    instances = re.search(r"\((\d+) instances\)", f("L1d cache") or "")
    if instances:
        l1 //= int(instances.group(1))
    l2 = parse_size(f("L2 cache"))
    inst2 = re.search(r"\((\d+) instances\)", f("L2 cache") or "")
    if inst2:
        l2 //= int(inst2.group(1))
    return {
        "cpu_model": f("Model name") or "unknown",
        "P_phys": sockets * cores,
        "P_log": mp.cpu_count(),
        "line_bytes": int(sh("getconf LEVEL1_DCACHE_LINESIZE") or 64),
        "l1d_bytes": l1, "l2_bytes": l2, "l3_bytes": parse_size(f("L3 cache")),
        "raw": sh("lscpu | grep -E 'Model name|Socket|Core|Thread|L1d|L2|L3'"),
    }


def detect_windows():
    raw = sh('powershell -NoProfile -Command "Get-CimInstance Win32_Processor | '
             'Select Name,NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize | Format-List"')
    def f(key):
        m = re.search(rf"{key}\s*:\s*(.+)", raw)
        return m.group(1).strip() if m else ""
    return {
        "cpu_model": f("Name") or platform.processor(),
        "P_phys": int(f("NumberOfCores") or mp.cpu_count()),
        "P_log": int(f("NumberOfLogicalProcessors") or mp.cpu_count()),
        "line_bytes": 64, "l1d_bytes": 0,
        "l2_bytes": int(f("L2CacheSize") or 0) * 1024,
        "l3_bytes": int(f("L3CacheSize") or 0) * 1024,
        "raw": raw,
    }


def main():
    out = []
    sysname = platform.system()
    h = {"Darwin": detect_mac, "Linux": detect_linux, "Windows": detect_windows}.get(sysname, detect_linux)()
    h["os"] = f"Python {platform.python_version()} / {sysname} {platform.release()} ({platform.machine()})"
    h["P_log_mp"] = mp.cpu_count()

    log(out, "=== Phase 0: hardware baseline ===")
    log(out, h["raw"])
    for k in ("cpu_model", "P_phys", "P_log", "line_bytes", "l1d_bytes", "l2_bytes", "l3_bytes", "os"):
        log(out, f"{k:12s}: {h[k]}")
    if "p_cores" in h:
        log(out, f"hybrid     : {h['p_cores']} P-cores + {h['e_cores']} E-cores")

    # Timer noise floor
    n = scale(2_000_000, 200_000)
    pc = time.perf_counter_ns
    t0 = pc()
    for _ in range(n):
        pc()
    t1 = pc()
    avg_ns = (t1 - t0) / n
    deltas = []
    last = pc()
    for _ in range(100_000):
        now = pc()
        if now != last:
            deltas.append(now - last)
        last = now
    min_tick = float(min(deltas)) if deltas else 0.0
    log(out, f"mp.cpu_count()             : {mp.cpu_count()}")
    log(out, f"perf_counter_ns calls      : {n}")
    log(out, f"avg call latency (ns)      : {avg_ns:.4f}")
    log(out, f"min observable tick (ns)   : {min_tick:.4f}")
    tag = emit_verify_tag("PHASE0", STUDENT_ID, avg_ns, min_tick)
    log(out, tag)

    save_results("phase0", {"hardware": h, "timer_calls": n, "avg_call_ns": avg_ns,
                            "min_tick_ns": min_tick, "verify": tag, "log": out})


if __name__ == "__main__":
    main()
