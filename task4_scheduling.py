"""Phase 4 - skewed workload: static block vs cyclic vs dynamic chunk queue."""
import math
import multiprocessing as mp
import time

from harness import run_benchmark, emit_verify_tag
from lab_common import STUDENT_ID, save_results, scale, p_phys, log

N = scale(12_000, 3_000)
CHUNKS = [1, 4, 16, 64, 256, 1024]
WARMUP, TIMED = scale(0, 0), scale(3, 1)


def item(i: int) -> float:
    k = 1 + int((i / 60) ** 2)            # K(i) = 1 + floor((i/60)^2)
    x = 0.5 + (i % 7) * 0.01
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    for _ in range(k):
        x = sqrt(abs(sin(x) + cos(x)) + 0.5)
    return x


def w_block(t, p, n, times):
    t0 = time.perf_counter_ns()
    for i in range(t * n // p, (t + 1) * n // p):
        item(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def w_cyclic(t, p, n, times):
    t0 = time.perf_counter_ns()
    for i in range(t, n, p):
        item(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def w_dynamic(t, p, n, times, counter, chunk):
    t0 = time.perf_counter_ns()
    while True:
        with counter.get_lock():          # synchronised claim on the shared index
            start = counter.value
            counter.value = start + chunk
        if start >= n:
            break
        for i in range(start, min(start + chunk, n)):
            item(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def run_strategy(p, kind, chunk=0):
    records = []

    def once():
        times = mp.Array("d", p)
        if kind == "dynamic":
            counter = mp.Value("i", 0)
            procs = [mp.Process(target=w_dynamic, args=(t, p, N, times, counter, chunk)) for t in range(p)]
        else:
            fn = w_block if kind == "block" else w_cyclic
            procs = [mp.Process(target=fn, args=(t, p, N, times)) for t in range(p)]
        t0 = time.perf_counter_ns()
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        wall = (time.perf_counter_ns() - t0) / 1e6
        records.append((wall, list(times)))

    st = run_benchmark(once, WARMUP, TIMED)
    timed = records[WARMUP:]
    wall, tw = min(timed, key=lambda r: abs(r[0] - st.median_ms))   # run nearest the median
    lam = max(tw) / (sum(tw) / p)
    return {"T_wall": st.median_ms, "iqr": st.iqr_ms, "t_worker": tw, "lambda": lam}


def main():
    out = []
    P = p_phys()
    work = [1 + int((i / 60) ** 2) for i in range(N)]
    log(out, f"=== Phase 4: skewed workload N = {N}, p = P_phys = {P} ===")
    log(out, f"K(0) = {work[0]}, K(N-1) = {work[-1]}, total iterations = {sum(work)}")

    rows = []
    for label, kind, c in ([("Static block", "block", 0), ("Static cyclic", "cyclic", 0)] +
                           [(f"Dynamic C={c}", "dynamic", c) for c in CHUNKS]):
        r = run_strategy(P, kind, c)
        r.update({"label": label, "kind": kind, "chunk": c})
        rows.append(r)
        log(out, f"{label:16s} T_wall={r['T_wall']:10.2f} ms  IQR={r['iqr']:7.2f}  lambda={r['lambda']:.3f}  "
                 f"t_worker=[{', '.join(f'{x:.0f}' for x in r['t_worker'])}]")

    dyn = [r for r in rows if r["kind"] == "dynamic"]
    best = min(dyn, key=lambda r: r["T_wall"])
    c_star = best["chunk"]
    T1 = rows[0]["T_wall"]
    log(out, f"C* = {c_star}  speedup over static block = {T1 / best['T_wall']:.2f}x")
    tag = emit_verify_tag("PHASE4", STUDENT_ID, T1, best["T_wall"])
    log(out, f"T_wall(Strategy 1)={T1:.4f}  T_wall(Strategy 3, C*)={best['T_wall']:.4f}")
    log(out, tag)
    save_results("phase4", {"N": N, "P": P, "rows": rows, "c_star": c_star, "verify": tag, "log": out})


if __name__ == "__main__":
    main()
