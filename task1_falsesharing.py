"""Phase 1 - false sharing / cache-line contention + Karp-Flatt."""
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag
from lab_common import STUDENT_ID, save_results, scale, p_phys, p_log, attach_shm, log

M_TOTAL = scale(24_000_000, 600_000)
STRIDES = [1, 2, 4, 8, 16, 32]
WARMUP, TIMED = scale(2, 1), scale(7, 3)


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = attach_shm(shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc            # store to the shared cache line every iteration
    del arr                            # release buffer export before closing
    shm.close()


def make_run(shm_name, p, stride):
    iters = M_TOTAL // p

    def run():
        procs = [mp.Process(target=fs_worker, args=(shm_name, t * stride, iters)) for t in range(p)]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
    return run


def p_list(plog):
    ps, p = [], 1
    while p < plog:
        ps.append(p)
        p *= 2
    ps.append(plog)
    return ps


def main():
    out = []
    P, PL = p_phys(), p_log()
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        # ---------- Table 1A: stride sweep at p = P_phys ----------
        log(out, f"=== Phase 1A: stride sweep, p = P_phys = {P}, M = {M_TOTAL} ===")
        t1a = []
        for s in STRIDES:
            st = run_benchmark(make_run(shm.name, P, s), WARMUP, TIMED)
            mops = M_TOTAL / (st.median_ms / 1e3) / 1e6
            row = {"stride": s, "bytes": s * 8, "median_ms": st.median_ms,
                   "iqr_ms": st.iqr_ms, "min_ms": st.min_ms, "mops": mops}
            t1a.append(row)
            log(out, f"stride={s:2d} ({s*8:3d} B)  median={st.median_ms:10.2f} ms  "
                     f"IQR={st.iqr_ms:8.2f} ms  {mops:7.2f} Mops/s")

        # biggest relative drop between consecutive strides = coherence boundary
        drops = [(t1a[i - 1]["median_ms"] - t1a[i]["median_ms"]) / t1a[i - 1]["median_ms"]
                 for i in range(1, len(t1a))]
        k = max(range(len(drops)), key=lambda i: drops[i]) + 1
        boundary = t1a[k]["bytes"]
        log(out, f"Largest time drop: {t1a[k-1]['bytes']} B -> {boundary} B "
                 f"({drops[k-1]*100:.1f}%)  => estimated coherence line = {boundary} B")

        # ---------- Table 1B: process scaling with stride 1 and 16 ----------
        t1b = {}
        for s in (1, 16):
            log(out, f"=== Phase 1B: scaling sweep, stride = {s} ===")
            rows, T1 = [], None
            for p in p_list(PL):
                st = run_benchmark(make_run(shm.name, p, s), WARMUP, TIMED)
                if p == 1:
                    T1 = st.median_ms
                S = T1 / st.median_ms
                E = S / p
                e = ((1 / S) - (1 / p)) / (1 - 1 / p) if p >= 2 else None
                rows.append({"p": p, "median_ms": st.median_ms, "iqr_ms": st.iqr_ms,
                             "S": S, "E": E, "e": e})
                log(out, f"p={p:2d}  T={st.median_ms:10.2f} ms  S={S:6.3f}  E={E:6.3f}  "
                         f"e(p)={'-' if e is None else f'{e:.4f}'}")
            t1b[str(s)] = rows
    finally:
        shm.close()
        shm.unlink()

    m1 = next(r["median_ms"] for r in t1a if r["stride"] == 1)
    m16 = next(r["median_ms"] for r in t1a if r["stride"] == 16)
    tag = emit_verify_tag("PHASE1", STUDENT_ID, m1, m16)
    log(out, f"median_ms(stride=1)={m1:.4f}  median_ms(stride=16)={m16:.4f}")
    log(out, tag)
    save_results("phase1", {"P": P, "M": M_TOTAL, "table1a": t1a, "boundary_bytes": boundary,
                            "table1b": t1b, "m1": m1, "m16": m16, "verify": tag, "log": out})


if __name__ == "__main__":
    main()
