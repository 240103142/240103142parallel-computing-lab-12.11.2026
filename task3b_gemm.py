"""Phase 3B - progressive parallel GEMM optimisation + 10-point tile sweep."""
import multiprocessing as mp
from array import array
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag
from lab_common import STUDENT_ID, save_results, load_results, scale, hw, p_phys, attach_shm, log

N = scale(384, 96)
BS_LIST = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
WARMUP, TIMED = scale(1, 0), scale(3, 1)

_g = {}


def init_gemm(na, nb, nc, n):
    for k, nm in (("A", na), ("B", nb), ("C", nc)):
        shm = attach_shm(nm)
        _g["shm" + k] = shm
        _g[k] = shm.buf.cast("d")
    _g["N"] = n


def v1_naive(r0, r1):
    A, B, C, n = _g["A"], _g["B"], _g["C"], _g["N"]
    for i in range(r0, r1):
        io = i * n
        for j in range(n):
            dot = 0.0
            for k in range(n):
                dot += A[io + k] * B[k * n + j]       # B walked with stride N
            C[io + j] = dot


def v2_ikj(r0, r1):
    A, B, C, n = _g["A"], _g["B"], _g["C"], _g["N"]
    for i in range(r0, r1):
        io = i * n
        crow = [0.0] * n
        for k in range(n):
            aik = A[io + k]                           # hoisted
            brow = B[k * n:(k + 1) * n]               # contiguous row slice (zero-copy view)
            for j in range(n):
                crow[j] += aik * brow[j]              # unit stride
        C[io:io + n] = array("d", crow)


def v3_blocked(r0, r1, bs):
    A, B, C, n = _g["A"], _g["B"], _g["C"], _g["N"]
    cloc = [[0.0] * n for _ in range(r1 - r0)]
    for ii in range(r0, r1, bs):
        imax = min(ii + bs, r1)
        for jj in range(0, n, bs):
            jmax = min(jj + bs, n)
            for kk in range(0, n, bs):
                kmax = min(kk + bs, n)
                for i in range(ii, imax):              # micro-kernel order i-k-j
                    crow = cloc[i - r0]
                    io = i * n
                    for k in range(kk, kmax):
                        aik = A[io + k]                # hoisted out of j-loop
                        brow = B[k * n + jj:k * n + jmax]   # hoisted tile slice
                        j = jj
                        for b in brow:
                            crow[j] += aik * b
                            j += 1
    for i in range(r0, r1):
        C[i * n:(i + 1) * n] = array("d", cloc[i - r0])


def task(args):
    variant, r0, r1, bs = args
    if variant == 1:
        v1_naive(r0, r1)
    elif variant == 2:
        v2_ikj(r0, r1)
    else:
        v3_blocked(r0, r1, bs)
    return r1 - r0


def main():
    out = []
    P = p_phys()
    h = hw()
    roof = load_results("phase3a")
    nbytes = N * N * 8
    shms = [shared_memory.SharedMemory(create=True, size=nbytes) for _ in range(3)]
    A, B, C = (s.buf.cast("d") for s in shms)
    for idx in range(N * N):
        A[idx] = ((idx * 7) % 13) / 13.0 - 0.5
        B[idx] = ((idx * 11) % 17) / 17.0 - 0.5
    rows = [(t * N // P, (t + 1) * N // P) for t in range(P)]
    flops = 2 * N ** 3
    results = []
    try:
        with mp.Pool(P, initializer=init_gemm, initargs=(shms[0].name, shms[1].name, shms[2].name, N)) as pool:
            def bench(variant, bs=0):
                jobs = [(variant, r0, r1, bs) for r0, r1 in rows]
                st = run_benchmark(lambda: pool.map(task, jobs, chunksize=1), WARMUP, TIMED)
                g = flops / ((st.median_ms / 1e3) * 1e9)
                return st, g

            log(out, f"=== Phase 3B: GEMM N = {N}, P_phys = {P} ===")
            st, g = bench(1)
            ref1 = list(C)
            results.append({"name": "Variant 1 (naive i-j-k)", "bs": None, "ms": st.median_ms, "iqr": st.iqr_ms, "gflops": g})
            log(out, f"V1 naive ijk   : {st.median_ms:10.2f} ms  {g:.5f} GFLOP/s")
            st, g = bench(2)
            ref2 = list(C)
            err12 = max(abs(x - y) for x, y in zip(ref1, ref2))
            results.append({"name": "Variant 2 (i-k-j)", "bs": None, "ms": st.median_ms, "iqr": st.iqr_ms, "gflops": g})
            log(out, f"V2 ikj         : {st.median_ms:10.2f} ms  {g:.5f} GFLOP/s   max|V1-V2|={err12:.2e}")
            for bs in BS_LIST:
                for i in range(N * N):
                    C[i] = 0.0
                st, g = bench(3, bs)
                err = max(abs(x - y) for x, y in zip(C, ref2))
                ok = err < 1e-9
                results.append({"name": "Variant 3 (blocked)", "bs": bs, "ms": st.median_ms, "iqr": st.iqr_ms,
                                "gflops": g, "max_err": err, "ok": ok})
                log(out, f"V3 BS={bs:3d}     : {st.median_ms:10.2f} ms  {g:.5f} GFLOP/s   "
                         f"max|V3-V2|={err:.2e} {'OK' if ok else 'FAIL'}")
    finally:
        del A, B, C
        for s in shms:
            s.close()
            s.unlink()

    v3 = [r for r in results if r["bs"]]
    best = max(v3, key=lambda r: r["gflops"])
    bs_star = best["bs"]
    l1, l2 = h.get("l1d_bytes", 0), h.get("l2_bytes", 0)
    for r in v3:
        r["W_tile"] = 3 * r["bs"] ** 2 * 8
        r["fits"] = "L1d" if l1 and r["W_tile"] <= l1 else ("L2" if l2 and r["W_tile"] <= l2 else "> L2")

    BW, Ps = roof["BW"], roof["P_sust"]
    def attain(I):
        return min(Ps, I * BW)
    I_naive, I_tiled = 0.125, bs_star / 12
    mapping = [
        {"name": "Variant 1", "I": I_naive, "gflops": results[0]["gflops"]},
        {"name": "Variant 2", "I": I_naive, "gflops": results[1]["gflops"]},
        {"name": f"Variant 3 (BS*={bs_star})", "I": I_tiled, "gflops": best["gflops"]},
    ]
    for m in mapping:
        m["attainable"] = attain(m["I"])
        m["pct"] = 100 * m["gflops"] / m["attainable"]
        m["bound"] = "memory" if m["I"] < roof["ridge"] else "compute"
        log(out, f"{m['name']:22s} I={m['I']:.4f}  attainable={m['attainable']:.4f} GFLOP/s  "
                 f"achieved={m['gflops']:.5f}  ({m['pct']:.1f}%, {m['bound']}-bound region)")

    tag = emit_verify_tag("PHASE3", STUDENT_ID, results[0]["gflops"], best["gflops"])
    log(out, f"BS* = {bs_star}  W_tile = {3*bs_star**2*8} B")
    log(out, f"GFLOP/s V1={results[0]['gflops']:.4f}  V3(BS*)={best['gflops']:.4f}")
    log(out, tag)
    save_results("phase3b", {"N": N, "P": P, "results": results, "bs_star": bs_star, "mapping": mapping,
                             "verify": tag, "log": out})


if __name__ == "__main__":
    main()
