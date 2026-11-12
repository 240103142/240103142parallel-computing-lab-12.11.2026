"""Phase 3A - empirical machine roofline: DRAM bandwidth + Python FMA ceiling."""
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag
from lab_common import STUDENT_ID, save_results, scale, p_phys, attach_shm, log

BUF_BYTES = scale(96 * 10**6, 8 * 10**6)
FLOPS_PER_WORKER = scale(80_000_000, 4_000_000)
WARMUP, TIMED = scale(2, 1), scale(7, 3)

_g = {}


def init_bw(name_a, name_b):
    _g["a"] = attach_shm(name_a)
    _g["b"] = attach_shm(name_b)
    _g["mva"] = _g["a"].buf
    _g["mvb"] = _g["b"].buf


def copy_slice(rng):
    s, e = rng
    _g["mva"][s:e] = _g["mvb"][s:e]          # native memcpy: 1 read + 1 write per byte
    return e - s


def fma_worker(iters):
    a, b = 0.999999, 1e-6
    x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = 0.1
    x8 = x9 = x10 = x11 = x12 = x13 = x14 = x15 = 0.2
    for _ in range(iters):                    # 16 FMAs x 2 FLOPs = 32 FLOPs / iteration
        x0 = x0 * a + b; x1 = x1 * a + b; x2 = x2 * a + b; x3 = x3 * a + b
        x4 = x4 * a + b; x5 = x5 * a + b; x6 = x6 * a + b; x7 = x7 * a + b
        x8 = x8 * a + b; x9 = x9 * a + b; x10 = x10 * a + b; x11 = x11 * a + b
        x12 = x12 * a + b; x13 = x13 * a + b; x14 = x14 * a + b; x15 = x15 * a + b
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15


def main():
    out = []
    P = p_phys()
    A = shared_memory.SharedMemory(create=True, size=BUF_BYTES)
    B = shared_memory.SharedMemory(create=True, size=BUF_BYTES)
    try:
        B.buf[:BUF_BYTES] = bytes(range(256)) * (BUF_BYTES // 256) + bytes(BUF_BYTES % 256)
        ranges = [(t * BUF_BYTES // P, (t + 1) * BUF_BYTES // P) for t in range(P)]
        with mp.Pool(P, initializer=init_bw, initargs=(A.name, B.name)) as pool:
            pool.map(copy_slice, ranges)      # touch pages once
            st = run_benchmark(lambda: pool.map(copy_slice, ranges, chunksize=1), WARMUP, TIMED)
        assert bytes(A.buf[:4096]) == bytes(B.buf[:4096])
    finally:
        for s in (A, B):
            s.close()
            s.unlink()
    bw = 2 * BUF_BYTES / ((st.median_ms / 1e3) * 1e9)
    log(out, f"=== Phase 3A: roofline (P_phys = {P}) ===")
    log(out, f"Copy {BUF_BYTES/1e6:.0f} MB (x2 buffers): median {st.median_ms:.3f} ms, "
             f"IQR {st.iqr_ms:.3f} ms  => BW_sustained = {bw:.3f} GB/s")

    iters = FLOPS_PER_WORKER // 32
    with mp.Pool(P) as pool:
        pool.map(fma_worker, [1000] * P)
        st2 = run_benchmark(lambda: pool.map(fma_worker, [iters] * P, chunksize=1), scale(1, 0), scale(5, 3))
    gflops = P * iters * 32 / ((st2.median_ms / 1e3) * 1e9)
    log(out, f"FMA loop {P} x {iters*32:.3g} FLOPs: median {st2.median_ms:.2f} ms  "
             f"=> P_sustained = {gflops:.4f} GFLOP/s")
    ridge = gflops / bw
    log(out, f"Ridge point I* = P/BW = {ridge:.6f} FLOP/Byte")
    save_results("phase3a", {"P": P, "buf_bytes": BUF_BYTES, "bw_ms": st.median_ms, "bw_iqr": st.iqr_ms,
                             "BW": bw, "fma_ms": st2.median_ms, "P_sust": gflops, "ridge": ridge, "log": out})


if __name__ == "__main__":
    main()
