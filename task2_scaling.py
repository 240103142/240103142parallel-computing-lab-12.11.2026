"""Phase 2 - strong (Amdahl) vs weak (Gustafson) scaling."""
import math
import multiprocessing as mp
import statistics
import time

from harness import run_benchmark, emit_verify_tag
from lab_common import STUDENT_ID, save_results, scale, hw, p_phys, p_log, log

N_STRONG = scale(1_600_000, 80_000)
N0_WEAK = scale(250_000, 12_500)
WARMUP, TIMED = scale(1, 0), scale(5, 3)
SERIAL_FRAC = 0.05


def stage_a(n_serial: int) -> float:
    """Strictly serial prefix reduction: 12 transcendental ops per element."""
    acc, kept = 0.0, 0
    sin, sqrt = math.sin, math.sqrt
    for i in range(n_serial):
        v = 1.0 + i * 1e-5
        v = sin(v); v = sqrt(v * v + 1.0); v = sin(v); v = sqrt(v * v + 1.0)
        v = sin(v); v = sqrt(v * v + 1.0); v = sin(v); v = sqrt(v * v + 1.0)
        v = sin(v); v = sqrt(v * v + 1.0); v = sin(v); v = sqrt(v * v + 1.0)
        acc += v
        if acc > 0.0:
            kept += 1
    return acc + kept


def stage_b(rng) -> float:
    """Embarrassingly parallel kernel: 35 FMA + trig iterations per element, no shared writes."""
    start, end = rng
    sin = math.sin
    total = 0.0
    for i in range(start, end):
        x = 1e-6 * (i + 1)
        for k in range(35):
            x = x * 0.9999 + 1e-4 * sin(x + k)
        total += x
    return total


def split(n, p):
    return [(t * n // p, (t + 1) * n // p) for t in range(p)]


def trial_factory(pool, n, p, stage_a_times):
    ranges = split(n, p)
    n_serial = int(SERIAL_FRAC * n)

    def trial():
        t0 = time.perf_counter_ns()
        stage_a(n_serial)
        t1 = time.perf_counter_ns()
        stage_a_times.append((t1 - t0) / 1e6)
        pool.map(stage_b, ranges, chunksize=1)
    return trial


def measure(p, n):
    sa = []
    with mp.Pool(processes=p) as pool:          # created outside the timed region
        pool.map(stage_b, split(p * 2, p))     # make sure every worker is alive
        st = run_benchmark(trial_factory(pool, n, p, sa), WARMUP, TIMED)
    return st, statistics.median(sa[WARMUP:])


def main():
    out = []
    P, PL = p_phys(), p_log()
    h = hw()

    log(out, f"=== Phase 2A: strong scaling, N = {N_STRONG} ===")
    strong = []
    for p in range(1, PL + 1):
        st, ta = measure(p, N_STRONG)
        T1 = strong[0]["T"] if strong else st.median_ms
        S = T1 / st.median_ms
        strong.append({"p": p, "T": st.median_ms, "iqr": st.iqr_ms, "T_A": ta, "S": S, "E": S / p})
        log(out, f"p={p:2d}  T={st.median_ms:10.2f} ms  (StageA {ta:8.2f} ms)  S={S:6.3f}  E={S/p:6.3f}")

    log(out, f"=== Phase 2B: weak scaling, N0 = {N0_WEAK} per worker ===")
    weak = []
    for p in range(1, PL + 1):
        st, ta = measure(p, p * N0_WEAK)
        T1 = weak[0]["T"] if weak else st.median_ms
        alpha = ta / st.median_ms
        SG = p - alpha * (p - 1)
        weak.append({"p": p, "N": p * N0_WEAK, "T": st.median_ms, "iqr": st.iqr_ms, "T_A": ta,
                     "E_weak": T1 / st.median_ms, "alpha": alpha, "S_G": SG})
        log(out, f"p={p:2d}  N={p*N0_WEAK:9d}  T={st.median_ms:10.2f} ms  E_weak={T1/st.median_ms:6.3f}  "
                 f"alpha={alpha:.4f}  S_G={SG:6.3f}")

    # Amdahl analysis
    s = strong[0]["T_A"] / strong[0]["T"]
    S_inf = 1 / s
    S_amdahl_PL = 1 / (s + (1 - s) / PL)
    S_obs_PL = strong[-1]["S"]
    log(out, f"serial fraction s = {s:.4f}  S_inf = {S_inf:.3f}  "
             f"Amdahl S({PL}) = {S_amdahl_PL:.3f}  observed S({PL}) = {S_obs_PL:.3f}")

    # Inflection: P-core -> E-core cluster on Apple Silicon, or P_phys -> SMT siblings
    if h.get("p_cores") and h.get("e_cores"):
        B, kind = int(h["p_cores"]), "P-core -> E-core cluster"
    elif PL > P:
        B, kind = P, "physical -> SMT/Hyper-Threading"
    else:
        B, kind = None, "none (P_log == P_phys, homogeneous)"
    dS = [None] + [strong[i]["S"] - strong[i - 1]["S"] for i in range(1, len(strong))]
    infl = {"boundary": B, "kind": kind}
    if B and B < PL:
        before = [d for i, d in enumerate(dS) if d is not None and strong[i]["p"] <= B]
        after = [d for i, d in enumerate(dS) if d is not None and strong[i]["p"] > B]
        mb, ma = statistics.mean(before), statistics.mean(after)
        infl.update({"mean_dS_before": mb, "mean_dS_after": ma,
                     "drop_pct": (1 - ma / mb) * 100 if mb else 0.0,
                     "dS_at_B": dS[B - 1] if B >= 2 else None, "dS_at_B1": dS[B]})
        log(out, f"Inflection at p={B}->{B+1} ({kind}): mean marginal dS/worker "
                 f"{mb:.3f} -> {ma:.3f}  (drop {infl['drop_pct']:.1f}%)")
    else:
        log(out, f"No architectural boundary inside 1..{PL}: {kind}")

    S_G_PL = weak[-1]["S_G"]
    tag = emit_verify_tag("PHASE2", STUDENT_ID, S_obs_PL, S_G_PL)
    log(out, f"S_strong(P_log)={S_obs_PL:.4f}  S_Gustafson(P_log)={S_G_PL:.4f}")
    log(out, tag)
    save_results("phase2", {"N_strong": N_STRONG, "N0": N0_WEAK, "strong": strong, "weak": weak,
                            "dS": dS, "s": s, "S_inf": S_inf, "S_amdahl_PL": S_amdahl_PL,
                            "S_obs_PL": S_obs_PL, "S_G_PL": S_G_PL, "inflection": infl,
                            "verify": tag, "log": out})


if __name__ == "__main__":
    main()
