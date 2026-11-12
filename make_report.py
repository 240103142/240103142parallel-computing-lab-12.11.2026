"""Builds lab_report_<ID>.md from results/*.json (run after all phases)."""
import os

from lab_common import STUDENT_ID, HERE, load_results, md_table

SOURCES = ["harness.py", "lab_common.py", "task0_cal.py", "task1_falsesharing.py", "task2_scaling.py",
           "task3a_roofline.py", "task3b_gemm.py", "task4_scheduling.py", "run_all.py", "make_report.py"]


def kb(n):
    if not n:
        return "n/a (not reported)"
    return f"{n/(1<<20):g} MB" if n >= 1 << 20 else f"{n/1024:g} KB"


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def main():
    p0, p1, p2 = load_results("phase0"), load_results("phase1"), load_results("phase2")
    p3a, p3b, p4 = load_results("phase3a"), load_results("phase3b"), load_results("phase4")
    h = p0["hardware"]
    P, PL = int(h["P_phys"]), int(h["P_log_mp"])
    L = []
    w = L.append

    w(f"# Lab Report — Performance Evaluation and Optimization in Parallel Computing")
    w(f"\n**Student ID:** {STUDENT_ID}  \n**Platform:** {h['os']}\n")

    # ------------------------------------------------------------------ 1
    w("## 1. Hardware Baseline & Verification Hashes\n")
    hyb = f" ({h['p_cores']} P-cores + {h['e_cores']} E-cores)" if h.get("p_cores") else ""
    w(md_table(["Hardware Parameter", "Measured Value", "Hardware Parameter", "Measured Value"], [
        ["CPU Model String", h["cpu_model"], "L1 Data Cache (per core)", kb(h["l1d_bytes"])],
        ["Physical Cores (P_phys)", f"{P}{hyb}", "L2 Cache Size", kb(h["l2_bytes"])],
        ["Logical Threads (P_log)", PL, "L3 Cache Size (Shared)",
         kb(h["l3_bytes"]) if h["l3_bytes"] else "none exposed (Apple SoC uses a System Level Cache)"
         if h.get("p_cores") else kb(h["l3_bytes"])],
        ["Cache Line Size (L_bytes)", f"{h['line_bytes']} B", "Python Version & OS", h["os"]],
    ]))
    if h.get("e_l2_bytes"):
        w(f"\nE-core cluster: L1d {kb(h['e_l1d_bytes'])}, L2 {kb(h['e_l2_bytes'])} (shared per cluster). "
          f"The P-core values are listed in the table.")
    w("\nRaw OS output:\n```\n" + h["raw"] + "\n```\n")
    w("**Phase 0 timer calibration**\n```\n" + "\n".join(p0["log"][-4:]) + "\n```\n")
    w("**Verification hashes (verbatim):**\n```")
    for ph in (p0, p1, p2, p3b, p4):
        w(ph["verify"])
    w("```\n")

    # ------------------------------------------------------------------ 2
    w("## 2. Phase 1 — False Sharing & Karp-Flatt\n")
    w(f"M = {p1['M']:,} total RMW steps, p = P_phys = {P}.\n")
    w("**Table 1A — stride sensitivity**\n")
    w(md_table(["stride (words)", "separation (B)", "median (ms)", "IQR (ms)", "throughput (Mops/s)"],
               [[r["stride"], r["bytes"], f(r["median_ms"], 2), f(r["iqr_ms"], 2), f(r["mops"], 2)]
                for r in p1["table1a"]]))
    b = p1["boundary_bytes"]
    exp = h["line_bytes"]
    w(f"\nThe largest drop in execution time occurs when the slot separation reaches **{b} bytes**. "
      f"The OS reports a {exp}-byte cache line. " +
      ("This matches: once each process's slot sits on its own coherence line, the cores no longer "
       "steal the line from each other on every store." if b == exp else
       "The measured step is not exactly at the reported line size. In pure Python each iteration costs "
       "tens of nanoseconds of bytecode dispatch, which partly hides the coherence-miss latency, and on "
       "Apple Silicon the adjacent-line prefetcher / 128-byte coherence granule can blur the boundary; the "
       "trend (slow at small strides, fast once slots are line-separated) is what the data shows.") + "\n")

    w("**Table 1B — Karp-Flatt diagnostic series**\n")
    rows = []
    s1, s16 = p1["table1b"]["1"], p1["table1b"]["16"]
    for a, c in zip(s1, s16):
        rows.append([a["p"], f(a["median_ms"], 1), f(a["S"]), f(a["E"]), f(a["e"], 4),
                     f(c["median_ms"], 1), f(c["S"]), f(c["E"]), f(c["e"], 4)])
    w(md_table(["p", "T (s=1) ms", "S (s=1)", "E (s=1)", "e(p) (s=1)",
                "T (s=16) ms", "S (s=16)", "E (s=16)", "e(p) (s=16)"], rows))
    e1 = [r["e"] for r in s1 if r["e"] is not None]
    e16 = [r["e"] for r in s16 if r["e"] is not None]
    if e1 and e16:
        w(f"\n**Forensic explanation.** With stride = 1, e(p) goes from {e1[0]:.4f} to {e1[-1]:.4f} as p grows; "
          f"with stride = 16 it goes from {e16[0]:.4f} to {e16[-1]:.4f}. "
          "The worker code is identical and has no serial section, so any e(p) above ~0 is overhead that "
          "behaves like serial work. Under stride = 1 all p slots live in the same cache line(s): every store "
          "needs the line in the Modified state, so the coherence protocol invalidates the copy held by every "
          "other core and the next writer has to fetch it back across the interconnect. Those transfers are "
          "serialised on the line, and their number grows with the number of writers, so the measured serial "
          "fraction e(p) rises with p instead of staying flat. With stride = 16 (128 B) each slot owns its line, "
          "stores hit the local L1 in Modified state, and the remaining e(p) is mostly process start-up/join cost.\n")

    # ------------------------------------------------------------------ 3
    w("## 3. Phase 2 — Strong vs Weak Scaling\n")
    w(f"**Experiment 2A — strong scaling** (N = {p2['N_strong']:,})\n")
    w(md_table(["p", "T (ms)", "Stage A (ms)", "S_strong", "E_strong", "ΔS (marginal)"],
               [[r["p"], f(r["T"], 1), f(r["T_A"], 1), f(r["S"]), f(r["E"]), f(d)]
                for r, d in zip(p2["strong"], p2["dS"])]))
    w(f"\n**Experiment 2B — weak scaling** (N0 = {p2['N0']:,} per worker)\n")
    w(md_table(["p", "N(p)", "T (ms)", "Stage A (ms)", "E_weak", "α", "S_Gustafson"],
               [[r["p"], f"{r['N']:,}", f(r["T"], 1), f(r["T_A"], 1), f(r["E_weak"]), f(r["alpha"], 4), f(r["S_G"])]
                for r in p2["weak"]]))
    w(f"\n**Amdahl analysis.** Serial fraction s = T_StageA(1)/T_1 = **{p2['s']:.4f}**. "
      f"Asymptotic limit S∞ = 1/s = **{p2['S_inf']:.2f}**. Amdahl prediction at p = {PL}: "
      f"S = 1/(s + (1−s)/{PL}) = **{p2['S_amdahl_PL']:.3f}**; observed S = **{p2['S_obs_PL']:.3f}** "
      f"({100*p2['S_obs_PL']/p2['S_amdahl_PL']:.1f}% of the Amdahl bound). The gap comes from what Amdahl "
      "ignores: Pool task dispatch/result pickling, unequal core speeds and shared power/thermal budget.\n")
    infl = p2["inflection"]
    if infl.get("drop_pct") is not None:
        B = infl["boundary"]
        w(f"**Architectural inflection ({infl['kind']}).** Up to p = {B} each added worker raised speedup by "
          f"{infl['mean_dS_before']:.3f} on average; beyond p = {B} the average marginal gain is "
          f"{infl['mean_dS_after']:.3f}, a **{infl['drop_pct']:.1f}% drop** in marginal speedup per added worker "
          f"(single step p={B}→{B+1}: ΔS = {f(infl['dS_at_B1'])}). " +
          ("Workers beyond the performance cluster are scheduled on efficiency cores, which run at lower clock "
           "and narrower issue width, so each one adds only a fraction of a P-core's throughput and Stage B's "
           "equal partition makes the fast cores wait for the slow ones at pool.map's join."
           if "E-core" in infl["kind"] else
           "Extra workers share execution ports and L1/L2 with their SMT sibling, so they add little throughput.")
          + "\n")
    else:
        w(f"**Architectural inflection.** {infl['kind']} — no SMT or P/E boundary exists inside 1..{PL}.\n")
    w(f"Gustafson scaled speedup at p = {PL}: **{p2['S_G_PL']:.3f}**, versus strong-scaling "
      f"{p2['S_obs_PL']:.3f}: growing the problem with p keeps the serial share small, so weak scaling "
      "stays much closer to linear.\n")

    # ------------------------------------------------------------------ 4
    w("## 4. Phase 3 — Empirical Roofline & GEMM Sweep\n")
    w(md_table(["Quantity", "Value"], [
        ["BW_sustained (2×%d MB memcpy, %d procs)" % (p3a["buf_bytes"] // 10**6, p3a["P"]), f"{p3a['BW']:.3f} GB/s"],
        ["P_sustained (16-way unrolled FMA, %d procs)" % p3a["P"], f"{p3a['P_sust']:.4f} GFLOP/s"],
        ["Ridge point I* = P/BW", f"{p3a['ridge']:.6f} FLOP/Byte"],
    ]))
    w(f"\n**GEMM N = {p3b['N']}, {p3b['P']} worker processes**\n")
    rows = []
    for r in p3b["results"]:
        rows.append([r["name"], r["bs"] or "—", f(r["ms"], 1), f(r["gflops"], 5),
                     kb(r["W_tile"]) if r.get("W_tile") else "—", r.get("fits", "—"),
                     ("OK " + f"{r['max_err']:.1e}") if r.get("ok") else ("FAIL" if "ok" in r else "—")])
    w(md_table(["Variant", "BS", "median (ms)", "GFLOP/s", "W_tile = 3·BS²·8", "fits in", "max|C3−C2|"], rows))
    bs = p3b["bs_star"]
    W = 3 * bs * bs * 8
    w(f"\n**Cache-capacity analysis.** The best tile is **BS* = {bs}**, footprint W_tile = {W:,} B "
      f"({kb(W)}), against L1d = {kb(h['l1d_bytes'])} and L2 = {kb(h['l2_bytes'])}. "
      "Very small tiles lose because the six-level loop nest re-enters Python loops and re-creates slices "
      "very often (loop overhead dominates); very large tiles stop fitting the inner cache levels and approach "
      "Variant 2's behaviour. In CPython a float in a list is a pointer to a heap object, so the real "
      "working set is larger than 8 B/element, which pushes the optimum toward tiles that fit comfortably "
      "inside L1d/L2 rather than filling them.\n")
    w("**Roofline mapping** (attainable = min(P_sustained, I × BW))\n")
    w(md_table(["Kernel", "I (FLOP/B)", "Attainable (GFLOP/s)", "Achieved (GFLOP/s)", "% of attainable", "region"],
               [[m["name"], f(m["I"], 4), f(m["attainable"], 4), f(m["gflops"], 5), f(m["pct"], 1), m["bound"]]
                for m in p3b["mapping"]]))
    w(f"\nI_naive ≈ 0.125 FLOP/B and I_tiled(BS*) = BS*/12 = {bs/12:.3f} FLOP/B. "
      f"Because the interpreter-bound compute ceiling is so low, the ridge point I* = {p3a['ridge']:.6f} "
      "lies far to the left of every kernel, so in pure Python all three variants sit in the compute "
      "(interpreter-dispatch) bound region. The speedups from Variant 1 → 2 → 3 therefore come mainly from "
      "removing index arithmetic and strided memoryview accesses from the hot loop, with cache locality as a "
      "secondary effect. Variant 2 is mapped at the naive intensity as a conservative estimate.\n")

    # ------------------------------------------------------------------ 5
    w("## 5. Phase 4 — Scheduling & Load Imbalance\n")
    w(f"N = {p4['N']:,} items, K(i) = 1 + ⌊(i/60)²⌋, p = {p4['P']}.\n")
    w(md_table(["Strategy", "T_wall (ms)", "IQR (ms)", "max t_worker (ms)", "mean t_worker (ms)", "λ"],
               [[r["label"], f(r["T_wall"], 1), f(r["iqr"], 1), f(max(r["t_worker"]), 1),
                 f(sum(r["t_worker"]) / len(r["t_worker"]), 1), f(r["lambda"], 3)] for r in p4["rows"]]))
    rr = {r["label"]: r for r in p4["rows"]}
    c1, c1024 = rr["Dynamic C=1"], rr["Dynamic C=1024"]
    cs = p4["c_star"]
    w(f"\n**Trade-off.** Static block gives λ = {rr['Static block']['lambda']:.2f}: the last worker owns the "
      "heaviest part of the triangle while the others idle at the join. Cyclic interleaving spreads heavy and "
      f"light items evenly (λ = {rr['Static cyclic']['lambda']:.2f}). In the dynamic queue, C = 1 balances "
      f"almost perfectly (λ = {c1['lambda']:.3f}) but every item costs a lock acquire on the shared counter, "
      "whose cache line bounces between cores (T_wall = %.1f ms). C = 1024 claims only ~%d chunks in total, so "
      "synchronisation is negligible, but the last chunks are huge and the tail is unbalanced "
      "(λ = %.3f, T_wall = %.1f ms). The best chunk on this CPU is **C* = %d** (T_wall = %.1f ms, λ = %.3f), "
      "%.2f× faster than static block.\n" % (c1["T_wall"], -(-p4["N"] // 1024), c1024["lambda"], c1024["T_wall"],
                                             cs, rr[f"Dynamic C={cs}"]["T_wall"], rr[f"Dynamic C={cs}"]["lambda"],
                                             rr["Static block"]["T_wall"] / rr[f"Dynamic C={cs}"]["T_wall"]))

    # ------------------------------------------------------------------ logs + code
    w("## Appendix A — Raw terminal telemetry\n")
    for name, ph in (("Phase 0", p0), ("Phase 1", p1), ("Phase 2", p2), ("Phase 3A", p3a),
                     ("Phase 3B", p3b), ("Phase 4", p4)):
        w(f"### {name}\n```\n" + "\n".join(ph["log"]) + "\n```\n")
    w("## Appendix B — Python 3 source code\n")
    for src in SOURCES:
        path = os.path.join(HERE, src)
        if os.path.exists(path):
            with open(path) as fh:
                w(f"### {src}\n```python\n{fh.read()}\n```\n")

    out = os.path.join(HERE, f"lab_report_{STUDENT_ID}.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print(f"Report written: {out}")


if __name__ == "__main__":
    main()
