# Performance Evaluation and Optimization in Parallel Computing — Python Lab

Pure standard-library Python 3 (no pip packages). Covers false sharing, strong/weak scaling,
an empirical roofline with GEMM tiling, and skewed-workload scheduling.

## Setup
1. In `lab_common.py`, set `STUDENT_ID` to your own ID.
2. Plug in the laptop and close other applications for stable measurements.
3. Disconnect from the network, as required by the lab protocol.

## Run (macOS Terminal)
```bash
cd parallel_lab
caffeinate -i python3 run_all.py   # all phases + report, about 15–30 min
```
`caffeinate` keeps the Mac awake. To restart from a given phase: `python3 run_all.py 2`.

Phases can also be run one by one: `python3 task0_cal.py`, `python3 task1_falsesharing.py`,
`python3 task2_scaling.py`, `python3 task3a_roofline.py`, `python3 task3b_gemm.py`,
`python3 task4_scheduling.py`, then `python3 make_report.py`.

## Output
* `lab_report_<ID>.md` — the submission file: hardware table, all five VERIFY_HASH lines,
  every table, analysis, raw terminal logs and the full source code appendix.
* `results/*.json` — raw measurements for each phase.

## Files
| File | Purpose |
|---|---|
| `harness.py` | Timing, statistics and VERIFY_HASH generator (as given in the lab sheet) |
| `lab_common.py` | Student ID, shared-memory helpers, result storage |
| `task0_cal.py` | Phase 0 — hardware audit and timer calibration |
| `task1_falsesharing.py` | Phase 1 — false sharing stride sweep and Karp-Flatt metric |
| `task2_scaling.py` | Phase 2 — strong (Amdahl) vs weak (Gustafson) scaling |
| `task3a_roofline.py` | Phase 3A — DRAM bandwidth and compute ceiling |
| `task3b_gemm.py` | Phase 3B — naive / reordered / blocked GEMM and tile-size sweep |
| `task4_scheduling.py` | Phase 4 — static block, cyclic and dynamic chunk scheduling |
| `run_all.py` | Runs all phases in order |
| `make_report.py` | Builds the Markdown report |

## Notes
* Apple Silicon: 128-byte coherence granule, no SMT (P_log = P_phys), but a performance/efficiency
  core boundary, which Phase 2 detects and analyses.
* Apple chips expose no L3 (they use a System Level Cache); the report states this.
* Do not set `LAB_QUICK=1` for real runs — it shrinks problem sizes for smoke testing only.
