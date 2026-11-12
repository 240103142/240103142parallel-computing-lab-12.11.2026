"""Runs every phase in order, then builds the report.
Usage:  python3 run_all.py            (full lab)
        python3 run_all.py 2          (start again from phase 2)
"""
import subprocess
import sys
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [("0", "task0_cal.py"), ("1", "task1_falsesharing.py"), ("2", "task2_scaling.py"),
         ("3", "task3a_roofline.py"), ("3", "task3b_gemm.py"), ("4", "task4_scheduling.py")]

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "0"
    for ph, script in STEPS:
        if ph < start:
            continue
        print(f"\n######## {script} ########", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, script)])
        if r.returncode:
            sys.exit(f"{script} failed (exit {r.returncode})")
        print(f"-- {script} done in {time.time() - t0:.0f} s", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "make_report.py")], check=True)
