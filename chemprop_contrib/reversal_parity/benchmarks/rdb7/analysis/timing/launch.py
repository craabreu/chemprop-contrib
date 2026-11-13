"""Run NPROC copies of a timing job at once, as run3.sh does, optionally capping CPU threads.

Usage: python launch.py MODE KIND NPROC THREADS EPOCHS
  MODE     bench   -> /home/charlles/data/rdb7/bench2.py KIND fold strict EPOCHS <scratch outdir>
           profile -> profile_steps.py KIND EPOCHS
  THREADS  0 keeps the default; N sets OMP_NUM_THREADS=N and MKL_NUM_THREADS=N
"""
import json
import os
import pathlib
import subprocess
import sys
import time

mode, kind, nproc, threads, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
here = pathlib.Path(__file__).parent
env = dict(os.environ)
if threads:
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(threads)
tag = f"{mode}-{kind}-n{nproc}-t{threads}"
procs = []
t0 = time.time()
for fold in range(nproc):
    if mode == "bench":
        out = here / "runs" / tag / f"f{fold}"
        cmd = [sys.executable, "/home/charlles/data/rdb7/bench2.py", kind, str(fold), "strict", epochs, str(out)]
    else:
        cmd = [sys.executable, str(here / "profile_steps.py"), kind, epochs]
    procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  cwd="/tmp"))
outs = [p.communicate()[0] for p in procs]
print(f"== {tag}: {nproc} at once, {time.time() - t0:.0f} s in all")
if mode == "bench":
    times = [json.loads((here / "runs" / tag / f"f{f}" / "result.json").read_text())["train_time_s"]
             for f in range(nproc)]
    print(f"   train_time_s per process: {times}  -> {sum(times) / len(times) / int(epochs):.2f} s/epoch")
else:
    for o in outs[:1]:  # one process's breakdown is representative
        print("\n".join(o.strip().splitlines()[-2 * int(epochs) - 1:]))
    walls = [[float(l.split("wall")[1].split("s")[0]) for l in o.splitlines() if l.startswith("sync=")] for o in outs]
    print("   wall per epoch, each process:", walls)
