"""Queue for the (m, h) rerun and the learning curve; one batch of five folds on the GPU at a time.

Usage: python driver.py rerun     parity_paired, 100 epochs, strict then paper, via the user's run3.sh
                                  -> /home/charlles/data/rdb7/jpca_{strict,paper}_v3
       python driver.py lcurve    waits for the rerun, then bench_lc.py for 10%, 25%, 50% of the strict
                                  training set, parity_paired and stock_aug, OMP_NUM_THREADS=1 for both
                                  -> /home/charlles/data/rdb7/lcurve_strict/frac<F>/<model>-f<fold>
"""
import os
import pathlib
import shutil
import subprocess
import sys
import time

here = pathlib.Path(__file__).parent
rdb7 = pathlib.Path("/home/charlles/data/rdb7")
marker = here / "rerun.done"
log = open(here / f"driver-{sys.argv[1]}.log", "a", buffering=1)


def say(msg):
    log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


if sys.argv[1] == "rerun":
    for split in ("strict", "paper"):
        tag = f"jpca_{split}_v3"
        assert not (rdb7 / tag).exists(), tag
        t0 = time.time()
        rc = subprocess.run(["bash", str(rdb7 / "run3.sh"), "parity_paired", "100", split, tag, "0", "1", "2", "3", "4"],
                            cwd="/tmp").returncode
        done = sum((rdb7 / tag / f"parity_paired-f{f}" / "result.json").exists() for f in range(5))
        say(f"{tag}: rc {rc}, {done}/5 results, {time.time() - t0:.0f} s")
    marker.write_text("done\n")
else:
    while not marker.exists():
        time.sleep(20)
    out = rdb7 / "lcurve_strict_md"  # (m, d) basis, commit 42887e4; lcurve_strict is a stopped (m, h)-era start
    assert not out.exists(), out
    out.mkdir(exist_ok=True)
    for name in ("bench_lc.py", "make_bench_lc.py"):
        shutil.copy(here / name, out / name)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # stock_aug first: it does not use ParityCGRFeaturizer, so it cannot depend on the (m, d)/(m, h) choice
    for model in ("stock_aug", "parity_paired"):
        for frac in ("0.1", "0.25", "0.5"):
            d = out / f"frac{frac}"
            (d / "logs").mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            procs = []
            for fold in range(5):
                run = d / f"{model}-f{fold}"
                if (run / "result.json").exists():
                    continue
                cmd = [sys.executable, str(out / "bench_lc.py"), model, str(fold), "strict", "100", frac, str(run)]
                procs.append(subprocess.Popen(cmd, env=env, cwd="/tmp", stderr=subprocess.STDOUT,
                                              stdout=open(d / "logs" / f"{model}-f{fold}.log", "w")))
            rcs = [p.wait() for p in procs]
            done = sum((d / f"{model}-f{f}" / "result.json").exists() for f in range(5))
            say(f"frac {frac} {model}: rcs {rcs}, {done}/5 results, {time.time() - t0:.0f} s")
say("finished")
