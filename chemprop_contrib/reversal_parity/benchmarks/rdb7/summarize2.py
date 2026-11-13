"""Summarize bench2 results: mean ± sd over folds. Usage: python summarize2.py TAG"""
import json
import pathlib
import sys

import numpy as np

root = pathlib.Path(__file__).parent / sys.argv[1]
res = [json.loads(p.read_text()) for p in sorted(root.glob("*/result.json"))]
keys = ["all_Ea_MAE", "all_Ea_RMSE", "fwd_Ea_MAE", "rev_Ea_MAE", "all_dE_MAE", "all_dE_RMSE",
        "antisym_dE_mean", "consistency_MAE", "best_epoch", "train_time_s"]
models = ["stock", "stock_aug", "parity", "parity_marcus", "parity_paired"]
print(f"## {sys.argv[1]}\n\n| metric (kcal/mol) | " + " | ".join(models) + " |")
print("|---|" + "---|" * len(models))
for k in keys:
    cells = []
    for m in models:
        v = np.array([r[k] for r in res if r["model"] == m], dtype=float)
        if not len(v):
            cells.append("-")
        elif abs(v.mean()) < 1e-3:
            cells.append(f"{v.mean():.0e}")
        else:
            cells.append(f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}")
    print(f"| {k} | " + " | ".join(cells) + " |")
print("\nfolds:", {m: sorted(r["fold"] for r in res if r["model"] == m) for m in models})
