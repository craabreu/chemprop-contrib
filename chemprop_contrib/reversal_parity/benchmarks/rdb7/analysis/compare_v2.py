"""Compare the v2 rerun of the parity models with the earlier runs (and the stock baselines)."""
import json
import pathlib

import numpy as np

root = pathlib.Path("/home/charlles/data/rdb7")
KEYS = ["all_Ea_MAE", "fwd_Ea_MAE", "rev_Ea_MAE", "all_Ea_RMSE", "all_dE_MAE", "antisym_dE_mean", "consistency_MAE"]


def load(tag, model):
    rs = []
    for f in range(5):
        p = root / tag / f"{model}-f{f}" / "result.json"
        if p.exists():
            rs.append(json.loads(p.read_text()))
    return rs


for split in ("strict", "paper"):
    print(f"\n== {split} split (mean ± std over folds; kcal/mol) ==")
    print(f"{'model':24s} {'n':>2s} " + " ".join(f"{k:>16s}" for k in KEYS) + f" {'epoch':>6s} {'time_s':>7s}")
    for model, tags in (("stock", [f"jpca_{split}"]), ("stock_aug", [f"jpca_{split}"]),
                        ("parity_paired", [f"jpca_{split}", f"jpca_{split}_v2"])):
        for tag in tags:
            rs = load(tag, model)
            if not rs:
                continue
            cells = []
            for k in KEYS:
                v = np.array([r[k] for r in rs])
                cells.append(f"{v.mean():9.2e}" if k in ("antisym_dE_mean", "consistency_MAE")
                             else f"{v.mean():7.2f} ± {v.std():5.2f}")
            label = model + (" (v2)" if tag.endswith("_v2") else "")
            ep = np.mean([r["best_epoch"] for r in rs])
            t = np.mean([r["train_time_s"] for r in rs])
            print(f"{label:24s} {len(rs):2d} " + " ".join(f"{c:>16s}" for c in cells) + f" {ep:6.1f} {t:7.0f}")
    # per-fold differences, old vs v2
    for model in ("parity_paired",):
        old, new = load(f"jpca_{split}", model), load(f"jpca_{split}_v2", model)
        if len(old) == len(new) == 5:
            d = [n["all_Ea_MAE"] - o["all_Ea_MAE"] for o, n in zip(old, new)]
            print(f"  {model}: v2 - old all_Ea_MAE per fold " + " ".join(f"{x:+.2f}" for x in d)
                  + f"   best epochs old {[o['best_epoch'] for o in old]} v2 {[n['best_epoch'] for n in new]}")
