"""Tables for the (m, h) rerun and the learning curve (read-only on /home/charlles/data/rdb7).

Usage: python analyze.py rerun | lcurve
Metrics pool the forward and reverse test reactions, as bench2.py's all_* keys do. "projected" is the
post hoc symmetrization of stock_aug: dE <- (dE(x) - dE(xbar)) / 2, then the nearest point with
Ea_f - Ea_r = dE.
"""
import json
import pathlib
import sys

import numpy as np

root = pathlib.Path("/home/charlles/data/rdb7")
COLS = ["Ea MAE", "Ea RMSE", "dE MAE", "dE RMSE"]


def metrics(ea_f, ea_r, de_f, de_r, z):
    e_ea = np.concatenate((ea_f - z["Yf"][:, 0], ea_r - z["Yr"][:, 0]))
    e_de = np.concatenate((de_f - z["Yf"][:, 1], de_r - z["Yr"][:, 1]))
    return [np.abs(e_ea).mean(), np.sqrt((e_ea**2).mean()), np.abs(e_de).mean(), np.sqrt((e_de**2).mean())]


def load(d):
    """Raw and projected metrics, plus result.json, for one run directory (None if unfinished)."""
    if not (d / "result.json").exists():
        return None
    z = np.load(d / "preds.npz")
    Pf, Pr = z["Pf"], z["Pr"]
    raw = metrics(Pf[:, 0], Pr[:, 0], Pf[:, 1], Pr[:, 1], z)
    de = 0.5 * (Pf[:, 1] - Pr[:, 1])
    r = (Pf[:, 0] - Pr[:, 0] - de) / 3
    proj = metrics(Pf[:, 0] - r, Pr[:, 0] + r, de + r, -(de + r), z)
    res = json.loads((d / "result.json").read_text())
    assert abs(raw[0] - res["all_Ea_MAE"]) < 1e-9 and abs(raw[2] - res["all_dE_MAE"]) < 1e-9
    return dict(raw=raw, proj=proj, res=res)


def runs(base, model):
    return [load(base / f"{model}-f{f}") for f in range(5)]


def row(label, rs, key, extra=True):
    rs = [r for r in rs if r is not None]
    if not rs:
        return f"{label:30s} (no results)"
    v = np.array([r[key] for r in rs])
    s = f"{label:30s} {len(rs)} " + " ".join(f"{m:6.2f} ± {sd:4.2f}" for m, sd in zip(v.mean(0), v.std(0)))
    if extra:
        res = [r["res"] for r in rs]
        s += (f"  {np.mean([x['antisym_dE_mean'] for x in res]):8.1e} {np.mean([x['consistency_MAE'] for x in res]):8.1e}"
              f"  {np.mean([x['best_epoch'] for x in res]):6.1f} {np.mean([x['train_time_s'] for x in res]):6.0f}")
    return s


def diffs(label, a, b, ka="raw", kb="raw"):
    if any(x is None for x in a + b):
        return f"  {label}: incomplete"
    out = []
    for j, name in ((0, "Ea MAE"), (2, "dE MAE")):
        d = np.array([x[ka][j] - y[kb][j] for x, y in zip(a, b)])
        out.append(f"{name} " + " ".join(f"{x:+.2f}" for x in d) + f" (mean {d.mean():+.2f})")
    return f"  {label}: " + "; ".join(out)


HEAD = f"{'':30s} n " + " ".join(f"{c:>13s}" for c in COLS) + "   antisym  consist   epoch time_s"

if sys.argv[1] == "rerun":
    for split in ("strict", "paper"):
        print(f"\n== {split} (mean ± std over folds, kcal/mol)\n{HEAD}")
        stock, aug = runs(root / f"jpca_{split}", "stock"), runs(root / f"jpca_{split}", "stock_aug")
        v2, v3 = runs(root / f"jpca_{split}_v2", "parity_paired"), runs(root / f"jpca_{split}_v3", "parity_paired")
        print(row("stock", stock, "raw"))
        print(row("stock_aug", aug, "raw"))
        print(row("stock_aug, projected", aug, "proj", extra=False))
        print(row("parity_paired (m, d) [v2]", v2, "raw"))
        print(row("parity_paired (m, h) [v3]", v3, "raw"))
        print(diffs("per fold, (m, h) - (m, d)", v3, v2))
        print(diffs("per fold, (m, h) - stock_aug", v3, aug))
        print(diffs("per fold, (m, h) - projected stock_aug", v3, aug, kb="proj"))
        print("  best epochs (m, h):", [r["res"]["best_epoch"] for r in v3 if r], " (m, d):",
              [r["res"]["best_epoch"] for r in v2 if r])
else:
    lc = root / "lcurve_strict_md"
    for frac in ("0.1", "0.25", "0.5", "1"):
        if frac == "1":  # full data: the (m, d) rerun at 42887e4 (v2) and the saved stock_aug runs
            aug, par = runs(root / "jpca_strict", "stock_aug"), runs(root / "jpca_strict_v2", "parity_paired")
        else:
            aug, par = runs(lc / f"frac{frac}", "stock_aug"), runs(lc / f"frac{frac}", "parity_paired")
        n = [r["res"].get("n_train_reactions") for r in par if r]
        print(f"\n== {float(frac):.0%} of the strict training set (reactions per fold: {n})\n{HEAD}")
        print(row("stock_aug", aug, "raw"))
        print(row("stock_aug, projected", aug, "proj", extra=False))
        print(row("parity_paired", par, "raw"))
        print(diffs("per fold, parity_paired - stock_aug", par, aug))
        print(diffs("per fold, parity_paired - projected stock_aug", par, aug, kb="proj"))
        print("  best epochs parity_paired:", [r["res"]["best_epoch"] for r in par if r], " stock_aug:",
              [r["res"]["best_epoch"] for r in aug if r])
