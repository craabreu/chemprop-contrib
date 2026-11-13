"""Post hoc symmetrization of the saved stock_aug predictions (read-only on /home/charlles/data/rdb7).

Step 1 (Reynolds): dE <- (dE(x) - dE(xbar)) / 2, exactly odd.
Step 2 (projection): move (Ea_f, Ea_r, dE) to the nearest point with Ea_f - Ea_r = dE.
Both are orthogonal projections onto linear subspaces that contain the truth, so the summed squared
error over a reaction and its reverse cannot increase.
"""
import json
import pathlib

import numpy as np

root = pathlib.Path("/home/charlles/data/rdb7")


def metrics(ea_f, ea_r, de, y):
    e_ea = np.concatenate((ea_f - y["ea_f"], ea_r - y["ea_r"]))
    e_de = de - y["de"]
    return [np.abs(e_ea).mean(), np.sqrt((e_ea**2).mean()), np.abs(e_de).mean(), np.sqrt((e_de**2).mean())]


for split in ("strict", "paper"):
    rows = {"stock_aug": [], "stock_aug, dE symmetrized": [], "stock_aug, projected": [], "parity_paired": []}
    check = []
    for fold in range(5):
        for model in ("stock_aug", "parity_paired"):
            tag = f"jpca_{split}_v2" if model == "parity_paired" else f"jpca_{split}"  # v2: rerun
            d = root / tag / f"{model}-f{fold}"
            z = np.load(d / "preds.npz")
            if fold == 0 and split == "strict" and model == "stock_aug":
                print("keys:", {k: z[k].shape for k in z.files})
            Pf, Pr, Yf, Yr = z["Pf"], z["Pr"], z["Yf"], z["Yr"]
            y = {"ea_f": Yf[:, 0], "ea_r": Yr[:, 0], "de": Yf[:, 1]}
            ea_f, ea_r = Pf[:, 0], Pr[:, 0]
            raw = metrics(ea_f, ea_r, Pf[:, 1], y)
            rows[model].append(raw)
            res = json.loads((d / "result.json").read_text())
            check.append(abs(raw[0] - res["all_Ea_MAE"]))
            if model == "stock_aug":
                de = 0.5 * (Pf[:, 1] - Pr[:, 1])
                rows["stock_aug, dE symmetrized"].append(metrics(ea_f, ea_r, de, y))
                r = (ea_f - ea_r - de) / 3
                rows["stock_aug, projected"].append(metrics(ea_f - r, ea_r + r, de + r, y))
    print(f"\n== {split} (mean ± std over 5 folds, kcal/mol); max |my Ea MAE - result.json| = {max(check):.1e}")
    print(f"{'':28s} {'Ea MAE':>13s} {'Ea RMSE':>13s} {'dE MAE':>13s} {'dE RMSE':>13s}")
    for name, v in rows.items():
        v = np.array(v)
        print(f"{name:28s} " + " ".join(f"{m:6.2f} ± {s:4.2f}" for m, s in zip(v.mean(0), v.std(0))))
    a, p = np.array(rows["stock_aug, projected"]), np.array(rows["parity_paired"])
    print("per-fold Ea MAE, parity_paired - projected stock_aug:", " ".join(f"{x:+.2f}" for x in p[:, 0] - a[:, 0]))
