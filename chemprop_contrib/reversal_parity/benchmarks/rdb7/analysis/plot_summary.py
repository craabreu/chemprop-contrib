"""Summary figures of the RDB7 benchmark, from the results committed next to this folder.

  figures/accuracy.png        barrier and dE MAE per model on both splits (full training set)
  figures/consistency.png     antisymmetry violation and forward/reverse pair consistency, log scale
  figures/learning_curve.png  MAE against training-set size, and per-fold differences of parity_paired
  figures/parity_{strict,paper}.png   predicted against reference barrier and dE, and each model's
                              Ea_fwd - Ea_rev against its own dE, for stock, stock_aug and parity_paired

Errors pool both directions of every test reaction. "stock_aug, symmetrized" is stock_aug corrected after
training: dE <- (dE(x) - dE(xbar)) / 2, then the nearest point with Ea_fwd - Ea_rev = dE.
Needs numpy and matplotlib (the package's test environment lacks matplotlib; any Python with both works).
Usage: python plot_summary.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = pathlib.Path(__file__).resolve().parent.parent
out = root / "figures"
out.mkdir(exist_ok=True)

# entity colors, fixed across figures (reference palette slots 1-3; the stock baseline in neutral gray)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COLOR = {"parity_paired": "#2a78d6", "stock_aug": "#eb6834", "stock_aug, symmetrized": "#1baf7a", "stock": MUTED}
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False,
    "legend.labelcolor": INK2,
})


def metrics(d):
    """Pooled-direction metrics of one run directory, raw and symmetrized."""
    z = np.load(d / "preds.npz")
    Pf, Pr, Yf, Yr = z["Pf"], z["Pr"], z["Yf"], z["Yr"]

    def score(ea_f, ea_r, de_f, de_r):
        e_ea = np.concatenate([ea_f - Yf[:, 0], ea_r - Yr[:, 0]])
        e_de = np.concatenate([de_f - Yf[:, 1], de_r - Yr[:, 1]])
        return {"Ea": np.abs(e_ea).mean(), "dE": np.abs(e_de).mean(),
                "antisym": np.abs(de_f + de_r).mean(), "consistency": np.abs(ea_f - ea_r - de_f).mean()}

    raw = score(Pf[:, 0], Pr[:, 0], Pf[:, 1], Pr[:, 1])
    de = 0.5 * (Pf[:, 1] - Pr[:, 1])
    r = (Pf[:, 0] - Pr[:, 0] - de) / 3
    sym = score(Pf[:, 0] - r, Pr[:, 0] + r, de + r, -(de + r))
    return raw, sym


def runs(tag, model):
    """[(raw, symmetrized)] over folds 0-4."""
    return [metrics(root / tag / f"{model}-f{f}") for f in range(5)]


def collect(tags):
    """{label: {metric: array over folds}} for stock, stock_aug, its symmetrized form and parity_paired."""
    stock_tag, aug_tag, par_tag = tags
    res = {}
    for label, tag, model, which in (("stock", stock_tag, "stock", 0), ("stock_aug", aug_tag, "stock_aug", 0),
                                     ("stock_aug, symmetrized", aug_tag, "stock_aug", 1),
                                     ("parity_paired", par_tag, "parity_paired", 0)):
        rs = [r[which] for r in runs(tag, model)]
        res[label] = {k: np.array([r[k] for r in rs]) for k in rs[0]}
    return res


full = {split: collect((f"jpca_{split}", f"jpca_{split}", f"jpca_{split}_v2")) for split in ("strict", "paper")}

# 1. accuracy ------------------------------------------------------------------------------------------------
SHOWN = ["stock_aug", "stock_aug, symmetrized", "parity_paired"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for ax, (metric, title) in zip(axes, (("Ea", "Barrier MAE (kcal/mol)"), ("dE", "Reaction energy MAE (kcal/mol)"))):
    for j, split in enumerate(("strict", "paper")):
        for k, label in enumerate(SHOWN):
            v = full[split][label][metric]
            y = j + (k - 1) * 0.22
            ax.errorbar(v.mean(), y, xerr=v.std(), fmt="o", ms=7, color=COLOR[label], mec=SURFACE, mew=1.5,
                        ecolor=COLOR[label], elinewidth=2, capsize=0, label=label if j == 0 else None, zorder=3)
            ax.text(v.mean() + v.std() + 0.06, y, f"{v.mean():.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks([0, 1], ["strict split", "paper split"])
    ax.set_ylim(1.55, -0.55)
    ax.set_title(title)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
s = {split: full[split]["stock"] for split in full}
axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle("Accuracy on RDB7 (5 folds, mean ± std; both directions of every test reaction scored)",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.text(0.01, -0.04, f"Off scale: stock (forward reactions only) has barrier MAE {s['strict']['Ea'].mean():.1f} "
         f"(strict) / {s['paper']['Ea'].mean():.1f} (paper) and reaction energy MAE {s['strict']['dE'].mean():.1f} / "
         f"{s['paper']['dE'].mean():.1f}, because it never learns the reverse direction.", color=INK2, fontsize=9)
fig.savefig(out / "accuracy.png", dpi=160, bbox_inches="tight")

# 2. consistency ---------------------------------------------------------------------------------------------
MODELS = ["stock", "stock_aug", "parity_paired"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
titles = {"antisym": "Antisymmetry violation  $\\langle|\\Delta E(x) + \\Delta E(\\bar x)|\\rangle$",
          "consistency": "Pair inconsistency  $\\langle|E_{a,\\mathrm{fwd}} - E_{a,\\mathrm{rev}} - \\Delta E|\\rangle$"}
for ax, metric in zip(axes, ("antisym", "consistency")):
    ticks, labels = [], []
    for k, label in enumerate(MODELS):
        for j, split in enumerate(("strict", "paper")):
            y = k * 2.6 + j
            v = full[split][label][metric].mean()
            ax.barh(y, v, height=0.8, color=COLOR[label], left=1e-8, zorder=3)
            ax.text(v * 1.6, y, f"{v:.1e}" if v < 1e-3 else f"{v:.1f}", va="center", fontsize=8.5, color=INK2)
            ticks.append(y)
            labels.append(f"{label} · {split}")
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e3)
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_title(titles[metric], fontsize=10.5)
    ax.set_xlabel("kcal/mol (log scale)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
fig.suptitle("Consistency between the two directions of a reaction (RDB7 test sets, mean over 5 folds)",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.text(0.01, -0.04, "parity_paired is exact by construction; its residual is float32 round-off. "
         "stock_aug symmetrized after training is also exact (0).", color=INK2, fontsize=9)
fig.savefig(out / "consistency.png", dpi=160, bbox_inches="tight")

# 3. learning curve ------------------------------------------------------------------------------------------
FRACS = [0.1, 0.25, 0.5, 1.0]
lc = {}
for f in FRACS:
    tags = ((f"jpca_strict",) * 2 + ("jpca_strict_v2",) if f == 1.0
            else (None, f"lcurve_strict_md/frac{f:g}", f"lcurve_strict_md/frac{f:g}"))
    res = {}
    for label, tag, model, which in (("stock_aug", tags[1], "stock_aug", 0),
                                     ("stock_aug, symmetrized", tags[1], "stock_aug", 1),
                                     ("parity_paired", tags[2], "parity_paired", 0)):
        rs = [r[which] for r in runs(tag, model)]
        res[label] = {k: np.array([r[k] for r in rs]) for k in rs[0]}
    lc[f] = res
n_full = 10137  # training reactions per strict fold
x = np.array(FRACS) * 100
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True, sharex=True)
for row, (metric, name) in enumerate((("Ea", "Barrier MAE"), ("dE", "Reaction energy MAE"))):
    ax = axes[row, 0]
    for label in SHOWN:
        m = np.array([lc[f][label][metric].mean() for f in FRACS])
        ax.plot(x, m, color=COLOR[label], lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, label=label, zorder=3)
    ax.set_ylabel(f"{name} (kcal/mol)")
    ax.set_title(f"{name}, mean over 5 folds")
    ax.grid(color=GRID, lw=0.8)
    ax = axes[row, 1]
    ax.axhline(0, color=INK2, lw=1)
    for label, dx in (("stock_aug", -0.03), ("stock_aug, symmetrized", 0.03)):
        d = np.array([lc[f]["parity_paired"][metric] - lc[f][label][metric] for f in FRACS])
        for i in range(5):
            ax.scatter(x * (1 + dx), d[:, i], s=22, color=COLOR[label], alpha=0.45, edgecolors="none", zorder=2)
        ax.plot(x * (1 + dx), d.mean(1), color=COLOR[label], lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5,
                label=f"parity_paired − {label}", zorder=3)
    ax.set_ylabel("difference (kcal/mol)")
    ax.set_title(f"{name}: parity_paired minus baseline (dots: folds)")
    ax.grid(color=GRID, lw=0.8)
for ax in axes[1]:
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{p:g}%\n({round(p / 100 * n_full):,})" for p in x])
    ax.minorticks_off()
    ax.set_xlabel("training set (reactions per fold)")
axes[0, 0].legend(fontsize=9)
axes[0, 1].legend(fontsize=9, loc="upper right")
for ax in axes[:, 1]:
    ax.text(0.98, 0.03, "below zero: parity_paired better", transform=ax.transAxes, ha="right", color=INK2,
            fontsize=8.5)
fig.suptitle("Learning curve on the RDB7 strict split (same subsets and number of updates for every model)",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.text(0.01, -0.02, "The fold-to-fold spread (about ±0.5 kcal/mol) is shared by all models, since it reflects how "
         "hard each fold's test set is; the paired per-fold differences at right separate the models.",
         color=INK2, fontsize=9)
fig.savefig(out / "learning_curve.png", dpi=160, bbox_inches="tight")
# 4. parity plots --------------------------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

BLUES = LinearSegmentedColormap.from_list(  # reference sequential ramp, steps 100 -> 700
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def pooled(tag, model):
    """Predictions and targets of folds 0-4 pooled, with both directions of every test reaction."""
    P, Y = [], []
    for f in range(5):
        z = np.load(root / tag / f"{model}-f{f}" / "preds.npz")
        P.append(np.vstack([z["Pf"], z["Pr"]]))
        Y.append(np.vstack([z["Yf"], z["Yr"]]))
        # self-consistency: predicted Ea_fwd - Ea_rev against predicted dE, forward orientation
        P.append(None)
        Y.append(np.column_stack([z["Pf"][:, 0] - z["Pr"][:, 0], z["Pf"][:, 1]]))
    pred = np.vstack([p for p in P if p is not None])
    true = np.vstack([y for y, p in zip(Y, P) if p is not None])
    cons = np.vstack([y for y, p in zip(Y, P) if p is None])
    return pred, true, cons


for split in ("strict", "paper"):
    cols = [("stock", f"jpca_{split}", "stock"), ("stock_aug", f"jpca_{split}", "stock_aug"),
            ("parity_paired", f"jpca_{split}_v2", "parity_paired")]
    data = {label: pooled(tag, model) for label, tag, model in cols}
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.8), constrained_layout=True)
    rows = [("Barrier $E_a$", 0), ("Reaction energy $\\Delta E$", 1), ("Self-consistency", None)]
    norm = LogNorm(1, 400)
    for r, (row_name, col) in enumerate(rows):
        if col is not None:
            vals = np.concatenate([np.concatenate([d[0][:, col], d[1][:, col]]) for d in data.values()])
        else:
            vals = np.concatenate([d[2].ravel() for d in data.values()])
        lo, hi = np.percentile(vals, [0.1, 99.9])
        pad = 0.04 * (hi - lo)
        lim = (lo - pad, hi + pad)
        for c, (label, _, _) in enumerate(cols):
            ax = axes[r, c]
            pred, true, cons = data[label]
            if col is not None:
                xv, yv = true[:, col], pred[:, col]
                xlabel, ylabel = "reference (kcal/mol)", "predicted (kcal/mol)"
            else:
                xv, yv = cons[:, 1], cons[:, 0]
                xlabel = "predicted $\\Delta E$ (kcal/mol)"
                ylabel = "predicted $E_{a,\\mathrm{fwd}} - E_{a,\\mathrm{rev}}$"
            hb = ax.hexbin(xv, yv, gridsize=60, extent=(*lim, *lim), cmap=BLUES, norm=norm, mincnt=1,
                           linewidths=0.2, edgecolors=SURFACE, zorder=2)
            ax.plot(lim, lim, color=INK2, lw=1, zorder=3)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.grid(color=GRID, lw=0.8, zorder=0)
            e = yv - xv
            if col is not None:
                ss = 1 - (e**2).sum() / ((xv - xv.mean()) ** 2).sum()
                note = f"MAE {np.abs(e).mean():.2f}\nRMSE {np.sqrt((e**2).mean()):.2f}\n$R^2$ {ss:.3f}"
            else:
                note = f"mean |gap| {np.abs(e).mean():.2g}"
            ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", color=INK2, fontsize=9)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel if c == 0 else "", fontsize=9)
            ax.set_title(f"{label}: {row_name}", fontsize=10.5)
    cb = fig.colorbar(hb, ax=axes, shrink=0.35, location="right", pad=0.01)
    cb.set_label("predictions per hexagon (log scale)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Parity plots, RDB7 {split} split: test sets of the 5 folds pooled, both directions of every "
                 "reaction", x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, -0.015, "A reaction can be in more than one fold's test set. Bottom row: each model's own "
             "forward-minus-reverse barrier against its own reaction energy; the identity line is the "
             "thermodynamic constraint.", color=INK2, fontsize=9)
    fig.savefig(out / f"parity_{split}.png", dpi=150, bbox_inches="tight")

print("wrote", *sorted(p.name for p in out.glob("*.png")))
