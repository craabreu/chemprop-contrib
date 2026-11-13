"""Learning-curve variant of bench2.py: trains on a random fraction of the training set.

Differences from bench2.py (generated from it by make_bench_lc.py):
- trains on the first round(FRACTION * n) reactions of a seeded permutation of the fold's training
  set (seed 1000 + fold), so the subsets are nested across fractions and identical for all models;
  stock_aug adds the reverses of the same reactions. Validation and test sets are the full ones.
- constant number of gradient updates: with K = round(1 / FRACTION), trains for EPOCHS * K epochs,
  warms up for 5 * K epochs and validates every K epochs, so the learning-rate schedule and the
  number of validation checks match the full-data run update for update.
- imports chemprop_contrib from an export of commit 42887e4 (the (m, d) feature basis), not from
  the worktree, and records the package path in result.json
Usage: python bench_lc.py MODEL FOLD SPLIT EPOCHS FRACTION OUTDIR

Original docstring:

RDB7 benchmark on the protocol of Spiekermann, Pattanaik & Green, JPCA 2022 (CCSD(T) only).

- exact forward and reverse ZPE-corrected barriers (data/rdb7_exact.csv)
- chemprop v1 balanced scaffold split on the forward reactant, 85/5/10, folds 0-4,
  forward/reverse pairs kept in the same set (make_splits.py)
- their architecture and training settings (SI Tables S4, S5): hidden 900, depth 4,
  LeakyReLU, sum aggregation, Noam schedule 1e-4 -> 1e-3 -> 1e-5 with 5 warm-up epochs,
  gradient clip 10, batch size 50
- not reproduced: ring-size features, dH as an FFN input, DFT pretraining, ensembling

Tasks: Ea (forward barrier dE0) and dE_rxn = E0(P) - E0(R), kcal/mol.
  stock          chemprop CGR, forward reactions only, [Ea, dE_rxn]
  stock_aug      chemprop CGR, forward + reverse reactions (the paper's approach), [Ea, dE_rxn]
  parity         ParityEquivariantMPNN, forward only, [Ea FREE, dE_rxn ODD]
  parity_marcus  ParityEquivariantMPNN, forward only, [E_int EVEN, dE_rxn ODD],
                 E_int = (Ea_fwd + Ea_rev) / 2, Ea = E_int + dE_rxn / 2 (exact)
  parity_paired  ParityEquivariantMPNN, forward only, [Ea paired with dE_rxn, dE_rxn ODD]:
                 the same decomposition inside the model, trained on Ea itself
Usage: python bench2.py MODEL FOLD SPLIT EPOCHS OUTDIR
"""
import csv
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/claude-1000/-home-charlles-github-repos-chemprop-contrib--claude-worktrees-"
                "reaction-parity/d614dc31-7a4a-4fe1-9021-e4ced7195e61/scratchpad/lcurve/md_42887e4")
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import torch
from chemprop.data import ReactionDatapoint, ReactionDataset, build_dataloader
from chemprop.featurizers import CondensedGraphOfReactionFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, RegressionFFN, SumAggregation
from chemprop.nn.transforms import UnscaleTransform
from sklearn.preprocessing import StandardScaler

from chemprop_contrib.reversal_parity import ParityCGRFeaturizer, ParityEquivariantMPNN, ParityTargetTransform

kind, fold, split, epochs, frac, outdir = (sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),
                                            float(sys.argv[5]), pathlib.Path(sys.argv[6]))
K = round(1 / frac)
outdir.mkdir(parents=True, exist_ok=True)
here = pathlib.Path("/home/charlles/data/rdb7")
PARITY, MARCUS = kind.startswith("parity"), kind == "parity_marcus"

rows = {int(r["idx"]): r for r in csv.DictReader(open(here / "data/rdb7_exact.csv"))}
sets = json.loads((here / f"splits_{split}.json").read_text())[fold]
n_full = len(sets["train"])
perm = np.random.default_rng(1000 + int(fold)).permutation(n_full)
sets["train"] = [sets["train"][j] for j in perm[: round(frac * n_full)]]


def fwd(i):
    r = rows[i]
    return f"{r['rsmi']}>>{r['psmi']}", float(r["dE0_fwd"]), float(r["dE0_rxn"]), float(r["dE0_rev"])


def rev(i):
    r = rows[i]
    return f"{r['psmi']}>>{r['rsmi']}", float(r["dE0_rev"]), -float(r["dE0_rxn"]), float(r["dE0_fwd"])


feat = ParityCGRFeaturizer() if PARITY else CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")


def dset(items, train_targets=False):
    # (smiles, Ea, dE_rxn, Ea of the opposite direction)
    if train_targets and MARCUS:
        ys = [[(ea + ea_o) / 2, de] for _, ea, de, ea_o in items]
    else:
        ys = [[ea, de] for _, ea, de, _ in items]
    dps = [ReactionDatapoint.from_smi(it[0], y=y, keep_h=True, add_h=False) for it, y in zip(items, ys)]
    d = ReactionDataset(dps, featurizer=feat)
    d.cache = True
    return d


tr_items = [fwd(i) for i in sets["train"]]
va_items = [fwd(i) for i in sets["val"]]
if kind == "stock_aug":
    tr_items += [rev(i) for i in sets["train"]]
    va_items += [rev(i) for i in sets["val"]]
train, val = dset(tr_items, True), dset(va_items, True)
test_f, test_r = dset([fwd(i) for i in sets["test"]]), dset([rev(i) for i in sets["test"]])

if PARITY:
    spec = {"parity": dict(odd=[1]), "parity_marcus": dict(odd=[1], even=[0]),
            "parity_paired": dict(paired_difference={0: 1})}[kind]
    target = ParityTargetTransform.fit(train.Y, **spec)
    scaler = target.to_standard_scaler()
else:
    scaler = StandardScaler().fit(train.Y)
train.normalize_targets(scaler)
val.normalize_targets(scaler)

torch.manual_seed(0)
mp = BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=900, depth=4, activation="leakyrelu")
pred = RegressionFFN(n_tasks=2, input_dim=900, hidden_dim=900, n_layers=1, activation="leakyrelu",
                     output_transform=None if PARITY else UnscaleTransform.from_standard_scaler(scaler))
lr = dict(warmup_epochs=5 * K, init_lr=1e-4, max_lr=1e-3, final_lr=1e-5)
if PARITY:
    model = ParityEquivariantMPNN.from_featurizer(feat, mp, SumAggregation(), pred, target, **lr)
else:
    model = MPNN(mp, SumAggregation(), pred, **lr)

tl = build_dataloader(train, batch_size=50, num_workers=0, seed=0)
vl = build_dataloader(val, batch_size=50, num_workers=0, shuffle=False)
ckpt = ModelCheckpoint(dirpath=outdir / "ckpt", monitor="val_loss", mode="min", save_top_k=1)
trainer = pl.Trainer(max_epochs=epochs * K, check_val_every_n_epoch=K, accelerator="gpu", devices=1, logger=False, callbacks=[ckpt],
                     gradient_clip_val=10.0, enable_progress_bar=False, enable_model_summary=False)
t0 = time.time()
trainer.fit(model, tl, vl)
train_time = time.time() - t0
best = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)  # our own file
model.load_state_dict(best["state_dict"])


def predict(d):
    out = trainer.predict(model, build_dataloader(d, batch_size=256, num_workers=0, shuffle=False))
    P = torch.cat(out).numpy().reshape(-1, 2).astype(float)
    if MARCUS:
        P[:, 0] += P[:, 1] / 2
    return P


Pf, Pr = predict(test_f), predict(test_r)
Yf, Yr = test_f.Y, test_r.Y
mae = lambda a, b: float(np.mean(np.abs(a - b)))
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
Pa, Ya = np.vstack([Pf, Pr]), np.vstack([Yf, Yr])
res = {
    "model": kind, "fold": int(fold), "split": split, "best_epoch": int(best["epoch"]),
    "fraction": frac, "n_train_reactions": len(sets["train"]), "epochs_run": epochs * K, "val_every": K,
    "train_time_s": round(train_time, 1),
    "package": sys.modules["chemprop_contrib.reversal_parity.featurizers"].__file__,
    # the paper's test sets contain both directions, so 'all' is the comparable number
    "all_Ea_MAE": mae(Pa[:, 0], Ya[:, 0]), "all_Ea_RMSE": rmse(Pa[:, 0], Ya[:, 0]),
    "fwd_Ea_MAE": mae(Pf[:, 0], Yf[:, 0]), "rev_Ea_MAE": mae(Pr[:, 0], Yr[:, 0]),
    "all_dE_MAE": mae(Pa[:, 1], Ya[:, 1]), "all_dE_RMSE": rmse(Pa[:, 1], Ya[:, 1]),
    "antisym_dE_mean": float(np.mean(np.abs(Pf[:, 1] + Pr[:, 1]))),
    "consistency_MAE": mae(Pf[:, 0] - Pr[:, 0], Pf[:, 1]),
}
np.savez(outdir / "preds.npz", Pf=Pf, Pr=Pr, Yf=Yf, Yr=Yr, test_idx=np.array(sets["test"]))
(outdir / "result.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
