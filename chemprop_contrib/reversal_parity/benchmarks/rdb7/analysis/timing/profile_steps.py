"""Where one training epoch of bench2.py's stock_aug and parity_paired models spends its time.

Same data, models and batch size as /home/charlles/data/rdb7/bench2.py (strict split, fold 0), but a
hand-written training loop with a timer around each stage. GPU stages are timed twice: with a
synchronize after each stage (wall time the stage really takes) and, in a second epoch, without any
synchronize, to get the CPU-side time of the whole step (what the Python thread is busy with).

Usage: python profile_steps.py KIND [EPOCHS]        KIND: stock_aug | parity_paired
"""
import csv
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import torch
from chemprop.data import ReactionDatapoint, ReactionDataset, build_dataloader
from chemprop.featurizers import CondensedGraphOfReactionFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, RegressionFFN, SumAggregation
from chemprop.nn.transforms import UnscaleTransform
from sklearn.preprocessing import StandardScaler

from chemprop_contrib.reversal_parity import ParityCGRFeaturizer, ParityEquivariantMPNN, ParityTargetTransform

kind = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
PARITY = kind.startswith("parity")
data = pathlib.Path("/home/charlles/data/rdb7")
rows = {int(r["idx"]): r for r in csv.DictReader(open(data / "data/rdb7_exact.csv"))}
sets = json.loads((data / "splits_strict.json").read_text())["0"]

items = [(f"{rows[i]['rsmi']}>>{rows[i]['psmi']}", float(rows[i]["dE0_fwd"]), float(rows[i]["dE0_rxn"]))
         for i in sets["train"]]
if kind == "stock_aug":
    items += [(f"{rows[i]['psmi']}>>{rows[i]['rsmi']}", float(rows[i]["dE0_rev"]), -float(rows[i]["dE0_rxn"]))
              for i in sets["train"]]
feat = ParityCGRFeaturizer() if PARITY else CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
train = ReactionDataset([ReactionDatapoint.from_smi(s, y=[ea, de], keep_h=True, add_h=False) for s, ea, de in items],
                        featurizer=feat)
train.cache = True
if PARITY:
    target = ParityTargetTransform.fit(train.Y, paired_difference={0: 1})
    scaler = target.to_standard_scaler()
else:
    scaler = StandardScaler().fit(train.Y)
train.normalize_targets(scaler)

torch.manual_seed(0)
mp = BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=900, depth=4, activation="leakyrelu")
pred = RegressionFFN(n_tasks=2, input_dim=900, hidden_dim=900, n_layers=1, activation="leakyrelu",
                     output_transform=None if PARITY else UnscaleTransform.from_standard_scaler(scaler))
if PARITY:
    model = ParityEquivariantMPNN.from_featurizer(feat, mp, SumAggregation(), pred, target)
else:
    model = MPNN(mp, SumAggregation(), pred)
dev = torch.device("cuda")
model.to(dev).train()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
tl = build_dataloader(train, batch_size=50, num_workers=0, seed=0)


def to_dev(batch):
    bmg, V_d, X_d, *rest = batch
    bmg.to(dev)
    return (bmg, V_d, X_d, *[None if t is None else t.to(dev) for t in rest])


def loss_of(batch):
    if PARITY:
        return model.criterion(*model._loss_inputs(batch))
    bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
    mask = targets.isfinite()
    preds = model.predictor.train_step(model.fingerprint(bmg, V_d, X_d))
    return model.criterion(preds, targets.nan_to_num(nan=0.0), mask, weights, lt_mask, gt_mask)


def epoch(sync):
    wait = torch.cuda.synchronize if sync else (lambda: None)
    T = dict(collate=0.0, to_gpu=0.0, forward=0.0, backward=0.0, optim=0.0)
    n = atoms = edges = 0
    it = iter(tl)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    while True:
        t = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        T["collate"] += time.perf_counter() - t
        t = time.perf_counter(); batch = to_dev(batch); wait(); T["to_gpu"] += time.perf_counter() - t
        atoms += batch[0].V.shape[0]; edges += batch[0].E.shape[0]
        t = time.perf_counter(); loss = loss_of(batch); wait(); T["forward"] += time.perf_counter() - t
        t = time.perf_counter(); opt.zero_grad(); loss.backward(); wait(); T["backward"] += time.perf_counter() - t
        t = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step(); wait()
        T["optim"] += time.perf_counter() - t
        n += 1
    torch.cuda.synchronize()
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    return n, atoms, edges, wall, cpu, T


epoch(True)  # warm-up (CUDA init, dataset cache)
print(f"{kind}: threads {torch.get_num_threads()}, d_v {feat.atom_fdim}, d_e {feat.bond_fdim}, "
      f"{len(train)} datapoints")
for sync in [True] * (epochs - 1) + [False]:
    n, atoms, edges, wall, cpu, T = epoch(sync)
    stages = "  ".join(f"{k} {v:5.2f}s ({1e3 * v / n:5.2f} ms/step)" for k, v in T.items())
    print(f"sync={sync!s:5s} steps {n}  atoms/step {atoms / n:6.0f}  wall {wall:5.2f}s  "
          f"process CPU {cpu:5.2f}s\n   {stages}")
