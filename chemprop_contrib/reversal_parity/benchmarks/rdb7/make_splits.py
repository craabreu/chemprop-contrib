"""Scaffold splits following Spiekermann, Pattanaik & Green, J. Phys. Chem. A 2022, 126, 3976.

Chemprop v1 'scaffold_balanced' split (their fork kspieks/chemprop@barrier_prediction,
chemprop/data/scaffold.py) on the forward-reaction reactant, 85/5/10, seeds 0-4 for five
folds. Each reverse reaction joins its forward reaction's set, so pairs never straddle sets.

--scaffold paper : Murcko scaffold of the atom-mapped reactant, as their fork computes it
--scaffold strict: atom maps cleared first, so a scaffold is a scaffold regardless of mapping
Writes splits_<scaffold>.json: {fold: {"train": [...], "val": [...], "test": [...]}} of RDB7 idx.
"""
import argparse
import csv
import json
import pathlib
from collections import defaultdict
from random import Random

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
p = argparse.ArgumentParser()
p.add_argument("--scaffold", choices=["paper", "strict"], default="paper")
args = p.parse_args()
here = pathlib.Path(__file__).parent
rows = list(csv.DictReader(open(here / "data/rdb7_exact.csv")))
idx = [int(r["idx"]) for r in rows]


def scaffold(smi):
    if args.scaffold == "paper":  # their make_mol(s, keep_h=True)
        m = Chem.MolFromSmiles(smi, sanitize=False)
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    else:
        m = Chem.MolFromSmiles(smi)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        m = Chem.RemoveHs(m)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m)


scaf_to_pos = defaultdict(set)
for pos, r in enumerate(rows):
    scaf_to_pos[scaffold(r["rsmi"])].add(pos)

sizes = (0.85, 0.05, 0.10)
n = len(rows)
train_size, val_size, test_size = (s * n for s in sizes)
folds = {}
for seed in range(5):
    # chemprop v1 scaffold_split(balanced=True), verbatim logic
    big, small = [], []
    for s in scaf_to_pos.values():
        (big if len(s) > val_size / 2 or len(s) > test_size / 2 else small).append(s)
    random = Random(seed)
    random.seed(seed)
    random.shuffle(big)
    random.shuffle(small)
    train, val, test = [], [], []
    for s in big + small:
        if len(train) + len(s) <= train_size:
            train += s
        elif len(val) + len(s) <= val_size:
            val += s
        else:
            test += s
    folds[seed] = {k: sorted(idx[i] for i in v) for k, v in (("train", train), ("val", val), ("test", test))}
    print(f"fold {seed}: train {len(train)}, val {len(val)}, test {len(test)}")
print(f"{len(scaf_to_pos)} scaffolds ({args.scaffold})")
(here / f"splits_{args.scaffold}.json").write_text(json.dumps(folds))
