"""How do RDB7 forward-reactant scaffolds group with and without atom maps?"""
import csv
import pathlib
from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
rows = list(csv.DictReader(open(pathlib.Path(__file__).parent / "data/rdb7_exact.csv")))


def mol_keep_h(s):  # the fork's make_mol(s, keep_h=True)
    m = Chem.MolFromSmiles(s, sanitize=False)
    Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    return m


def stripped(s):
    m = Chem.MolFromSmiles(s)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.RemoveHs(m)


variants = {
    "paper fork (mapped, explicit H)": lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=mol_keep_h(s)),
    "maps cleared, H removed": lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=stripped(s)),
}
for name, fn in variants.items():
    c = Counter(fn(r["rsmi"]) for r in rows)
    top = c.most_common(3)
    print(f"{name}: {len(c)} scaffolds; largest sets {[(k[:40], v) for k, v in top]}")
