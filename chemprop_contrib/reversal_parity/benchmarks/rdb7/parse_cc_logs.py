"""Parse CCSD(T)-F12a energies from the RDB7 Molpro logs (untrusted data: read as text only).

Usage: python -I parse_cc_logs.py QM_LOGS_DIR CSV OUT_CSV
"""
import csv
import pathlib
import re
import sys

import numpy as np

HARTREE = 627.5094740631  # kcal/mol
logs, csv_in, out = map(pathlib.Path, sys.argv[1:4])
pat = re.compile(r"^\s*!CCSD\(T\)-F12a total energy\s+(-?\d+\.\d+)\s*$")

rows = list(csv.DictReader(open(csv_in, newline="")))
records, problems = [], []
for r in rows:
    i = int(r["idx"])
    d = logs / f"rxn{i:06d}"
    E = {}
    for tag in ("r", "p", "ts"):
        # bimolecular products are split into one log per fragment: p000002_0.log, p000002_1.log
        files = [d / f"{tag}{i:06d}.log"] if (d / f"{tag}{i:06d}.log").is_file() else \
            sorted(d.glob(f"{tag}{i:06d}_*.log"))
        if not files:
            problems.append((i, tag, "missing"))
            E[tag] = None
            continue
        total = 0.0
        for f in files:
            vals = [float(m.group(1)) for line in f.read_text(errors="replace").splitlines()
                    if (m := pat.match(line))]
            if len(vals) != 1:
                problems.append((i, f.name, f"{len(vals)} energies"))
                total = None
                break
            total += vals[0]
        E[tag] = total
    if any(E.get(t) is None for t in ("r", "p", "ts")):
        continue
    records.append({
        "idx": i, "rsmi": r["rsmi"], "psmi": r["psmi"],
        "dE0": float(r["dE0"]), "dHrxn298": float(r["dHrxn298"]),
        "dEel_ts_r": (E["ts"] - E["r"]) * HARTREE,
        "dEel_ts_p": (E["ts"] - E["p"]) * HARTREE,
        "dEel_rxn": (E["p"] - E["r"]) * HARTREE,
        "n_prod_frags": r["psmi"].count(".") + 1,
    })

print(f"reactions: {len(rows)}, parsed: {len(records)}, problems: {len(problems)}")
for p in problems[:10]:
    print("  ", p)
zpe = np.array([x["dE0"] - x["dEel_ts_r"] for x in records])
print(f"dE0 - dEel(TS-R) [ZPE part of forward barrier]: mean {zpe.mean():.2f}, sd {zpe.std():.2f}, "
      f"min {zpe.min():.2f}, max {zpe.max():.2f}")
gap = np.array([x["dHrxn298"] - x["dEel_rxn"] for x in records])
print(f"dHrxn298 - dEel(P-R): mean {gap.mean():.2f}, sd {gap.std():.2f}, min {gap.min():.2f}, max {gap.max():.2f}")
for k in (1, 2, 3):
    sel = [x for x in records if x["n_prod_frags"] == k]
    if sel:
        g = np.array([x["dHrxn298"] - x["dEel_rxn"] for x in sel])
        print(f"  {k} product fragment(s): n={len(sel)}, dH-dEel mean {g.mean():.2f}, sd {g.std():.2f}")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
