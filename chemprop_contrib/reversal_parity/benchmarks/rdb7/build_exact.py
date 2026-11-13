"""Build exact forward/reverse ZPE-corrected CCSD(T)-F12a barriers for RDB7.

E0 = E_CCSD(T) + s * ZPE(wB97X-D3), with s fitted so that forward barriers reproduce the
published dE0. The reverse barrier of each reaction is E0(TS) - E0(P), with the same s.
Writes data/rdb7_exact.csv.
"""
import csv
import pathlib

import numpy as np

here = pathlib.Path(__file__).parent / "data"
cc = {int(r["idx"]): r for r in csv.DictReader(open(here / "cc_energies.csv"))}
zpe: dict[int, dict[str, float]] = {}
for r in csv.DictReader(open(here / "zpe.csv")):
    d = zpe.setdefault(int(r["idx"]), {"r": 0.0, "p": 0.0, "ts": 0.0})
    d[r["species"]] += float(r["zpe_kcal"])

idx = sorted(cc)
dEel_f = np.array([float(cc[i]["dEel_ts_r"]) for i in idx])
dEel_r = np.array([float(cc[i]["dEel_ts_p"]) for i in idx])
dZ_f = np.array([zpe[i]["ts"] - zpe[i]["r"] for i in idx])
dZ_r = np.array([zpe[i]["ts"] - zpe[i]["p"] for i in idx])
dE0 = np.array([float(cc[i]["dE0"]) for i in idx])

# least-squares scale factor for dE0 = dEel_f + s * dZ_f
s = float(np.dot(dZ_f, dE0 - dEel_f) / np.dot(dZ_f, dZ_f))
resid = dE0 - (dEel_f + s * dZ_f)
print(f"fitted ZPE scale factor s = {s:.5f}")
print(f"forward dE0 residual: max |r| = {np.abs(resid).max():.4f}, RMS = {np.sqrt((resid**2).mean()):.4f} kcal/mol")
for s_try in (0.975, 0.984, 0.986, 0.9864, 0.988):
    r_try = dE0 - (dEel_f + s_try * dZ_f)
    print(f"  s = {s_try}: max |r| = {np.abs(r_try).max():.4f}")

Ea_r = dEel_r + s * dZ_r
dE0_rxn = dE0 - Ea_r  # E0(P) - E0(R)
print(f"reverse barriers: min {Ea_r.min():.2f}, n negative {int((Ea_r < 0).sum())}")
dH = np.array([float(cc[i]["dHrxn298"]) for i in idx])
err = (dE0 - dH) - Ea_r
print(f"error of the old approximation dE0 - dH298 vs exact reverse barrier: MAE {np.abs(err).mean():.2f}, max {np.abs(err).max():.2f}")

with open(here / "rdb7_exact.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["idx", "rsmi", "psmi", "dE0_fwd", "dE0_rev", "dE0_rxn", "dHrxn298"])
    for k, i in enumerate(idx):
        w.writerow([i, cc[i]["rsmi"], cc[i]["psmi"], f"{dE0[k]:.5f}", f"{Ea_r[k]:.5f}",
                    f"{dE0_rxn[k]:.5f}", cc[i]["dHrxn298"]])
print(f"wrote {len(idx)} reactions")
