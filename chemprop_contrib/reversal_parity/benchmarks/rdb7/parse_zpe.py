"""Stream ZPEs out of the RDB7 wB97X-D3 Q-Chem archive without extracting it.

Logs are untrusted data: they are read as text and only one number per file is kept.
Usage: python -I parse_zpe.py WB97XD3_TAR_GZ OUT_CSV
"""
import csv
import re
import sys
import tarfile

tar_path, out = sys.argv[1:3]
name_pat = re.compile(r"^qm_logs/rxn(\d{6})/(r|p|ts)\d{6}(?:_(\d+))?\.log$")
zpe_pat = re.compile(rb"Zero point vibrational energy:\s+(-?\d+\.\d+)\s+kcal/mol")

rows, n_freq_missing = [], 0
with tarfile.open(tar_path, mode="r|gz") as tf:
    for m in tf:
        if not m.isfile():
            continue
        g = name_pat.match(m.name)
        if not g:
            continue
        data = tf.extractfile(m).read()
        vals = zpe_pat.findall(data)
        if not vals:
            n_freq_missing += 1
        rows.append({"idx": int(g[1]), "species": g[2], "frag": int(g[3]) if g[3] else 0,
                     "zpe_kcal": float(vals[-1]) if vals else "", "n_zpe_lines": len(vals)})

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["idx"], r["species"], r["frag"])))
print(f"logs: {len(rows)}, without a ZPE line: {n_freq_missing}")
